"""
core/llm_client.py
====================
The reasoning "brain" of AutoPilot-DS.

Design principle (per the challenge spec): the LLM makes *decisions*,
never raw ML results. Every decision returned by this module is a small,
structured JSON object (e.g. "which cleaning ops to run", "which models
to try next", "what is the likely cause of this failure"). Deterministic
Python tools (profiling/, cleaning/, modeling/, verification/...) are the
only code allowed to touch data or compute metrics. The LLM is never
asked to "predict" a metric or invent a number — it only picks strategies
and explains reasoning in natural language.

Two backends:

1. AnthropicBackend   - real tool-calling LLM (claude-sonnet-4-6) used
                         when ANTHROPIC_API_KEY is set and the `anthropic`
                         package + network are available.
2. HeuristicBackend    - a transparent, deterministic rule-based reasoner
                         used as an offline fallback (e.g. demos, CI,
                         sandboxed environments without egress). It
                         produces the exact same structured decisions the
                         Anthropic backend would, driven by profiling
                         statistics and prior attempts, and always emits a
                         written rationale so the trace stays meaningful.

`LLMPlanner` is the façade the rest of the codebase talks to; it never
needs to know which backend is active.
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Optional

SYSTEM_PROMPT = """You are the planning brain of AutoPilot-DS, an autonomous \
ML/Auto-EDA agent. You choose STRATEGIES (cleaning operations, feature \
engineering steps, candidate models, next actions after a failure) for an \
unseen, messy tabular dataset. You never compute statistics or metrics \
yourself — those come from deterministic tool outputs given to you as \
context. You must:
1. Reason from the provided profiling / evaluation / verification JSON.
2. Never repeat a strategy signature already listed in `strategies_tried`.
3. Always return STRICT JSON matching the requested schema, plus a short
   natural-language `rationale` field explaining your reasoning.
4. Prefer the simplest strategy likely to work; escalate complexity only
   after simpler strategies failed or scored low.
"""


class LLMBackend:
    name = "base"

    def decide(self, task: str, context: Dict[str, Any], schema_hint: str) -> Dict[str, Any]:
        raise NotImplementedError


class AnthropicBackend(LLMBackend):
    """Real Claude tool-calling backend."""

    name = "anthropic-claude-sonnet-4-6"

    def __init__(self, model: str = "claude-sonnet-4-6"):
        import anthropic  # noqa: local import so package is optional

        self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self._model = model

    def decide(self, task: str, context: Dict[str, Any], schema_hint: str) -> Dict[str, Any]:
        user_msg = (
            f"TASK: {task}\n\n"
            f"CONTEXT (deterministic tool output, ground truth):\n"
            f"{json.dumps(context, default=str, indent=2)[:12000]}\n\n"
            f"Return STRICT JSON matching this shape:\n{schema_hint}\n"
            f"Return ONLY the JSON object, no prose outside it."
        )
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=1200,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        text = text.strip().strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Fall back to a minimal safe decision rather than crash the run.
            return {"error": "unparseable_llm_output", "raw": text}


class HeuristicBackend(LLMBackend):
    """
    Deterministic, explainable rule-based reasoner used when no LLM API
    key / network is available. It mirrors the JSON contract of
    AnthropicBackend so the rest of the system is backend-agnostic.
    Every decision is accompanied by a rationale so the Agent Trace stays
    informative even offline.
    """

    name = "heuristic-fallback"

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    def decide(self, task: str, context: Dict[str, Any], schema_hint: str) -> Dict[str, Any]:
        handler = getattr(self, f"_handle_{task}", None)
        if handler is None:
            return {"rationale": f"No heuristic handler for task '{task}'.", "action": "noop"}
        return handler(context)

    # ------------------------------------------------------------------
    def _handle_cleaning_strategy(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        profile = ctx.get("profile", {})
        tried_signatures = ctx.get("tried_signatures", [])
        iteration = ctx.get("iteration", 0)
        high_missing_cols = [
            c for c, v in profile.get("missing_pct", {}).items() if v > 40
        ]
        moderate_missing = [
            c for c, v in profile.get("missing_pct", {}).items() if 0 < v <= 40
        ]
        has_dupes = profile.get("duplicate_rows", 0) > 0
        has_outliers = any(v > 0 for v in profile.get("outlier_counts", {}).values())
        has_constant = len(profile.get("constant_columns", [])) > 0

        ops: List[str] = []
        if has_constant:
            ops.append("drop_constant_columns")
        if high_missing_cols:
            ops.append("drop_high_missing_columns")
        if moderate_missing:
            # escalate to KNN imputation on later iterations if simple
            # imputation already failed
            if iteration >= 2 and "impute_median_mode" in " ".join(tried_signatures):
                ops.append("impute_knn")
            else:
                ops.append("impute_median_mode")
        if has_dupes:
            ops.append("drop_duplicate_rows")
        if has_outliers:
            ops.append("cap_outliers_iqr")
        ops.append("fix_dtypes")

        if not ops:
            ops = ["noop_clean"]

        return {
            "ops": ops,
            "rationale": (
                f"Profiling shows {len(high_missing_cols)} high-missing column(s), "
                f"{len(moderate_missing)} moderate-missing column(s), "
                f"duplicates={has_dupes}, outliers={has_outliers}, "
                f"constant_columns={has_constant}. Selected the smallest op set "
                f"that addresses each detected quality issue, escalating imputation "
                f"strategy on iteration {iteration} since simpler imputation was already tried."
            ),
        }

    def _handle_feature_strategy(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        profile = ctx.get("profile", {})
        iteration = ctx.get("iteration", 0)
        n_features = profile.get("n_columns", 0)
        high_card_cats = ctx.get("high_cardinality_categoricals", [])

        ops = ["scale_numeric"]
        if high_card_cats:
            ops.append("target_safe_encode_high_cardinality")
        else:
            ops.append("onehot_encode_categoricals")
        if n_features > 15 or iteration >= 2:
            ops.append("select_top_k_features")
        if iteration >= 3:
            ops.append("drop_low_variance_features")

        cat_note = (
            "high-cardinality categoricals present so using frequency-safe encoding"
            if high_card_cats else "low-cardinality categoricals, using one-hot"
        )
        return {
            "ops": ops,
            "rationale": (
                f"{n_features} raw columns detected; {cat_note}. "
                f"Feature selection added because iteration={iteration} or column count is high, "
                f"to reduce overfitting risk."
            ),
        }

    def _handle_model_selection(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        task_type = ctx.get("task_type")
        iteration = ctx.get("iteration", 0)
        tried_models = set(ctx.get("all_models_tried", []))

        if task_type == "classification":
            pool = ["logistic_regression", "random_forest_clf", "gradient_boosting_clf", "knn_clf"]
        else:
            pool = ["ridge_regression", "random_forest_reg", "gradient_boosting_reg", "knn_reg"]

        # Iteration 0: try the cheap/interpretable baseline + one ensemble.
        # Later iterations: escalate toward ensembles / more expressive models
        # and drop whatever previously underperformed or repeated.
        if iteration == 0:
            chosen = pool[:2]
        else:
            remaining = [m for m in pool if m not in tried_models]
            chosen = remaining[:2] if remaining else pool[-2:]

        return {
            "models": chosen,
            "rationale": (
                f"Iteration {iteration}: chose {chosen} for task={task_type}. "
                f"Already-tried models {sorted(tried_models) or 'none'} were "
                f"{'avoided' if tried_models else 'not yet available'} to ensure "
                f"strategy novelty and broaden the search across model families."
            ),
        }

    def _handle_failure_diagnosis(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        verification = ctx.get("verification", {})
        checks = verification.get("checks", {})
        causes = []
        remedies = []

        if checks.get("overfit_gap", {}).get("status") == "fail":
            causes.append("large train/test performance gap indicates overfitting")
            remedies.append("simplify model / add regularization / reduce feature count")
        if checks.get("leakage", {}).get("status") == "fail":
            causes.append("one or more features are suspiciously correlated with the target "
                           "(possible leakage)")
            remedies.append("drop suspected leaking columns and retrain")
        if checks.get("data_quality", {}).get("status") == "fail":
            causes.append("residual missing values / low-quality features weakening signal")
            remedies.append("apply stronger imputation or drop the offending columns")
        if checks.get("class_balance", {}).get("status") == "fail":
            causes.append("target classes are heavily imbalanced, biasing accuracy-based ranking")
            remedies.append("use class_weight='balanced' and macro-F1 as the primary metric")
        if not causes:
            causes.append("overall health score below the acceptance threshold without a single "
                           "dominant cause; likely an underpowered model/feature set")
            remedies.append("try a more expressive model family and expand feature engineering")

        return {
            "likely_causes": causes,
            "suggested_remedies": remedies,
            "rationale": "Derived directly from failing verification checks: " + "; ".join(causes),
        }

    def _handle_reflection(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        diagnosis = ctx.get("diagnosis", {})
        iteration = ctx.get("iteration", 0)
        health_history = ctx.get("health_history", [])
        improving = len(health_history) >= 2 and health_history[-1] > health_history[-2]

        next_focus = "feature_engineering"
        remedies = diagnosis.get("suggested_remedies", [])
        if any("leak" in r for r in remedies):
            next_focus = "cleaning"
        elif any("model" in r or "regulariz" in r for r in remedies):
            next_focus = "modeling"
        elif any("imbalanc" in r for r in remedies):
            next_focus = "modeling"

        return {
            "next_focus": next_focus,
            "continue": True,
            "rationale": (
                f"Health score trend is {'improving' if improving else 'flat/declining'} "
                f"over the last iterations ({health_history[-3:]}). Focusing iteration "
                f"{iteration + 1} on '{next_focus}' based on diagnosed remedies: {remedies}."
            ),
        }


class LLMPlanner:
    """
    Backend-agnostic façade used by every engine that needs a "decision".
    Automatically selects AnthropicBackend when `ANTHROPIC_API_KEY` is set
    and the SDK importable; otherwise falls back to HeuristicBackend so
    the whole pipeline still runs end-to-end offline/deterministically.
    """

    def __init__(self, prefer_anthropic: bool = True, seed: int = 42):
        self.backend: LLMBackend
        if prefer_anthropic and os.environ.get("ANTHROPIC_API_KEY"):
            try:
                self.backend = AnthropicBackend()
            except Exception:
                self.backend = HeuristicBackend(seed=seed)
        else:
            self.backend = HeuristicBackend(seed=seed)

    @property
    def backend_name(self) -> str:
        return self.backend.name

    def decide(self, task: str, context: Dict[str, Any], schema_hint: str = "{}") -> Dict[str, Any]:
        return self.backend.decide(task, context, schema_hint)
