"""
core/strategy_selector.py
============================
Strategy Selector.

Bridges the LLM Planner's individual decisions (cleaning ops, feature
ops, model choices) into one coherent StrategyPlan for the iteration,
and guarantees novelty: if the assembled plan has already been tried
(per `StateManager.has_tried`), it perturbs the plan (adds an escalation
op / swaps a model) until it is distinct or a small retry budget is
exhausted.
"""

from __future__ import annotations

from typing import Any, Dict, List

from core.llm_client import LLMPlanner
from core.state import AgentState, StateManager

CLEANING_SCHEMA = """{"ops": ["string", ...], "rationale": "string"}"""
FEATURE_SCHEMA = """{"ops": ["string", ...], "rationale": "string"}"""
MODEL_SCHEMA = """{"models": ["string", ...], "rationale": "string"}"""


class StrategySelector:
    def __init__(self, planner: LLMPlanner):
        self.planner = planner

    def propose(self, state: AgentState) -> Dict[str, Any]:
        profile = state["profile_report"]
        iteration = state.get("iteration", 0)
        tried_signatures = [s["signature"] for s in state.get("strategies_tried", [])]

        cleaning_decision = self.planner.decide(
            "cleaning_strategy",
            {"profile": profile, "tried_signatures": tried_signatures, "iteration": iteration},
            CLEANING_SCHEMA,
        )
        feature_decision = self.planner.decide(
            "feature_strategy",
            {
                "profile": profile,
                "iteration": iteration,
                "high_cardinality_categoricals": profile.get("high_cardinality_categoricals", []),
            },
            FEATURE_SCHEMA,
        )
        all_models_tried: List[str] = []
        for s in state.get("strategies_tried", []):
            all_models_tried.extend(s["strategy"].get("models", []))

        model_decision = self.planner.decide(
            "model_selection",
            {
                "task_type": state.get("task_type"),
                "iteration": iteration,
                "all_models_tried": all_models_tried,
            },
            MODEL_SCHEMA,
        )

        strategy = {
            "cleaning_ops": cleaning_decision.get("ops", ["fix_dtypes"]),
            "feature_ops": feature_decision.get("ops", ["scale_numeric"]),
            "models": model_decision.get("models", []),
            "rationale": {
                "cleaning": cleaning_decision.get("rationale", ""),
                "feature": feature_decision.get("rationale", ""),
                "models": model_decision.get("rationale", ""),
            },
        }

        # novelty guarantee: escalate if this exact plan was already tried
        attempts = 0
        while StateManager.has_tried(state, strategy) and attempts < 3:
            strategy = self._perturb(strategy, state, attempts)
            attempts += 1

        return strategy

    # ------------------------------------------------------------------
    def _perturb(self, strategy: Dict[str, Any], state: AgentState, attempt: int) -> Dict[str, Any]:
        """Force novelty when the planner (heuristic or LLM) proposed a
        plan identical to a previous attempt."""
        strategy = dict(strategy)
        escalations = ["impute_knn", "cap_outliers_iqr", "drop_high_missing_columns"]
        for op in escalations:
            if op not in strategy["cleaning_ops"]:
                strategy["cleaning_ops"] = strategy["cleaning_ops"] + [op]
                break
        if "select_top_k_features" not in strategy["feature_ops"]:
            strategy["feature_ops"] = strategy["feature_ops"] + ["select_top_k_features"]

        task_type = state.get("task_type")
        from modeling.model_challenger import ModelChallenger

        pool = ModelChallenger().available_models(task_type)
        used = set(strategy["models"])
        for m in pool:
            if m not in used:
                strategy["models"] = strategy["models"] + [m]
                break
        strategy["rationale"]["perturbation"] = (
            f"Auto-perturbed on attempt {attempt + 1} because the proposed plan "
            f"duplicated a previously tried strategy signature."
        )
        return strategy
