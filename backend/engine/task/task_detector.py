"""
task/task_detector.py
========================
Task Detector.

Deterministically infers the target column and ML task type
(classification vs regression) from column names, dtypes, and
cardinality — while allowing a user override to take precedence.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

TARGET_NAME_HINTS = (
    "target", "label", "class", "y", "outcome", "result",
    "churn", "price", "score", "rating", "sales", "response",
)


class TaskDetector:
    CLASSIFICATION_MAX_UNIQUE_RATIO = 0.05
    CLASSIFICATION_MAX_UNIQUE_ABS = 20

    def detect(
        self,
        df: pd.DataFrame,
        target_override: Optional[str] = None,
        task_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        target = target_override if target_override in df.columns else self._infer_target(df)
        task_type = task_override or self._infer_task_type(df, target)

        meta: Dict[str, Any] = {
            "target_source": "user_override" if target_override in df.columns else "auto_detected",
            "task_source": "user_override" if task_override else "auto_detected",
        }

        series = df[target].dropna()
        if task_type == "classification":
            vc = series.astype(str).value_counts(normalize=True)
            meta["n_classes"] = int(series.nunique())
            meta["class_balance"] = vc.round(3).to_dict()
            meta["is_imbalanced"] = bool(vc.max() > 0.8) if len(vc) else False
        else:
            meta["target_mean"] = float(series.mean()) if len(series) else None
            meta["target_std"] = float(series.std()) if len(series) else None
            meta["target_skew"] = float(series.skew()) if len(series) > 2 else None

        return {"target_column": target, "task_type": task_type, "meta": meta}

    # ------------------------------------------------------------------
    def _infer_target(self, df: pd.DataFrame) -> str:
        cols_lower = {c: c.lower() for c in df.columns}
        for hint in TARGET_NAME_HINTS:
            for col, low in cols_lower.items():
                if hint == low or low.endswith(f"_{hint}") or low.startswith(f"{hint}_"):
                    return col
        # fallback: last column, excluding obvious ID-like columns
        candidates = [c for c in df.columns if "id" not in c.lower()]
        return candidates[-1] if candidates else df.columns[-1]

    def _infer_task_type(self, df: pd.DataFrame, target: str) -> str:
        series = df[target].dropna()
        n = len(series)
        if n == 0:
            return "classification"
        n_unique = series.nunique()
        if not pd.api.types.is_numeric_dtype(series):
            return "classification"
        unique_ratio = n_unique / n
        if n_unique <= self.CLASSIFICATION_MAX_UNIQUE_ABS and unique_ratio <= self.CLASSIFICATION_MAX_UNIQUE_RATIO:
            return "classification"
        if n_unique <= 10 and set(series.unique()) <= set(range(0, 11)):
            return "classification"
        return "regression"
