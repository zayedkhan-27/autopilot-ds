"""
modeling/evaluator.py
========================
Evaluator.

Computes standard, deterministic metrics for every trained candidate and
ranks them. This is intentionally dumb and mechanical — no LLM involved —
so results are always trustworthy and reproducible. The Independent
Verifier (a separate module) re-checks these numbers rather than trusting
them blindly.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


class Evaluator:
    def evaluate_all(self, trained: Dict[str, Any], task_type: str) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}
        for name, res in trained.items():
            if res.get("status") != "ok":
                metrics[name] = {"status": "failed", "error": res.get("error")}
                continue
            metrics[name] = self._evaluate_one(res, task_type)
        return metrics

    def rank(self, metrics: Dict[str, Any], task_type: str) -> list:
        primary = "f1_macro" if task_type == "classification" else "r2"
        valid = {k: v for k, v in metrics.items() if v.get("status") == "ok"}
        return sorted(valid.keys(), key=lambda k: valid[k]["test"][primary], reverse=True)

    # ------------------------------------------------------------------
    def _evaluate_one(self, res: Dict[str, Any], task_type: str) -> Dict[str, Any]:
        if task_type == "classification":
            train_m = self._classification_metrics(res["y_train"], res["y_pred_train"])
            test_m = self._classification_metrics(res["y_test"], res["y_pred_test"])
            primary = "f1_macro"
        else:
            train_m = self._regression_metrics(res["y_train"], res["y_pred_train"])
            test_m = self._regression_metrics(res["y_test"], res["y_pred_test"])
            primary = "r2"

        gap = train_m[primary] - test_m[primary]
        return {
            "status": "ok",
            "train": train_m,
            "test": test_m,
            "cv": res.get("cv_scores", {}),
            "overfit_gap": round(float(gap), 4),
            "fit_time_sec": res.get("fit_time_sec"),
            "primary_metric_name": primary,
        }

    @staticmethod
    def _classification_metrics(y_true, y_pred) -> Dict[str, float]:
        out = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        }
        try:
            if len(np.unique(y_true)) == 2:
                out["roc_auc"] = float(roc_auc_score(y_true, y_pred))
        except Exception:
            pass
        return out

    @staticmethod
    def _regression_metrics(y_true, y_pred) -> Dict[str, float]:
        out = {
            "r2": float(r2_score(y_true, y_pred)),
            "rmse": float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
        }
        try:
            out["mape"] = float(mean_absolute_percentage_error(y_true, y_pred))
        except Exception:
            out["mape"] = None
        return out
