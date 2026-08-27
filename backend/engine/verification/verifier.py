"""
verification/verifier.py
===========================
Independent Verifier.

Deliberately separate from the Evaluator. Where the Evaluator computes
metrics, the Verifier interrogates *whether those metrics can be
trusted*: does the train/test gap suggest overfitting, is there feature-
target leakage, is validation methodology sound, are results stable
across seeds, is data quality good enough to support the claimed
performance. Each check returns pass/warn/fail plus evidence, feeding
directly into the Pipeline Health Engine and the Failure Diagnoser.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

OVERFIT_GAP_WARN = 0.08
OVERFIT_GAP_FAIL = 0.18
LEAKAGE_CORR_WARN = 0.95


class IndependentVerifier:
    def verify(
        self,
        df: pd.DataFrame,
        target_column: str,
        task_type: str,
        profile: Dict[str, Any],
        metrics: Dict[str, Any],
        best_model_name: str,
        reproducibility: Dict[str, Any] | None,
    ) -> Dict[str, Any]:
        checks: Dict[str, Any] = {}

        checks["data_quality"] = self._check_data_quality(profile)
        checks["overfit_gap"] = self._check_overfit_gap(metrics, best_model_name)
        checks["leakage"] = self._check_leakage(df, target_column, task_type)
        checks["validation_soundness"] = self._check_validation(metrics, best_model_name)
        checks["reproducibility"] = self._check_reproducibility(reproducibility)
        if task_type == "classification":
            checks["class_balance"] = self._check_class_balance(df, target_column)

        n_fail = sum(1 for c in checks.values() if c["status"] == "fail")
        n_warn = sum(1 for c in checks.values() if c["status"] == "warn")
        overall = "fail" if n_fail > 0 else ("warn" if n_warn > 0 else "pass")

        return {
            "checks": checks,
            "overall_status": overall,
            "n_fail": n_fail,
            "n_warn": n_warn,
        }

    # ------------------------------------------------------------------
    def _check_data_quality(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        total_missing = profile.get("total_missing_pct", 0)
        status = "pass"
        if total_missing > 15:
            status = "fail"
        elif total_missing > 5:
            status = "warn"
        return {"status": status, "evidence": {"total_missing_pct": total_missing}}

    def _check_overfit_gap(self, metrics: Dict[str, Any], best_model_name: str) -> Dict[str, Any]:
        m = metrics.get(best_model_name, {})
        gap = abs(m.get("overfit_gap", 0)) if m.get("status") == "ok" else None
        if gap is None:
            return {"status": "fail", "evidence": {"reason": "no valid metrics for best model"}}
        status = "pass"
        if gap >= OVERFIT_GAP_FAIL:
            status = "fail"
        elif gap >= OVERFIT_GAP_WARN:
            status = "warn"
        return {"status": status, "evidence": {"train_test_gap": round(gap, 4)}}

    def _check_leakage(self, df: pd.DataFrame, target_column: str, task_type: str) -> Dict[str, Any]:
        if task_type != "regression":
            return {"status": "pass", "evidence": {"reason": "leakage-by-correlation only checked for regression targets"}}
        numeric_df = df.select_dtypes(include=[np.number])
        if target_column not in numeric_df.columns or numeric_df.shape[1] < 2:
            return {"status": "pass", "evidence": {}}
        corr = numeric_df.corr(numeric_only=True)[target_column].drop(target_column, errors="ignore")
        suspicious = corr[corr.abs() >= LEAKAGE_CORR_WARN]
        status = "fail" if len(suspicious) > 0 else "pass"
        return {
            "status": status,
            "evidence": {"suspicious_columns": suspicious.round(3).to_dict()},
        }

    def _check_validation(self, metrics: Dict[str, Any], best_model_name: str) -> Dict[str, Any]:
        m = metrics.get(best_model_name, {})
        cv = m.get("cv", {}) if m.get("status") == "ok" else {}
        if not cv or cv.get("mean") is None:
            return {"status": "warn", "evidence": {"reason": "cross-validation unavailable"}}
        std = cv.get("std", 0) or 0
        status = "pass" if std < 0.15 else ("warn" if std < 0.3 else "fail")
        return {"status": status, "evidence": {"cv_mean": cv.get("mean"), "cv_std": std}}

    def _check_reproducibility(self, reproducibility: Dict[str, Any] | None) -> Dict[str, Any]:
        if not reproducibility:
            return {"status": "warn", "evidence": {"reason": "reproducibility check not run"}}
        status = "pass" if reproducibility.get("stable") else "fail"
        return {"status": status, "evidence": reproducibility}

    def _check_class_balance(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        vc = df[target_column].astype(str).value_counts(normalize=True)
        if len(vc) == 0:
            return {"status": "warn", "evidence": {}}
        max_share = float(vc.max())
        status = "fail" if max_share > 0.9 else ("warn" if max_share > 0.75 else "pass")
        return {"status": status, "evidence": {"majority_class_share": round(max_share, 3)}}
