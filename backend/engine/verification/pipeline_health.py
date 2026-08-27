"""
verification/pipeline_health.py
==================================
Pipeline Health Engine.

Combines data quality, model performance, validation soundness,
robustness, and reproducibility into a single 0-100 Pipeline Health
Score. This score is what the Stop Controller checks against the
acceptance threshold, and what the "Validation / Pipeline Health"
dashboard tab visualizes.

Weights (sum to 1.0):
    data_quality        0.20
    model_performance    0.35
    validation           0.20
    robustness            0.15   (overfit gap + leakage checks)
    reproducibility       0.10
"""

from __future__ import annotations

from typing import Any, Dict

WEIGHTS = {
    "data_quality": 0.20,
    "model_performance": 0.35,
    "validation": 0.20,
    "robustness": 0.15,
    "reproducibility": 0.10,
}

STATUS_SCORE = {"pass": 100.0, "warn": 55.0, "fail": 10.0}


class PipelineHealthEngine:
    def compute(
        self,
        verification_report: Dict[str, Any],
        metrics: Dict[str, Any],
        best_model_name: str,
        task_type: str,
    ) -> Dict[str, Any]:
        checks = verification_report.get("checks", {})

        data_quality = STATUS_SCORE.get(checks.get("data_quality", {}).get("status"), 50.0)
        validation = STATUS_SCORE.get(checks.get("validation_soundness", {}).get("status"), 50.0)
        reproducibility = STATUS_SCORE.get(checks.get("reproducibility", {}).get("status"), 50.0)

        robustness_components = [
            STATUS_SCORE.get(checks.get("overfit_gap", {}).get("status"), 50.0),
            STATUS_SCORE.get(checks.get("leakage", {}).get("status"), 50.0),
        ]
        if "class_balance" in checks:
            robustness_components.append(STATUS_SCORE.get(checks["class_balance"]["status"], 50.0))
        robustness = sum(robustness_components) / len(robustness_components)

        model_performance = self._model_performance_score(metrics, best_model_name, task_type)

        overall = (
            data_quality * WEIGHTS["data_quality"]
            + model_performance * WEIGHTS["model_performance"]
            + validation * WEIGHTS["validation"]
            + robustness * WEIGHTS["robustness"]
            + reproducibility * WEIGHTS["reproducibility"]
        )

        return {
            "overall_score": round(overall, 2),
            "components": {
                "data_quality": round(data_quality, 2),
                "model_performance": round(model_performance, 2),
                "validation": round(validation, 2),
                "robustness": round(robustness, 2),
                "reproducibility": round(reproducibility, 2),
            },
            "weights": WEIGHTS,
        }

    @staticmethod
    def _model_performance_score(metrics: Dict[str, Any], best_model_name: str, task_type: str) -> float:
        m = metrics.get(best_model_name, {})
        if m.get("status") != "ok":
            return 0.0
        test = m["test"]
        if task_type == "classification":
            raw = test.get("f1_macro", 0.0)
        else:
            raw = max(0.0, test.get("r2", 0.0))
        return float(min(1.0, max(0.0, raw)) * 100)
