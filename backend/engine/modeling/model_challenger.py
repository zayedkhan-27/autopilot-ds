"""
modeling/model_challenger.py
===============================
Model Challenger.

A deliberately small, curated model library (per the challenge spec: "a
small but effective model library rather than attempting to support every
ML algorithm"). The Strategy Selector / LLM Planner picks names from this
registry; this module is the only place that knows how to instantiate the
actual scikit-learn estimator, keeping model construction deterministic
and reproducible (fixed random_state everywhere).
"""

from __future__ import annotations

from typing import Any, Dict

from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

RANDOM_STATE = 42

CLASSIFICATION_REGISTRY: Dict[str, Any] = {
    "logistic_regression": lambda: LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
    ),
    "random_forest_clf": lambda: RandomForestClassifier(
        n_estimators=200, max_depth=None, class_weight="balanced", random_state=RANDOM_STATE
    ),
    "gradient_boosting_clf": lambda: GradientBoostingClassifier(random_state=RANDOM_STATE),
    "knn_clf": lambda: KNeighborsClassifier(n_neighbors=7),
}

REGRESSION_REGISTRY: Dict[str, Any] = {
    "ridge_regression": lambda: Ridge(alpha=1.0, random_state=RANDOM_STATE),
    "random_forest_reg": lambda: RandomForestRegressor(
        n_estimators=200, max_depth=None, random_state=RANDOM_STATE
    ),
    "gradient_boosting_reg": lambda: GradientBoostingRegressor(random_state=RANDOM_STATE),
    "knn_reg": lambda: KNeighborsRegressor(n_neighbors=7),
}


class ModelChallenger:
    """Instantiates fresh, deterministic estimators by registry name."""

    def registry_for(self, task_type: str) -> Dict[str, Any]:
        return CLASSIFICATION_REGISTRY if task_type == "classification" else REGRESSION_REGISTRY

    def build(self, model_name: str, task_type: str):
        registry = self.registry_for(task_type)
        if model_name not in registry:
            raise ValueError(f"Unknown model '{model_name}' for task '{task_type}'")
        return registry[model_name]()

    def available_models(self, task_type: str):
        return list(self.registry_for(task_type).keys())
