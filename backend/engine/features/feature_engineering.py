"""
features/feature_engineering.py
==================================
Feature Engineering Engine.

Turns a named list of feature ops (chosen by the Strategy Selector / LLM
Planner) into a deterministic scikit-learn ColumnTransformer, plus
leakage-safety pruning (dropping ID-like columns, near-duplicate-of-target
columns) that always runs regardless of the plan, since leaking columns
must never reach the model.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

class _SafeSelectKBest(SelectKBest):
    """SelectKBest that clamps k to the actual number of incoming features
    at fit time, so it never raises when the post-encoding feature space
    is smaller than the requested k (e.g. after aggressive one-hot
    expansion on a small dataset)."""

    def fit(self, X, y=None):
        n_features = X.shape[1]
        self.k = min(self.k, n_features) if isinstance(self.k, int) else self.k
        return super().fit(X, y)


VALID_OPS = {
    "scale_numeric",
    "onehot_encode_categoricals",
    "target_safe_encode_high_cardinality",
    "select_top_k_features",
    "drop_low_variance_features",
}


class FeatureEngineeringEngine:
    def plan_and_build(
        self,
        df: pd.DataFrame,
        target_column: str,
        task_type: str,
        ops: List[str],
        profile: Dict[str, Any],
    ) -> Tuple[ColumnTransformer, Dict[str, Any]]:
        report: Dict[str, Any] = {"ops_applied": [], "dropped_for_leakage": []}

        feature_df = df.drop(columns=[target_column])

        # --- leakage-safety pruning (always applied, non-negotiable) ---
        id_like = [
            c for c in feature_df.columns
            if feature_df[c].nunique(dropna=True) == len(feature_df) and len(feature_df) > 1
        ]
        feature_df = feature_df.drop(columns=id_like)
        report["dropped_for_leakage"] = id_like

        numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = feature_df.select_dtypes(exclude=[np.number]).columns.tolist()

        high_card_cols = [c for c in categorical_cols if feature_df[c].nunique(dropna=True) > 50]
        low_card_cols = [c for c in categorical_cols if c not in high_card_cols]

        numeric_steps = [("impute", SimpleImputer(strategy="median"))]
        if "scale_numeric" in ops:
            numeric_steps.append(("scale", StandardScaler()))
            report["ops_applied"].append("scale_numeric")
        numeric_pipeline = Pipeline(numeric_steps)

        transformers = []
        if numeric_cols:
            transformers.append(("num", numeric_pipeline, numeric_cols))

        if low_card_cols:
            cat_pipeline = Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore")),
            ])
            transformers.append(("cat_low_card", cat_pipeline, low_card_cols))
            if "onehot_encode_categoricals" in ops:
                report["ops_applied"].append("onehot_encode_categoricals")

        if high_card_cols:
            if "target_safe_encode_high_cardinality" in ops:
                # Ordinal encoding is used as a leakage-safe stand-in for target
                # encoding: it fits only on training folds inside the pipeline,
                # never uses target statistics directly, avoiding target leakage
                # while still handling very high cardinality without exploding
                # the feature space via one-hot.
                cat_pipeline = Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                ])
                report["ops_applied"].append("target_safe_encode_high_cardinality")
            else:
                cat_pipeline = Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                ])
            transformers.append(("cat_high_card", cat_pipeline, high_card_cols))

        if not transformers:
            # Degenerate case: nothing usable left. Fabricate a passthrough on
            # numeric_cols (empty) so the pipeline still builds cleanly.
            transformers.append(("num", numeric_pipeline, []))

        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

        report["numeric_columns"] = numeric_cols
        report["low_cardinality_categoricals"] = low_card_cols
        report["high_cardinality_categoricals"] = high_card_cols
        report["n_input_features"] = len(numeric_cols) + len(categorical_cols)
        report["feature_selection_requested"] = (
            "select_top_k_features" in ops or "drop_low_variance_features" in ops
        )
        return preprocessor, report

    def selection_step(self, ops: List[str], task_type: str, n_features_hint: int = 20):
        """Optional post-encoding feature selection step, appended to the
        model pipeline (after the ColumnTransformer) when requested.
        `n_features_hint` is the approximate raw feature count (pre-one-hot
        expansion); k is capped so SelectKBest never errors on a smaller
        post-encoding feature space than requested."""
        steps = []
        if "drop_low_variance_features" in ops:
            steps.append(("variance_threshold", VarianceThreshold(threshold=0.0)))
        if "select_top_k_features" in ops:
            score_func = f_classif if task_type == "classification" else f_regression
            k = max(5, n_features_hint * 3)
            steps.append(("select_k_best", _SafeSelectKBest(score_func=score_func, k=k)))
        return steps
