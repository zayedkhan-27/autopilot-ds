"""
modeling/trainer.py
=====================
Trainer.

Deterministic fit/predict execution. Given a preprocessing
ColumnTransformer + a list of candidate model names, builds full
sklearn Pipelines (preprocess -> [selection] -> model), trains each on a
fixed, reproducible train/test split (and cross-validation for a more
robust performance estimate), and returns fitted pipelines + raw
predictions. No metric interpretation happens here — that is the
Evaluator's job — the Trainer only produces reproducible artifacts.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from features.feature_engineering import FeatureEngineeringEngine
from modeling.model_challenger import RANDOM_STATE, ModelChallenger

CV_FOLDS = 5


class Trainer:
    def __init__(self):
        self.challenger = ModelChallenger()
        self.fe_engine = FeatureEngineeringEngine()

    def train_candidates(
        self,
        df: pd.DataFrame,
        target_column: str,
        task_type: str,
        preprocessor,
        feature_ops: List[str],
        feature_report: Dict[str, Any],
        model_names: List[str],
        test_size: float = 0.2,
    ) -> Dict[str, Any]:
        X = df.drop(columns=[target_column])
        # keep only columns the preprocessor actually knows about
        keep_cols = (
            feature_report.get("numeric_columns", [])
            + feature_report.get("low_cardinality_categoricals", [])
            + feature_report.get("high_cardinality_categoricals", [])
        )
        X = X[keep_cols] if keep_cols else X
        y = df[target_column]

        stratify = y if task_type == "classification" else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=stratify
        )

        results: Dict[str, Any] = {}
        n_hint = max(1, len(keep_cols))

        for name in model_names:
            t0 = time.time()
            estimator = self.challenger.build(name, task_type)
            steps = [("preprocess", clone(preprocessor))]
            steps += self.fe_engine.selection_step(feature_ops, task_type, n_features_hint=n_hint)
            steps.append(("model", estimator))
            pipeline = Pipeline(steps)

            try:
                pipeline.fit(X_train, y_train)
                y_pred_train = pipeline.predict(X_train)
                y_pred_test = pipeline.predict(X_test)

                cv_scores = self._cross_validate(pipeline, X_train, y_train, task_type)

                results[name] = {
                    "status": "ok",
                    "pipeline": pipeline,
                    "fit_time_sec": round(time.time() - t0, 3),
                    "y_train": y_train,
                    "y_pred_train": y_pred_train,
                    "y_test": y_test,
                    "y_pred_test": y_pred_test,
                    "cv_scores": cv_scores,
                    "X_test_index": X_test.index,
                }
            except Exception as exc:  # noqa: BLE001 - surfaced to trace/diagnoser
                results[name] = {
                    "status": "failed",
                    "error": str(exc),
                    "fit_time_sec": round(time.time() - t0, 3),
                }
        return results

    # ------------------------------------------------------------------
    def _cross_validate(self, pipeline: Pipeline, X, y, task_type: str) -> Dict[str, Any]:
        try:
            if task_type == "classification":
                splitter = StratifiedKFold(n_splits=min(CV_FOLDS, y.value_counts().min() or 2),
                                            shuffle=True, random_state=RANDOM_STATE)
                scoring = "f1_macro"
            else:
                splitter = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
                scoring = "r2"
            scores = cross_val_score(pipeline, X, y, cv=splitter, scoring=scoring)
            return {
                "scoring": scoring,
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "folds": [float(s) for s in scores],
            }
        except Exception as exc:  # noqa: BLE001
            return {"scoring": None, "mean": None, "std": None, "error": str(exc)}

    def reproducibility_check(
        self, df, target_column, task_type, preprocessor, feature_ops, feature_report, model_name
    ) -> Dict[str, Any]:
        """Re-run training twice with different (but fixed) seeds for the
        train/test split to verify metric stability -> feeds the
        Independent Verifier's reproducibility check."""
        scores = []
        for seed in (RANDOM_STATE, RANDOM_STATE + 1):
            X = df.drop(columns=[target_column])
            keep_cols = (
                feature_report.get("numeric_columns", [])
                + feature_report.get("low_cardinality_categoricals", [])
                + feature_report.get("high_cardinality_categoricals", [])
            )
            X = X[keep_cols] if keep_cols else X
            y = df[target_column]
            stratify = y if task_type == "classification" else None
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=seed, stratify=stratify
            )
            estimator = self.challenger.build(model_name, task_type)
            n_hint = max(1, len(keep_cols))
            steps = [("preprocess", clone(preprocessor))]
            steps += self.fe_engine.selection_step(feature_ops, task_type, n_features_hint=n_hint)
            steps.append(("model", estimator))
            pipeline = Pipeline(steps)
            pipeline.fit(X_train, y_train)
            pred = pipeline.predict(X_test)
            if task_type == "classification":
                from sklearn.metrics import f1_score
                scores.append(float(f1_score(y_test, pred, average="macro")))
            else:
                from sklearn.metrics import r2_score
                scores.append(float(r2_score(y_test, pred)))
        spread = max(scores) - min(scores)
        return {"seed_scores": scores, "spread": spread, "stable": spread < 0.1}
