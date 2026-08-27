"""
profiling/profiler.py
=======================
Dataset Profiler.

Purely deterministic — no LLM calls here. Produces a JSON-serializable
profile dict that becomes the primary "context" object the LLM Planner
reasons over for cleaning / feature-engineering decisions.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


class DatasetProfiler:
    """Computes a structured statistical profile of a raw DataFrame."""

    def profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        n_rows, n_cols = df.shape
        missing_counts = df.isna().sum()
        missing_pct = (missing_counts / max(n_rows, 1) * 100).round(2)

        dtypes = {c: str(t) for c, t in df.dtypes.items()}
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        datetime_cols = self._detect_datetime_columns(df)

        cardinality = {c: int(df[c].nunique(dropna=True)) for c in df.columns}
        constant_cols = [c for c, n in cardinality.items() if n <= 1]
        id_like_cols = [
            c for c in df.columns
            if cardinality.get(c, 0) == n_rows and n_rows > 1
            and (df[c].dtype == object or "id" in c.lower())
        ]

        outlier_counts = self._iqr_outlier_counts(df, numeric_cols)
        duplicate_rows = int(df.duplicated().sum())

        corr_matrix = None
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr(numeric_only=True).round(3).to_dict()

        skewness = {
            c: round(float(df[c].skew()), 3)
            for c in numeric_cols
            if df[c].notna().sum() > 2
        }

        high_cardinality_categoricals = [
            c for c in categorical_cols if cardinality.get(c, 0) > 50
        ]

        report: Dict[str, Any] = {
            "n_rows": n_rows,
            "n_columns": n_cols,
            "dtypes": dtypes,
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "datetime_columns": datetime_cols,
            "missing_counts": missing_counts.to_dict(),
            "missing_pct": missing_pct.to_dict(),
            "cardinality": cardinality,
            "constant_columns": constant_cols,
            "id_like_columns": id_like_cols,
            "high_cardinality_categoricals": high_cardinality_categoricals,
            "outlier_counts": outlier_counts,
            "duplicate_rows": duplicate_rows,
            "skewness": skewness,
            "correlation_matrix": corr_matrix,
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 ** 2), 3),
            "total_missing_pct": round(float(missing_counts.sum() / (n_rows * n_cols) * 100), 2)
            if n_rows and n_cols else 0.0,
        }
        return report

    # ------------------------------------------------------------------
    @staticmethod
    def _detect_datetime_columns(df: pd.DataFrame) -> List[str]:
        detected = []
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                detected.append(c)
                continue
            if df[c].dtype == object:
                sample = df[c].dropna().astype(str).head(20)
                if len(sample) == 0:
                    continue
                try:
                    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
                    if parsed.notna().mean() > 0.8:
                        detected.append(c)
                except Exception:
                    continue
        return detected

    @staticmethod
    def _iqr_outlier_counts(df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, int]:
        counts = {}
        for c in numeric_cols:
            series = df[c].dropna()
            if len(series) < 4:
                counts[c] = 0
                continue
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                counts[c] = 0
                continue
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            counts[c] = int(((series < lower) | (series > upper)).sum())
        return counts
