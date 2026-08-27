"""
cleaning/cleaning_engine.py
=============================
Data Cleaning Engine.

Executes a *plan* (an ordered list of operation names) chosen upstream by
the Strategy Selector / LLM Planner. Every operation here is a pure,
deterministic, well-tested transformation — the LLM never edits data
directly, it only picks which of these named ops to run and in what
order. Each op logs a structured entry describing what changed, which
feeds both the trace and the Pipeline Health data-quality score.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

VALID_OPS = {
    "drop_constant_columns",
    "drop_high_missing_columns",
    "impute_median_mode",
    "impute_knn",
    "drop_duplicate_rows",
    "cap_outliers_iqr",
    "fix_dtypes",
    "noop_clean",
}


class DataCleaningEngine:
    HIGH_MISSING_THRESHOLD = 40.0  # percent

    def apply(
        self, df: pd.DataFrame, ops: List[str], target_column: str | None = None
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        working = df.copy()
        log: List[Dict[str, Any]] = []
        for op in ops:
            if op not in VALID_OPS:
                log.append({"op": op, "status": "skipped", "reason": "unknown_op"})
                continue
            before_shape = working.shape
            method = getattr(self, f"_op_{op}")
            working, detail = method(working, target_column)
            after_shape = working.shape
            log.append({
                "op": op,
                "status": "applied",
                "before_shape": before_shape,
                "after_shape": after_shape,
                "detail": detail,
            })
        return working, log

    # ------------------------------------------------------------------
    def _op_noop_clean(self, df, target):
        return df, "No cleaning operations required."

    def _op_drop_constant_columns(self, df, target):
        const_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1 and c != target]
        return df.drop(columns=const_cols), {"dropped": const_cols}

    def _op_drop_high_missing_columns(self, df, target):
        pct = df.isna().mean() * 100
        drop_cols = [c for c in df.columns if pct[c] > self.HIGH_MISSING_THRESHOLD and c != target]
        return df.drop(columns=drop_cols), {"dropped": drop_cols, "threshold_pct": self.HIGH_MISSING_THRESHOLD}

    def _op_impute_median_mode(self, df, target):
        filled = {}
        for c in df.columns:
            if c == target or df[c].isna().sum() == 0:
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                val = df[c].median()
            else:
                mode = df[c].mode(dropna=True)
                val = mode.iloc[0] if len(mode) else "missing"
            df[c] = df[c].fillna(val)
            filled[c] = str(val)
        return df, {"filled_with": filled}

    def _op_impute_knn(self, df, target):
        from sklearn.impute import KNNImputer

        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target]
        cat_cols = [c for c in df.columns if c not in numeric_cols and c != target]

        detail = {"numeric_knn_imputed": numeric_cols, "categorical_mode_imputed": []}
        if numeric_cols:
            imputer = KNNImputer(n_neighbors=5)
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        for c in cat_cols:
            if df[c].isna().sum() > 0:
                mode = df[c].mode(dropna=True)
                val = mode.iloc[0] if len(mode) else "missing"
                df[c] = df[c].fillna(val)
                detail["categorical_mode_imputed"].append(c)
        return df, detail

    def _op_drop_duplicate_rows(self, df, target):
        n_before = len(df)
        df = df.drop_duplicates()
        return df, {"rows_removed": n_before - len(df)}

    def _op_cap_outliers_iqr(self, df, target):
        capped = {}
        for c in df.select_dtypes(include=[np.number]).columns:
            if c == target:
                continue
            series = df[c].dropna()
            if len(series) < 4:
                continue
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n_capped = int(((df[c] < lower) | (df[c] > upper)).sum())
            if n_capped:
                df[c] = df[c].clip(lower, upper)
                capped[c] = n_capped
        return df, {"columns_capped": capped}

    def _op_fix_dtypes(self, df, target):
        converted = []
        for c in df.columns:
            if df[c].dtype == object:
                coerced = pd.to_numeric(df[c], errors="coerce")
                if coerced.notna().mean() > 0.9 and df[c].notna().any():
                    df[c] = coerced
                    converted.append(c)
        return df, {"converted_to_numeric": converted}
