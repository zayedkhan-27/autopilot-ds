"""
profiling/eda_engine.py
=========================
Auto-EDA Engine.

Deterministic visualization tool. Given a DataFrame and the Dataset
Profiler's report, it builds a set of Plotly figures that the Streamlit
"Auto-EDA" tab renders directly, plus a compact textual summary the LLM
Planner can use as extra context without touching raw data itself.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class AutoEDAEngine:
    MAX_CATEGORIES_PLOTTED = 15
    MAX_NUMERIC_PLOTTED = 12

    def run(
        self,
        df: pd.DataFrame,
        profile: Dict[str, Any],
        target_column: Optional[str] = None,
    ) -> Dict[str, Any]:
        figures: Dict[str, go.Figure] = {}

        figures["missingness"] = self._missingness_heatmap(df)

        numeric_cols = profile.get("numeric_columns", [])[: self.MAX_NUMERIC_PLOTTED]
        cat_cols = profile.get("categorical_columns", [])[: self.MAX_NUMERIC_PLOTTED]

        for col in numeric_cols:
            figures[f"dist_{col}"] = self._numeric_distribution(df, col)

        for col in cat_cols:
            figures[f"cat_{col}"] = self._categorical_bar(df, col)

        if len(numeric_cols) >= 2:
            figures["correlation_heatmap"] = self._correlation_heatmap(df, numeric_cols)

        if target_column and target_column in df.columns:
            figures["target_distribution"] = self._target_distribution(df, target_column)

        summary = self._text_summary(profile)
        return {"figures": figures, "summary": summary}

    # ------------------------------------------------------------------
    def _missingness_heatmap(self, df: pd.DataFrame) -> go.Figure:
        miss = df.isna().astype(int)
        fig = px.imshow(
            miss.T,
            aspect="auto",
            color_continuous_scale=["#eef2f7", "#d62728"],
            labels=dict(x="Row index", y="Column", color="Missing"),
            title="Missingness Map",
        )
        fig.update_layout(height=max(300, 20 * len(df.columns)))
        return fig

    def _numeric_distribution(self, df: pd.DataFrame, col: str) -> go.Figure:
        fig = px.histogram(df, x=col, marginal="box", nbins=40, title=f"Distribution of {col}")
        return fig

    def _categorical_bar(self, df: pd.DataFrame, col: str) -> go.Figure:
        vc = df[col].astype(str).value_counts().head(self.MAX_CATEGORIES_PLOTTED)
        fig = px.bar(x=vc.index, y=vc.values, title=f"Top categories: {col}",
                     labels={"x": col, "y": "count"})
        return fig

    def _correlation_heatmap(self, df: pd.DataFrame, numeric_cols) -> go.Figure:
        corr = df[numeric_cols].corr(numeric_only=True)
        fig = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            title="Correlation Heatmap",
        )
        return fig

    def _target_distribution(self, df: pd.DataFrame, target: str) -> go.Figure:
        series = df[target].dropna()
        if pd.api.types.is_numeric_dtype(series) and series.nunique() > 20:
            fig = px.histogram(df, x=target, nbins=40, title=f"Target distribution: {target}")
        else:
            vc = series.astype(str).value_counts().head(20)
            fig = px.bar(x=vc.index, y=vc.values, title=f"Target class balance: {target}",
                         labels={"x": target, "y": "count"})
        return fig

    def _text_summary(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "shape": f"{profile.get('n_rows')} rows x {profile.get('n_columns')} columns",
            "total_missing_pct": profile.get("total_missing_pct"),
            "duplicate_rows": profile.get("duplicate_rows"),
            "n_numeric": len(profile.get("numeric_columns", [])),
            "n_categorical": len(profile.get("categorical_columns", [])),
            "n_datetime": len(profile.get("datetime_columns", [])),
            "constant_columns": profile.get("constant_columns", []),
            "high_cardinality_categoricals": profile.get("high_cardinality_categoricals", []),
        }
