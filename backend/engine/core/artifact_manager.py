"""
core/artifact_manager.py
===========================
Artifact Manager.

Handles all filesystem output for a run: cleaned dataset, best model +
preprocessing pipeline, metrics, EDA figures, agent trace, and a final
self-contained HTML report. Called once at the end of a successful (or
terminated) run by the orchestrator's `finalize` node.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

import joblib
import pandas as pd

from core.trace import AgentTrace


class ArtifactManager:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_all(self, state: Dict[str, Any]) -> Dict[str, str]:
        paths: Dict[str, str] = {}

        # cleaned dataset
        df: pd.DataFrame = state.get("df")
        if df is not None:
            p = os.path.join(self.output_dir, "cleaned_dataset.csv")
            df.to_csv(p, index=False)
            paths["cleaned_dataset"] = p

        # best model pipeline (preprocessing + model, single joblib object)
        best_name = state.get("best_model_name")
        trained = state.get("trained_models", {})
        if best_name and best_name in trained and trained[best_name].get("status") == "ok":
            p = os.path.join(self.output_dir, "best_model_pipeline.joblib")
            joblib.dump(trained[best_name]["pipeline"], p)
            paths["best_model_pipeline"] = p

        # metrics
        p = os.path.join(self.output_dir, "metrics.json")
        with open(p, "w") as f:
            json.dump(_json_safe(state.get("metrics", {})), f, indent=2)
        paths["metrics"] = p

        # pipeline health / verification
        p = os.path.join(self.output_dir, "pipeline_health.json")
        with open(p, "w") as f:
            json.dump(_json_safe({
                "pipeline_health": state.get("pipeline_health", {}),
                "verification_report": state.get("verification_report", {}),
            }), f, indent=2)
        paths["pipeline_health"] = p

        # EDA figures -> combined HTML
        eda_figs = state.get("eda_figures", {}).get("figures", {})
        if eda_figs:
            p = os.path.join(self.output_dir, "eda_report.html")
            self._write_eda_html(eda_figs, p)
            paths["eda_report"] = p

        # agent trace
        trace = AgentTrace(state)
        p = os.path.join(self.output_dir, "agent_trace.json")
        with open(p, "w") as f:
            json.dump(_json_safe(trace.events), f, indent=2)
        paths["agent_trace_json"] = p

        p = os.path.join(self.output_dir, "agent_trace.md")
        with open(p, "w") as f:
            f.write(trace.to_markdown())
        paths["agent_trace_md"] = p

        # final report
        p = os.path.join(self.output_dir, "final_report.html")
        self._write_final_report(state, p)
        paths["final_report"] = p

        state.setdefault("artifacts", {}).update(paths)
        return paths

    # ------------------------------------------------------------------
    def _write_eda_html(self, figures: Dict[str, Any], path: str) -> None:
        parts = ["<html><head><title>AutoPilot-DS EDA Report</title></head><body>",
                 "<h1>AutoPilot-DS — Auto-EDA Report</h1>"]
        for name, fig in figures.items():
            parts.append(f"<h2>{name}</h2>")
            parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))
        parts.append("</body></html>")
        with open(path, "w") as f:
            f.write("\n".join(parts))

    def _write_final_report(self, state: Dict[str, Any], path: str) -> None:
        health = state.get("pipeline_health", {})
        metrics = state.get("metrics", {})
        best = state.get("best_model_name")
        verification = state.get("verification_report", {})
        task_type = state.get("task_type")
        target = state.get("target_column")

        rows = []
        for name, m in metrics.items():
            if m.get("status") != "ok":
                rows.append(f"<tr><td>{name}</td><td colspan='4'>FAILED: {m.get('error')}</td></tr>")
                continue
            primary = m["primary_metric_name"]
            rows.append(
                f"<tr><td>{name}{' (best)' if name == best else ''}</td>"
                f"<td>{m['test'].get(primary):.4f}</td>"
                f"<td>{m['train'].get(primary):.4f}</td>"
                f"<td>{m['overfit_gap']:.4f}</td>"
                f"<td>{m.get('fit_time_sec')}</td></tr>"
            )

        checks_html = "".join(
            f"<li><b>{k}</b>: {v['status'].upper()} — {v.get('evidence')}</li>"
            for k, v in verification.get("checks", {}).items()
        )

        html = f"""
        <html><head><title>AutoPilot-DS Final Report</title>
        <style>
        body {{ font-family: -apple-system, Arial, sans-serif; margin: 40px; color: #1a1a2e; }}
        table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #1a1a2e; color: white; }}
        .score {{ font-size: 48px; font-weight: bold; color: #2a9d8f; }}
        h1 {{ color: #1a1a2e; }}
        </style></head>
        <body>
        <h1>AutoPilot-DS — Final Pipeline Report</h1>
        <p><b>Dataset:</b> {state.get('dataset_name')} &nbsp; | &nbsp;
           <b>Task:</b> {task_type} &nbsp; | &nbsp;
           <b>Target:</b> {target} &nbsp; | &nbsp;
           <b>Iterations run:</b> {state.get('iteration')} &nbsp; | &nbsp;
           <b>Status:</b> {state.get('status')}</p>

        <h2>Pipeline Health Score</h2>
        <div class="score">{health.get('overall_score', 'N/A')} / 100</div>
        <p>Components: {health.get('components', {})}</p>

        <h2>Model Comparison</h2>
        <table>
        <tr><th>Model</th><th>Test primary metric</th><th>Train primary metric</th>
            <th>Overfit gap</th><th>Fit time (s)</th></tr>
        {''.join(rows)}
        </table>

        <h2>Independent Verification</h2>
        <ul>{checks_html}</ul>

        <h2>Best Model</h2>
        <p><b>{best}</b></p>
        </body></html>
        """
        with open(path, "w") as f:
            f.write(html)


def _json_safe(obj: Any) -> Any:
    """Recursively strip non-JSON-serializable objects (DataFrames, numpy
    types, fitted estimators) so state can be dumped safely."""
    import numpy as np
    import pandas as pd

    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()
                if not isinstance(v, (pd.DataFrame, pd.Series))
                and not hasattr(v, "fit")}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return obj
