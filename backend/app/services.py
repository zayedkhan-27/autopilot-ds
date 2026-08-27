from pathlib import Path
from datetime import datetime
import sys

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from .models import Experiment, Dataset


# ============================================================
# Make the original AutoPilot engine importable
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parents[1]
ENGINE_DIR = BACKEND_DIR / "engine"

if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))


# ============================================================
# Convert NumPy / Pandas values into JSON-safe Python values
# ============================================================

def json_safe(value):
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, dict):
        return {
            str(key): json_safe(val)
            for key, val in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]

    # Don't try to serialize ML models/pipelines/DataFrames
    if isinstance(value, pd.DataFrame):
        return {
            "rows": len(value),
            "columns": list(value.columns),
        }

    if isinstance(value, pd.Series):
        return value.tolist()

    return str(value)


# ============================================================
# Extract only useful result data from the ML AgentState
# ============================================================

def safe_state(state: dict) -> dict:
    return {
        "run_id": json_safe(state.get("run_id")),
        "dataset_name": json_safe(state.get("dataset_name")),
        "status": json_safe(state.get("status")),

        "target_column": json_safe(state.get("target_column")),
        "task_type": json_safe(state.get("task_type")),

        "original_shape": json_safe(state.get("original_shape")),

        "profile_report": json_safe(
            state.get("profile_report")
        ),

        "eda_summary": json_safe(
            state.get("eda_summary")
        ),

        "cleaning_log": json_safe(
            state.get("cleaning_log")
        ),

        "feature_report": json_safe(
            state.get("feature_report")
        ),

        "metrics": json_safe(
            state.get("metrics")
        ),

        "best_model_name": json_safe(
            state.get("best_model_name")
        ),

        "verification_report": json_safe(
            state.get("verification_report")
        ),

        "pipeline_health": json_safe(
            state.get("pipeline_health")
        ),

        "trace": json_safe(
            state.get("trace", [])
        ),

        "iteration": json_safe(
            state.get("iteration")
        ),
    }


# ============================================================
# Read uploaded dataset
# ============================================================

def load_dataset(file_path: str) -> pd.DataFrame:

    path = Path(file_path)

    extension = path.suffix.lower()

    if extension == ".csv":
        return pd.read_csv(path)

    if extension == ".xlsx":
        return pd.read_excel(path)

    raise ValueError(
        f"Unsupported dataset format: {extension}"
    )


# ============================================================
# Run AutoPilot ML experiment
# ============================================================

def run_experiment(
    db: Session,
    experiment: Experiment,
    dataset: Dataset,
):

    experiment.status = "RUNNING"
    experiment.error_message = None

    db.commit()
    db.refresh(experiment)

    try:

        # Import AFTER adding engine directory to sys.path
        from core.orchestrator import AutoPilotDSAgent

        # ----------------------------------------------------
        # Load dataset
        # ----------------------------------------------------

        df = load_dataset(dataset.file_path)

        if df.empty:
            raise ValueError(
                "The uploaded dataset contains no rows."
            )

        if len(df.columns) < 2:
            raise ValueError(
                "The dataset must contain at least two columns."
            )

        # ----------------------------------------------------
        # Create AutoPilot agent
        # ----------------------------------------------------

        agent = AutoPilotDSAgent(
            prefer_anthropic=True,
            max_iterations=experiment.max_iterations,
        )

        # ----------------------------------------------------
        # Run ML pipeline
        # ----------------------------------------------------

        state = agent.run(
            df=df,
            dataset_name=dataset.original_filename,
            target_override=experiment.target_column,
            task_override=experiment.task_type,
            max_iterations=experiment.max_iterations,
            raw_df_path=dataset.file_path,
        )

        # ----------------------------------------------------
        # Prepare database-safe result
        # ----------------------------------------------------

        result = safe_state(state)

        pipeline_status = state.get("status")

        if pipeline_status == "failed":
            experiment.status = "FAILED"
        else:
            experiment.status = "COMPLETED"

        experiment.target_column = state.get(
            "target_column"
        )

        experiment.task_type = state.get(
            "task_type"
        )

        experiment.best_model_name = state.get(
            "best_model_name"
        )

        # ----------------------------------------------------
        # Pipeline health
        # ----------------------------------------------------

        health = state.get(
            "pipeline_health",
            {},
        )

        if isinstance(health, dict):
            score = health.get("overall_score")

            if score is not None:
                experiment.health_score = float(score)

        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        experiment.result_json = result

        experiment.completed_at = datetime.utcnow()

        db.commit()
        db.refresh(experiment)

        return experiment

    except Exception as exc:

        experiment.status = "FAILED"

        experiment.error_message = str(exc)

        experiment.completed_at = datetime.utcnow()

        db.commit()
        db.refresh(experiment)

        print(
            "\n"
            "================ AUTOPILOT ERROR ================\n"
            f"{type(exc).__name__}: {exc}\n"
            "=================================================\n"
        )

        raise