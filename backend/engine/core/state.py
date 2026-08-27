"""
core/state.py
==============
State Manager.

Defines the single source of truth (`AgentState`) that flows through every
node of the LangGraph orchestrator. Every engine reads from and writes to
this object so the whole run is reproducible, inspectable, and resumable.

The state is a plain dict (TypedDict) because LangGraph's StateGraph works
best with dict-like state and native reducers. `StateManager` wraps a few
convenience helpers (snapshotting, strategy-signature bookkeeping, trace
shortcuts) around the raw dict so nodes don't repeat bookkeeping logic.
"""

from __future__ import annotations

import copy
import time
import uuid
from typing import Any, Dict, List, Optional, TypedDict


class AgentState(TypedDict, total=False):
    # ----- run identity -----
    run_id: str
    created_at: float
    dataset_name: str

    # ----- raw / working data -----
    raw_df_path: str            # path to original uploaded csv (never mutated)
    df: Any                     # current working pandas.DataFrame
    original_shape: tuple

    # ----- user inputs -----
    target_override: Optional[str]
    task_override: Optional[str]

    # ----- profiling -----
    profile_report: Dict[str, Any]

    # ----- EDA -----
    eda_figures: Dict[str, Any]        # name -> plotly Figure
    eda_summary: Dict[str, Any]

    # ----- task detection -----
    target_column: Optional[str]
    task_type: Optional[str]           # "classification" | "regression"
    task_meta: Dict[str, Any]

    # ----- cleaning -----
    cleaning_log: List[Dict[str, Any]]
    active_cleaning_plan: Dict[str, Any]

    # ----- feature engineering -----
    # ----- feature engineering -----
    feature_plan: Dict[str, Any]
    feature_report: Dict[str, Any]
    _preprocessor: Any

    # ----- strategy bookkeeping -----
    strategies_tried: List[Dict[str, Any]]   # every StrategyPlan signature tried
    current_strategy: Dict[str, Any]
    iteration: int
    max_iterations: int

    # ----- modeling -----
    candidate_models: List[str]
    trained_models: Dict[str, Any]      # name -> {pipeline, fit_time, ...}
    metrics: Dict[str, Any]             # name -> metric dict
    best_model_name: Optional[str]
    best_health_score: float
    health_history: List[float]

    # ----- verification -----
    verification_report: Dict[str, Any]
    pipeline_health: Dict[str, Any]
    diagnosis: Dict[str, Any]
    reflection_log: List[Dict[str, Any]]

    # ----- control -----
    stop_reason: Optional[str]
    should_continue: bool
    status: str   # "running" | "success" | "stopped_max_iter" | "stopped_no_improvement" | "failed"

    # ----- trace -----
    trace: List[Dict[str, Any]]

    # ----- artifacts -----
    artifacts: Dict[str, str]   # logical name -> filesystem path


def new_state(dataset_name: str, max_iterations: int = 5) -> AgentState:
    """Factory for a fresh AgentState at the start of a run."""
    return AgentState(
        run_id=str(uuid.uuid4())[:8],
        created_at=time.time(),
        dataset_name=dataset_name,
        target_override=None,
        task_override=None,
        profile_report={},
        eda_figures={},
        eda_summary={},
        target_column=None,
        task_type=None,
        task_meta={},
        cleaning_log=[],
        active_cleaning_plan={},
        feature_plan={},
        feature_report={},
        _preprocessor=None,
        strategies_tried=[],
        current_strategy={},
        iteration=0,
        max_iterations=max_iterations,
        candidate_models=[],
        trained_models={},
        metrics={},
        best_model_name=None,
        best_health_score=0.0,
        health_history=[],
        verification_report={},
        pipeline_health={},
        diagnosis={},
        reflection_log=[],
        stop_reason=None,
        should_continue=True,
        status="running",
        trace=[],
        artifacts={},
    )


class StateManager:
    """
    Thin convenience wrapper around AgentState for nodes that want
    structured helpers instead of poking the dict directly. LangGraph
    still owns the canonical dict; this class is stateless w.r.t. storage
    (it always operates on the state dict passed to it) so it stays safe
    to use from multiple nodes / threads.
    """

    @staticmethod
    def snapshot(state: AgentState) -> AgentState:
        """Deep copy for safe rollback / comparison across iterations."""
        return copy.deepcopy(state)

    @staticmethod
    def strategy_signature(strategy: Dict[str, Any]) -> str:
        """
        Build a stable signature for a strategy so the Stop Controller and
        Strategy Selector can detect duplicate attempts.
        """
        cleaning = tuple(sorted(strategy.get("cleaning_ops", [])))
        features = tuple(sorted(strategy.get("feature_ops", [])))
        models = tuple(sorted(strategy.get("models", [])))
        return f"clean={cleaning}|feat={features}|models={models}"

    @staticmethod
    def has_tried(state: AgentState, strategy: Dict[str, Any]) -> bool:
        sig = StateManager.strategy_signature(strategy)
        return sig in {s["signature"] for s in state.get("strategies_tried", [])}

    @staticmethod
    def register_strategy(state: AgentState, strategy: Dict[str, Any]) -> None:
        sig = StateManager.strategy_signature(strategy)
        state.setdefault("strategies_tried", []).append(
            {"iteration": state.get("iteration", 0), "signature": sig, "strategy": strategy}
        )

    @staticmethod
    def record_health(state: AgentState, score: float) -> None:
        state.setdefault("health_history", []).append(score)
