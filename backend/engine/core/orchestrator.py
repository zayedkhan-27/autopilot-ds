"""
core/orchestrator.py
=======================
Agent Orchestrator + Planner.

Wires every engine into a LangGraph StateGraph implementing the
verifiable self-correction loop:

    profile -> eda -> detect_task -> [LOOP: clean -> engineer_features ->
        train -> evaluate -> verify -> compute_health -> stop_check]
        -> (loop back to clean/engineer_features/train depending on the
            Reflection Engine's chosen focus, or finalize)
    -> finalize

Every node:
  1. Logs a `decision` or `tool_call` trace event before acting.
  2. Calls exactly one deterministic engine (or the LLMPlanner for
     genuinely LLM-owned decisions).
  3. Logs an `observation` trace event with a compact summary of the
     result.

If the `langgraph` package is not installed (e.g. offline sandbox), a
`SimpleOrchestrator` fallback executes the identical node sequence and
conditional-edge logic as a plain Python loop, so the whole pipeline still
runs end-to-end without the dependency. Both share the same node
functions, so behavior is identical regardless of which runner is used.
"""

from __future__ import annotations

from typing import Any, Dict

from cleaning.cleaning_engine import DataCleaningEngine
from control.stop_controller import StopController
from core.llm_client import LLMPlanner
from core.state import AgentState, StateManager, new_state
from core.strategy_selector import StrategySelector
from core.trace import AgentTrace
from features.feature_engineering import FeatureEngineeringEngine
from modeling.evaluator import Evaluator
from modeling.trainer import Trainer
from profiling.eda_engine import AutoEDAEngine
from profiling.profiler import DatasetProfiler
from task.task_detector import TaskDetector
from verification.failure_diagnoser import FailureDiagnoser
from verification.pipeline_health import PipelineHealthEngine
from verification.reflection_engine import ReflectionEngine
from verification.verifier import IndependentVerifier


class AutoPilotDSAgent:
    """
    Owns one instance of every engine + the LangGraph (or fallback) graph.
    `run(df, dataset_name, target_override, task_override, max_iterations)`
    executes the full autonomous workflow and returns the final AgentState.
    """

    def __init__(self, prefer_anthropic: bool = True, max_iterations: int = 5):
        self.planner = LLMPlanner(prefer_anthropic=prefer_anthropic)
        self.profiler = DatasetProfiler()
        self.eda_engine = AutoEDAEngine()
        self.task_detector = TaskDetector()
        self.cleaning_engine = DataCleaningEngine()
        self.fe_engine = FeatureEngineeringEngine()
        self.strategy_selector = StrategySelector(self.planner)
        self.trainer = Trainer()
        self.evaluator = Evaluator()
        self.verifier = IndependentVerifier()
        self.health_engine = PipelineHealthEngine()
        self.diagnoser = FailureDiagnoser(self.planner)
        self.reflection_engine = ReflectionEngine(self.planner)
        self.stop_controller = StopController(max_iterations=max_iterations)
        self.max_iterations = max_iterations
        self._graph = self._build_graph()

    # ======================================================================
    # Public entrypoint
    # ======================================================================
    def run(
        self,
        df,
        dataset_name: str,
        target_override: str | None = None,
        task_override: str | None = None,
        max_iterations: int | None = None,
        raw_df_path: str | None = None,
    ) -> AgentState:
        state = new_state(dataset_name, max_iterations=max_iterations or self.max_iterations)
        state["df"] = df
        state["original_shape"] = df.shape
        state["raw_df_path"] = raw_df_path or ""
        state["target_override"] = target_override
        state["task_override"] = task_override

        trace = AgentTrace(state)
        trace.info("Orchestrator", "Run started",
                    f"dataset={dataset_name}, shape={df.shape}, "
                    f"llm_backend={self.planner.backend_name}")

        final_state = self._graph.invoke(state)
        return final_state

    # ======================================================================
    # Graph construction
    # ======================================================================
    def _build_graph(self):
        try:
            from langgraph.graph import END, StateGraph
        except ImportError:
            return _SimpleOrchestrator(self)

        g = StateGraph(AgentState)
        g.add_node("profile", self.node_profile)
        g.add_node("eda", self.node_eda)
        g.add_node("detect_task", self.node_detect_task)
        g.add_node("select_strategy", self.node_select_strategy)
        g.add_node("clean", self.node_clean)
        g.add_node("engineer_features", self.node_engineer_features)
        g.add_node("train_evaluate", self.node_train_evaluate)
        g.add_node("verify", self.node_verify)
        g.add_node("compute_health", self.node_compute_health)
        g.add_node("stop_check", self.node_stop_check)
        g.add_node("diagnose_reflect", self.node_diagnose_reflect)
        g.add_node("finalize", self.node_finalize)

        g.set_entry_point("profile")
        g.add_edge("profile", "eda")
        g.add_edge("eda", "detect_task")
        g.add_edge("detect_task", "select_strategy")
        g.add_edge("select_strategy", "clean")
        g.add_edge("clean", "engineer_features")
        g.add_edge("engineer_features", "train_evaluate")
        g.add_edge("train_evaluate", "verify")
        g.add_edge("verify", "compute_health")
        g.add_edge("compute_health", "stop_check")

        g.add_conditional_edges(
            "stop_check",
            lambda s: "finalize" if not s.get("should_continue", False) else "diagnose_reflect",
            {"finalize": "finalize", "diagnose_reflect": "diagnose_reflect"},
        )
        g.add_edge("diagnose_reflect", "select_strategy")
        g.add_edge("finalize", END)

        return g.compile()

    # ======================================================================
    # Nodes
    # ======================================================================
    def node_profile(self, state: AgentState) -> AgentState:
        trace = AgentTrace(state)
        trace.tool_call("DatasetProfiler", "Profiling raw dataset")
        report = self.profiler.profile(state["df"])
        state["profile_report"] = report
        trace.observation(
            "DatasetProfiler", "Profile complete",
            f"{report['n_rows']} rows x {report['n_columns']} cols, "
            f"{report['total_missing_pct']}% missing, "
            f"{report['duplicate_rows']} duplicate rows, "
            f"{len(report['constant_columns'])} constant columns.",
        )
        return state

    def node_eda(self, state: AgentState) -> AgentState:
        trace = AgentTrace(state)
        trace.tool_call("AutoEDAEngine", "Generating exploratory visualizations")
        target_hint = state.get("target_override")
        result = self.eda_engine.run(state["df"], state["profile_report"], target_hint)
        state["eda_figures"] = result
        state["eda_summary"] = result["summary"]
        trace.observation("AutoEDAEngine", "EDA complete",
                           f"{len(result['figures'])} figures generated.")
        return state

    def node_detect_task(self, state: AgentState) -> AgentState:
        trace = AgentTrace(state)
        trace.decision("TaskDetector", "Detecting target column and ML task type")
        result = self.task_detector.detect(
            state["df"], state.get("target_override"), state.get("task_override")
        )
        state["target_column"] = result["target_column"]
        state["task_type"] = result["task_type"]
        state["task_meta"] = result["meta"]
        trace.observation(
            "TaskDetector", f"target='{result['target_column']}', task={result['task_type']}",
            f"source={result['meta'].get('target_source')}; "
            + (f"classes={result['meta'].get('n_classes')}"
               if result["task_type"] == "classification"
               else f"target_std={result['meta'].get('target_std')}"),
        )
        return state

    def node_select_strategy(self, state: AgentState) -> AgentState:
        trace = AgentTrace(state)
        state["iteration"] = state.get("iteration", 0) + 1
        trace.decision(
            "Planner/StrategySelector",
            f"Selecting strategy for iteration {state['iteration']}",
        )
        strategy = self.strategy_selector.propose(state)
        state["current_strategy"] = strategy
        StateManager.register_strategy(state, strategy)
        trace.strategy_change(
            "StrategySelector",
            f"Iteration {state['iteration']} strategy chosen",
            detail=(
                f"cleaning={strategy['cleaning_ops']}, features={strategy['feature_ops']}, "
                f"models={strategy['models']}"
            ),
            rationale=strategy.get("rationale"),
        )
        return state

    def node_clean(self, state: AgentState) -> AgentState:
        trace = AgentTrace(state)
        strategy = state["current_strategy"]
        trace.tool_call("DataCleaningEngine", "Applying cleaning plan", detail=str(strategy["cleaning_ops"]))
        cleaned_df, log = self.cleaning_engine.apply(
            state["df"], strategy["cleaning_ops"], state.get("target_column")
        )
        state["df"] = cleaned_df
        state.setdefault("cleaning_log", []).append({"iteration": state["iteration"], "log": log})
        state["active_cleaning_plan"] = strategy
        trace.observation(
            "DataCleaningEngine", "Cleaning applied",
            f"shape now {cleaned_df.shape}", ops_log=log,
        )
        return state

    def node_engineer_features(self, state: AgentState) -> AgentState:
        trace = AgentTrace(state)
        strategy = state["current_strategy"]
        trace.tool_call("FeatureEngineeringEngine", "Building preprocessing pipeline",
                         detail=str(strategy["feature_ops"]))
        preprocessor, report = self.fe_engine.plan_and_build(
            state["df"], state["target_column"], state["task_type"],
            strategy["feature_ops"], state["profile_report"],
        )
        state["feature_plan"] = strategy["feature_ops"]
        state["feature_report"] = report
        state["_preprocessor"] = preprocessor  # transient, not JSON-serialized
        trace.observation(
            "FeatureEngineeringEngine", "Preprocessing pipeline built",
            f"{report['n_input_features']} input features, "
            f"{len(report['dropped_for_leakage'])} dropped for leakage risk.",
        )
        return state

    def node_train_evaluate(self, state: AgentState) -> AgentState:
        trace = AgentTrace(state)
        strategy = state["current_strategy"]
        trace.tool_call("Trainer", "Training candidate models", detail=str(strategy["models"]))
        trained = self.trainer.train_candidates(
            state["df"], state["target_column"], state["task_type"],
            state["_preprocessor"], strategy["feature_ops"], state["feature_report"],
            strategy["models"],
        )
        state["trained_models"] = trained
        failed = [n for n, r in trained.items() if r.get("status") != "ok"]
        if failed:
            trace.failure("Trainer", f"{len(failed)} model(s) failed to train", detail=str(failed))

        trace.tool_call("Evaluator", "Computing metrics for all candidates")
        metrics = self.evaluator.evaluate_all(trained, state["task_type"])
        state["metrics"] = metrics
        ranking = self.evaluator.rank(metrics, state["task_type"])
        best_name = ranking[0] if ranking else None
        state["best_model_name"] = best_name
        detail = ", ".join(
            f"{n}: {metrics[n]['test'].get(metrics[n]['primary_metric_name']):.4f}"
            for n in ranking
        ) if ranking else "no successful candidates"
        trace.observation("Evaluator", f"Best candidate: {best_name}", detail)
        return state

    def node_verify(self, state: AgentState) -> AgentState:
        trace = AgentTrace(state)
        best_name = state.get("best_model_name")
        if not best_name:
            state["verification_report"] = {"checks": {}, "overall_status": "fail", "n_fail": 1, "n_warn": 0}
            trace.failure("IndependentVerifier", "No trained model to verify")
            return state

        trace.tool_call("Trainer", "Running reproducibility check (dual-seed retrain)")
        reproducibility = self.trainer.reproducibility_check(
            state["df"], state["target_column"], state["task_type"], state["_preprocessor"],
            state["current_strategy"]["feature_ops"], state["feature_report"], best_name,
        )

        trace.tool_call("IndependentVerifier", "Running independent verification checks")
        report = self.verifier.verify(
            state["df"], state["target_column"], state["task_type"], state["profile_report"],
            state["metrics"], best_name, reproducibility,
        )
        state["verification_report"] = report
        fail_names = [k for k, v in report["checks"].items() if v["status"] == "fail"]
        warn_names = [k for k, v in report["checks"].items() if v["status"] == "warn"]
        (trace.failure if fail_names else trace.observation)(
            "IndependentVerifier", f"Verification: {report['overall_status'].upper()}",
            f"failed={fail_names}, warned={warn_names}",
        )
        return state

    def node_compute_health(self, state: AgentState) -> AgentState:
        trace = AgentTrace(state)
        trace.tool_call("PipelineHealthEngine", "Computing composite health score")
        health = self.health_engine.compute(
            state["verification_report"], state["metrics"], state.get("best_model_name"),
            state["task_type"],
        )
        state["pipeline_health"] = health
        StateManager.record_health(state, health["overall_score"])
        if health["overall_score"] > state.get("best_health_score", 0.0):
            state["best_health_score"] = health["overall_score"]
        trace.observation(
            "PipelineHealthEngine", f"Pipeline Health Score: {health['overall_score']}/100",
            str(health["components"]),
        )
        return state

    def node_stop_check(self, state: AgentState) -> AgentState:
        trace = AgentTrace(state)
        # peek at whether the *next* strategy would be a duplicate, without
        # registering it yet (registration happens in select_strategy).
        preview_strategy = self.strategy_selector.propose(state)
        is_duplicate = StateManager.has_tried(state, preview_strategy)

        decision = self.stop_controller.should_stop(
            state["iteration"], state.get("health_history", []), is_duplicate
        )
        state["should_continue"] = not decision["stop"]
        state["stop_reason"] = decision["reason"]
        state["status"] = decision["status"]
        trace.decision(
            "StopController",
            f"{'STOP' if decision['stop'] else 'CONTINUE'} (reason={decision['reason']})",
            f"iteration={state['iteration']}/{state['max_iterations']}, "
            f"health_history={state.get('health_history', [])}",
        )
        return state

    def node_diagnose_reflect(self, state: AgentState) -> AgentState:
        trace = AgentTrace(state)
        trace.tool_call("FailureDiagnoser", "Diagnosing cause of low pipeline health")
        diagnosis = self.diagnoser.diagnose(state["verification_report"], state["iteration"])
        state["diagnosis"] = diagnosis
        trace.diagnosis(
            "FailureDiagnoser", "Diagnosis complete",
            "; ".join(diagnosis.get("likely_causes", [])) or "no dominant cause identified",
            suggested_remedies=diagnosis.get("suggested_remedies", []),
        )

        trace.tool_call("ReflectionEngine", "Deciding focus for next iteration")
        reflection = self.reflection_engine.reflect(
            diagnosis, state["iteration"], state.get("health_history", [])
        )
        state.setdefault("reflection_log", []).append({"iteration": state["iteration"], **reflection})
        trace.reflection(
            "ReflectionEngine", f"Next focus: {reflection.get('next_focus')}",
            reflection.get("rationale", ""),
        )
        trace.retry("Orchestrator", f"Retrying with new strategy (iteration {state['iteration'] + 1})")
        return state

    def node_finalize(self, state: AgentState) -> AgentState:
        trace = AgentTrace(state)
        health = state.get("pipeline_health", {})
        trace.final(
            "Orchestrator", f"Run finished: status={state.get('status')}",
            f"best_model={state.get('best_model_name')}, "
            f"health_score={health.get('overall_score')}, "
            f"iterations={state['iteration']}, stop_reason={state.get('stop_reason')}",
        )
        state.pop("_preprocessor", None)
        return state


class _SimpleOrchestrator:
    """
    Fallback runner used when `langgraph` is not importable. Implements the
    exact same node sequence / loop condition as the LangGraph version so
    behavior is identical either way — only the execution engine differs.
    """

    def __init__(self, agent: "AutoPilotDSAgent"):
        self.agent = agent

    def invoke(self, state: AgentState) -> AgentState:
        a = self.agent
        state = a.node_profile(state)
        state = a.node_eda(state)
        state = a.node_detect_task(state)

        while True:
            state = a.node_select_strategy(state)
            state = a.node_clean(state)
            state = a.node_engineer_features(state)
            state = a.node_train_evaluate(state)
            state = a.node_verify(state)
            state = a.node_compute_health(state)
            state = a.node_stop_check(state)
            if not state.get("should_continue", False):
                break
            state = a.node_diagnose_reflect(state)

        state = a.node_finalize(state)
        return state
