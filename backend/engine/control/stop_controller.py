"""
control/stop_controller.py
=============================
Stop Controller.

Prevents infinite self-correction loops. Enforces:
  - max_iterations hard cap
  - health-score acceptance threshold (early success stop)
  - improvement threshold (stop if health stalls for N iterations)
  - duplicate-strategy detection (stop if planner has no novel move left)
"""

from __future__ import annotations

from typing import Any, Dict, List

HEALTH_ACCEPT_THRESHOLD = 75.0
MIN_IMPROVEMENT = 1.0          # minimum health-score points to count as "improving"
STALL_PATIENCE = 2             # iterations without improvement before stopping


class StopController:
    def __init__(
        self,
        max_iterations: int = 5,
        accept_threshold: float = HEALTH_ACCEPT_THRESHOLD,
        min_improvement: float = MIN_IMPROVEMENT,
        stall_patience: int = STALL_PATIENCE,
    ):
        self.max_iterations = max_iterations
        self.accept_threshold = accept_threshold
        self.min_improvement = min_improvement
        self.stall_patience = stall_patience

    def should_stop(
        self,
        iteration: int,
        health_history: List[float],
        next_strategy_is_duplicate: bool,
    ) -> Dict[str, Any]:
        if health_history and health_history[-1] >= self.accept_threshold:
            return {"stop": True, "reason": "health_threshold_met", "status": "success"}

        if iteration >= self.max_iterations:
            return {"stop": True, "reason": "max_iterations_reached", "status": "stopped_max_iter"}

        if next_strategy_is_duplicate:
            return {"stop": True, "reason": "no_novel_strategy_left", "status": "stopped_no_strategy"}

        if len(health_history) > self.stall_patience:
            recent = health_history[-(self.stall_patience + 1):]
            improvements = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]
            if all(imp < self.min_improvement for imp in improvements):
                return {"stop": True, "reason": "improvement_stalled", "status": "stopped_no_improvement"}

        return {"stop": False, "reason": None, "status": "running"}
