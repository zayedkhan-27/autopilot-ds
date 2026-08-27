"""
verification/reflection_engine.py
====================================
Reflection Engine.

After a Failure Diagnosis, the Reflection Engine decides where the *next*
iteration should focus (cleaning / feature_engineering / modeling) given
the diagnosed causes and the health-score trend across iterations. This
is what turns a single train-evaluate pass into a genuine self-correction
loop: it explicitly reasons about "why the previous attempt fell short"
before choosing what changes.
"""

from __future__ import annotations

from typing import Any, Dict, List

from core.llm_client import LLMPlanner

SCHEMA_HINT = """{
  "next_focus": "cleaning" | "feature_engineering" | "modeling",
  "continue": true | false,
  "rationale": "string"
}"""


class ReflectionEngine:
    def __init__(self, planner: LLMPlanner):
        self.planner = planner

    def reflect(
        self,
        diagnosis: Dict[str, Any],
        iteration: int,
        health_history: List[float],
    ) -> Dict[str, Any]:
        context = {
            "diagnosis": diagnosis,
            "iteration": iteration,
            "health_history": health_history,
        }
        decision = self.planner.decide("reflection", context, SCHEMA_HINT)
        decision.setdefault("next_focus", "feature_engineering")
        decision.setdefault("continue", True)
        return decision
