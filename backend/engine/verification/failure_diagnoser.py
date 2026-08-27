"""
verification/failure_diagnoser.py
====================================
Failure Diagnoser.

Consumes the Independent Verifier's structured checks and asks the LLM
Planner to explain *why* the pipeline is failing/underperforming and what
class of remedy to try. The diagnoser itself does no data processing; it
only assembles context and calls `LLMPlanner.decide(...)`, keeping the
decision auditable in the trace.
"""

from __future__ import annotations

from typing import Any, Dict

from core.llm_client import LLMPlanner

SCHEMA_HINT = """{
  "likely_causes": ["string", ...],
  "suggested_remedies": ["string", ...],
  "rationale": "string"
}"""


class FailureDiagnoser:
    def __init__(self, planner: LLMPlanner):
        self.planner = planner

    def diagnose(self, verification_report: Dict[str, Any], iteration: int) -> Dict[str, Any]:
        context = {"verification": verification_report, "iteration": iteration}
        decision = self.planner.decide("failure_diagnosis", context, SCHEMA_HINT)
        decision.setdefault("likely_causes", [])
        decision.setdefault("suggested_remedies", [])
        return decision
