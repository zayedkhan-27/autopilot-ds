"""
core/trace.py
==============
Agent Trace System.

Every node in the graph emits structured trace events through this module
instead of ad-hoc print()/logging calls. The trace is the backbone of the
"Live Agent Trace" dashboard tab: it must show decisions, tool calls,
observations, failures, reflections, strategy changes, retries, and final
decisions, in chronological order, with enough detail to reconstruct why
the agent did what it did.

Event types
-----------
decision        - the LLM/planner chose an action
tool_call       - a deterministic Python tool was invoked
observation     - the result of a tool call, summarized
failure         - a check failed / metric below threshold / exception
diagnosis       - Failure Diagnoser's explanation of a failure
reflection      - Reflection Engine's reasoning about what to try next
strategy_change - Strategy Selector switched approach
retry           - a loop iteration restarted
final           - final decision / run completion
info            - generic narration
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

EVENT_TYPES = {
    "decision", "tool_call", "observation", "failure",
    "diagnosis", "reflection", "strategy_change", "retry", "final", "info",
}


@dataclass
class TraceEvent:
    step: int
    iteration: int
    event_type: str
    actor: str            # e.g. "Planner", "DataCleaningEngine", "Verifier"
    title: str
    detail: str
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "iteration": self.iteration,
            "type": self.event_type,
            "actor": self.actor,
            "title": self.title,
            "detail": self.detail,
            "payload": self.payload,
            "timestamp": self.timestamp,
        }


class AgentTrace:
    """
    Append-only trace log bound to a run. Wraps `state["trace"]` (a plain
    list of dicts) so it can be trivially serialized to JSON / rendered in
    Streamlit / exported without any custom (de)serialization code.
    """

    def __init__(self, state: Dict[str, Any]):
        self._state = state
        self._state.setdefault("trace", [])

    @property
    def events(self) -> List[Dict[str, Any]]:
        return self._state["trace"]

    def log(
        self,
        event_type: str,
        actor: str,
        title: str,
        detail: str = "",
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if event_type not in EVENT_TYPES:
            event_type = "info"
        ev = TraceEvent(
            step=len(self._state["trace"]) + 1,
            iteration=self._state.get("iteration", 0),
            event_type=event_type,
            actor=actor,
            title=title,
            detail=detail,
            payload=payload or {},
        )
        self._state["trace"].append(ev.to_dict())
        return ev.to_dict()

    # convenience shortcuts -------------------------------------------------
    def decision(self, actor, title, detail="", **payload):
        return self.log("decision", actor, title, detail, payload)

    def tool_call(self, actor, title, detail="", **payload):
        return self.log("tool_call", actor, title, detail, payload)

    def observation(self, actor, title, detail="", **payload):
        return self.log("observation", actor, title, detail, payload)

    def failure(self, actor, title, detail="", **payload):
        return self.log("failure", actor, title, detail, payload)

    def diagnosis(self, actor, title, detail="", **payload):
        return self.log("diagnosis", actor, title, detail, payload)

    def reflection(self, actor, title, detail="", **payload):
        return self.log("reflection", actor, title, detail, payload)

    def strategy_change(self, actor, title, detail="", **payload):
        return self.log("strategy_change", actor, title, detail, payload)

    def retry(self, actor, title, detail="", **payload):
        return self.log("retry", actor, title, detail, payload)

    def final(self, actor, title, detail="", **payload):
        return self.log("final", actor, title, detail, payload)

    def info(self, actor, title, detail="", **payload):
        return self.log("info", actor, title, detail, payload)

    # rendering ---------------------------------------------------------
    def to_markdown(self) -> str:
        lines = ["# Agent Execution Trace\n"]
        for ev in self.events:
            ts = time.strftime("%H:%M:%S", time.localtime(ev["timestamp"]))
            lines.append(
                f"**[{ev['step']:03d} | iter {ev['iteration']} | {ts}] "
                f"{ev['type'].upper()} — {ev['actor']}: {ev['title']}**"
            )
            if ev["detail"]:
                lines.append(f"> {ev['detail']}")
            if ev["payload"]:
                lines.append(f"`{ev['payload']}`")
            lines.append("")
        return "\n".join(lines)
