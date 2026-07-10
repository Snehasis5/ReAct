"""
Strict data schemas for every object that flows through the agent loop.

Nothing in agent.py or monitor.py passes around bare dicts for state that
matters - it's all typed here so a malformed LLM response or tool result
fails fast with a validation error instead of corrupting downstream state.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Failure taxonomy
# ---------------------------------------------------------------------------

class FailureMode(str, Enum):
    NONE = "none"
    TOOL_FAILURE = "tool_failure"                # the tool itself errored / crashed / timed out
    RESULT_INCONSISTENCY = "result_inconsistency"  # tool "succeeded" but output contradicts the goal
    GOAL_DRIFT = "goal_drift"                     # plan has wandered from the original goal


class RecoveryStrategy(str, Enum):
    RETRY_WITH_BACKOFF = "retry_with_backoff"     # tool_failure: same action, same tool, again
    REGENERATE_ACTION = "regenerate_action"       # result_inconsistency: re-derive the action with error context
    REPLAN_FROM_MEMORY = "replan_from_memory"     # goal_drift: rebuild the remaining plan from working memory
    ABANDON_SUBTASK = "abandon_subtask"           # replanning budget exhausted


class StepStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    UNRESOLVABLE = "unresolvable"


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

class PlanStep(BaseModel):
    id: str
    description: str
    status: StepStatus = StepStatus.PENDING
    attempts: int = 0


class Plan(BaseModel):
    goal: str
    steps: list[PlanStep]


# ---------------------------------------------------------------------------
# Tool I/O
# ---------------------------------------------------------------------------

class ToolCall(BaseModel):
    tool_name: str
    tool_input: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    tool_name: str
    success: bool
    output: Optional[dict[str, Any]] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# ReAct trace
# ---------------------------------------------------------------------------

class ReActTrace(BaseModel):
    step_id: str
    thought: str
    action: Optional[ToolCall] = None
    observation: Optional[str] = None


# ---------------------------------------------------------------------------
# Self-monitoring
# ---------------------------------------------------------------------------

class SelfEvaluation(BaseModel):
    step_id: str
    action_succeeded: bool
    result_makes_sense: bool
    still_on_track: bool
    failure_mode: FailureMode = FailureMode.NONE
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)


class RecoveryAction(BaseModel):
    step_id: str
    failure_mode: FailureMode
    strategy: RecoveryStrategy
    rationale: str
    new_plan_steps: Optional[list[PlanStep]] = None


# ---------------------------------------------------------------------------
# Working memory - explicit structure, not just raw conversation history
# ---------------------------------------------------------------------------

class MemoryEntry(BaseModel):
    step_id: str
    kind: str  # "observation" | "fact" | "artifact" | "correction" | "unresolved"
    content: str
    timestamp: datetime = Field(default_factory=_now)


class WorkingMemory(BaseModel):
    goal: str
    entries: list[MemoryEntry] = Field(default_factory=list)
    completed_step_ids: list[str] = Field(default_factory=list)
    unresolved_step_ids: list[str] = Field(default_factory=list)

    def add(self, step_id: str, kind: str, content: str) -> None:
        self.entries.append(MemoryEntry(step_id=step_id, kind=kind, content=content))

    def as_context_string(self, max_entries: int = 25) -> str:
        """Rendered view of memory for injection into LLM prompts."""
        lines = [f"GOAL: {self.goal}"]
        if self.completed_step_ids:
            lines.append(f"COMPLETED STEPS: {', '.join(self.completed_step_ids)}")
        if self.unresolved_step_ids:
            lines.append(f"UNRESOLVED (abandoned) STEPS: {', '.join(self.unresolved_step_ids)}")
        lines.append("MEMORY LOG (most recent last):")
        for e in self.entries[-max_entries:]:
            lines.append(f"  [{e.step_id}][{e.kind}] {e.content}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Structured run log (what gets persisted to disk + rendered in the viewer)
# ---------------------------------------------------------------------------

class StepLog(BaseModel):
    step_id: str
    reasoning: ReActTrace
    tool_result: Optional[ToolResult] = None
    self_evaluation: Optional[SelfEvaluation] = None
    recovery: Optional[RecoveryAction] = None
    timestamp: datetime = Field(default_factory=_now)


class RunLog(BaseModel):
    run_id: str
    goal: str
    self_correction_enabled: bool
    plan: Plan
    steps: list[StepLog] = Field(default_factory=list)
    self_corrections: int = 0
    unresolved_subtasks: list[str] = Field(default_factory=list)
    final_output: Optional[str] = None
    completed: bool = False
    started_at: datetime = Field(default_factory=_now)
    ended_at: Optional[datetime] = None

    def duration_seconds(self) -> Optional[float]:
        if self.ended_at is None:
            return None
        return (self.ended_at - self.started_at).total_seconds()
