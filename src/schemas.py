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
#
# Five modes, each with its own distinct recovery strategy (see monitor.py
# for the full rationale and detection logic):
#
#   tool_failure               -> retry_with_backoff       (same action, same tool, again - transient failure)
#   dependency_missing         -> adapt_approach            (route around a capability the environment doesn't have)
#   incomplete_implementation  -> continue_implementation   (complete the stub using the current artifact as a base)
#   result_inconsistency       -> regenerate_action         (re-derive the action w/ error context)
#   goal_drift                 -> replan_from_memory        (rebuild remaining plan from memory)
#
# `dependency_missing` is detected deterministically (regex over the raw
# tool error); the rest go through the LLM judge.
# ---------------------------------------------------------------------------

class FailureMode(str, Enum):
    NONE = "none"
    TOOL_FAILURE = "tool_failure"                           # the tool itself errored / crashed / timed out
    DEPENDENCY_MISSING = "dependency_missing"                # missing package/binary/capability - no amount of retrying helps
    INCOMPLETE_IMPLEMENTATION = "incomplete_implementation"  # tool "succeeded" but the artifact it wrote is a stub
    RESULT_INCONSISTENCY = "result_inconsistency"            # tool "succeeded" but output otherwise contradicts the goal
    GOAL_DRIFT = "goal_drift"                                # plan has wandered from the original goal


class RecoveryStrategy(str, Enum):
    RETRY_WITH_BACKOFF = "retry_with_backoff"           # tool_failure: replay the IDENTICAL action, same tool, after a backoff
    ADAPT_APPROACH = "adapt_approach"                   # dependency_missing: re-reason around the missing capability entirely
    CONTINUE_IMPLEMENTATION = "continue_implementation"  # incomplete_implementation: finish the stub, using the current artifact as a base
    REGENERATE_ACTION = "regenerate_action"             # result_inconsistency: re-derive the action with error context
    REPLAN_FROM_MEMORY = "replan_from_memory"           # goal_drift: rebuild the remaining plan from working memory
    ABANDON_SUBTASK = "abandon_subtask"                 # replanning budget exhausted, regardless of failure mode


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
    # Explicit, structured file state - the ground truth of what's actually on
    # disk in the sandbox right now, keyed by filename. This is what lets the
    # agent build a file *incrementally* across steps: write_file always
    # overwrites, so the reasoner needs to see the CURRENT content before it
    # can correctly extend it, not just a "bytes_written: N" observation. It's
    # also what lets the JUDGE catch incomplete_implementation - it can look
    # at the actual code, not just a success flag.
    files: dict[str, str] = Field(default_factory=dict)

    def add(self, step_id: str, kind: str, content: str) -> None:
        self.entries.append(MemoryEntry(step_id=step_id, kind=kind, content=content))

    def remember_file(self, filename: str, content: str) -> None:
        """Record the authoritative current content of a file the agent wrote.
        Called after every successful write_file so later steps can see it."""
        self.files[filename] = content

    def as_context_string(self, max_entries: int = 25, max_file_chars: int = 6000) -> str:
        """Rendered view of memory for injection into LLM prompts."""
        lines = [f"GOAL: {self.goal}"]
        if self.completed_step_ids:
            lines.append(f"COMPLETED STEPS: {', '.join(self.completed_step_ids)}")
        if self.unresolved_step_ids:
            lines.append(f"UNRESOLVED (abandoned) STEPS: {', '.join(self.unresolved_step_ids)}")
        lines.append("MEMORY LOG (most recent last):")
        for e in self.entries[-max_entries:]:
            lines.append(f"  [{e.step_id}][{e.kind}] {e.content}")
        if self.files:
            lines.append("\nCURRENT FILE CONTENTS IN SANDBOX (this is what's actually on disk right "
                          "now - write_file OVERWRITES a file, so if you are extending one of these, "
                          "your new tool_input.content must include ALL of this plus your addition, "
                          "not just the new part):")
            for fname, content in self.files.items():
                truncated = content if len(content) <= max_file_chars else (
                    content[:max_file_chars] + f"\n... [truncated, {len(content)} chars total]"
                )
                lines.append(f"--- {fname} ---\n{truncated}\n--- end {fname} ---")
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