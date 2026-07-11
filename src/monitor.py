"""
The Judge / Monitor.

This is deliberately a *separate* component from the reasoning step that
proposed the action - it does not get to mark its own homework using the
same context/framing that produced the action. It receives the original
goal, the plan, working memory (including the CURRENT ARTIFACTS - the
actual up-to-date content of every file the agent has written, not just a
path/byte-count), and the raw tool result, and must decide three
independent questions:

  1. action_succeeded   - did the tool call itself work (no exception/timeout)?
  2. result_makes_sense - does the *content* of the result (and, critically,
                           the actual code in the current artifacts) look
                           consistent with what this step was trying to
                           accomplish?
  3. still_on_track      - given everything so far, is the plan still valid,
                           or has the agent drifted from the original goal?

Failure taxonomy (5 modes, each with a *distinct* recovery strategy):

  tool_failure               -> retry_with_backoff       (same action, same tool, again - transient failure)
  dependency_missing         -> adapt_approach            (route around a capability the environment doesn't have)
  incomplete_implementation  -> continue_implementation   (complete the stub using the current artifact as a base)
  result_inconsistency       -> regenerate_action         (re-derive the action w/ error context)
  goal_drift                 -> replan_from_memory        (rebuild remaining plan from memory)

`dependency_missing` is detected deterministically (regex over the raw tool
error) because it's an objective, unambiguous signal that doesn't need an
LLM's opinion - same reasoning as the tool_failure fast-path below. The
other modes require semantic judgment (is this code actually a stub? has
the plan actually drifted?) and go through the LLM judge, which now has
real code to look at instead of just a byte count.

When a subtask exceeds the replanning budget, the strategy becomes
abandon_subtask regardless of failure mode - this is what prevents infinite
loops (requirement: "maximum replanning budget").
"""
from __future__ import annotations

import re

from .llm import get_llm_client
from .schemas import (
    FailureMode,
    Plan,
    PlanStep,
    RecoveryAction,
    RecoveryStrategy,
    SelfEvaluation,
    ToolResult,
    WorkingMemory,
)

# Objective, environment-level signals that no amount of retrying will fix -
# the fix is to stop doing the thing that needs the missing capability.
_DEPENDENCY_MISSING_PATTERNS = [
    r"No module named",
    r"ModuleNotFoundError",
    r"ImportError",
    r"command not found",
    r"is not recognized as an internal or external command",
    r"No such file or directory: '.*(pip|python[0-9.]*|pytest)'",
]
_DEPENDENCY_MISSING_RE = re.compile("|".join(_DEPENDENCY_MISSING_PATTERNS), re.IGNORECASE)

JUDGE_SYSTEM_PROMPT = """You are JUDGE, an independent self-evaluation module for an autonomous \
coding agent. You did NOT propose the action being evaluated - you are checking someone else's work. \
Be skeptical, and pay close attention to the CURRENT ARTIFACTS section of working memory - it contains \
the actual, real, up-to-date content of every file the agent has written. A tool call can report \
"success" while the artifact it produced is a stub - that is still a failure. Return ONLY a JSON object \
with this exact shape, no prose:

{
  "action_succeeded": bool,       // did the tool call complete without error?
  "result_makes_sense": bool,     // does the tool's OUTPUT, and the actual artifact content, plausibly satisfy this step's intent?
  "still_on_track": bool,         // is the overall plan still a valid path to the ORIGINAL goal?
  "failure_mode": "none" | "tool_failure" | "dependency_missing" | "incomplete_implementation" | "result_inconsistency" | "goal_drift",
  "rationale": string,            // one or two sentences, concrete, citing the evidence
  "confidence": number            // 0.0-1.0
}

Rules:
- If the tool call raised an error / exception unrelated to a missing package/binary -> "tool_failure".
- If the tool succeeded but the CURRENT ARTIFACTS content for this step's file contains a stub pattern
  (e.g. "TODO", "NotImplementedError", a bare "pass" as an entire function body, a placeholder
  "return []" / "return None" / "raise NotImplementedError" where the step description requires real,
  complete logic) -> "incomplete_implementation". Judge the code, not just whether write_file returned
  success - a stub that compiles is still incomplete.
- If the tool succeeded but its output contradicts or fails to satisfy the step's intent in some OTHER
  way (e.g. tests report failures, exit_code != 0 for a step that required success, output is empty when
  content was expected) -> "result_inconsistency".
- If the plan itself has wandered away from the original goal (e.g. steps no longer serve the stated
  goal, or working memory shows the agent solving the wrong problem) -> "goal_drift".
- Otherwise -> "none".
- Note: "dependency_missing" is handled deterministically before you're ever called for tool errors -
  you will not normally need to assign it yourself, but if you see clear evidence of a missing
  package/binary in a *successful* tool's output (unusual, but possible), you may still use it.
"""


class Monitor:
    def __init__(self, llm=None):
        self.llm = llm or get_llm_client()

    def evaluate(
        self,
        goal: str,
        plan: Plan,
        memory: WorkingMemory,
        step: PlanStep,
        tool_result: ToolResult,
    ) -> SelfEvaluation:
        if not tool_result.success:
            error_text = tool_result.error or ""

            # Deterministic fast-path #1: missing dependency/capability. No
            # amount of retrying fixes this - it needs a different approach,
            # so it gets its own mode rather than falling into tool_failure
            # where retry_with_backoff would just burn the replanning budget
            # retrying the exact same doomed action.
            if _DEPENDENCY_MISSING_RE.search(error_text):
                return SelfEvaluation(
                    step_id=step.id,
                    action_succeeded=False,
                    result_makes_sense=False,
                    still_on_track=True,
                    failure_mode=FailureMode.DEPENDENCY_MISSING,
                    rationale=f"Tool '{tool_result.tool_name}' failed due to a missing dependency/capability in this environment: {error_text}",
                    confidence=0.9,
                )

            # Deterministic fast-path #2: any other hard tool crash - no LLM
            # opinion needed, it's unambiguously a transient tool_failure.
            return SelfEvaluation(
                step_id=step.id,
                action_succeeded=False,
                result_makes_sense=False,
                still_on_track=False,
                failure_mode=FailureMode.TOOL_FAILURE,
                rationale=f"Tool '{tool_result.tool_name}' raised an error: {error_text}",
                confidence=0.95,
            )

        ctx = (
            f"CURRENT STEP ID: {step.id}\n"
            f"CURRENT STEP DESCRIPTION: {step.description}\n\n"
            f"PLAN STEPS:\n" + "\n".join(f"  [{s.id}] {s.description} (status={s.status})" for s in plan.steps) +
            f"\n\n{memory.as_context_string()}\n\n"
            f"TOOL RESULT (this action):\n{tool_result.model_dump()}\n"
        )
        judgment = self.llm.chat_json(
            [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT + "\nJUDGE"},
                {"role": "user", "content": ctx},
            ],
            temperature=0.0,
        )
        judgment["step_id"] = step.id
        return SelfEvaluation(**judgment)


_STRATEGY_MAP = {
    FailureMode.TOOL_FAILURE: RecoveryStrategy.RETRY_WITH_BACKOFF,
    FailureMode.DEPENDENCY_MISSING: RecoveryStrategy.ADAPT_APPROACH,
    FailureMode.INCOMPLETE_IMPLEMENTATION: RecoveryStrategy.CONTINUE_IMPLEMENTATION,
    FailureMode.RESULT_INCONSISTENCY: RecoveryStrategy.REGENERATE_ACTION,
    FailureMode.GOAL_DRIFT: RecoveryStrategy.REPLAN_FROM_MEMORY,
}


def decide_recovery(
    evaluation: SelfEvaluation,
    step: PlanStep,
    max_attempts: int,
) -> RecoveryAction | None:
    """Pure decision function - no side effects, easy to unit test."""
    if evaluation.failure_mode == FailureMode.NONE:
        return None

    if step.attempts >= max_attempts:
        return RecoveryAction(
            step_id=step.id,
            failure_mode=evaluation.failure_mode,
            strategy=RecoveryStrategy.ABANDON_SUBTASK,
            rationale=(
                f"Exceeded replanning budget ({max_attempts} attempts) for step '{step.id}'. "
                f"Last failure mode: {evaluation.failure_mode.value}. Marking unresolvable and continuing."
            ),
        )

    return RecoveryAction(
        step_id=step.id,
        failure_mode=evaluation.failure_mode,
        strategy=_STRATEGY_MAP[evaluation.failure_mode],
        rationale=evaluation.rationale,
    )