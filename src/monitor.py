"""
The Judge / Monitor.

This is deliberately a *separate* component from the reasoning step that
proposed the action - it does not get to mark its own homework using the
same context/framing that produced the action. It receives only: the
original goal, the plan, working memory, and the raw tool result, and must
decide three independent questions:

  1. action_succeeded   - did the tool call itself work (no exception/timeout)?
  2. result_makes_sense - does the *content* of the result look consistent
                           with what this step was trying to accomplish?
  3. still_on_track      - given everything so far, is the plan still valid,
                           or has the agent drifted from the original goal?

Failure taxonomy (exactly 3 modes, each with a *distinct* recovery strategy):

  tool_failure          -> retry_with_backoff   (same action, same tool, again)
  result_inconsistency  -> regenerate_action    (re-derive the action w/ error context)
  goal_drift            -> replan_from_memory   (rebuild remaining plan from memory)

When a subtask exceeds the replanning budget, the strategy becomes
abandon_subtask regardless of failure mode - this is what prevents infinite
loops (requirement: "maximum replanning budget").
"""
from __future__ import annotations

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

JUDGE_SYSTEM_PROMPT = """You are JUDGE, an independent self-evaluation module for an autonomous \
coding agent. You did NOT propose the action being evaluated - you are checking someone else's work. \
Be skeptical. Return ONLY a JSON object with this exact shape, no prose:

{
  "action_succeeded": bool,       // did the tool call complete without error?
  "result_makes_sense": bool,     // does the tool's OUTPUT content plausibly satisfy this step's intent?
  "still_on_track": bool,         // is the overall plan still a valid path to the ORIGINAL goal?
  "failure_mode": "none" | "tool_failure" | "result_inconsistency" | "goal_drift",
  "rationale": string,            // one or two sentences, concrete, citing the evidence
  "confidence": number            // 0.0-1.0
}

Rules:
- If the tool call raised an error / exception -> failure_mode = "tool_failure".
- If the tool succeeded but its output contradicts or fails to satisfy the step's
  intent (e.g. tests report failures, output is empty when content was expected,
  exit_code != 0 for a step that required success) -> failure_mode = "result_inconsistency".
- If the plan itself has wandered away from the original goal (e.g. steps no longer
  serve the stated goal, or working memory shows the agent solving the wrong problem)
  -> failure_mode = "goal_drift".
- Otherwise -> failure_mode = "none".
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
        # Fast deterministic path: a hard tool crash doesn't need an LLM
        # opinion - it's unambiguously a tool_failure.
        if not tool_result.success:
            return SelfEvaluation(
                step_id=step.id,
                action_succeeded=False,
                result_makes_sense=False,
                still_on_track=False,
                failure_mode=FailureMode.TOOL_FAILURE,
                rationale=f"Tool '{tool_result.tool_name}' raised an error: {tool_result.error}",
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
