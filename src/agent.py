"""
The agent loop.

Agent.run(goal, self_correction=True, on_event=callback) -> RunLog

Flow per step:
  1. REASON  - LLM produces a visible thought + a proposed tool call (ReAct).
  2. ACT     - the tool is actually invoked (call_tool), real subprocess execution.
  3. OBSERVE - the raw tool result becomes an observation string in memory.
  4. SELF-EVAL (if self_correction) - Monitor judges the result independently.
  5. RECOVER (if a failure was detected) - one of:
       retry_with_backoff   -> re-run the exact same step
       regenerate_action    -> re-reason with the error appended to memory as context
       replan_from_memory   -> LLM rebuilds the *remaining* plan from working memory
       abandon_subtask      -> mark step unresolvable, move on (replanning budget hit)

`on_event(dict)` is called synchronously at every observable moment so a
caller (the CLI, the eval harness, or the websocket chat server) can render
a live trace without needing to know anything about the internals.

When self_correction=False (the baseline used for comparison), the agent
still executes tools for real, but skips steps 4-5 entirely: it accepts
whatever the tool returns - success or failure - and barrels on to the next
step. This is what "no self-correction" means in practice, and it's the
control condition run_evaluation.py compares against.
"""
from __future__ import annotations

import uuid
from typing import Callable, Optional

from .config import settings
from .llm import get_llm_client
from .monitor import Monitor, decide_recovery
from .schemas import (
    FailureMode,
    Plan,
    PlanStep,
    ReActTrace,
    RecoveryStrategy,
    RunLog,
    StepLog,
    StepStatus,
    ToolCall,
    WorkingMemory,
)
from .tools import call_tool, tool_catalog_text

EventCallback = Optional[Callable[[dict], None]]

PLANNER_SYSTEM_PROMPT = """You are PLANNER, the planning module of an autonomous coding agent. \
Given a goal, produce a plan of AT LEAST 5 concrete, ordered, executable steps that involve writing \
code, executing it, and validating it (not just talking about it). Return ONLY JSON:

{"steps": [{"id": "s1", "description": "..."}, {"id": "s2", "description": "..."}, ...]}

PLANNER"""

REASONER_SYSTEM_PROMPT = f"""You are REASONER, the ReAct reasoning module of an autonomous coding agent. \
Given the goal, the plan, working memory, and the current step, produce ONE visible reasoning thought \
and exactly ONE tool call to make progress on the current step. Return ONLY JSON:

{{"thought": "...", "tool_name": "...", "tool_input": {{...}}}}

Available tools:
{tool_catalog_text()}

If context includes a PRIOR ERROR for this step, your thought must explicitly account for it and your \
new tool_input must be different/corrected, not a blind repeat.

REASONER"""

REPLAN_SYSTEM_PROMPT = """You are REPLAN, invoked when the agent has drifted from its original goal. \
Given the goal and working memory (including what has already been done), produce a SHORT new plan \
(1-4 steps) for the REMAINING work only, grounded in what memory shows is actually still needed. \
Return ONLY JSON: {"steps": [{"id": "...", "description": "..."}, ...]}

REPLAN"""

FINAL_SYNTHESIS_SYSTEM_PROMPT = """You are the final synthesis module of an autonomous coding agent. \
Given the goal and the full working memory, write a 2-4 sentence final answer describing what was \
built, whether it works, and any caveats (e.g. unresolved subtasks). Plain text, no JSON.

FINAL_SYNTHESIS"""


def _emit(on_event: EventCallback, event_type: str, **data):
    if on_event:
        on_event({"type": event_type, **data})


class Agent:
    def __init__(self, llm=None):
        self.llm = llm or get_llm_client()
        self.monitor = Monitor(self.llm)
        self.max_attempts = settings.max_replan_attempts

    # -- planning -------------------------------------------------------

    def _plan(self, goal: str) -> Plan:
        data = self.llm.chat_json(
            [
                {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                {"role": "user", "content": f"GOAL: {goal}"},
            ],
            temperature=0.2,
        )
        steps = [PlanStep(id=s["id"], description=s["description"]) for s in data["steps"]]
        return Plan(goal=goal, steps=steps)

    def _replan(self, goal: str, memory: WorkingMemory) -> list[PlanStep]:
        data = self.llm.chat_json(
            [
                {"role": "system", "content": REPLAN_SYSTEM_PROMPT},
                {"role": "user", "content": memory.as_context_string()},
            ],
            temperature=0.2,
        )
        return [PlanStep(id=s["id"], description=s["description"]) for s in data["steps"]]

    # -- ReAct reasoning --------------------------------------------------

    def _reason(self, goal: str, plan: Plan, memory: WorkingMemory, step: PlanStep,
                prior_error: Optional[str] = None) -> ReActTrace:
        ctx = (
            f"GOAL: {goal}\n"
            f"CURRENT STEP ID: {step.id}\n"
            f"CURRENT STEP DESCRIPTION: {step.description}\n"
            f"ATTEMPT: {step.attempts + 1}\n"
        )
        if prior_error:
            ctx += f"PRIOR ERROR FOR THIS STEP: {prior_error}\n"
        ctx += f"\n{memory.as_context_string()}"

        data = self.llm.chat_json(
            [
                {"role": "system", "content": REASONER_SYSTEM_PROMPT},
                {"role": "user", "content": ctx},
            ],
            temperature=0.2,
        )
        action = ToolCall(tool_name=data["tool_name"], tool_input=data.get("tool_input", {}))
        return ReActTrace(step_id=step.id, thought=data["thought"], action=action)

    # -- main loop ----------------------------------------------------------

    def run(self, goal: str, self_correction: bool = True, on_event: EventCallback = None) -> RunLog:
        run_id = uuid.uuid4().hex[:12]
        plan = self._plan(goal)
        memory = WorkingMemory(goal=goal)
        run_log = RunLog(run_id=run_id, goal=goal, self_correction_enabled=self_correction, plan=plan)

        _emit(on_event, "plan_created", run_id=run_id, plan=plan.model_dump())

        pending: list[PlanStep] = list(plan.steps)
        i = 0
        MAX_TOTAL_STEPS = 40  # hard ceiling so a pathological loop can't run forever
        total_executed = 0

        while i < len(pending) and total_executed < MAX_TOTAL_STEPS:
            step = pending[i]
            step.status = StepStatus.IN_PROGRESS
            total_executed += 1

            prior_error = None
            last_eval_step_log = None

            reasoning = self._reason(goal, plan, memory, step, prior_error=prior_error)
            _emit(on_event, "thought", step_id=step.id, thought=reasoning.thought)
            _emit(on_event, "action", step_id=step.id, action=reasoning.action.model_dump())

            tool_result = call_tool(reasoning.action.tool_name, reasoning.action.tool_input)
            reasoning.observation = str(tool_result.output if tool_result.success else tool_result.error)
            _emit(on_event, "observation", step_id=step.id, tool_result=tool_result.model_dump())

            step_log = StepLog(step_id=step.id, reasoning=reasoning, tool_result=tool_result)
            memory.add(step.id, "observation", f"{reasoning.action.tool_name} -> {reasoning.observation[:400]}")

            if not self_correction:
                # Baseline: no evaluation, no recovery - just record and move on,
                # regardless of whether the tool actually succeeded.
                step.status = StepStatus.DONE if tool_result.success else StepStatus.UNRESOLVABLE
                if not tool_result.success:
                    run_log.unresolved_subtasks.append(step.id)
                run_log.steps.append(step_log)
                i += 1
                continue

            # --- self-correction path ---
            evaluation = self.monitor.evaluate(goal, plan, memory, step, tool_result)
            step_log.self_evaluation = evaluation
            _emit(on_event, "self_evaluation", step_id=step.id, evaluation=evaluation.model_dump())

            if evaluation.failure_mode == FailureMode.NONE:
                step.status = StepStatus.DONE
                memory.completed_step_ids.append(step.id)
                run_log.steps.append(step_log)
                i += 1
                continue

            # A failure was detected - decide + apply recovery.
            step.attempts += 1
            recovery = decide_recovery(evaluation, step, self.max_attempts)
            step_log.recovery = recovery
            run_log.self_corrections += 1
            _emit(on_event, "recovery", step_id=step.id, recovery=recovery.model_dump())
            run_log.steps.append(step_log)

            if recovery.strategy == RecoveryStrategy.ABANDON_SUBTASK:
                step.status = StepStatus.UNRESOLVABLE
                memory.unresolved_step_ids.append(step.id)
                memory.add(step.id, "unresolved", recovery.rationale)
                run_log.unresolved_subtasks.append(step.id)
                _emit(on_event, "step_abandoned", step_id=step.id, reason=recovery.rationale)
                i += 1
                continue

            if recovery.strategy == RecoveryStrategy.RETRY_WITH_BACKOFF:
                # Re-run the exact same step id (loop does not advance i).
                memory.add(step.id, "correction", f"Retrying after tool_failure: {evaluation.rationale}")
                continue

            if recovery.strategy == RecoveryStrategy.REGENERATE_ACTION:
                # Re-reason with explicit error context, still same step id.
                memory.add(step.id, "correction", f"Regenerating action after result_inconsistency: {evaluation.rationale}")
                error_ctx = evaluation.rationale
                reasoning2 = self._reason(goal, plan, memory, step, prior_error=error_ctx)
                _emit(on_event, "thought", step_id=step.id, thought=reasoning2.thought, regenerated=True)
                _emit(on_event, "action", step_id=step.id, action=reasoning2.action.model_dump(), regenerated=True)
                tool_result2 = call_tool(reasoning2.action.tool_name, reasoning2.action.tool_input)
                reasoning2.observation = str(tool_result2.output if tool_result2.success else tool_result2.error)
                _emit(on_event, "observation", step_id=step.id, tool_result=tool_result2.model_dump(), regenerated=True)
                memory.add(step.id, "observation", f"[regenerated] {reasoning2.action.tool_name} -> {reasoning2.observation[:400]}")
                eval2 = self.monitor.evaluate(goal, plan, memory, step, tool_result2)
                _emit(on_event, "self_evaluation", step_id=step.id, evaluation=eval2.model_dump(), regenerated=True)
                run_log.steps.append(StepLog(
                    step_id=step.id, reasoning=reasoning2, tool_result=tool_result2, self_evaluation=eval2,
                ))
                if eval2.failure_mode == FailureMode.NONE:
                    step.status = StepStatus.DONE
                    memory.completed_step_ids.append(step.id)
                    i += 1
                # else: loop continues on same step, will re-evaluate against attempts budget
                continue

            if recovery.strategy == RecoveryStrategy.REPLAN_FROM_MEMORY:
                memory.add(step.id, "correction", f"Replanning from memory after goal_drift: {evaluation.rationale}")
                new_steps = self._replan(goal, memory)
                _emit(on_event, "replan", step_id=step.id, new_steps=[s.model_dump() for s in new_steps])
                # Replace the remaining (not-yet-executed) pending steps with the new plan.
                pending = pending[: i + 1] + new_steps
                step.status = StepStatus.DONE  # the drifted step itself is superseded by the new plan
                i += 1
                continue

        # Final synthesis
        final_text = self.llm.chat(
            [
                {"role": "system", "content": FINAL_SYNTHESIS_SYSTEM_PROMPT},
                {"role": "user", "content": memory.as_context_string(max_entries=50)},
            ],
            temperature=0.2,
        )
        run_log.final_output = final_text
        run_log.completed = len(run_log.unresolved_subtasks) == 0
        from datetime import datetime, timezone
        run_log.ended_at = datetime.now(timezone.utc)
        _emit(on_event, "final", run_id=run_id, final_output=final_text, run_log=run_log.model_dump(mode="json"))
        return run_log
