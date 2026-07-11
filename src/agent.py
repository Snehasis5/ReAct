"""
The agent loop.

Agent.run(goal, self_correction=True, on_event=callback) -> RunLog

Flow per step:
  1. REASON  - LLM produces a visible thought + a proposed tool call (ReAct).
  2. ACT     - the tool is actually invoked (call_tool), real subprocess execution.
  3. OBSERVE - the raw tool result becomes an observation string in memory.
  4. SELF-EVAL (if self_correction) - Monitor judges the result independently.
  5. RECOVER (if a failure was detected) - one of five DISTINCT strategies,
     one per failure mode (see monitor.py for the full taxonomy):

       retry_with_backoff       (tool_failure)              -> replay the IDENTICAL
           action against the SAME tool after a short backoff. No re-reasoning:
           a transient failure means the action was fine, the environment
           hiccuped, so the fix is "do the exact same thing again."

       adapt_approach            (dependency_missing)        -> re-reason with an
           explicit instruction to NOT use the same tool/library again and to
           find a different route to the step's goal. Unlike a retry, the
           action itself must change.

       continue_implementation   (incomplete_implementation) -> re-reason with
           an explicit instruction to replace the stub with a complete,
           working implementation, building on the CURRENT FILE CONTENTS
           already visible in working memory.

       regenerate_action         (result_inconsistency)      -> re-reason with
           the judge's rationale appended as error context, letting the
           reasoner re-derive whatever it thinks is a better action.

       replan_from_memory        (goal_drift)                -> throw out the
           rest of the ORIGINAL plan and have the LLM rebuild just the
           remaining steps from what working memory shows is actually needed.

       abandon_subtask            -> mark step unresolvable, move on
           (replanning budget hit, regardless of failure mode).

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

import time
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
    SelfEvaluation,
    StepLog,
    StepStatus,
    ToolCall,
    ToolResult,
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
and exactly ONE tool call to make progress on the current step.

Your ENTIRE response must be a single JSON object and nothing else - no prose before or after it, no \
markdown code fences, and critically: do NOT write the tool call as a Python-style function call like \
`write_file(filename="x.py", content="...")`. That is WRONG. The tool call belongs INSIDE the JSON, as \
the "tool_name" and "tool_input" fields, exactly like this shape:

{{"thought": "...", "tool_name": "write_file", "tool_input": {{"filename": "x.py", "content": "..."}}}}

Available tools:
{tool_catalog_text()}

If context includes a PRIOR ERROR / CORRECTIVE NOTE for this step, your thought must explicitly account \
for it and your new tool_input must be different/corrected accordingly, not a blind repeat. In \
particular: if the note says a dependency/tool is unavailable, do NOT propose that same tool again - \
pick a genuinely different route. If the note says an implementation is incomplete/a stub, your \
tool_input.content must be the FULL file with real, complete logic, not just the missing fragment.

IMPORTANT - write_file OVERWRITES the entire file, it does not append or patch. Working memory below \
may include a "CURRENT FILE CONTENTS IN SANDBOX" section showing what's already on disk. If this step \
is extending, fixing, or building on a file that already exists, your tool_input.content MUST be the \
FULL updated file (everything already there, plus your change) - never just the new fragment, or you \
will silently delete everything earlier steps wrote.

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
            ctx += f"PRIOR ERROR / CORRECTIVE NOTE FOR THIS STEP: {prior_error}\n"
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

    # -- resilience wrappers ------------------------------------------------
    #
    # The REASONER and JUDGE calls are external LLM calls and can fail for the
    # same mundane reasons a tool call can: rate limits, transient API errors,
    # a model returning nothing usable even after chat_json's own repair
    # retries. call_tool() already contains tool crashes as a ToolResult
    # rather than letting them raise; these two wrappers give the reasoning
    # and judging calls that same treatment, so an LLM hiccup becomes a
    # tool_failure that flows through the EXISTING recovery machinery
    # (retry_with_backoff -> ... -> abandon_subtask) instead of raising an
    # exception that kills the entire run after a single bad completion.

    def _safe_reason(self, goal: str, plan: Plan, memory: WorkingMemory, step: PlanStep,
                      prior_error: Optional[str] = None) -> tuple[ReActTrace, Optional[ToolResult]]:
        """Returns (reasoning, forced_failure). forced_failure is None on success,
        or a ToolResult(success=False) if the REASONER call itself failed - in
        which case reasoning.action is a harmless placeholder that was never
        actually executed."""
        try:
            return self._reason(goal, plan, memory, step, prior_error=prior_error), None
        except Exception as e:  # noqa: BLE001
            reasoning = ReActTrace(
                step_id=step.id,
                thought=f"[reasoning module failed before an action could be proposed] {e}",
                action=ToolCall(tool_name="reason", tool_input={}),
            )
            forced_failure = ToolResult(
                tool_name="reason", success=False,
                error=f"REASONER LLM call failed: {type(e).__name__}: {e}",
            )
            reasoning.observation = forced_failure.error
            return reasoning, forced_failure

    def _safe_evaluate(self, goal: str, plan: Plan, memory: WorkingMemory, step: PlanStep,
                        tool_result: ToolResult) -> SelfEvaluation:
        try:
            return self.monitor.evaluate(goal, plan, memory, step, tool_result)
        except Exception as e:  # noqa: BLE001
            return SelfEvaluation(
                step_id=step.id,
                action_succeeded=tool_result.success,
                result_makes_sense=False,
                still_on_track=False,
                failure_mode=FailureMode.TOOL_FAILURE,
                rationale=f"JUDGE LLM call failed and could not evaluate this step: {type(e).__name__}: {e}",
                confidence=0.5,
            )

    # -- shared "re-reason once, act, evaluate" cycle ------------------------
    #
    # adapt_approach, continue_implementation, and regenerate_action all share
    # the same mechanics (re-reason with a corrective hint baked into
    # prior_error, execute whatever comes out, evaluate it, log it) - they
    # differ only in the WORDING of the hint given to the reasoner, which is
    # what actually makes them behave differently (avoid-this-tool vs.
    # complete-the-stub vs. fix-what's-wrong). retry_with_backoff is
    # deliberately NOT built on this helper - see _retry_identical_action.

    def _run_correction_cycle(self, goal: str, plan: Plan, memory: WorkingMemory, step: PlanStep,
                               corrective_hint: str, tag: str, on_event: EventCallback,
                               run_log: RunLog) -> SelfEvaluation:
        reasoning2, forced_failure2 = self._safe_reason(goal, plan, memory, step, prior_error=corrective_hint)
        _emit(on_event, "thought", step_id=step.id, thought=reasoning2.thought, tag=tag)
        if forced_failure2 is not None:
            tool_result2 = forced_failure2
            _emit(on_event, "observation", step_id=step.id, tool_result=tool_result2.model_dump(), tag=tag)
        else:
            _emit(on_event, "action", step_id=step.id, action=reasoning2.action.model_dump(), tag=tag)
            tool_result2 = call_tool(reasoning2.action.tool_name, reasoning2.action.tool_input)
            reasoning2.observation = str(tool_result2.output if tool_result2.success else tool_result2.error)
            _emit(on_event, "observation", step_id=step.id, tool_result=tool_result2.model_dump(), tag=tag)
        memory.add(step.id, "observation", f"[{tag}] {reasoning2.action.tool_name} -> {reasoning2.observation[:400]}")
        if reasoning2.action.tool_name == "write_file" and tool_result2.success:
            fname2 = reasoning2.action.tool_input.get("filename")
            fcontent2 = reasoning2.action.tool_input.get("content")
            if fname2 is not None and fcontent2 is not None:
                memory.remember_file(fname2, fcontent2)
        eval2 = self._safe_evaluate(goal, plan, memory, step, tool_result2)
        _emit(on_event, "self_evaluation", step_id=step.id, evaluation=eval2.model_dump(), tag=tag)
        run_log.steps.append(StepLog(
            step_id=step.id, reasoning=reasoning2, tool_result=tool_result2, self_evaluation=eval2,
        ))
        return eval2

    # -- retry_with_backoff: replay, don't re-reason -------------------------
    #
    # Distinct on purpose from the cycle above: a tool_failure means the
    # ACTION was fine and the ENVIRONMENT hiccuped (network blip, transient
    # storage error), so the correct fix is to do the exact same thing again,
    # not to ask the reasoner to invent something new. If there's no concrete
    # prior action to replay (the REASONER call itself is what failed), we
    # fall back to letting the main loop re-reason from scratch next
    # iteration - there's nothing to replay in that case.

    def _retry_identical_action(self, goal: str, plan: Plan, memory: WorkingMemory, step: PlanStep,
                                 reasoning: ReActTrace, on_event: EventCallback,
                                 run_log: RunLog) -> Optional[SelfEvaluation]:
        backoff_seconds = min(2 ** max(step.attempts - 1, 0), 8)
        time.sleep(backoff_seconds)
        _emit(on_event, "action", step_id=step.id, action=reasoning.action.model_dump(), tag="retried")
        tool_result = call_tool(reasoning.action.tool_name, reasoning.action.tool_input)
        observation = str(tool_result.output if tool_result.success else tool_result.error)
        _emit(on_event, "observation", step_id=step.id, tool_result=tool_result.model_dump(), tag="retried")
        memory.add(step.id, "observation", f"[retried] {reasoning.action.tool_name} -> {observation[:400]}")
        if reasoning.action.tool_name == "write_file" and tool_result.success:
            fname = reasoning.action.tool_input.get("filename")
            fcontent = reasoning.action.tool_input.get("content")
            if fname is not None and fcontent is not None:
                memory.remember_file(fname, fcontent)
        evaluation = self._safe_evaluate(goal, plan, memory, step, tool_result)
        _emit(on_event, "self_evaluation", step_id=step.id, evaluation=evaluation.model_dump(), tag="retried")
        run_log.steps.append(StepLog(
            step_id=step.id, reasoning=reasoning, tool_result=tool_result, self_evaluation=evaluation,
        ))
        return evaluation

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

            reasoning, forced_failure = self._safe_reason(goal, plan, memory, step, prior_error=prior_error)
            _emit(on_event, "thought", step_id=step.id, thought=reasoning.thought)

            if forced_failure is not None:
                tool_result = forced_failure
                _emit(on_event, "observation", step_id=step.id, tool_result=tool_result.model_dump())
            else:
                _emit(on_event, "action", step_id=step.id, action=reasoning.action.model_dump())
                tool_result = call_tool(reasoning.action.tool_name, reasoning.action.tool_input)
                reasoning.observation = str(tool_result.output if tool_result.success else tool_result.error)
                _emit(on_event, "observation", step_id=step.id, tool_result=tool_result.model_dump())

            step_log = StepLog(step_id=step.id, reasoning=reasoning, tool_result=tool_result)
            memory.add(step.id, "observation", f"{reasoning.action.tool_name} -> {reasoning.observation[:400]}")
            if reasoning.action.tool_name == "write_file" and tool_result.success:
                fname = reasoning.action.tool_input.get("filename")
                fcontent = reasoning.action.tool_input.get("content")
                if fname is not None and fcontent is not None:
                    memory.remember_file(fname, fcontent)

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
            evaluation = self._safe_evaluate(goal, plan, memory, step, tool_result)
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
                memory.add(step.id, "correction",
                           f"Retrying identical action after tool_failure (attempt {step.attempts}): {evaluation.rationale}")
                if forced_failure is None:
                    # A real action actually ran and failed transiently - replay
                    # that exact action rather than asking the reasoner to
                    # invent something new.
                    eval_r = self._retry_identical_action(goal, plan, memory, step, reasoning, on_event, run_log)
                    if eval_r.failure_mode == FailureMode.NONE:
                        step.status = StepStatus.DONE
                        memory.completed_step_ids.append(step.id)
                        i += 1
                    # else: still failing, loop continues on same step, will
                    # re-evaluate against the attempts budget next pass.
                # else: the REASONER call itself is what failed, so there is no
                # concrete action to replay - fall through and let the top of
                # the loop re-reason from scratch next iteration.
                continue

            if recovery.strategy == RecoveryStrategy.ADAPT_APPROACH:
                hint = (
                    f"A REQUIRED DEPENDENCY/CAPABILITY IS MISSING IN THIS ENVIRONMENT: {evaluation.rationale} "
                    f"Do NOT propose the same tool/library again - it will fail the same way every time. "
                    f"Propose a genuinely different approach to this step's goal that avoids the missing "
                    f"capability entirely (a different tool, a different library, or a manual workaround)."
                )
                memory.add(step.id, "correction", f"Adapting approach after dependency_missing: {evaluation.rationale}")
                eval2 = self._run_correction_cycle(goal, plan, memory, step, hint, "adapted", on_event, run_log)
                if eval2.failure_mode == FailureMode.NONE:
                    step.status = StepStatus.DONE
                    memory.completed_step_ids.append(step.id)
                    i += 1
                continue

            if recovery.strategy == RecoveryStrategy.CONTINUE_IMPLEMENTATION:
                hint = (
                    f"THE CURRENT IMPLEMENTATION IS A STUB, NOT REAL LOGIC: {evaluation.rationale} "
                    f"The CURRENT FILE CONTENTS section above shows exactly what's on disk right now - "
                    f"use it as your base. Your next tool_input.content must be the FULL file with every "
                    f"stub/placeholder/TODO replaced by complete, working logic, not just the missing piece."
                )
                memory.add(step.id, "correction", f"Continuing implementation after incomplete_implementation: {evaluation.rationale}")
                eval2 = self._run_correction_cycle(goal, plan, memory, step, hint, "continued", on_event, run_log)
                if eval2.failure_mode == FailureMode.NONE:
                    step.status = StepStatus.DONE
                    memory.completed_step_ids.append(step.id)
                    i += 1
                continue

            if recovery.strategy == RecoveryStrategy.REGENERATE_ACTION:
                hint = evaluation.rationale
                memory.add(step.id, "correction", f"Regenerating action after result_inconsistency: {evaluation.rationale}")
                eval2 = self._run_correction_cycle(goal, plan, memory, step, hint, "regenerated", on_event, run_log)
                if eval2.failure_mode == FailureMode.NONE:
                    step.status = StepStatus.DONE
                    memory.completed_step_ids.append(step.id)
                    i += 1
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
        try:
            final_text = self.llm.chat(
                [
                    {"role": "system", "content": FINAL_SYNTHESIS_SYSTEM_PROMPT},
                    {"role": "user", "content": memory.as_context_string(max_entries=50)},
                ],
                temperature=0.2,
            )
        except Exception as e:  # noqa: BLE001
            # All step work is already recorded in run_log.steps regardless -
            # a failure here shouldn't discard a completed run, just note
            # that the prose summary specifically couldn't be generated.
            n_done = len(memory.completed_step_ids)
            n_unresolved = len(run_log.unresolved_subtasks)
            final_text = (
                f"[final synthesis call failed: {type(e).__name__}: {e}] "
                f"{n_done} step(s) completed, {n_unresolved} unresolved. "
                f"See run_log.steps for the full trace."
            )
        run_log.final_output = final_text
        run_log.completed = len(run_log.unresolved_subtasks) == 0
        from datetime import datetime, timezone
        run_log.ended_at = datetime.now(timezone.utc)
        _emit(on_event, "final", run_id=run_id, final_output=final_text, run_log=run_log.model_dump(mode="json"))
        return run_log