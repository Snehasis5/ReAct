import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("LLM_MODE", "mock")

from src.monitor import decide_recovery
from src.schemas import FailureMode, PlanStep, RecoveryStrategy, SelfEvaluation


def _eval(failure_mode):
    return SelfEvaluation(
        step_id="s1", action_succeeded=False, result_makes_sense=False,
        still_on_track=False, failure_mode=failure_mode, rationale="test", confidence=0.9,
    )


def test_tool_failure_maps_to_retry_with_backoff():
    step = PlanStep(id="s1", description="d", attempts=0)
    rec = decide_recovery(_eval(FailureMode.TOOL_FAILURE), step, max_attempts=3)
    assert rec.strategy == RecoveryStrategy.RETRY_WITH_BACKOFF


def test_result_inconsistency_maps_to_regenerate_action():
    step = PlanStep(id="s1", description="d", attempts=0)
    rec = decide_recovery(_eval(FailureMode.RESULT_INCONSISTENCY), step, max_attempts=3)
    assert rec.strategy == RecoveryStrategy.REGENERATE_ACTION


def test_goal_drift_maps_to_replan_from_memory():
    step = PlanStep(id="s1", description="d", attempts=0)
    rec = decide_recovery(_eval(FailureMode.GOAL_DRIFT), step, max_attempts=3)
    assert rec.strategy == RecoveryStrategy.REPLAN_FROM_MEMORY


def test_budget_exhausted_forces_abandon_regardless_of_failure_mode():
    step = PlanStep(id="s1", description="d", attempts=3)
    rec = decide_recovery(_eval(FailureMode.TOOL_FAILURE), step, max_attempts=3)
    assert rec.strategy == RecoveryStrategy.ABANDON_SUBTASK


def test_no_failure_means_no_recovery():
    step = PlanStep(id="s1", description="d", attempts=0)
    ev = SelfEvaluation(step_id="s1", action_succeeded=True, result_makes_sense=True,
                         still_on_track=True, failure_mode=FailureMode.NONE, rationale="ok", confidence=0.9)
    assert decide_recovery(ev, step, max_attempts=3) is None
