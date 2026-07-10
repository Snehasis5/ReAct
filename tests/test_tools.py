import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("LLM_MODE", "mock")

from src.tools import call_tool


def test_write_and_execute_python():
    r = call_tool("write_file", {"filename": "t_add.py", "content": "print(1+2)"})
    assert r.success
    r2 = call_tool("execute_python", {"filename": "t_add.py", "timeout": 5})
    assert r2.success
    assert r2.output["exit_code"] == 0
    assert "3" in r2.output["stdout"]


def test_execute_missing_file_is_graceful_tool_failure():
    r = call_tool("execute_python", {"filename": "does_not_exist_xyz.py"})
    assert r.success is False
    assert "does not exist" in r.error


def test_unknown_tool_is_graceful():
    r = call_tool("not_a_real_tool", {})
    assert r.success is False
    assert "unknown tool" in r.error


def test_invalid_schema_is_graceful():
    r = call_tool("write_file", {"filename": "x.py"})  # missing required 'content'
    assert r.success is False
    assert "invalid tool input schema" in r.error


def test_run_tests_reports_pass_fail_counts():
    call_tool("write_file", {"filename": "t_mod.py", "content": "def add(a,b): return a+b"})
    test_code = (
        "from t_mod import add\n"
        "def test_ok():\n    assert add(1,2) == 3\n"
        "def test_bad():\n    assert add(1,1) == 3\n"
    )
    call_tool("write_file", {"filename": "test_t_mod.py", "content": test_code})
    r = call_tool("run_tests", {"filename": "test_t_mod.py", "timeout": 20})
    assert r.success
    assert r.output["passed"] == 1
    assert r.output["failed"] == 1


def test_flaky_tool_fails_intermittently_across_many_calls():
    call_tool("write_file", {"filename": "report.txt", "content": "ok"})
    results = [call_tool("flaky_read_report", {"filename": "report.txt"}) for _ in range(60)]
    successes = sum(1 for r in results if r.success)
    failures = sum(1 for r in results if not r.success)
    # With a ~35% failure rate over 60 calls, both buckets should be non-trivial.
    assert successes > 0
    assert failures > 0
