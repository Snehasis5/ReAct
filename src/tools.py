"""
Typed tools for the code-generation-and-execution domain, plus the
subprocess-based sandbox runtime.

Every tool has a Pydantic input schema and a Pydantic output schema.
`call_tool()` is the single entrypoint the agent uses - it validates input,
runs the tool, and *never* lets a raw exception escape: it always returns a
ToolResult(success=False, error=...) instead, so a bad/unexpected schema or
a crashing tool degrades gracefully rather than taking down the agent loop.

One tool (`flaky_read_report`) is intentionally unreliable: it simulates a
transient storage error on ~35% of calls. This is the "intentionally broken
tool that fails intermittently" required by the assignment, used to prove
the agent's tool_failure detection + retry-with-backoff recovery path.
"""
from __future__ import annotations

import os
import random
import re
import subprocess
import sys

from pydantic import BaseModel, ValidationError

from .config import settings
from .schemas import ToolResult

SANDBOX = settings.sandbox_dir
FLAKY_FAILURE_RATE = 0.35


def _sandbox_path(filename: str) -> str:
    # Guard against path traversal outside the sandbox.
    safe = os.path.normpath(filename).lstrip(os.sep)
    if ".." in safe.split(os.sep):
        raise ValueError(f"unsafe path: {filename}")
    return os.path.join(SANDBOX, safe)


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------

class WriteFileInput(BaseModel):
    filename: str
    content: str


class WriteFileOutput(BaseModel):
    path: str
    bytes_written: int


def write_file(inp: WriteFileInput) -> WriteFileOutput:
    path = _sandbox_path(inp.filename)
    os.makedirs(os.path.dirname(path) or SANDBOX, exist_ok=True)
    with open(path, "w") as f:
        f.write(inp.content)
    return WriteFileOutput(path=path, bytes_written=len(inp.content.encode("utf-8")))


# ---------------------------------------------------------------------------
# execute_python
# ---------------------------------------------------------------------------

class ExecutePythonInput(BaseModel):
    filename: str
    timeout: int = 10


class ExecutePythonOutput(BaseModel):
    exit_code: int
    stdout: str
    stderr: str


def execute_python(inp: ExecutePythonInput) -> ExecutePythonOutput:
    path = _sandbox_path(inp.filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist")
    try:
        proc = subprocess.run(
            [sys.executable, os.path.basename(path)],
            capture_output=True, text=True, timeout=inp.timeout, cwd=SANDBOX,
        )
        return ExecutePythonOutput(exit_code=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
    except subprocess.TimeoutExpired:
        return ExecutePythonOutput(exit_code=-1, stdout="", stderr=f"TIMEOUT after {inp.timeout}s")


# ---------------------------------------------------------------------------
# run_tests
# ---------------------------------------------------------------------------

class RunTestsInput(BaseModel):
    filename: str
    timeout: int = 20


class RunTestsOutput(BaseModel):
    exit_code: int
    passed: int
    failed: int
    stdout: str
    stderr: str


def run_tests(inp: RunTestsInput) -> RunTestsOutput:
    path = _sandbox_path(inp.filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", os.path.basename(path), "-q"],
        capture_output=True, text=True, timeout=inp.timeout, cwd=SANDBOX,
    )
    out = proc.stdout
    passed_m = re.search(r"(\d+) passed", out)
    failed_m = re.search(r"(\d+) failed", out)
    return RunTestsOutput(
        exit_code=proc.returncode,
        passed=int(passed_m.group(1)) if passed_m else 0,
        failed=int(failed_m.group(1)) if failed_m else (0 if proc.returncode == 0 else 1),
        stdout=out,
        stderr=proc.stderr,
    )


# ---------------------------------------------------------------------------
# flaky_read_report  (the intentionally broken tool)
# ---------------------------------------------------------------------------

class ReadReportInput(BaseModel):
    filename: str


class ReadReportOutput(BaseModel):
    content: str


def flaky_read_report(inp: ReadReportInput) -> ReadReportOutput:
    if random.random() < FLAKY_FAILURE_RATE:
        raise ConnectionError("simulated transient storage error: connection reset while reading artifact store")
    path = _sandbox_path(inp.filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    with open(path) as f:
        return ReadReportOutput(content=f.read())


# ---------------------------------------------------------------------------
# Registry + dispatcher
# ---------------------------------------------------------------------------

TOOL_REGISTRY: dict[str, dict] = {
    "write_file": {
        "func": write_file, "input": WriteFileInput, "output": WriteFileOutput,
        "description": "Write source code to a file in the sandbox workspace.",
    },
    "execute_python": {
        "func": execute_python, "input": ExecutePythonInput, "output": ExecutePythonOutput,
        "description": "Execute a python file in the sandbox and capture stdout/stderr/exit_code.",
    },
    "run_tests": {
        "func": run_tests, "input": RunTestsInput, "output": RunTestsOutput,
        "description": "Run pytest against a test file in the sandbox and get pass/fail counts.",
    },
    "flaky_read_report": {
        "func": flaky_read_report, "input": ReadReportInput, "output": ReadReportOutput,
        "description": (
            "Read back a previously written file as a 'report'. UNRELIABLE: fails "
            "intermittently (~35% of calls) with a simulated transient storage error."
        ),
    },
}


def tool_catalog_text() -> str:
    lines = []
    for name, spec in TOOL_REGISTRY.items():
        lines.append(f"- {name}({', '.join(spec['input'].model_fields.keys())}): {spec['description']}")
    return "\n".join(lines)


def call_tool(tool_name: str, tool_input: dict) -> ToolResult:
    spec = TOOL_REGISTRY.get(tool_name)
    if spec is None:
        return ToolResult(tool_name=tool_name, success=False, error=f"unknown tool '{tool_name}'")

    try:
        parsed_input = spec["input"](**tool_input)
    except ValidationError as e:
        # Unexpected/malformed schema from the LLM's proposed action - handled
        # gracefully, not a crash.
        return ToolResult(tool_name=tool_name, success=False, error=f"invalid tool input schema: {e}")

    try:
        result = spec["func"](parsed_input)
        return ToolResult(tool_name=tool_name, success=True, output=result.model_dump())
    except Exception as e:  # noqa: BLE001 - tools may raise anything; we contain it here
        return ToolResult(tool_name=tool_name, success=False, error=f"{type(e).__name__}: {e}")
