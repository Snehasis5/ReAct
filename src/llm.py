"""
OpenRouter orchestration client.

Two implementations behind one interface (`chat`, `chat_json`, `chat_stream`):

- LiveLLMClient   -> real calls to OpenRouter via the OpenAI-compatible SDK.
- MockLLMClient   -> deterministic offline stub used when LLM_MODE=mock.
                     Lets the whole agent loop (planning, ReAct, self-eval,
                     recovery) be exercised and unit-tested without network
                     access or burning API credits. This is what
                     run_evaluation.py uses by default in CI/sandboxed envs.

agent.py and monitor.py only ever talk to `get_llm_client()` - they don't
know or care which implementation is behind it.
"""
from __future__ import annotations

import json
import random
import re
import time
from typing import Callable, Optional

from .config import settings

Message = dict[str, str]
StreamCallback = Callable[[str], None]


def extract_json(text: str) -> dict:
    """Best-effort extraction of a JSON object from an LLM response that may
    be wrapped in markdown fences or surrounded by prose."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    else:
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        candidate = text[brace_start:brace_end + 1] if (brace_start != -1 and brace_end != -1 and brace_end > brace_start) else text
    return json.loads(candidate)


REPAIR_INSTRUCTION = (
    "Your previous response was not valid JSON - it looked like prose, markdown, or "
    "a Python-style function call (e.g. `write_file(filename=..., content=...)`) instead "
    "of the required JSON object. Convert your PREVIOUS answer into ONLY a single valid "
    "JSON object matching the schema you were given. Output the JSON object and nothing "
    "else - no explanation, no code fences, no function-call syntax."
)


class LiveLLMClient:
    def __init__(self):
        from openai import OpenAI  # local import: not needed in mock mode
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.openrouter_api_key,
        )
        self.model = settings.openrouter_model

    def chat(self, messages: list[Message], temperature: float = 0.2, max_retries: int = 2) -> str:
        last_err: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, messages=messages, temperature=temperature,
                )
            except Exception as e:  # network blip, rate limit, etc.
                last_err = e
                if attempt < max_retries:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise

            choice = resp.choices[0] if resp.choices else None
            content = (choice.message.content or "").strip() if choice else ""
            if content:
                return content

            # Some OpenRouter models (particularly free-tier / reasoning models)
            # occasionally return an empty `content` field - either a transient
            # blip, or the real answer landed in a separate `reasoning` field
            # instead. Neither case is "the model gave a real, malformed
            # answer" (that's what chat_json's repair-prompt loop is for) -
            # this is the completion itself coming back empty, so retry the
            # call rather than propagating an empty string downstream.
            reasoning_field = getattr(choice.message, "reasoning", None) if choice else None
            if reasoning_field and str(reasoning_field).strip():
                return str(reasoning_field).strip()

            last_err = ValueError("LLM returned empty content")
            if attempt < max_retries:
                time.sleep(1.5 * (attempt + 1))
                continue

        raise last_err or ValueError("LLM returned empty content after retries")

    def chat_stream(self, messages: list[Message], on_token: StreamCallback,
                     temperature: float = 0.2) -> str:
        stream = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=temperature, stream=True,
        )
        full = []
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                full.append(delta)
                on_token(delta)
        return "".join(full)

    def chat_json(self, messages: list[Message], temperature: float = 0.0, max_repair_attempts: int = 2) -> dict:
        working_messages = list(messages)
        last_text = ""
        last_error: Optional[Exception] = None

        for attempt in range(max_repair_attempts + 1):
            text = self.chat(working_messages, temperature=temperature)
            last_text = text
            try:
                return extract_json(text)
            except Exception as e:  # noqa: BLE001
                last_error = e
                if attempt < max_repair_attempts:
                    # Give the model its own bad output back and ask it to fix the
                    # format - this recovers from models (especially small/free ones)
                    # that ignore "return only JSON" and instead write prose or
                    # pseudo-function-call syntax.
                    working_messages = working_messages + [
                        {"role": "assistant", "content": text},
                        {"role": "user", "content": REPAIR_INSTRUCTION},
                    ]
                    continue

        raise ValueError(
            f"LLM did not return valid JSON after {max_repair_attempts + 1} attempt(s). "
            f"Raw output: {last_text!r}"
        ) from last_error


class MockLLMClient:
    """
    Deterministic stub LLM for offline runs.

    It doesn't "understand" the goal - it pattern-matches on which prompt
    role/content is asking for what (planner / reasoner / judge / recovery /
    synthesis) via markers in the system prompt, and returns plausible,
    schema-valid responses. It also deliberately injects a controlled
    inconsistency on the "recovery test" goal so the self-correction path is
    exercised during evaluation.
    """

    def __init__(self, seed: int = 7):
        self._rng = random.Random(seed)

    def _last_user_content(self, messages: list[Message]) -> str:
        for m in reversed(messages):
            if m["role"] == "user":
                return m["content"]
        return ""

    def _system_marker(self, messages: list[Message]) -> str:
        for m in messages:
            if m["role"] == "system":
                return m["content"]
        return ""

    def chat(self, messages: list[Message], temperature: float = 0.2) -> str:
        sys_prompt = self._system_marker(messages)
        user = self._last_user_content(messages)

        if "FINAL_SYNTHESIS" in sys_prompt:
            return (
                "Task complete. The agent wrote a Python module implementing the "
                "requested function(s), executed it in the sandbox, and validated "
                "behavior with pytest. See working memory for step-by-step evidence."
            )
        return "Mock LLM: no matching handler for this prompt."

    def chat_stream(self, messages, on_token: StreamCallback, temperature: float = 0.2) -> str:
        text = self.chat(messages, temperature=temperature)
        for word in text.split(" "):
            piece = word + " "
            on_token(piece)
        return text

    def chat_json(self, messages: list[Message], temperature: float = 0.0) -> dict:
        sys_prompt = self._system_marker(messages)
        user = self._last_user_content(messages)

        if "PLANNER" in sys_prompt:
            return self._mock_plan(user)
        if "REASONER" in sys_prompt:
            return self._mock_reason(user)
        if "JUDGE" in sys_prompt:
            return self._mock_judge(user)
        if "REPLAN" in sys_prompt:
            return self._mock_replan(user)
        raise ValueError(f"MockLLMClient: no JSON handler for prompt: {sys_prompt[:80]}")

    # -- mock behaviors -----------------------------------------------------

    def _mock_plan(self, goal: str) -> dict:
        return {
            "steps": [
                {"id": "s1", "description": f"Write a Python module implementing: {goal}"},
                {"id": "s2", "description": "Execute the module to confirm it runs without errors"},
                {"id": "s3", "description": "Write a pytest test file covering the core behavior"},
                {"id": "s4", "description": "Run the test suite and confirm all tests pass"},
                {"id": "s5", "description": "Read back the execution report to confirm final state"},
            ]
        }

    def _mock_reason(self, ctx: str) -> dict:
        step_id = "s1"
        m = re.search(r"CURRENT STEP ID:\s*(\S+)", ctx)
        if m:
            step_id = m.group(1)
        attempt = 1
        m2 = re.search(r"ATTEMPT:\s*(\d+)", ctx)
        if m2:
            attempt = int(m2.group(1))

        if step_id == "s1":
            code = (
                "def target_function(*args, **kwargs):\n"
                "    \"\"\"Auto-generated by mock reasoning step.\"\"\"\n"
                "    return sum(args) if args else 0\n"
            )
            return {
                "thought": "I need to create the module first before I can execute or test it.",
                "tool_name": "write_file",
                "tool_input": {"filename": "solution.py", "content": code},
            }
        if step_id == "s2":
            return {
                "thought": "The module exists now. Let's execute it to make sure it imports/runs cleanly.",
                "tool_name": "execute_python",
                "tool_input": {"filename": "solution.py", "timeout": 10},
            }
        if step_id == "s3":
            test_code = (
                "from solution import target_function\n\n"
                "def test_sum():\n"
                "    assert target_function(1, 2, 3) == 6\n\n"
                "def test_empty():\n"
                "    assert target_function() == 0\n"
            )
            return {
                "thought": "Now I'll write a small pytest file to verify correctness, not just that it runs.",
                "tool_name": "write_file",
                "tool_input": {"filename": "test_solution.py", "content": test_code},
            }
        if step_id == "s4":
            return {
                "thought": "Run the test suite to confirm the implementation is actually correct.",
                "tool_name": "run_tests",
                "tool_input": {"filename": "test_solution.py", "timeout": 20},
            }
        if step_id == "s5":
            return {
                "thought": (
                    "As a final check I want to read back the report artifact to confirm "
                    "the recorded run state before closing out the goal."
                ),
                "tool_name": "flaky_read_report",
                "tool_input": {"filename": "test_solution.py"},
            }
        return {
            "thought": "Re-attempting the previous action with adjusted input after a failure.",
            "tool_name": "execute_python",
            "tool_input": {"filename": "solution.py", "timeout": 10},
        }

    def _mock_judge(self, ctx: str) -> dict:
        step_id = "s1"
        m = re.search(r"CURRENT STEP ID:\s*(\S+)", ctx)
        if m:
            step_id = m.group(1)

        tool_failed = "\"success\": false" in ctx or "'success': False" in ctx

        if tool_failed:
            return {
                "action_succeeded": False,
                "result_makes_sense": False,
                "still_on_track": False,
                "failure_mode": "tool_failure",
                "rationale": "The tool call raised an error rather than returning a result.",
                "confidence": 0.9,
            }

        if step_id == "s4" and "exit_code\": 1" in ctx:
            return {
                "action_succeeded": True,
                "result_makes_sense": False,
                "still_on_track": True,
                "failure_mode": "result_inconsistency",
                "rationale": "pytest ran but reported failing tests, which contradicts the goal of a working implementation.",
                "confidence": 0.85,
            }

        return {
            "action_succeeded": True,
            "result_makes_sense": True,
            "still_on_track": True,
            "failure_mode": "none",
            "rationale": "Tool executed successfully and the output is consistent with the current step's intent.",
            "confidence": 0.8,
        }

    def _mock_replan(self, ctx: str) -> dict:
        return {
            "steps": [
                {"id": "r1", "description": "Rewrite solution.py with a corrected implementation based on the failing test output"},
                {"id": "r2", "description": "Re-run the test suite to confirm the fix"},
            ]
        }


def get_llm_client():
    if settings.llm_mode == "mock":
        return MockLLMClient()
    return LiveLLMClient()