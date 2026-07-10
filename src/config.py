"""
Centralized configuration + environment validation.

Fails loudly and early (at import time) if required config is missing or
malformed, rather than letting the agent limp along and fail deep inside
a tool call with a confusing stack trace.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class Settings:
    openrouter_api_key: str
    openrouter_model: str
    llm_mode: str  # "live" | "mock"
    max_replan_attempts: int = 3
    tool_timeout_seconds: int = 15
    sandbox_dir: Path = field(default_factory=lambda: ROOT_DIR / "sandbox_workspace")
    log_dir: Path = field(default_factory=lambda: ROOT_DIR / "logs")

    def validate(self) -> None:
        if self.llm_mode == "live":
            if not self.openrouter_api_key or self.openrouter_api_key.startswith("sk-or-v1-your-key"):
                raise ConfigError(
                    "OPENROUTER_API_KEY is missing or still set to the placeholder value in .env. "
                    "Set a real key, or set LLM_MODE=mock to run offline with the stub LLM."
                )
            if not self.openrouter_model:
                raise ConfigError("OPENROUTER_MODEL is not set in .env.")
        if self.max_replan_attempts < 1:
            raise ConfigError("MAX_REPLAN_ATTEMPTS must be >= 1.")


def _load_settings() -> Settings:
    mode = os.getenv("LLM_MODE", "live").strip().lower()
    if mode not in ("live", "mock"):
        raise ConfigError(f"LLM_MODE must be 'live' or 'mock', got: {mode!r}")

    settings = Settings(
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY", "").strip(),
        openrouter_model=os.getenv("OPENROUTER_MODEL", "openrouter/free").strip(),
        llm_mode=mode,
        max_replan_attempts=int(os.getenv("MAX_REPLAN_ATTEMPTS", "3")),
        tool_timeout_seconds=int(os.getenv("TOOL_TIMEOUT_SECONDS", "15")),
    )
    settings.validate()
    settings.sandbox_dir.mkdir(parents=True, exist_ok=True)
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    return settings


settings = _load_settings()
