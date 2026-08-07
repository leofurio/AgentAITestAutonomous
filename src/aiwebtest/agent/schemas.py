"""Pydantic models describing a test run, its steps, assertions and final report."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(UTC)


class Verdict(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"


class StepKind(StrEnum):
    REASONING = "reasoning"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


class ModelUsage(BaseModel):
    """One model as a run actually used it, with what it consumed.

    A run can involve more than one: the browser-driving agent and the instruction
    normalizer may sit on different providers and models (see ``normalizer_settings``).
    The provider adapters accumulate into these objects while the run is in flight, so
    a run that crashes still reports what it spent up to that point.
    """

    role: str = "agent"  # "agent" (browser loop) | "normalizer" (instruction pre-pass)
    provider: str = ""
    model: str = ""
    # Request-shaping settings, when the adapter actually sends them (effort is
    # Anthropic-only today); None means "not applicable to this provider".
    effort: str | None = None
    max_tokens: int | None = None
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    # Prompt-caching counters. Providers that report no usage leave every count at 0.
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class AssertionResult(BaseModel):
    description: str
    condition: str
    expected: str | None = None
    actual: str | None = None
    passed: bool
    timestamp: datetime = Field(default_factory=_now)
    screenshot_path: str | None = None


class Step(BaseModel):
    index: int
    kind: StepKind
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    result_summary: str | None = None
    text: str | None = None
    screenshot_path: str | None = None
    error: str | None = None
    # Stable descriptor of the targeted element (id/name/role/text), captured at
    # action time so the generated replay can use a robust locator instead of the
    # ephemeral ordinal ref.
    locator_hint: dict[str, Any] | None = None
    timestamp: datetime = Field(default_factory=_now)


class TestReport(BaseModel):
    run_id: str
    instruction: str
    # Canonical rewrite of `instruction` produced by the normalizer pass, if it ran.
    normalized_instruction: str | None = None
    target_url: str | None = None
    # Headline model of the run (the browser-driving agent); `models` below carries the
    # per-role detail, including the normalizer when it runs on a different model.
    model: str
    models: list[ModelUsage] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=_now)
    finished_at: datetime | None = None
    verdict: Verdict = Verdict.ERROR
    summary: str = ""
    steps: list[Step] = Field(default_factory=list)
    assertions: list[AssertionResult] = Field(default_factory=list)
