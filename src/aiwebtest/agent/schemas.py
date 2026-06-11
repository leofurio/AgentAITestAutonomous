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
    target_url: str | None = None
    model: str
    started_at: datetime = Field(default_factory=_now)
    finished_at: datetime | None = None
    verdict: Verdict = Verdict.ERROR
    summary: str = ""
    steps: list[Step] = Field(default_factory=list)
    assertions: list[AssertionResult] = Field(default_factory=list)
