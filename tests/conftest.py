"""Shared test fixtures: settings, a fake Anthropic client, and a headless browser page."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

from aiwebtest.asyncio_compat import configure_windows_event_loop_policy
from aiwebtest.config import (
    AgentConfig,
    BrowserConfig,
    ReportConfig,
    Settings,
    ViewportConfig,
)

configure_windows_event_loop_policy()

FIXTURE_SITE = Path(__file__).parent / "fixtures" / "site"
LOGIN_URL = (FIXTURE_SITE / "login.html").as_uri()


# --- Fake Anthropic client -------------------------------------------------------

@dataclass
class FakeTextBlock:
    text: str
    type: str = "text"


@dataclass
class FakeToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]
    type: str = "tool_use"


@dataclass
class FakeMessage:
    content: list[Any]
    stop_reason: str


class _FakeStream:
    def __init__(self, message: FakeMessage) -> None:
        self._message = message

    async def __aenter__(self) -> _FakeStream:
        return self

    async def __aexit__(self, *exc) -> None:
        return None

    @property
    def text_stream(self):
        # Mirror the SDK: a property returning an async iterator of text deltas.
        return self._iter_text()

    async def _iter_text(self):
        for block in self._message.content:
            if getattr(block, "type", None) == "text":
                yield block.text

    async def get_final_message(self) -> FakeMessage:
        return self._message


class _FakeMessages:
    def __init__(self, turns: list[FakeMessage]) -> None:
        self._turns = list(turns)

    def stream(self, **kwargs: Any) -> _FakeStream:
        if not self._turns:
            raise AssertionError("FakeAnthropicClient ran out of scripted turns")
        return _FakeStream(self._turns.pop(0))


class FakeAnthropicClient:
    """Returns scripted assistant turns; ignores the request payload."""

    def __init__(self, turns: list[FakeMessage]) -> None:
        self.messages = _FakeMessages(turns)


@dataclass
class _Counter:
    n: int = 0

    def next_id(self) -> str:
        self.n += 1
        return f"toolu_{self.n}"


def tool_turn(name: str, tool_input: dict[str, Any], text: str = "") -> FakeMessage:
    """Build a single-tool assistant turn (with optional leading reasoning text)."""
    counter = tool_turn._counter  # type: ignore[attr-defined]
    blocks: list[Any] = []
    if text:
        blocks.append(FakeTextBlock(text=text))
    blocks.append(FakeToolUseBlock(id=counter.next_id(), name=name, input=tool_input))
    return FakeMessage(content=blocks, stop_reason="tool_use")


tool_turn._counter = _Counter()  # type: ignore[attr-defined]


# --- Settings --------------------------------------------------------------------

@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        anthropic_api_key="test-key",
        model="claude-opus-4-8",
        effort="high",
        max_tokens=2048,
        browser=BrowserConfig(headless=True, channel=None, viewport=ViewportConfig()),
        agent=AgentConfig(max_steps=20, include_screenshots=True, allowed_domains=[]),
        report=ReportConfig(output_dir=str(tmp_path / "runs")),
        data={},
    )


# --- Headless browser page -------------------------------------------------------

@pytest_asyncio.fixture
async def browser_page(tmp_path: Path):
    from playwright.async_api import async_playwright

    pw = await async_playwright().start()
    browser = await pw.chromium.launch(headless=True)
    context = await browser.new_context()
    page = await context.new_page()
    try:
        yield page
    finally:
        await context.close()
        await browser.close()
        await pw.stop()
