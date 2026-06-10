"""End-to-end: the full AgentLoop driving a real headless browser via a fake LLM.

This exercises loop → toolset → browser → report without any Anthropic API cost,
which is what makes it runnable in CI / the cloud container.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

import pytest

from aiwebtest.agent.events import EventBus
from aiwebtest.agent.loop import AgentLoop
from aiwebtest.agent.schemas import Verdict
from aiwebtest.browser.snapshot import take_snapshot
from tests.conftest import LOGIN_URL, FakeAnthropicClient, tool_turn

pytestmark = pytest.mark.asyncio


async def _discover_refs(browser_page):
    await browser_page.goto(LOGIN_URL)
    _, elements = await take_snapshot(browser_page)
    user = next(e.ref for e in elements if e.role == "text")
    pwd = next(e.ref for e in elements if e.role == "password")
    btn = next(e.ref for e in elements if "Log in" in e.name)
    return user, pwd, btn


async def _run(settings, client, run_id="run1"):
    bus = EventBus()
    run_dir = settings.output_dir / run_id
    loop = AgentLoop(
        client=client, settings=settings, run_id=run_id,
        instruction="Log in and verify the welcome message.",
        target_url=LOGIN_URL, data={}, bus=bus, run_dir=run_dir,
    )
    events: list[dict] = []

    async def drain():
        async for e in bus:
            events.append(e)

    drain_task = asyncio.create_task(drain())
    report = await loop.run()
    await drain_task
    return report, events, run_dir


async def test_successful_login_run(settings, browser_page):
    user, pwd, btn = await _discover_refs(browser_page)
    turns = [
        tool_turn("navigate", {"url": LOGIN_URL}, text="Opening the login page."),
        tool_turn("get_page_snapshot", {}),
        tool_turn("type_text", {"ref": user, "text": "demo"}),
        tool_turn("type_text", {"ref": pwd, "text": "secret"}),
        tool_turn("click", {"ref": btn}),
        tool_turn("assert_that", {
            "description": "dashboard welcome message is shown",
            "condition": "text_contains", "expected": "Welcome, demo",
        }),
        tool_turn("finish_test", {"verdict": "pass", "summary": "Login succeeded."}),
    ]
    report, events, run_dir = await _run(settings, FakeAnthropicClient(turns))

    assert report.verdict == Verdict.PASS
    assert len(report.assertions) == 1 and report.assertions[0].passed

    # Report artifacts written to disk.
    data = json.loads((run_dir / "report.json").read_text())
    assert data["verdict"] == "pass"
    assert (run_dir / "report.html").exists()
    assert list((run_dir / "screenshots").glob("*.png")), "expected screenshots"

    # The WebSocket event sequence covers the lifecycle.
    types = [e["type"] for e in events]
    assert types[0] == "status"
    assert "reasoning" in types and "step" in types and "assertion" in types
    assert types[-2:] == ["report", "status"]
    report_evt = next(e for e in events if e["type"] == "report")
    assert report_evt["data"]["verdict"] == "pass"


async def test_generated_replay_script_runs(settings, browser_page):
    # The generated playwright_test.py must actually replay the run on its own —
    # this guards against the ephemeral-ref regression (refs wiped after re-render).
    user, pwd, btn = await _discover_refs(browser_page)
    turns = [
        tool_turn("navigate", {"url": LOGIN_URL}),
        tool_turn("get_page_snapshot", {}),
        tool_turn("type_text", {"ref": user, "text": "demo"}),
        tool_turn("type_text", {"ref": pwd, "text": "secret"}),
        tool_turn("click", {"ref": btn}),
        tool_turn("assert_that", {
            "description": "welcome shown", "condition": "text_contains",
            "expected": "Welcome, demo",
        }),
        tool_turn("finish_test", {"verdict": "pass", "summary": "ok"}),
    ]
    _report, _events, run_dir = await _run(settings, FakeAnthropicClient(turns), run_id="replay")
    script = run_dir / "playwright_test.py"
    assert script.exists()

    proc = await asyncio.create_subprocess_exec(
        sys.executable, str(script),
        cwd=str(run_dir),
        env={**os.environ, "AIWEBTEST_REPLAY_HEADLESS": "1"},
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
    assert proc.returncode == 0, stderr.decode(errors="replace")
    assert "PASS:" in stdout.decode(errors="replace")


async def test_failed_assertion_yields_fail_verdict(settings, browser_page):
    user, pwd, btn = await _discover_refs(browser_page)
    turns = [
        tool_turn("navigate", {"url": LOGIN_URL}),
        tool_turn("get_page_snapshot", {}),
        tool_turn("type_text", {"ref": user, "text": "demo"}),
        tool_turn("type_text", {"ref": pwd, "text": "WRONG"}),
        tool_turn("click", {"ref": btn}),
        tool_turn("assert_that", {
            "description": "welcome should appear", "condition": "text_contains",
            "expected": "Welcome, demo",
        }),
        # Agent wrongly declares pass; the failed assertion must override to fail.
        tool_turn("finish_test", {"verdict": "pass", "summary": "thought it worked"}),
    ]
    report, _events, _run_dir = await _run(settings, FakeAnthropicClient(turns), run_id="run2")
    assert report.verdict == Verdict.FAIL
