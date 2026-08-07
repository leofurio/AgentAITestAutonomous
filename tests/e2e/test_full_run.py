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
from tests.conftest import LOGIN_URL, FakeAnthropicClient, text_turn, tool_turn

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

    # The report details the model that drove the run, with a call count matching the
    # turns the loop actually consumed (no normalizer here, so a single entry).
    assert len(data["models"]) == 1
    agent_model = data["models"][0]
    assert agent_model["role"] == "agent"
    assert agent_model["provider"] == "anthropic"
    assert agent_model["model"] == settings.model
    assert agent_model["effort"] == "high" and agent_model["max_tokens"] == 2048
    assert agent_model["calls"] == len(turns)
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
    # The fixture inputs/button carry ids, so the replay must carry stable id
    # descriptors (resolve() turns them into [id="..."] at runtime) rather than
    # relying on the ephemeral ordinal ref.
    text = script.read_text()
    assert "'id': 'username'" in text or "'id': 'login-btn'" in text

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


async def test_generated_script_passes_under_pytest(settings, browser_page):
    # The same artifact is dual-use: `pytest playwright_test.py` must collect the
    # test_replay() entrypoint and pass — this is what makes a recorded run a
    # drop-in member of a CI suite.
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
    _report, _events, run_dir = await _run(settings, FakeAnthropicClient(turns), run_id="aspytest")

    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "pytest", "playwright_test.py", "-q",
        "-p", "no:cacheprovider", "--noconftest",
        cwd=str(run_dir),
        env={**os.environ, "AIWEBTEST_REPLAY_HEADLESS": "1"},
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=180)
    out = stdout.decode(errors="replace")
    assert proc.returncode == 0, out + stderr.decode(errors="replace")
    assert "1 passed" in out


async def test_suite_replays_a_recorded_run(settings, browser_page):
    # Suite integration end-to-end: a completed live run becomes a saved test whose
    # replay re-executes the REAL generated script in a subprocess — no LLM involved.
    from types import SimpleNamespace

    from aiwebtest.suite import SuiteRunner, SuiteStore

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
    await _run(settings, FakeAnthropicClient(turns), run_id="suiterec")

    store = SuiteStore(settings.output_dir / "suite.json")
    test = store.add("login flow", "Log in and verify the welcome message.",
                     target_url=LOGIN_URL, source_run_id="suiterec")
    runner = SuiteRunner(SimpleNamespace(settings=settings), store)

    record = await runner.run_test(test.test_id, mode="replay")

    assert (record.mode, record.verdict) == ("replay", "pass")
    assert "assertion" in record.summary  # summary comes from the replay's own report
    # The replay produced the same artifacts as a live run, in its own run dir.
    replay_dir = settings.output_dir / record.run_id
    assert json.loads((replay_dir / "report.json").read_text())["verdict"] == "pass"
    assert store.get(test.test_id).history[-1].verdict == "pass"


async def test_replay_continues_after_failed_assertion(settings, browser_page):
    # A failed assertion must not abort the replay: like the live agent run, the
    # remaining steps (including later assertions) still execute, and the script
    # ends with a FAIL verdict and exit code 1.
    turns = [
        tool_turn("navigate", {"url": LOGIN_URL}),
        tool_turn("get_page_snapshot", {}),
        tool_turn("assert_that", {
            "description": "text that does not exist", "condition": "text_contains",
            "expected": "No such text on this page",
        }),
        tool_turn("assert_that", {
            "description": "still on the login page", "condition": "url_contains",
            "expected": "login",
        }),
        tool_turn("finish_test", {"verdict": "fail", "summary": "one check failed"}),
    ]
    _report, _events, run_dir = await _run(settings, FakeAnthropicClient(turns), run_id="soft")
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
    out = stdout.decode(errors="replace")
    assert proc.returncode == 1, stderr.decode(errors="replace")
    assert "FAIL: text that does not exist" in out
    assert "PASS: still on the login page" in out   # executed AFTER the failure
    assert "REPLAY RESULT: FAIL - 1 of 2 assertion(s) failed" in out


async def test_normalizer_pass_rewrites_instruction(settings, browser_page):
    # With a normalizer factory wired in, the loop first rewrites the instruction into a
    # canonical spec, records it on the report, and emits a "normalized" event — without
    # disturbing the scripted browser-driving turns.
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
    canonical = "GOAL: log in\nSTEPS:\n1. navigate to login"

    bus = EventBus()
    run_dir = settings.output_dir / "norm"
    loop = AgentLoop(
        client=FakeAnthropicClient(turns), settings=settings, run_id="norm",
        instruction="log in pls", target_url=LOGIN_URL, data={}, bus=bus, run_dir=run_dir,
        normalizer_factory=lambda: FakeAnthropicClient([text_turn(canonical)]),
        normalizer_settings=settings,
    )
    events: list[dict] = []

    async def drain():
        async for e in bus:
            events.append(e)

    drain_task = asyncio.create_task(drain())
    report = await loop.run()
    await drain_task

    assert report.verdict == Verdict.PASS
    assert report.normalized_instruction == canonical
    assert report.instruction == "log in pls"  # original preserved

    norm_evt = next(e for e in events if e["type"] == "normalized")
    assert norm_evt["data"]["text"] == canonical
    data = json.loads((run_dir / "report.json").read_text())
    assert data["normalized_instruction"] == canonical

    # Both models the run used are reported separately, each with its own call count,
    # so a normalizer on a cheaper model is visible as its own line item.
    roles = {m["role"]: m for m in data["models"]}
    assert set(roles) == {"agent", "normalizer"}
    assert roles["agent"]["calls"] == len(turns)
    assert roles["normalizer"]["calls"] == 1


async def test_reminder_lets_agent_finish_after_prose_only_turn(settings, browser_page):
    # If the agent ends a turn with prose and no tool call, the loop nudges it to call
    # finish_test instead of silently passing. Here it complies on the next turn → PASS.
    turns = [
        tool_turn("navigate", {"url": LOGIN_URL}),
        text_turn("I believe the test is complete."),
        tool_turn("finish_test", {"verdict": "pass", "summary": "done"}),
    ]
    report, _events, _run_dir = await _run(settings, FakeAnthropicClient(turns), run_id="nudge")
    assert report.verdict == Verdict.PASS


async def test_no_finish_test_yields_error_not_pass(settings, browser_page):
    # An agent that never calls finish_test, even after the reminders, is inconclusive —
    # it must NOT be reported as an implicit pass. settings.agent.max_retries == 2, so the
    # loop reminds twice and gives up on the third prose-only turn.
    turns = [
        text_turn("Looks fine to me."),
        text_turn("Still looks fine."),
        text_turn("Yep, done."),
    ]
    report, _events, _run_dir = await _run(settings, FakeAnthropicClient(turns), run_id="noverdict")
    assert report.verdict == Verdict.ERROR
    assert "without calling finish_test" in report.summary


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
