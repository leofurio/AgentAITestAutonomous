"""End-to-end: in-process localized repair heals a recording with a stale locator.

Drives a real headless browser. The AI repair *decision* is stubbed with a callback (a
FakeAnthropicClient can't return a ref that depends on a live snapshot taken inside the
executor), so this proves the executor wiring — deterministic match fails → repair is
consulted → the flow completes and re-records with the corrected locator — without an
API key. RepairAgent's own model parsing is covered in tests/unit/test_replay_repair.py.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from aiwebtest.agent.events import EventBus
from aiwebtest.agent.loop import AgentLoop
from aiwebtest.agent.schemas import StepKind, Verdict
from aiwebtest.browser.snapshot import take_snapshot
from aiwebtest.replay.executor import LocalizedReplayer
from tests.conftest import LOGIN_URL, FakeAnthropicClient, tool_turn

pytestmark = pytest.mark.asyncio


async def _make_recording(settings, browser_page, run_id):
    """Run the agent loop once against the fixture login to get a passing recording."""
    await browser_page.goto(LOGIN_URL)
    _, elements = await take_snapshot(browser_page)
    user = next(e.ref for e in elements if e.role == "text")
    pwd = next(e.ref for e in elements if e.role == "password")
    btn = next(e.ref for e in elements if "Log in" in e.name)
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
    bus = EventBus()
    run_dir = settings.output_dir / run_id
    loop = AgentLoop(
        client=FakeAnthropicClient(turns), settings=settings, run_id=run_id,
        instruction="Log in and verify the welcome message.",
        target_url=LOGIN_URL, data={}, bus=bus, run_dir=run_dir,
    )
    drain = asyncio.create_task(_drain(bus))
    report = await loop.run()
    await drain
    return report


async def _make_visible_assert_recording(settings, browser_page, run_id):
    """A recording whose only assertion checks an element *by ref* (so its target can
    be corrupted to simulate a vanished element)."""
    await browser_page.goto(LOGIN_URL)
    _, elements = await take_snapshot(browser_page)
    btn = next(e.ref for e in elements if "Log in" in e.name)
    turns = [
        tool_turn("navigate", {"url": LOGIN_URL}),
        tool_turn("get_page_snapshot", {}),
        tool_turn("assert_that", {
            "description": "login button visible", "condition": "visible", "ref": btn,
        }),
        tool_turn("finish_test", {"verdict": "pass", "summary": "ok"}),
    ]
    bus = EventBus()
    run_dir = settings.output_dir / run_id
    loop = AgentLoop(
        client=FakeAnthropicClient(turns), settings=settings, run_id=run_id,
        instruction="Check the login button is visible.",
        target_url=LOGIN_URL, data={}, bus=bus, run_dir=run_dir,
    )
    drain = asyncio.create_task(_drain(bus))
    report = await loop.run()
    await drain
    return report


async def _drain(bus):
    async for _ in bus:
        pass


def _break_click_locator(report):
    """Corrupt the click step's descriptor so no live element matches it."""
    click = next(
        s for s in report.steps
        if s.kind == StepKind.TOOL_CALL and s.tool_name == "click"
    )
    click.locator_hint = {
        "id": "gone-btn", "testid": "", "attr_name": "",
        "role": "button", "name": "Submit the order", "tag": "button",
    }
    return report


async def test_localized_repair_heals_a_stale_locator(settings, browser_page):
    report = await _make_recording(settings, browser_page, "rec_heal")
    assert report.verdict == Verdict.PASS
    _break_click_locator(report)

    async def repair(step, outline, elements):
        # Stand-in for the agent: pick the real "Log in" control from the live page.
        return next((e.ref for e in elements if "Log in" in e.name), None)

    run_id = "heal_ok"
    replayer = LocalizedReplayer(
        report=report, browser=settings.browser,
        output_dir=settings.output_dir / run_id, run_id=run_id, repair=repair,
    )
    outcome = await replayer.run()

    assert (outcome.verdict, outcome.repaired) == ("pass", True)

    # The heal produced a normal recording, and its report.json passed.
    heal_dir = settings.output_dir / run_id
    assert json.loads((heal_dir / "report.json").read_text())["verdict"] == "pass"
    # The re-recorded script carries the *corrected* locator (the fixture button id),
    # so the next deterministic replay works without any model.
    script = (heal_dir / "playwright_test.py").read_text()
    assert "'id': 'login-btn'" in script
    assert "gone-btn" not in script


async def test_without_repair_a_stale_locator_errors(settings, browser_page):
    # Same broken recording, but no repair callback: the deterministic match fails and
    # the heal errors (which is what escalates to a full agent re-run in the suite).
    report = await _make_recording(settings, browser_page, "rec_noheal")
    _break_click_locator(report)

    run_id = "heal_err"
    replayer = LocalizedReplayer(
        report=report, browser=settings.browser,
        output_dir=settings.output_dir / run_id, run_id=run_id, repair=None,
    )
    outcome = await replayer.run()

    assert (outcome.verdict, outcome.repaired) == ("error", False)
    assert "Could not resolve" in outcome.summary


async def test_localized_repair_does_not_mask_a_vanished_assertion_target(settings, browser_page):
    # An assertion whose target element is gone must surface as a FAIL (a real
    # regression), never be AI-repaired into a false pass. The repair callback must not
    # even be consulted for an assertion.
    report = await _make_visible_assert_recording(settings, browser_page, "rec_assert")
    assert report.verdict == Verdict.PASS
    assert_step = next(
        s for s in report.steps
        if s.kind == StepKind.TOOL_CALL and s.tool_name == "assert_that"
    )
    assert_step.locator_hint = {
        "id": "gone", "testid": "", "attr_name": "",
        "role": "button", "name": "Not here", "tag": "button",
    }

    called = False

    async def repair(step, outline, elements):
        nonlocal called
        called = True
        return next((e.ref for e in elements if "Log in" in e.name), None)

    run_id = "heal_assert"
    replayer = LocalizedReplayer(
        report=report, browser=settings.browser,
        output_dir=settings.output_dir / run_id, run_id=run_id, repair=repair,
    )
    outcome = await replayer.run()

    assert (outcome.verdict, outcome.repaired) == ("fail", False)
    assert called is False  # assertions are never re-pointed by the AI
    assert json.loads(
        (settings.output_dir / run_id / "report.json").read_text()
    )["verdict"] == "fail"


async def test_localized_repair_replays_a_valid_recording_without_repair(settings, browser_page):
    # When nothing is broken, the executor just replays in-process and passes, touching
    # no model at all (repaired stays False).
    report = await _make_recording(settings, browser_page, "rec_clean")

    called = False

    async def repair(step, outline, elements):
        nonlocal called
        called = True
        return None

    run_id = "heal_clean"
    replayer = LocalizedReplayer(
        report=report, browser=settings.browser,
        output_dir=settings.output_dir / run_id, run_id=run_id, repair=repair,
    )
    outcome = await replayer.run()

    assert (outcome.verdict, outcome.repaired) == ("pass", False)
    assert called is False  # deterministic match covered every step
