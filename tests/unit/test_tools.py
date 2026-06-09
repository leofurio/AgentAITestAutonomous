"""BrowserToolset: tool success/error paths, assertions and the domain guardrail."""

from __future__ import annotations

import pytest

from aiwebtest.browser.snapshot import take_snapshot
from aiwebtest.browser.tools import BrowserToolset
from tests.conftest import LOGIN_URL

pytestmark = pytest.mark.asyncio


async def _toolset(browser_page, tmp_path, allowed=None):
    return BrowserToolset(
        page=browser_page,
        screenshot_dir=tmp_path / "shots",
        allowed_domains=allowed or [],
        include_screenshots=True,
    )


async def _ref_for(browser_page, predicate):
    _, elements = await take_snapshot(browser_page)
    for e in elements:
        if predicate(e):
            return e.ref
    raise AssertionError("no matching element")


async def test_navigate_and_snapshot(browser_page, tmp_path):
    ts = await _toolset(browser_page, tmp_path)
    out = await ts.dispatch("navigate", {"url": LOGIN_URL})
    assert not out.is_error
    snap = await ts.dispatch("get_page_snapshot", {"include_screenshot": True})
    assert not snap.is_error
    assert snap.screenshot_path is not None


async def test_login_flow_and_assertion(browser_page, tmp_path):
    ts = await _toolset(browser_page, tmp_path)
    await ts.dispatch("navigate", {"url": LOGIN_URL})

    user_ref = await _ref_for(browser_page, lambda e: e.role == "text")
    pass_ref = await _ref_for(browser_page, lambda e: e.role == "password")
    btn_ref = await _ref_for(browser_page, lambda e: "Log in" in e.name)

    await ts.dispatch("type_text", {"ref": user_ref, "text": "demo"})
    await ts.dispatch("type_text", {"ref": pass_ref, "text": "secret"})
    await ts.dispatch("click", {"ref": btn_ref})

    out = await ts.dispatch(
        "assert_that",
        {"description": "welcome shown", "condition": "text_contains", "expected": "Welcome, demo"},
    )
    assert out.assertion is not None
    assert out.assertion.passed is True


async def test_failed_assertion_records_fail(browser_page, tmp_path):
    ts = await _toolset(browser_page, tmp_path)
    await ts.dispatch("navigate", {"url": LOGIN_URL})
    out = await ts.dispatch(
        "assert_that",
        {"description": "nope", "condition": "text_contains", "expected": "NotOnThisPage"},
    )
    assert out.assertion.passed is False


async def test_stale_ref_returns_error_not_raise(browser_page, tmp_path):
    ts = await _toolset(browser_page, tmp_path)
    await ts.dispatch("navigate", {"url": LOGIN_URL})
    out = await ts.dispatch("click", {"ref": "e9999"})  # nonexistent ref
    assert out.is_error is True
    assert "snapshot" in out.summary.lower()


async def test_domain_guard_blocks_offsite(browser_page, tmp_path):
    ts = await _toolset(browser_page, tmp_path, allowed=["example.com"])
    out = await ts.dispatch("navigate", {"url": "https://evil.test/page"})
    assert out.is_error is True
    assert "outside the allowed domains" in out.summary
