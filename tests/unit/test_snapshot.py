"""Snapshot ref-mapping against the fixture login page."""

from __future__ import annotations

import pytest

from aiwebtest.browser.snapshot import ref_selector, take_snapshot
from tests.conftest import LOGIN_URL

pytestmark = pytest.mark.asyncio


async def test_snapshot_assigns_resolvable_refs(browser_page):
    await browser_page.goto(LOGIN_URL)
    outline, elements = await take_snapshot(browser_page)

    assert "[ref=" in outline
    refs = {e.ref for e in elements}
    assert refs, "expected at least one interactive element"

    # Every ref must resolve back to exactly one element in the DOM.
    for ref in refs:
        count = await browser_page.locator(ref_selector(ref)).count()
        assert count == 1

    # The login button should be captured with its accessible name.
    assert any("Log in" in e.name for e in elements)


async def test_outline_lists_inputs(browser_page):
    await browser_page.goto(LOGIN_URL)
    _, elements = await take_snapshot(browser_page)
    roles = {e.role for e in elements}
    assert "text" in roles or "password" in roles
