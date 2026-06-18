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


async def test_snapshot_clears_stale_refs_on_retag(browser_page):
    # Re-tagging after a DOM change must not leave stale ref attributes: otherwise a ref
    # id (e.g. e2) ends up on more than one element and ref-based locators hit a
    # strict-mode "resolved to N elements" error.
    await browser_page.goto(LOGIN_URL)
    await take_snapshot(browser_page)
    # Insert a new element at the top so re-tagging shifts the ref-id assignment.
    await browser_page.evaluate(
        "() => { const b = document.createElement('button'); b.textContent = 'New'; "
        "document.body.insertBefore(b, document.body.firstChild); }"
    )
    _, elements = await take_snapshot(browser_page)

    for ref in {e.ref for e in elements}:
        count = await browser_page.locator(ref_selector(ref)).count()
        assert count == 1, f"ref {ref} resolved to {count} elements"
