"""Deterministically match a recorded locator hint against a live page snapshot.

This is the in-process twin of the candidate chain baked into the generated replay
script (``report/playwright_codegen.py``): given the stable descriptor captured at
record time (testid / id / name / role / text) it returns the ref id of the first
live snapshot element that satisfies the strongest available criterion. The recorded
ref itself is deliberately *not* used — it is an ordinal from a different snapshot and
matching on it is what makes a stale recording silently target the wrong element. When
nothing matches, the caller escalates to the AI repair pass.
"""

from __future__ import annotations

import re

from ..browser.snapshot import SnapshotElement

# Mirror of the ROLE_MAP / ROLE_OK baked into playwright_codegen's generated helper
# body. Kept in sync by test_replay_matcher.test_role_maps_mirror_codegen.
ROLE_MAP = {
    "text": "textbox", "email": "textbox", "password": "textbox", "search": "textbox",
    "tel": "textbox", "url": "textbox", "number": "spinbutton",
    "checkbox": "checkbox", "radio": "radio",
}
ROLE_OK = {"button", "link", "heading", "textbox", "checkbox", "radio", "tab", "menuitem"}

_NAME_ATTR_TAGS = {"input", "select", "textarea"}


def _norm(text: str | None) -> str:
    """Collapse whitespace and lowercase, matching Playwright's loose text matching."""
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def _mapped_role(role: str | None) -> str:
    return ROLE_MAP.get(role or "", role or "")


def match_ref(hint: dict | None, elements: list[SnapshotElement]) -> str | None:
    """Return the ref of the live element best matching ``hint``, or None.

    Candidates are tried in the same priority order as the generated replay's
    ``_candidates`` (testid, id, name attribute, role+name, tag+text, text); within a
    priority the first element in snapshot order wins, mirroring Playwright ``.first``.
    """
    if not hint:
        return None

    testid = hint.get("testid")
    if testid:
        for el in elements:
            if el.testid == testid:
                return el.ref

    el_id = hint.get("id")
    if el_id:
        for el in elements:
            if el.el_id == el_id:
                return el.ref

    attr_name = hint.get("attr_name")
    if attr_name and hint.get("tag") in _NAME_ATTR_TAGS:
        for el in elements:
            if el.attr_name == attr_name:
                return el.ref

    name = _norm(hint.get("name"))
    role = _mapped_role(hint.get("role"))
    if name and role in ROLE_OK:
        for el in elements:
            if _mapped_role(el.role) == role and name in _norm(el.name):
                return el.ref

    tag = hint.get("tag")
    if name and tag:
        for el in elements:
            if el.tag == tag and name in _norm(el.name):
                return el.ref

    if name:
        for el in elements:
            if name in _norm(el.name):
                return el.ref

    return None
