"""Deterministic hint→live-ref matching: the in-process twin of the replay candidate chain.

The matcher decides, without a model, whether a recorded descriptor still points at a
live element; only when it returns None does the AI repair pass get involved.
"""

from __future__ import annotations

from aiwebtest.browser.snapshot import SnapshotElement
from aiwebtest.replay.matcher import ROLE_MAP, ROLE_OK, match_ref


def _el(ref, *, tag="", role="", name="", el_id="", attr_name="", testid=""):
    return SnapshotElement(
        ref=ref, tag=tag, role=role, name=name, value="",
        el_id=el_id, attr_name=attr_name, testid=testid,
    )


def test_role_maps_mirror_codegen():
    # The matcher and the generated replay must agree on locator priority, or an
    # in-process heal could target a different element than the deterministic replay.
    from aiwebtest.report.playwright_codegen import _ROLE_MAP, _ROLE_OK

    assert ROLE_MAP == _ROLE_MAP
    assert ROLE_OK == _ROLE_OK


def test_testid_beats_everything():
    els = [
        _el("e1", el_id="x", role="button", name="Save"),
        _el("e2", testid="save-btn", role="button", name="Save"),
    ]
    assert match_ref({"testid": "save-btn", "id": "x", "name": "Save"}, els) == "e2"


def test_id_match():
    els = [_el("e1", el_id="username"), _el("e2", el_id="password")]
    assert match_ref({"id": "password"}, els) == "e2"


def test_name_attribute_only_for_form_controls():
    els = [_el("e1", tag="input", attr_name="email"), _el("e2", tag="input", attr_name="pw")]
    assert match_ref({"attr_name": "pw", "tag": "input"}, els) == "e2"
    # attr_name is ignored when the recorded tag is not a form control.
    assert match_ref({"attr_name": "pw", "tag": "div"}, els) is None


def test_role_and_name_with_role_mapping():
    # A recorded "password" input role maps to the ARIA "textbox", as in the replay.
    els = [
        _el("e1", tag="input", role="text", name="Username"),
        _el("e2", tag="input", role="password", name="Password"),
    ]
    assert match_ref({"role": "password", "name": "Password", "tag": "input"}, els) == "e2"


def test_role_and_name_is_case_insensitive_substring():
    els = [_el("e1", tag="button", role="button", name="Log in to continue")]
    assert match_ref({"role": "button", "name": "log in", "tag": "button"}, els) == "e1"


def test_tag_and_text_fallback_when_role_mismatches():
    # The snapshot labels every <a> as role "link", but the recorded role may differ;
    # tag + visible text still matches, mirroring the replay's :has-text candidate.
    els = [_el("e1", tag="a", role="link", name="Next page")]
    assert match_ref({"role": "menuitem", "name": "Next", "tag": "a"}, els) == "e1"


def test_plain_text_fallback():
    els = [_el("e1", tag="span", role="generic", name="Order confirmed")]
    assert match_ref({"role": "", "name": "confirmed", "tag": ""}, els) == "e1"


def test_returns_first_element_in_snapshot_order():
    els = [
        _el("e1", tag="button", role="button", name="Delete"),
        _el("e2", tag="button", role="button", name="Delete"),
    ]
    assert match_ref({"role": "button", "name": "Delete", "tag": "button"}, els) == "e1"


def test_stale_hint_matches_nothing():
    # The whole point: a descriptor that no longer exists returns None, so the caller
    # escalates to AI repair instead of silently hitting the wrong element.
    els = [_el("e1", tag="button", role="button", name="Log in", el_id="login-btn")]
    stale = {"id": "old-submit", "testid": "", "attr_name": "",
             "role": "button", "name": "Submit order", "tag": "button"}
    assert match_ref(stale, els) is None


def test_none_and_empty_hint():
    els = [_el("e1", el_id="x", name="hi")]
    assert match_ref(None, els) is None
    assert match_ref({}, els) is None


def test_ref_is_never_used_to_match():
    # A recorded ref must not be trusted: it is an ordinal from a different snapshot.
    els = [_el("e1", name="something")]
    assert match_ref({"ref": "e1"}, els) is None
