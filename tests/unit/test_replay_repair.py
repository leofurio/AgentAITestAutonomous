"""RepairAgent: the focused LLM pass that re-points one broken step to a live ref."""

from __future__ import annotations

from aiwebtest.agent.providers import ensure_agent_client
from aiwebtest.agent.schemas import Step, StepKind
from aiwebtest.browser.snapshot import SnapshotElement
from aiwebtest.replay.repair import RepairAgent, describe_step, parse_ref
from tests.conftest import FakeAnthropicClient, text_turn


def _agent(turns, settings):
    # Repair is handed an already-wrapped client (as the suite does via
    # ensure_agent_client), so the raw fake SDK client gains a .complete adapter.
    return RepairAgent(ensure_agent_client(FakeAnthropicClient(turns), settings), settings)


def _click_step(**hint):
    base = {"id": "", "testid": "", "attr_name": "", "role": "button",
            "name": "Log in", "tag": "button"}
    base.update(hint)
    return Step(index=0, kind=StepKind.TOOL_CALL, tool_name="click",
                tool_input={"ref": "e9"}, locator_hint=base)


def _el(ref, *, name="", role="", tag="", el_id="", testid=""):
    return SnapshotElement(ref=ref, tag=tag, role=role, name=name, value="",
                           el_id=el_id, attr_name="", testid=testid)


def test_parse_ref_accepts_only_live_refs():
    assert parse_ref("e12", {"e12", "e3"}) == "e12"
    assert parse_ref("The answer is e3.", {"e3"}) == "e3"
    assert parse_ref("e99", {"e1", "e2"}) is None      # not on the live page
    assert parse_ref("NONE", {"e1"}) is None
    assert parse_ref("", {"e1"}) is None


def test_describe_step_reads_the_recorded_intent():
    desc = describe_step(_click_step(name="Log in", role="button"))
    assert "Click" in desc and "Log in" in desc

    typed = Step(index=0, kind=StepKind.TOOL_CALL, tool_name="type_text",
                 tool_input={"ref": "e1", "text": "demo"},
                 locator_hint={"role": "text", "name": "Username", "tag": "input"})
    d2 = describe_step(typed)
    assert "Type into" in d2 and "Username" in d2 and "demo" in d2


async def test_repair_returns_the_models_ref(settings):
    els = [_el("e1", name="Home"), _el("e3", name="Sign in", role="button", tag="button")]
    assert await _agent([text_turn("e3")], settings).repair(_click_step(), "outline", els) == "e3"


async def test_repair_rejects_a_hallucinated_ref(settings):
    els = [_el("e1", name="Home")]
    # The model named a ref that is not on the page: reject rather than click blindly.
    assert await _agent([text_turn("e42")], settings).repair(_click_step(), "outline", els) is None


async def test_repair_honors_a_none_answer(settings):
    els = [_el("e1", name="Home"), _el("e2", name="About")]
    assert await _agent([text_turn("NONE")], settings).repair(_click_step(), "outline", els) is None


async def test_repair_with_no_live_elements_returns_none(settings):
    assert await _agent([text_turn("e1")], settings).repair(_click_step(), "outline", []) is None
