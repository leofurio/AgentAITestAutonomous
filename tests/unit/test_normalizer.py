"""The instruction normalizer rewrites a free-form request into a canonical text spec."""

from __future__ import annotations

import pytest

from aiwebtest.agent.normalizer import (
    InstructionNormalizer,
    _build_request,
    _to_canonical_spec,
)
from aiwebtest.config import Settings, normalizer_settings


class _FakeClient:
    """Records the request and returns a scripted text-only assistant turn."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[tuple] = []

    async def complete(self, messages, tools, system_prompt):
        self.calls.append((messages, tools, system_prompt))
        return {"role": "assistant", "content": [{"type": "text", "text": self.text}]}


def _settings() -> Settings:
    return Settings(model="claude-opus-4-8", effort="high", max_tokens=1024)


@pytest.mark.asyncio
async def test_normalize_returns_canonical_spec():
    # Model output uses mixed labels/bullets; the normalizer re-serializes it into the
    # exact canonical form (fixed sections, renumbered steps) for deterministic bytes.
    raw = """
    Objective: log in
    Steps:
    - navigate to login
      check: username and password fields are visible
    2) type {username} and {password}
    * click submit
      CHECK: welcome message shown
    """
    client = _FakeClient(raw)
    normalizer = InstructionNormalizer(client, _settings())

    out = await normalizer.normalize(
        "log in pls", target_url="https://example.test", data={"username": "demo"}
    )

    assert out == (
        "GOAL: log in\n"
        "STEPS:\n"
        "1. navigate to login\n"
        "   CHECK: username and password fields are visible\n"
        "2. type {username} and {password}\n"
        "3. click submit\n"
        "   CHECK: welcome message shown"
    )
    # No tools are offered to the normalizer — it only produces text.
    messages, tools, system_prompt = client.calls[0]
    assert tools == []
    assert "goal:" in system_prompt.lower()
    request = messages[0]["content"]
    assert "log in pls" in request
    assert "https://example.test" in request


@pytest.mark.asyncio
async def test_normalize_strips_markdown_code_fence():
    client = _FakeClient("```\nGOAL: x\nSTEPS:\n1. go\n```")
    normalizer = InstructionNormalizer(client, _settings())

    out = await normalizer.normalize("do x")
    assert out == "GOAL: x\nSTEPS:\n1. go"


@pytest.mark.asyncio
async def test_normalize_falls_back_to_original_when_no_steps():
    client = _FakeClient("GOAL: x")  # objective but no steps
    normalizer = InstructionNormalizer(client, _settings())

    out = await normalizer.normalize("  keep me  ")
    assert out == "keep me"


@pytest.mark.asyncio
async def test_normalize_does_not_leak_data_values():
    client = _FakeClient("GOAL: x\nSTEPS:\n1. go")
    normalizer = InstructionNormalizer(client, _settings())

    await normalizer.normalize("do it", data={"password": "s3cret"})

    request = client.calls[0][0][0]["content"]
    assert "s3cret" not in request  # values are masked; only keys are exposed
    assert "password" in request


@pytest.mark.asyncio
async def test_normalize_falls_back_to_original_on_empty_output():
    client = _FakeClient("")
    normalizer = InstructionNormalizer(client, _settings())

    out = await normalizer.normalize("  keep me  ")
    assert out == "keep me"


def test_to_canonical_spec_rejects_missing_objective_or_steps():
    assert _to_canonical_spec("just prose with no structure") is None
    assert _to_canonical_spec("STEPS:\n1. go") is None        # no objective
    assert _to_canonical_spec("GOAL: x") is None              # no steps
    assert _to_canonical_spec("GOAL: x\nSTEPS:") is None      # header but no step lines


def test_to_canonical_spec_treats_lines_after_goal_as_steps():
    # No explicit STEPS header: bullets right after GOAL are steps; CHECKS is optional.
    out = _to_canonical_spec("GOAL: do it\n- first\n- second")
    assert out == "GOAL: do it\nSTEPS:\n1. first\n2. second"


def test_to_canonical_spec_ignores_preamble_before_goal():
    # A chatty preamble before GOAL must not leak into the steps.
    raw = "Sure! Here is the normalized spec:\nGOAL: log in\nSTEPS:\n1. go to login"
    out = _to_canonical_spec(raw)
    assert out == "GOAL: log in\nSTEPS:\n1. go to login"



def test_checks_stay_with_the_step_they_follow():
    # The point of attaching checks: in a stateful flow most outcomes are observable
    # only at one moment, so each must be asserted where it actually holds. Collected
    # into a trailing list they are unverifiable — the login form is gone by the end.
    raw = """GOAL: complete SauceDemo purchase
STEPS:
1. navigate to https://www.saucedemo.com/
   CHECK: Username and Password fields are visible
2. type {username} into Username
3. click Login
   CHECK: product listing page shows products
4. click Add to Cart for Sauce Labs Backpack
   CHECK: cart badge shows 1
5. click Finish
   CHECK: confirmation shows "Thank you for your order!"
"""
    out = _to_canonical_spec(raw)

    assert out == (
        "GOAL: complete SauceDemo purchase\n"
        "STEPS:\n"
        "1. navigate to https://www.saucedemo.com/\n"
        "   CHECK: Username and Password fields are visible\n"
        "2. type {username} into Username\n"
        "3. click Login\n"
        "   CHECK: product listing page shows products\n"
        "4. click Add to Cart for Sauce Labs Backpack\n"
        "   CHECK: cart badge shows 1\n"
        "5. click Finish\n"
        '   CHECK: confirmation shows "Thank you for your order!"'
    )
    # No check drifts to the end, where it could no longer be evaluated.
    assert "CHECKS:" not in out


def test_a_step_may_carry_several_checks_or_none():
    out = _to_canonical_spec(
        "GOAL: x\nSTEPS:\n1. open\n2. submit\n   CHECK: a\n   CHECK: b"
    )
    assert out == "GOAL: x\nSTEPS:\n1. open\n2. submit\n   CHECK: a\n   CHECK: b"


def test_trailing_checks_list_is_kept_at_the_end():
    # A model that ignores the format still produces a usable spec: its unplaced checks
    # stay trailing rather than being pinned to a step we would have to guess.
    out = _to_canonical_spec("GOAL: x\nSTEPS:\n1. go\n2. click\nCHECKS:\n- a\n- b")
    assert out == "GOAL: x\nSTEPS:\n1. go\n2. click\nCHECKS:\n- a\n- b"


def test_orphan_and_empty_checks_never_corrupt_the_steps():
    # A check before any step has nothing to attach to, and a bare "CHECK:" names no
    # outcome — neither may end up masquerading as a step.
    out = _to_canonical_spec("GOAL: x\nCHECK: orphan\nSTEPS:\n1. go\nCHECK:")
    assert out == "GOAL: x\nSTEPS:\n1. go\nCHECKS:\n- orphan"


def test_normalizer_prompt_demands_checks_under_their_step():
    from aiwebtest.agent.normalizer import NORMALIZE_SYSTEM_PROMPT

    assert "CHECK:" in NORMALIZE_SYSTEM_PROMPT
    assert "NEVER gather the checks into a list at the end" in NORMALIZE_SYSTEM_PROMPT


def test_agent_is_told_to_assert_each_check_in_place():
    # The format change is inert unless the agent asserts at that point in the flow.
    from aiwebtest.agent.prompts import SYSTEM_PROMPT

    assert "assert it immediately after" in SYSTEM_PROMPT
    assert "Do not defer checks to the end" in SYSTEM_PROMPT


def test_build_request_omits_optional_sections():
    request = _build_request("just text", target_url=None, data=None)
    assert request.startswith("Test request to normalize:")
    assert "Target site" not in request
    assert "test data" not in request.lower()


def test_normalizer_settings_overrides_provider_and_model():
    base = Settings(
        agent_provider="anthropic",
        model="claude-opus-4-8",
        normalizer_provider="openai",
        normalizer_model="gpt-mini",
    )
    derived = normalizer_settings(base)
    assert derived.agent_provider == "openai"
    assert derived.model == "gpt-mini"
    # Original is untouched.
    assert base.agent_provider == "anthropic"


def test_normalizer_settings_falls_back_to_run_provider_and_model():
    base = Settings(agent_provider="anthropic", model="claude-opus-4-8")
    derived = normalizer_settings(base)
    assert derived.agent_provider == "anthropic"
    assert derived.model == "claude-opus-4-8"


def test_normalizer_settings_dedicated_keys_override_inherited():
    base = Settings(
        agent_provider="openai",
        OPENAI_API_KEY="run-key",
        NORMALIZER_OPENAI_API_KEY="norm-key",
    )
    derived = normalizer_settings(base)
    assert derived.openai_api_key == "norm-key"  # dedicated key wins for the normalizer
    assert base.openai_api_key == "run-key"      # main run keeps its own key


def test_normalizer_settings_inherits_provider_key_when_no_dedicated_key():
    base = Settings(agent_provider="anthropic", ANTHROPIC_API_KEY="run-key")
    derived = normalizer_settings(base)
    assert derived.anthropic_api_key == "run-key"


def test_normalizer_settings_caps_tokens_and_effort_for_optimization():
    base = Settings(
        model="claude-opus-4-8",
        max_tokens=8192,
        effort="high",
        normalizer_max_tokens=512,
        normalizer_effort="low",
    )
    derived = normalizer_settings(base)
    # The normalizer runs under a tighter budget to optimize tokens...
    assert derived.max_tokens == 512
    assert derived.effort == "low"
    # ...without touching the main run's budget.
    assert base.max_tokens == 8192
    assert base.effort == "high"


def test_normalizer_effort_falls_back_to_run_effort_when_empty():
    base = Settings(effort="high", normalizer_effort="")
    derived = normalizer_settings(base)
    assert derived.effort == "high"
