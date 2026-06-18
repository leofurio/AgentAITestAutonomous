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
    2) type {username} and {password}
    * click submit
    Expected results:
    - welcome message shown
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
        "2. type {username} and {password}\n"
        "3. click submit\n"
        "CHECKS:\n"
        "- welcome message shown"
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
