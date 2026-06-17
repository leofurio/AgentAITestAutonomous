"""The instruction normalizer rewrites a free-form request into a canonical spec."""

from __future__ import annotations

import pytest

from aiwebtest.agent.normalizer import InstructionNormalizer, _build_request
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
async def test_normalize_returns_canonical_text():
    client = _FakeClient("Objective: log in.\nSteps:\n1. navigate")
    normalizer = InstructionNormalizer(client, _settings())

    out = await normalizer.normalize(
        "log in pls", target_url="https://example.test", data={"username": "demo"}
    )

    assert out == "Objective: log in.\nSteps:\n1. navigate"
    # No tools are offered to the normalizer — it only produces text.
    messages, tools, system_prompt = client.calls[0]
    assert tools == []
    assert "normalize" in system_prompt.lower()
    request = messages[0]["content"]
    assert "log in pls" in request
    assert "https://example.test" in request


@pytest.mark.asyncio
async def test_normalize_does_not_leak_data_values():
    client = _FakeClient("Objective: x")
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
