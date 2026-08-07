"""Provider adapters normalize model-specific tool calling into the AgentLoop shape."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from aiwebtest.agent.providers import (
    AnthropicAgentClient,
    OpenAIAgentClient,
    OpenRouterAgentClient,
    _normalize_anthropic_blocks,
    _with_cache_breakpoint,
    usage_tracker,
)
from aiwebtest.config import Settings


def test_cache_breakpoint_marks_last_block_without_mutating_history():
    messages = [
        {"role": "user", "content": "open the page"},  # plain string — skipped
        {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": []},
                {"type": "tool_result", "tool_use_id": "t2", "content": []},
            ],
        },
    ]
    out = _with_cache_breakpoint(messages)
    # Last block of the last message gets the breakpoint...
    assert out[-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    # ...and the original history is untouched (no cache_control leaked in).
    assert "cache_control" not in messages[-1]["content"][-1]


def test_cache_breakpoint_skips_string_content():
    messages = [{"role": "user", "content": "just text"}]
    assert _with_cache_breakpoint(messages) == messages


def test_anthropic_normalization_preserves_thinking_blocks():
    # Thinking blocks (with signature) must round-trip untouched: with adaptive
    # thinking + tool use, the API rejects follow-up requests that drop them.
    blocks = [
        SimpleNamespace(type="thinking", thinking="plan the click", signature="sig123"),
        SimpleNamespace(type="redacted_thinking", data="opaque"),
        SimpleNamespace(type="text", text="Clicking the login button."),
        SimpleNamespace(type="tool_use", id="toolu_1", name="click", input={"ref": "e1"}),
    ]
    normalized = _normalize_anthropic_blocks(blocks)
    assert normalized[0] == {
        "type": "thinking",
        "thinking": "plan the click",
        "signature": "sig123",
    }
    assert normalized[1] == {"type": "redacted_thinking", "data": "opaque"}
    assert normalized[2]["type"] == "text"
    assert normalized[3]["name"] == "click"


class _FakeResponses:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            id=f"resp_{len(self.calls)}",
            output_text="I will inspect the page." if len(self.calls) == 1 else "",
            output=[
                SimpleNamespace(
                    type="function_call",
                    call_id="call_1",
                    name="navigate",
                    arguments='{"url":"https://example.test"}',
                )
            ],
            usage=SimpleNamespace(
                input_tokens=100,
                output_tokens=20,
                input_tokens_details=SimpleNamespace(cached_tokens=80),
            ),
        )


class _FakeOpenAI:
    def __init__(self):
        self.responses = _FakeResponses()


class _FakeOpenRouterCompletions:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="I will inspect the page.",
                        tool_calls=[
                            SimpleNamespace(
                                id="call_1",
                                function=SimpleNamespace(
                                    name="navigate",
                                    arguments='{"url":"https://example.test"}',
                                ),
                            )
                        ],
                    )
                )
            ],
            usage=SimpleNamespace(prompt_tokens=250, completion_tokens=40),
        )


class _FakeOpenRouterChat:
    def __init__(self):
        self.completions = _FakeOpenRouterCompletions()


class _FakeOpenRouter:
    def __init__(self):
        self.chat = _FakeOpenRouterChat()


def _settings() -> Settings:
    return Settings(
        agent_provider="openai",
        model="gpt-test",
        effort="high",
        max_tokens=1024,
    )


@pytest.mark.asyncio
async def test_openai_provider_maps_tools_and_function_calls():
    client = _FakeOpenAI()
    provider = OpenAIAgentClient(client, _settings())

    message = await provider.complete(
        [{"role": "user", "content": "Open the page"}],
        [
            {
                "name": "navigate",
                "description": "Navigate",
                "input_schema": {
                    "type": "object",
                    "properties": {"url": {"type": "string"}},
                    "required": ["url"],
                },
            }
        ],
        "system",
    )

    first_call = client.responses.calls[0]
    assert first_call["tools"][0]["type"] == "function"
    assert first_call["tools"][0]["parameters"]["required"] == ["url"]
    assert first_call["instructions"] == "system"
    assert message["content"][0] == {"type": "text", "text": "I will inspect the page."}
    assert message["content"][1]["name"] == "navigate"
    assert message["content"][1]["input"] == {"url": "https://example.test"}


@pytest.mark.asyncio
async def test_openai_provider_sends_tool_results_with_previous_response():
    client = _FakeOpenAI()
    provider = OpenAIAgentClient(client, _settings())
    provider.previous_response_id = "resp_1"

    await provider.complete(
        [
            {"role": "user", "content": "Open the page"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_1",
                        "content": [{"type": "text", "text": "Loaded"}],
                        "is_error": False,
                    }
                ],
            },
        ],
        [],
        "system",
    )

    call = client.responses.calls[0]
    assert call["previous_response_id"] == "resp_1"
    assert "instructions" not in call
    assert call["input"] == [
        {"type": "function_call_output", "call_id": "call_1", "output": "Loaded"}
    ]


@pytest.mark.asyncio
async def test_openrouter_provider_maps_tools_and_function_calls():
    client = _FakeOpenRouter()
    provider = OpenRouterAgentClient(client, _settings())

    message = await provider.complete(
        [{"role": "user", "content": "Open the page"}],
        [
            {
                "name": "navigate",
                "description": "Navigate",
                "input_schema": {
                    "type": "object",
                    "properties": {"url": {"type": "string"}},
                    "required": ["url"],
                },
            }
        ],
        "system",
    )

    first_call = client.chat.completions.calls[0]
    assert first_call["messages"][0] == {"role": "system", "content": "system"}
    assert first_call["messages"][1] == {"role": "user", "content": "Open the page"}
    assert first_call["tools"][0]["type"] == "function"
    assert first_call["tools"][0]["function"]["parameters"]["required"] == ["url"]
    assert first_call["tool_choice"] == "auto"
    assert message["content"][0] == {"type": "text", "text": "I will inspect the page."}
    assert message["content"][1]["id"] == "call_1"
    assert message["content"][1]["name"] == "navigate"
    assert message["content"][1]["input"] == {"url": "https://example.test"}


@pytest.mark.asyncio
async def test_openrouter_provider_sends_tool_results_as_chat_tool_messages():
    client = _FakeOpenRouter()
    provider = OpenRouterAgentClient(client, _settings())

    await provider.complete(
        [
            {"role": "user", "content": "Open the page"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Opening the page."},
                    {
                        "type": "tool_use",
                        "id": "call_1",
                        "name": "navigate",
                        "input": {"url": "https://example.test"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_1",
                        "content": [{"type": "text", "text": "Loaded"}],
                        "is_error": False,
                    }
                ],
            },
        ],
        [],
        "system",
    )

    messages = client.chat.completions.calls[0]["messages"]
    assert messages[2] == {
        "role": "assistant",
        "content": "Opening the page.",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "navigate",
                    "arguments": '{"url": "https://example.test"}',
                },
            }
        ],
    }
    assert messages[3] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "Loaded",
    }


@pytest.mark.asyncio
async def test_openrouter_omits_tools_when_none_given():
    # The normalizer pass calls with no tools: tools/tool_choice must be omitted, or
    # several OpenAI-compatible backends reject the call / return empty content.
    client = _FakeOpenRouter()
    provider = OpenRouterAgentClient(client, _settings())

    await provider.complete([{"role": "user", "content": "normalize this"}], [], "system")

    call = client.chat.completions.calls[0]
    assert "tools" not in call
    assert "tool_choice" not in call


@pytest.mark.asyncio
async def test_openai_omits_tools_when_none_given():
    client = _FakeOpenAI()
    provider = OpenAIAgentClient(client, _settings())

    await provider.complete([{"role": "user", "content": "normalize this"}], [], "system")

    assert "tools" not in client.responses.calls[0]


@pytest.mark.asyncio
async def test_openai_provider_accumulates_token_usage():
    # The report details what each model consumed, so every turn's usage must add up.
    provider = OpenAIAgentClient(_FakeOpenAI(), _settings())

    await provider.complete([{"role": "user", "content": "go"}], [], "system")
    await provider.complete([{"role": "user", "content": "go"}], [], "system")

    assert provider.usage.provider == "openai"
    assert provider.usage.model == "gpt-test"
    assert provider.usage.max_tokens == 1024
    assert provider.usage.calls == 2
    assert provider.usage.input_tokens == 200
    assert provider.usage.output_tokens == 40
    assert provider.usage.cache_read_tokens == 160  # nested input_tokens_details
    assert provider.usage.total_tokens == 240


@pytest.mark.asyncio
async def test_openrouter_provider_maps_chat_completion_token_names():
    # Chat Completions spells the same counters prompt_/completion_tokens.
    provider = OpenRouterAgentClient(_FakeOpenRouter(), _settings())

    await provider.complete([{"role": "user", "content": "go"}], [], "system")

    assert provider.usage.provider == "openrouter"
    assert provider.usage.input_tokens == 250
    assert provider.usage.output_tokens == 40


@pytest.mark.asyncio
async def test_anthropic_provider_records_cache_counters():
    message = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="ok")],
        usage=SimpleNamespace(
            input_tokens=10, output_tokens=5,
            cache_read_input_tokens=900, cache_creation_input_tokens=1500,
        ),
    )

    class _Stream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

        @property
        def text_stream(self):
            return self._iter()

        async def _iter(self):
            yield "ok"

        async def get_final_message(self):
            return message

    client = SimpleNamespace(messages=SimpleNamespace(stream=lambda **kwargs: _Stream()))
    provider = AnthropicAgentClient(client, _settings())

    await provider.complete([{"role": "user", "content": "go"}], [], "system")

    # Prompt caching is the point of the request shape; the report shows both sides.
    assert provider.usage.cache_read_tokens == 900
    assert provider.usage.cache_write_tokens == 1500
    assert provider.usage.effort == "high"  # only Anthropic sends effort


def test_usage_tracker_falls_back_for_clients_without_one():
    # Custom/test clients expose no tracker; the report must still name the model.
    settings = _settings()
    tracker = usage_tracker(object(), settings, "normalizer")

    assert tracker.role == "normalizer"
    assert (tracker.provider, tracker.model, tracker.calls) == ("openai", "gpt-test", 0)

    # An adapter's own tracker is returned as-is, only labelled with its role.
    provider = OpenAIAgentClient(_FakeOpenAI(), settings)
    assert usage_tracker(provider, settings, "agent") is provider.usage
    assert provider.usage.role == "agent"
