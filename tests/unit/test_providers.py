"""Provider adapters normalize model-specific tool calling into the AgentLoop shape."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from aiwebtest.agent.providers import OpenAIAgentClient, OpenRouterAgentClient
from aiwebtest.config import Settings


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
            ]
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
