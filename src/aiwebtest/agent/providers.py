"""LLM provider adapters for the browser-driving agent loop."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from ..config import Settings

AgentBlock = dict[str, Any]
AgentMessage = dict[str, Any]


class AgentClient(Protocol):
    async def complete(
        self,
        messages: list[AgentMessage],
        tool_schemas: list[dict[str, Any]],
        system_prompt: str,
    ) -> AgentMessage:
        """Return the next assistant turn as normalized text/tool-use blocks."""


@dataclass
class AnthropicAgentClient:
    """Adapter for Anthropic Messages tool use."""

    client: Any
    settings: Settings

    async def complete(
        self,
        messages: list[AgentMessage],
        tool_schemas: list[dict[str, Any]],
        system_prompt: str,
    ) -> AgentMessage:
        kwargs: dict[str, Any] = {
            "model": self.settings.model,
            "max_tokens": self.settings.max_tokens,
            "system": system_prompt,
            "tools": tool_schemas,
            "messages": messages,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": self.settings.effort},
        }
        async with self.client.messages.stream(**kwargs) as stream:
            async for _ in stream.text_stream:
                # Text deltas are surfaced from the final message; iterate to drive streaming.
                pass
            message = await stream.get_final_message()
        return {"role": "assistant", "content": _normalize_anthropic_blocks(message.content)}


@dataclass
class OpenAIAgentClient:
    """Adapter for OpenAI Responses API function calling."""

    client: Any
    settings: Settings
    previous_response_id: str | None = None

    async def complete(
        self,
        messages: list[AgentMessage],
        tool_schemas: list[dict[str, Any]],
        system_prompt: str,
    ) -> AgentMessage:
        kwargs: dict[str, Any] = {
            "model": self.settings.model,
            "tools": [_to_openai_tool(tool) for tool in tool_schemas],
            "input": self._next_input(messages),
            "max_output_tokens": self.settings.max_tokens,
        }
        if self.previous_response_id:
            kwargs["previous_response_id"] = self.previous_response_id
        else:
            kwargs["instructions"] = system_prompt
        response = await self.client.responses.create(**kwargs)
        self.previous_response_id = getattr(response, "id", None)
        return {"role": "assistant", "content": _normalize_openai_output(response)}

    def _next_input(self, messages: list[AgentMessage]) -> Any:
        if not self.previous_response_id:
            first = messages[0]["content"]
            return first if isinstance(first, str) else _blocks_to_text(first)

        latest = messages[-1]["content"]
        if not isinstance(latest, list):
            return str(latest)
        return [
            {
                "type": "function_call_output",
                "call_id": block["tool_use_id"],
                "output": _blocks_to_text(block.get("content", [])),
            }
            for block in latest
            if block.get("type") == "tool_result"
        ]


@dataclass
class OpenRouterAgentClient:
    """Adapter for OpenRouter's OpenAI-compatible Chat Completions API."""

    client: Any
    settings: Settings

    async def complete(
        self,
        messages: list[AgentMessage],
        tool_schemas: list[dict[str, Any]],
        system_prompt: str,
    ) -> AgentMessage:
        completion = await self.client.chat.completions.create(
            model=self.settings.model,
            messages=_to_openai_chat_messages(messages, system_prompt),
            tools=[_to_openai_chat_tool(tool) for tool in tool_schemas],
            tool_choice="auto",
            max_tokens=self.settings.max_tokens,
        )
        return {"role": "assistant", "content": _normalize_openai_chat_completion(completion)}


def ensure_agent_client(client: Any, settings: Settings) -> AgentClient:
    """Keep tests/custom factories compatible while preferring explicit adapters."""
    if hasattr(client, "complete"):
        return client
    return AnthropicAgentClient(client=client, settings=settings)


def _normalize_anthropic_blocks(blocks: list[Any]) -> list[AgentBlock]:
    normalized: list[AgentBlock] = []
    for block in blocks:
        block_type = _get(block, "type")
        if block_type == "text":
            normalized.append({"type": "text", "text": _get(block, "text", "")})
        elif block_type == "tool_use":
            normalized.append(
                {
                    "type": "tool_use",
                    "id": _get(block, "id", ""),
                    "name": _get(block, "name", ""),
                    "input": dict(_get(block, "input", {}) or {}),
                }
            )
    return normalized


def _normalize_openai_output(response: Any) -> list[AgentBlock]:
    blocks: list[AgentBlock] = []
    output_text = getattr(response, "output_text", "")
    if output_text:
        blocks.append({"type": "text", "text": output_text})

    for item in getattr(response, "output", []) or []:
        item_type = getattr(item, "type", None)
        if item_type == "function_call":
            arguments = getattr(item, "arguments", "{}") or "{}"
            try:
                tool_input = json.loads(arguments)
            except json.JSONDecodeError:
                tool_input = {}
            blocks.append(
                {
                    "type": "tool_use",
                    "id": getattr(item, "call_id", None) or getattr(item, "id", ""),
                    "name": getattr(item, "name", ""),
                    "input": tool_input,
                }
            )
        elif item_type == "message" and not output_text:
            text = _openai_message_text(item)
            if text:
                blocks.append({"type": "text", "text": text})
    return blocks


def _normalize_openai_chat_completion(completion: Any) -> list[AgentBlock]:
    choices = getattr(completion, "choices", []) or []
    if not choices:
        return []

    message = getattr(choices[0], "message", None)
    if message is None:
        return []

    blocks: list[AgentBlock] = []
    content = getattr(message, "content", None)
    if isinstance(content, str) and content:
        blocks.append({"type": "text", "text": content})
    elif isinstance(content, list):
        text = _chat_content_parts_to_text(content)
        if text:
            blocks.append({"type": "text", "text": text})

    for tool_call in getattr(message, "tool_calls", []) or []:
        function = getattr(tool_call, "function", None)
        arguments = getattr(function, "arguments", "{}") or "{}"
        try:
            tool_input = json.loads(arguments)
        except json.JSONDecodeError:
            tool_input = {}
        blocks.append(
            {
                "type": "tool_use",
                "id": getattr(tool_call, "id", ""),
                "name": getattr(function, "name", "") if function is not None else "",
                "input": tool_input,
            }
        )
    return blocks


def _openai_message_text(item: Any) -> str:
    parts: list[str] = []
    for content in getattr(item, "content", []) or []:
        if getattr(content, "type", None) in {"output_text", "text"}:
            text = getattr(content, "text", "")
            if text:
                parts.append(text)
    return "\n".join(parts).strip()


def _to_openai_tool(tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "name": tool["name"],
        "description": tool.get("description", ""),
        "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
    }


def _to_openai_chat_tool(tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
        },
    }


def _to_openai_chat_messages(
    messages: list[AgentMessage],
    system_prompt: str,
) -> list[dict[str, Any]]:
    chat_messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    for message in messages:
        role = message["role"]
        content = message["content"]

        if role == "assistant":
            chat_messages.append(_assistant_to_openai_chat_message(content))
            continue

        has_tool_result = isinstance(content, list) and any(
            block.get("type") == "tool_result" for block in content
        )
        if has_tool_result:
            chat_messages.extend(_tool_results_to_openai_chat_messages(content))
        else:
            chat_messages.append({"role": role, "content": _content_to_text(content)})
    return chat_messages


def _assistant_to_openai_chat_message(content: Any) -> dict[str, Any]:
    if not isinstance(content, list):
        return {"role": "assistant", "content": str(content)}

    text = _blocks_to_text([block for block in content if block.get("type") == "text"])
    tool_calls = []
    for block in content:
        if block.get("type") != "tool_use":
            continue
        tool_calls.append(
            {
                "id": block["id"],
                "type": "function",
                "function": {
                    "name": block["name"],
                    "arguments": json.dumps(block.get("input", {}), ensure_ascii=True),
                },
            }
        )

    chat_message: dict[str, Any] = {"role": "assistant", "content": text or None}
    if tool_calls:
        chat_message["tool_calls"] = tool_calls
    return chat_message


def _tool_results_to_openai_chat_messages(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "role": "tool",
            "tool_call_id": block["tool_use_id"],
            "content": _blocks_to_text(block.get("content", [])),
        }
        for block in content
        if block.get("type") == "tool_result"
    ]


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return _blocks_to_text(content)
    return str(content)


def _chat_content_parts_to_text(content: list[Any]) -> str:
    parts: list[str] = []
    for part in content:
        if isinstance(part, dict):
            if part.get("type") == "text":
                parts.append(part.get("text", ""))
        elif getattr(part, "type", None) == "text":
            parts.append(getattr(part, "text", ""))
    return "\n".join(p for p in parts if p).strip()


def _blocks_to_text(blocks: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for block in blocks:
        if block.get("type") == "text":
            parts.append(block.get("text", ""))
        elif block.get("type") == "image":
            parts.append("[screenshot captured]")
        else:
            parts.append(json.dumps(block, ensure_ascii=True))
    return "\n".join(p for p in parts if p).strip()


def _get(block: Any, name: str, default: Any = None) -> Any:
    if isinstance(block, dict):
        return block.get(name, default)
    return getattr(block, name, default)
