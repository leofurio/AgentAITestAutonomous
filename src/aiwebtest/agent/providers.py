"""LLM provider adapters for the browser-driving agent loop."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..config import Settings
from ..logging_config import get_logger
from .schemas import ModelUsage

logger = get_logger("providers")

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
    usage: ModelUsage = field(init=False)

    def __post_init__(self) -> None:
        self.usage = _new_usage("anthropic", self.settings, self.settings.effort)

    async def complete(
        self,
        messages: list[AgentMessage],
        tool_schemas: list[dict[str, Any]],
        system_prompt: str,
    ) -> AgentMessage:
        kwargs: dict[str, Any] = {
            "model": self.settings.model,
            "max_tokens": self.settings.max_tokens,
            # Cache the stable tools+system prefix; top-level cache_control caches the
            # growing conversation tail. On a multi-step run every turn after the first
            # reads the prefix from cache instead of re-processing it — a large latency win.
            "system": [
                {"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}
            ],
            "messages": _with_cache_breakpoint(messages),
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": self.settings.effort},
        }
        # Omit tools entirely when there are none (e.g. the normalizer pass): an empty
        # tools array is rejected / yields no content on several APIs.
        if tool_schemas:
            kwargs["tools"] = tool_schemas
        _log_request("anthropic", self.settings.model, messages, tool_schemas)
        async with self.client.messages.stream(**kwargs) as stream:
            async for _ in stream.text_stream:
                # Text deltas are surfaced from the final message; iterate to drive streaming.
                pass
            message = await stream.get_final_message()
        _record_usage(self.usage, _get(message, "usage"))
        content = _normalize_anthropic_blocks(message.content)
        _log_response("anthropic", content)
        return {"role": "assistant", "content": content}


@dataclass
class OpenAIAgentClient:
    """Adapter for OpenAI Responses API function calling."""

    client: Any
    settings: Settings
    previous_response_id: str | None = None
    usage: ModelUsage = field(init=False)

    def __post_init__(self) -> None:
        # effort is not part of the Responses request built below, so it is not reported.
        self.usage = _new_usage("openai", self.settings, None)

    async def complete(
        self,
        messages: list[AgentMessage],
        tool_schemas: list[dict[str, Any]],
        system_prompt: str,
    ) -> AgentMessage:
        kwargs: dict[str, Any] = {
            "model": self.settings.model,
            "input": self._next_input(messages),
            "max_output_tokens": self.settings.max_tokens,
        }
        if tool_schemas:
            kwargs["tools"] = [_to_openai_tool(tool) for tool in tool_schemas]
        if self.previous_response_id:
            kwargs["previous_response_id"] = self.previous_response_id
        else:
            kwargs["instructions"] = system_prompt
        _log_request("openai", self.settings.model, messages, tool_schemas)
        response = await self.client.responses.create(**kwargs)
        _record_usage(self.usage, getattr(response, "usage", None))
        self.previous_response_id = getattr(response, "id", None)
        content = _normalize_openai_output(response)
        _log_response("openai", content)
        return {"role": "assistant", "content": content}

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
    usage: ModelUsage = field(init=False)

    def __post_init__(self) -> None:
        # effort has no Chat Completions equivalent here, so it is not reported.
        self.usage = _new_usage("openrouter", self.settings, None)

    async def complete(
        self,
        messages: list[AgentMessage],
        tool_schemas: list[dict[str, Any]],
        system_prompt: str,
    ) -> AgentMessage:
        _log_request("openrouter", self.settings.model, messages, tool_schemas)
        kwargs: dict[str, Any] = {
            "model": self.settings.model,
            "messages": _to_openai_chat_messages(messages, system_prompt),
            "max_tokens": self.settings.max_tokens,
        }
        # Only send tools/tool_choice when there are tools: an empty tools array with
        # tool_choice="auto" is rejected / returns empty content on several backends
        # (this is what made the tool-less normalizer pass come back with no content).
        if tool_schemas:
            kwargs["tools"] = [_to_openai_chat_tool(tool) for tool in tool_schemas]
            kwargs["tool_choice"] = "auto"
        completion = await self.client.chat.completions.create(**kwargs)
        _record_usage(self.usage, getattr(completion, "usage", None))
        content = _normalize_openai_chat_completion(completion)
        _log_response("openrouter", content)
        return {"role": "assistant", "content": content}


def ensure_agent_client(client: Any, settings: Settings) -> AgentClient:
    """Keep tests/custom factories compatible while preferring explicit adapters."""
    if hasattr(client, "complete"):
        return client
    return AnthropicAgentClient(client=client, settings=settings)


def usage_tracker(client: Any, settings: Settings, role: str) -> ModelUsage:
    """Return ``client``'s live usage tracker, labelled with its role in the run.

    The object is the adapter's own, so counts it accumulates *after* this call still
    reach whoever holds the reference (the report). Custom clients supplied by tests or
    user factories expose no tracker; they still get an entry describing the configured
    provider/model, so the report always states what drove the run — only the token
    counts stay at zero.
    """
    tracker = getattr(client, "usage", None)
    if not isinstance(tracker, ModelUsage):
        tracker = ModelUsage(
            provider=settings.agent_provider,
            model=settings.model,
            max_tokens=settings.max_tokens,
        )
    tracker.role = role
    return tracker


def _new_usage(provider: str, settings: Settings, effort: str | None) -> ModelUsage:
    """A zeroed usage tracker for one adapter, seeded with its request settings."""
    return ModelUsage(
        provider=provider,
        model=settings.model,
        effort=effort,
        max_tokens=settings.max_tokens,
    )


def _record_usage(tracker: ModelUsage, raw: Any) -> None:
    """Accumulate one API response's token counts into the adapter's tracker.

    Deliberately tolerant: the providers spell the same counters differently, and a
    response without usage must never break a run — it only costs us the numbers.
    """
    tracker.calls += 1
    if raw is None:
        return
    tracker.input_tokens += _usage_int(raw, "input_tokens", "prompt_tokens")
    tracker.output_tokens += _usage_int(raw, "output_tokens", "completion_tokens")
    tracker.cache_write_tokens += _usage_int(raw, "cache_creation_input_tokens")
    # Anthropic reports cache reads at the top level; the OpenAI-shaped APIs nest them.
    cached = _usage_int(raw, "cache_read_input_tokens")
    for details in ("input_tokens_details", "prompt_tokens_details"):
        if cached:
            break
        cached = _usage_int(_get(raw, details), "cached_tokens")
    tracker.cache_read_tokens += cached


def _usage_int(raw: Any, *names: str) -> int:
    """First integer among ``names`` on ``raw`` (dict or object), else 0."""
    if raw is None:
        return 0
    for name in names:
        value = _get(raw, name)
        if isinstance(value, int):
            return value
    return 0


def _with_cache_breakpoint(messages: list[AgentMessage]) -> list[AgentMessage]:
    """Return a request copy with a cache breakpoint on the last conversation block.

    Caching is a prefix match, so marking the latest block each turn caches the whole
    conversation prefix; the next turn reads it back instead of re-processing history.
    The stored history is left untouched (we copy only the final message/block).
    """
    if not messages:
        return messages
    last = messages[-1]
    content = last.get("content")
    if not isinstance(content, list) or not content:
        return messages  # first user turn is a plain string — system cache still applies
    new_content = list(content)
    new_content[-1] = {**new_content[-1], "cache_control": {"type": "ephemeral"}}
    return [*messages[:-1], {**last, "content": new_content}]


def _normalize_anthropic_blocks(blocks: list[Any]) -> list[AgentBlock]:
    normalized: list[AgentBlock] = []
    for block in blocks:
        block_type = _get(block, "type")
        if block_type == "text":
            normalized.append({"type": "text", "text": _get(block, "text", "")})
        elif block_type == "thinking":
            # Thinking blocks (and their signature) must be echoed back verbatim in
            # tool-use loops, or the API rejects the next request.
            normalized.append(
                {
                    "type": "thinking",
                    "thinking": _get(block, "thinking", ""),
                    "signature": _get(block, "signature", ""),
                }
            )
        elif block_type == "redacted_thinking":
            normalized.append({"type": "redacted_thinking", "data": _get(block, "data", "")})
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


def _log_request(
    provider: str,
    model: str,
    messages: list[AgentMessage],
    tool_schemas: list[dict[str, Any]],
) -> None:
    if not logger.isEnabledFor(10):  # logging.DEBUG
        return
    logger.debug(
        "%s call: model=%s messages=%d tools=%d", provider, model, len(messages),
        len(tool_schemas),
    )


def _log_response(provider: str, blocks: list[AgentBlock]) -> None:
    if not logger.isEnabledFor(10):  # logging.DEBUG
        return
    logger.debug("%s response: %s", provider, _summarize_blocks(blocks))


def _summarize_blocks(blocks: list[AgentBlock]) -> str:
    parts: list[str] = []
    for block in blocks:
        block_type = block.get("type")
        if block_type == "text":
            parts.append(f"text={block.get('text', '')[:300]!r}")
        elif block_type == "tool_use":
            args = json.dumps(block.get("input", {}), ensure_ascii=False)
            parts.append(f"tool_use={block.get('name')}({args})")
        elif block_type in {"thinking", "redacted_thinking"}:
            parts.append(block_type)
    return " | ".join(parts) if parts else "(no content)"
