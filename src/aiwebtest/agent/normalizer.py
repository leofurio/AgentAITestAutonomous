"""InstructionNormalizer: rewrite a free-form test request into a canonical, deterministic spec.

A small pre-pass LLM call takes the user's natural-language instruction (plus the target
URL and any test data) and rewrites it into a normalized, numbered, unambiguous form.
Feeding the agent loop this canonical text — instead of the raw prose — reduces run-to-run
variance: the same intent yields the same steps and the same assertions. The pass is
deliberately conservative: it clarifies and structures, it never invents steps or data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..config import Settings

NORMALIZE_SYSTEM_PROMPT = """\
You normalize web-application test requests. Given a free-form test description (and \
optionally a target URL and test data), rewrite it into a single canonical specification \
that another agent will execute against a real browser. Your only goal is to make the \
intent explicit, ordered, and unambiguous so that the same request always produces the \
same test — you are a rewriter, not a planner.

Rules:
- Preserve the original intent exactly. Do NOT add steps, pages, checks, or data that the \
request does not imply. Do NOT remove anything meaningful. If the request is vague, keep \
it as-is rather than inventing specifics.
- Decompose the request into atomic, ordered, imperative steps (one action per step).
- State every expected outcome as an explicit, verifiable assertion.
- Refer to provided test data only by its key in braces, e.g. {username}, {password}. \
Never inline secret values.
- Use stable, literal wording. Prefer neutral verbs: navigate, click, type, select, \
press, wait for, verify. Do not editorialize.
- Output ONLY the normalized specification in the exact structure below — no preamble, \
no explanations, no markdown fences.

Output structure:
Objective: <one sentence describing what the test verifies>
Steps:
1. <imperative action>
2. <imperative action>
...
Expected results:
- <verifiable assertion>
- <verifiable assertion>
...
"""


@dataclass
class InstructionNormalizer:
    """Wraps an agent client to produce a normalized instruction string."""

    client: Any
    settings: Settings

    async def normalize(
        self,
        instruction: str,
        target_url: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> str:
        """Return the canonical rewrite of the request, or the original on empty output."""
        request = _build_request(instruction, target_url, data)
        message = await self.client.complete(
            [{"role": "user", "content": request}], [], NORMALIZE_SYSTEM_PROMPT
        )
        normalized = _collect_text(message)
        return normalized or instruction.strip()


def _build_request(
    instruction: str,
    target_url: str | None,
    data: dict[str, Any] | None,
) -> str:
    parts = ["Test request to normalize:", instruction.strip()]
    if target_url:
        parts.append(f"\nTarget site: {target_url}")
    if data:
        parts.append(
            "\nAvailable test data (refer to keys in braces, never inline values):\n"
            + json.dumps({k: "<provided>" for k in data}, indent=2, ensure_ascii=False)
        )
    return "\n".join(parts)


def _collect_text(message: dict[str, Any]) -> str:
    parts = [b.get("text", "") for b in message.get("content", []) if b.get("type") == "text"]
    return "\n".join(p for p in parts if p).strip()
