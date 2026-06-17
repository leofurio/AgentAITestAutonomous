"""InstructionNormalizer: rewrite a free-form test request into a canonical, deterministic spec.

A small pre-pass LLM call takes the user's natural-language instruction (plus the target
URL and any test data) and rewrites it into a normalized, numbered, unambiguous form.
Feeding the agent loop this canonical text — instead of the raw prose — reduces run-to-run
variance: the same intent yields the same steps and the same assertions. The pass is
deliberately conservative: it clarifies and structures, it never invents steps or data.

It is also a token optimizer: the rewrite is kept compact (terse imperative steps, no
filler or restated values) so the downstream agent loop carries fewer input tokens on
every turn. The pass itself runs under a tight output budget and effort (see
``normalizer_settings``) to keep its own cost low.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..config import Settings

NORMALIZE_SYSTEM_PROMPT = """\
You normalize web-application test requests. Given a free-form test description (and \
optionally a target URL and test data), rewrite it into a single canonical specification \
that another agent will execute against a real browser. Your two goals: make the intent \
explicit, ordered, and unambiguous so the same request always produces the same test, and \
make the result as token-compact as possible — you are a rewriter and compressor, not a \
planner.

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

Token efficiency (minimize input tokens for the downstream agent):
- Be maximally concise. Use short imperative phrases, not full sentences. Drop articles \
and filler words where meaning is preserved.
- Do not repeat the target URL, data values, or the same element across steps. Merge \
trivially sequential actions only when they are unambiguous (e.g. "type {username} and \
{password}" is fine; never merge distinct verifications).
- Omit obvious mechanics the executing agent already knows (taking snapshots, waiting for \
loads) unless the request explicitly depends on them.
- No duplication between Steps and Expected results.
- Output ONLY the normalized specification in the exact structure below — no preamble, \
no explanations, no markdown fences.

Output structure:
Objective: <one short clause: what the test verifies>
Steps:
1. <terse imperative action>
2. <terse imperative action>
...
Expected results:
- <terse verifiable assertion>
- <terse verifiable assertion>
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
