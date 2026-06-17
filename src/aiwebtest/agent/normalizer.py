"""InstructionNormalizer: rewrite a free-form test request into a canonical, deterministic spec.

A small pre-pass LLM call takes the user's natural-language instruction (plus the target
URL and any test data) and rewrites it into a normalized, structured form. Feeding the
agent loop this canonical spec — instead of the raw prose — reduces run-to-run variance:
the same intent yields the same steps and the same assertions. The pass is deliberately
conservative: it clarifies and structures, it never invents steps or data.

The canonical form is a **compact JSON object** (`objective`, `steps`, `expected_results`).
JSON gives a rigid, machine-validatable shape — more deterministic than prose — and is
re-serialized minified with a fixed key order, so the same intent always yields the same
bytes. It is also a token optimizer: terse fields, no filler or restated values, kept under
a tight output budget/effort (see ``normalizer_settings``). If the model returns anything
that is not valid JSON of the expected shape, the pass falls back to the raw instruction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..config import Settings
from ..logging_config import get_logger

logger = get_logger("normalizer")

NORMALIZE_SYSTEM_PROMPT = """\
You normalize web-application test requests. Given a free-form test description (and \
optionally a target URL and test data), rewrite it into a single canonical JSON \
specification that another agent will execute against a real browser. Your two goals: make \
the intent explicit, ordered, and unambiguous so the same request always produces the same \
test, and make the result as token-compact as possible — you are a rewriter and compressor, \
not a planner.

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
- No duplication between steps and expected_results.

Output ONLY a single JSON object — no preamble, no explanation, no markdown fences — with \
exactly these keys:
{"objective": "<one short clause: what the test verifies>", "steps": ["<terse imperative \
action>", ...], "expected_results": ["<terse verifiable assertion>", ...]}
"""


@dataclass
class InstructionNormalizer:
    """Wraps an agent client to produce a normalized JSON instruction string."""

    client: Any
    settings: Settings

    async def normalize(
        self,
        instruction: str,
        target_url: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> str:
        """Return the canonical JSON rewrite, or the original instruction on any failure."""
        request = _build_request(instruction, target_url, data)
        logger.debug("normalize request (model=%s):\n%s", self.settings.model, request)
        message = await self.client.complete(
            [{"role": "user", "content": request}], [], NORMALIZE_SYSTEM_PROMPT
        )
        raw = _collect_text(message)
        logger.debug("normalize raw model output: %s", raw)
        canonical = _to_canonical_json(raw)
        if canonical is None:
            logger.debug("normalize: output not valid JSON of the expected shape; "
                         "falling back to the original instruction")
            return instruction.strip()
        logger.debug("normalize canonical JSON: %s", canonical)
        return canonical


def _to_canonical_json(text: str) -> str | None:
    """Parse, validate and minify the model output into a canonical JSON spec.

    Returns a minified JSON string with a fixed key order, or None when the output is not
    valid JSON of the expected shape (so the caller can fall back to the raw instruction).
    """
    payload = _strip_code_fence(text)
    if not payload:
        return None
    try:
        parsed = json.loads(payload)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None

    objective = parsed.get("objective")
    steps = parsed.get("steps")
    expected = parsed.get("expected_results")
    if not isinstance(objective, str) or not isinstance(steps, list):
        return None
    if expected is None:
        expected = []
    if not isinstance(expected, list):
        return None
    if not objective.strip() or not steps:
        return None

    canonical = {
        "objective": objective.strip(),
        "steps": [str(s).strip() for s in steps],
        "expected_results": [str(e).strip() for e in expected],
    }
    # Minified + fixed key order = deterministic bytes and minimal tokens.
    return json.dumps(canonical, ensure_ascii=False, separators=(",", ":"))


def _strip_code_fence(text: str) -> str:
    """Remove a leading/trailing markdown code fence if the model wrapped its JSON."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    # Drop the opening fence (e.g. ``` or ```json) and a trailing fence line if present.
    lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


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
