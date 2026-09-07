"""InstructionNormalizer: rewrite a free-form test request into a canonical, deterministic spec.

A small pre-pass LLM call takes the user's natural-language instruction (plus the target
URL and any test data) and rewrites it into a normalized, structured form. Feeding the
agent loop this canonical spec — instead of the raw prose — reduces run-to-run variance:
the same intent yields the same steps and the same checks. The pass is deliberately
conservative: it clarifies and structures, it never invents steps or data.

The canonical form is a **compact, line-oriented plain-text spec** in which every check
sits under the step after which it must hold::

    GOAL: <one clause>
    STEPS:
    1. <action>
       CHECK: <outcome true right after step 1>
    2. <action>

Attaching checks to their step is what makes them verifiable at all: in a stateful flow
most outcomes are observable only at one moment — a login form is gone once you sign in,
a cart badge resets at checkout — so a check collected into a trailing list has no moment
to be evaluated in, and the agent has to guess one. That guess is exactly the run-to-run
variance this pass exists to remove.

This is more token-efficient than JSON (no braces/quotes/repeated keys) and far more
reliable for models to emit than strict JSON, which cuts down on fallbacks. A tolerant
parser accepts common label/bullet variants and re-serializes them into the exact form
above — fixed section order, renumbered steps — so the same intent yields the same bytes.
The pass runs under a tight output budget/effort (see ``normalizer_settings``). If the
output has no objective or no steps, the pass falls back to the raw instruction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..config import Settings
from ..logging_config import get_logger

logger = get_logger("normalizer")

NORMALIZE_SYSTEM_PROMPT = """\
You normalize web-application test requests into a compact canonical spec for another \
agent to execute against a real browser. Make the intent explicit, ordered and unambiguous \
(same request -> same spec) and keep it minimal. You rewrite and compress; you never plan, \
add, or guess.

Hard rules:
- Preserve intent exactly: never add or drop steps, checks, or data. If the request is \
vague, keep it vague — do not invent specifics.
- One imperative action per step; one verifiable outcome per check.
- Reference test data only by its key in braces, e.g. {username}, {password}. Never inline \
secret values.
- Terse: short imperatives, drop articles and filler, never repeat the URL, an element, or \
a data value. Omit obvious mechanics (snapshots, waiting) unless the request depends on them.

Format — output EXACTLY this and NOTHING else. The first line MUST start with "GOAL:". \
No preamble, no closing remarks, no explanations, no JSON, no markdown fences:
GOAL: <one short clause>
STEPS:
1. <action>
   CHECK: <outcome that is true right after this step>
2. <action>

Attach every check to the step after which it holds, as an indented "CHECK:" line. A step \
may carry none, one, or several. NEVER gather the checks into a list at the end: most are \
true only at one moment — the login form is gone once you sign in, the cart badge resets \
at checkout — so a check parked at the end cannot be evaluated. If the request states no \
expected outcome, emit no CHECK lines at all.

Example:
GOAL: sign in succeeds
STEPS:
1. navigate to login
   CHECK: username and password fields are visible
2. type {username} and {password}
3. click sign in
   CHECK: dashboard shows welcome message
"""

_GOAL_LABELS = ("goal:", "objective:")
_CHECK_PREFIX = "check:"
_STEPS_HEADERS = ("steps", "step")
_CHECKS_HEADERS = (
    "checks", "check", "expected", "expected result", "expected results",
    "expected outcome", "expected outcomes", "assertions", "assertion",
)


@dataclass
class InstructionNormalizer:
    """Wraps an agent client to produce a normalized plain-text instruction string."""

    client: Any
    settings: Settings

    async def normalize(
        self,
        instruction: str,
        target_url: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> str:
        """Return the canonical spec, or the original instruction on any failure."""
        request = _build_request(instruction, target_url, data)
        logger.debug("normalize request (model=%s):\n%s", self.settings.model, request)
        message = await self.client.complete(
            [{"role": "user", "content": request}], [], NORMALIZE_SYSTEM_PROMPT
        )
        raw = _collect_text(message)
        logger.debug("normalize raw model output: %s", raw)
        canonical = _to_canonical_spec(raw)
        if canonical is None:
            logger.debug("normalize: output has no objective/steps; "
                         "falling back to the original instruction")
            return instruction.strip()
        logger.debug("normalize canonical spec:\n%s", canonical)
        return canonical


def _to_canonical_spec(text: str) -> str | None:
    """Parse the model output into the canonical GOAL/STEPS spec.

    Tolerant of label and bullet variants, and of a model that still emits one trailing
    checks list instead of attaching each check to its step. Returns the canonical string
    (fixed section order, renumbered steps, each check under the step it follows) or None
    when there is no objective or no steps, so the caller can fall back to the raw
    instruction.
    """
    payload = _strip_code_fence(text)
    if not payload:
        return None

    objective = ""
    steps: list[tuple[str, list[str]]] = []  # (action, checks that hold after it)
    unplaced: list[str] = []                 # checks with no step to attach to
    section = "steps"  # content after GOAL (before any header) is treated as steps

    for raw_line in payload.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        low = line.lower()

        if low.startswith(_GOAL_LABELS):
            objective = line.split(":", 1)[1].strip()
            section = "steps"
            continue
        if not objective:
            # Ignore any preamble the model emits before the GOAL line, so it never
            # leaks into the steps (a common cause of corrupted specs).
            continue
        header = low.rstrip(":").strip()
        if header in _STEPS_HEADERS:
            section = "steps"
            continue
        if header in _CHECKS_HEADERS:
            section = "checks"
            continue

        item = _strip_bullet(line)
        if not item:
            continue

        check = _as_check(item)  # None when the line is not a "CHECK:" line
        if section == "checks":
            # A model that ignored the format and emitted a trailing list. Keep those
            # checks at the end rather than inventing a step for them: guessing where
            # an unplaced check belongs is exactly the variance this pass removes.
            unplaced.append(check or item)
        elif check:
            (steps[-1][1] if steps else unplaced).append(check)
        elif check is None:
            steps.append((item, []))
        # else: a bare "CHECK:" naming no outcome — drop it.

    if not objective or not steps:
        return None

    lines = [f"GOAL: {objective}", "STEPS:"]
    for index, (action, step_checks) in enumerate(steps, 1):
        lines.append(f"{index}. {action}")
        lines += [f"   CHECK: {check}" for check in step_checks]
    if unplaced:
        lines.append("CHECKS:")
        lines += [f"- {check}" for check in unplaced]
    return "\n".join(lines)


def _as_check(item: str) -> str | None:
    """The outcome text when ``item`` is a "CHECK: ..." line, else None."""
    if not item.lower().startswith(_CHECK_PREFIX):
        return None
    return item[len(_CHECK_PREFIX):].strip()


def _strip_bullet(line: str) -> str:
    """Strip a leading list marker: '- ', '* ', '• ', '1. ', '2) ' etc."""
    text = line.lstrip("-*•").strip() if line[:1] in "-*•" else line
    # Numbered markers: digits followed by '.' or ')'.
    i = 0
    while i < len(text) and text[i].isdigit():
        i += 1
    if i > 0 and i < len(text) and text[i] in ".)":
        text = text[i + 1:].strip()
    return text.strip()


def _strip_code_fence(text: str) -> str:
    """Remove a leading/trailing markdown code fence if the model wrapped its output."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    # Drop the opening fence (e.g. ``` or ```text) and a trailing fence line if present.
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
