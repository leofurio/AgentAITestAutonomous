"""RepairAgent: a focused, tool-less LLM pass that re-points one broken step.

When a recorded step's locator no longer matches any live element, this pass shows the
agent the recorded *intent* and the current page's ref-indexed outline and asks for the
single ref that fulfils that intent (or NONE). It is deliberately narrow — one element,
one answer — so it stays cheap and its output is trivial to validate: the returned ref
must exist in the live snapshot, otherwise it is rejected.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ..agent.schemas import Step
from ..browser.snapshot import SnapshotElement
from ..config import Settings
from ..logging_config import get_logger

logger = get_logger("repair")

REPAIR_SYSTEM_PROMPT = """\
You repair one broken step of a recorded web-app test. A locator that used to work no \
longer matches any element on the live page (the page changed — a control was renamed, \
moved, or restructured). You are given the recorded action's intent and the current \
page's interactive elements, each with a ref id like e12.

Pick the single ref id of the element that best fulfils the recorded intent. Judge by \
role and visible name/label; ignore ordering. If no element on the page can plausibly \
fulfil the intent, answer NONE — never guess a wrong element.

Reply with ONLY the ref id (e.g. e12) or the word NONE. No punctuation, no explanation.
"""

_REF_RE = re.compile(r"\be\d+\b")

# How the recorded tool maps to a human verb for the intent line.
_VERBS = {
    "click": "Click",
    "type_text": "Type into",
    "select_option": "Select an option in",
    "wait_for": "Wait for",
    "get_text": "Read text from",
    "assert_that": "Verify",
}


def describe_step(step: Step) -> str:
    """A one-line, human description of the recorded action for the repair prompt."""
    hint = step.locator_hint or {}
    args = step.tool_input or {}
    verb = _VERBS.get(step.tool_name or "", "Act on")
    descriptor = (
        hint.get("name")
        or hint.get("testid")
        or hint.get("id")
        or hint.get("attr_name")
        or "element"
    )
    role = hint.get("role") or hint.get("tag") or "element"
    line = f'{verb} the {role} labelled "{descriptor}"'
    if step.tool_name == "type_text" and args.get("text"):
        line += f' (text: "{args["text"]}")'
    elif step.tool_name == "select_option" and args.get("value"):
        line += f' (value: "{args["value"]}")'
    return line


def parse_ref(text: str, valid: set[str]) -> str | None:
    """Extract a ref id from the model's reply, accepting it only if it exists live."""
    match = _REF_RE.search(text or "")
    if not match:
        return None
    ref = match.group(0)
    return ref if ref in valid else None


@dataclass
class RepairAgent:
    """Wraps an agent client to re-point a single broken step to a live ref."""

    client: Any
    settings: Settings

    async def repair(
        self, step: Step, outline: str, elements: list[SnapshotElement]
    ) -> str | None:
        """Return a live ref that fulfils ``step``'s intent, or None."""
        valid = {el.ref for el in elements}
        if not valid:
            return None
        request = (
            f"Recorded step to repair:\n{describe_step(step)}\n\n"
            f"Live page elements:\n{outline}\n\n"
            "Which ref matches the recorded step? Reply with the ref id or NONE."
        )
        logger.debug("repair request for %s step:\n%s", step.tool_name, request)
        message = await self.client.complete(
            [{"role": "user", "content": request}], [], REPAIR_SYSTEM_PROMPT
        )
        raw = _collect_text(message)
        ref = parse_ref(raw, valid)
        logger.debug("repair raw=%r -> ref=%s", raw, ref)
        return ref


def _collect_text(message: dict[str, Any]) -> str:
    parts = [b.get("text", "") for b in message.get("content", []) if b.get("type") == "text"]
    return "\n".join(p for p in parts if p).strip()
