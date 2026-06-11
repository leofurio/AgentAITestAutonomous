"""BrowserToolset: implements each tool Claude can call, plus the Anthropic schemas.

``dispatch(name, input)`` never raises into the agent loop — failures are returned as
structured error results so the model can self-correct (e.g. by re-snapshotting).
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from playwright.async_api import Error as PlaywrightError
from playwright.async_api import Page
from playwright.async_api import TimeoutError as PlaywrightTimeout

from ..agent.schemas import AssertionResult
from .snapshot import SnapshotElement, ref_selector, take_snapshot

# Anthropic tool definitions exposed to Claude.
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "navigate",
        "description": "Navigate the browser to a URL on the site under test.",
        "input_schema": {
            "type": "object",
            "properties": {"url": {"type": "string", "description": "Absolute URL to open"}},
            "required": ["url"],
        },
    },
    {
        "name": "get_page_snapshot",
        "description": (
            "Return a ref-indexed outline of the current page's interactive and visible "
            "elements. Call this after navigation or any action that changes the page."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "include_screenshot": {
                    "type": "boolean",
                    "description": "Also attach a screenshot of the viewport.",
                }
            },
        },
    },
    {
        "name": "click",
        "description": "Click an element identified by its snapshot ref id.",
        "input_schema": {
            "type": "object",
            "properties": {"ref": {"type": "string", "description": "Element ref, e.g. e12"}},
            "required": ["ref"],
        },
    },
    {
        "name": "type_text",
        "description": "Fill a text input/textarea identified by ref, optionally pressing Enter.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ref": {"type": "string"},
                "text": {"type": "string"},
                "submit": {"type": "boolean", "description": "Press Enter after typing."},
            },
            "required": ["ref", "text"],
        },
    },
    {
        "name": "select_option",
        "description": "Select an option (by value or label) in a <select> identified by ref.",
        "input_schema": {
            "type": "object",
            "properties": {"ref": {"type": "string"}, "value": {"type": "string"}},
            "required": ["ref", "value"],
        },
    },
    {
        "name": "press_key",
        "description": "Press a keyboard key globally (e.g. Enter, Tab, Escape).",
        "input_schema": {
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
        },
    },
    {
        "name": "wait_for",
        "description": "Wait for a condition before proceeding (anti-flakiness).",
        "input_schema": {
            "type": "object",
            "properties": {
                "state": {
                    "type": "string",
                    "enum": ["visible", "hidden", "text"],
                    "description": "visible/hidden wait on a ref; text waits for text on the page.",
                },
                "ref": {"type": "string"},
                "value": {"type": "string", "description": "Text to wait for when state=text."},
                "timeout_ms": {"type": "integer"},
            },
            "required": ["state"],
        },
    },
    {
        "name": "screenshot",
        "description": "Capture a screenshot of the page and attach it.",
        "input_schema": {
            "type": "object",
            "properties": {"full_page": {"type": "boolean"}},
        },
    },
    {
        "name": "get_text",
        "description": "Read the visible text of an element (by ref) or the whole page.",
        "input_schema": {
            "type": "object",
            "properties": {"ref": {"type": "string"}},
        },
    },
    {
        "name": "assert_that",
        "description": (
            "Record a deterministic verification. The harness evaluates the condition and "
            "stores the pass/fail result in the report."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "description": {"type": "string", "description": "What is being verified."},
                "condition": {
                    "type": "string",
                    "enum": ["visible", "text_contains", "url_contains", "value_equals"],
                },
                "ref": {"type": "string", "description": "Element ref for visible/value_equals."},
                "expected": {
                    "type": "string",
                    "description": "Expected text/substring/value (not needed for 'visible').",
                },
            },
            "required": ["description", "condition"],
        },
    },
    {
        "name": "finish_test",
        "description": "End the test and declare the overall verdict.",
        "input_schema": {
            "type": "object",
            "properties": {
                "verdict": {"type": "string", "enum": ["pass", "fail"]},
                "summary": {"type": "string"},
            },
            "required": ["verdict", "summary"],
        },
    },
]


@dataclass
class ToolOutcome:
    """Result of dispatching one tool call."""

    summary: str
    content_blocks: list[dict[str, Any]] = field(default_factory=list)
    is_error: bool = False
    screenshot_path: str | None = None
    assertion: AssertionResult | None = None
    finished: bool = False
    verdict: str | None = None
    locator_hint: dict[str, Any] | None = None


class DomainGuardError(Exception):
    """Raised internally when navigation would leave the allowed domain set."""


class BrowserToolset:
    def __init__(
        self,
        page: Page,
        screenshot_dir: Path,
        allowed_domains: list[str],
        include_screenshots: bool = True,
    ) -> None:
        self.page = page
        self.screenshot_dir = screenshot_dir
        self.allowed_domains = [d.lower() for d in allowed_domains]
        self.include_screenshots = include_screenshots
        self._shot_counter = 0
        # ref -> element descriptor from the most recent snapshot (for replay hints).
        self._elements_by_ref: dict[str, SnapshotElement] = {}

    # -- dispatch ---------------------------------------------------------------

    async def dispatch(self, name: str, tool_input: dict[str, Any]) -> ToolOutcome:
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            return ToolOutcome(summary=f"Unknown tool '{name}'", is_error=True,
                               content_blocks=[_text(f"Unknown tool: {name}")])
        try:
            outcome = await handler(tool_input)
        except DomainGuardError as exc:
            return ToolOutcome(summary=str(exc), is_error=True, content_blocks=[_text(str(exc))])
        except (PlaywrightTimeout, PlaywrightError) as exc:
            msg = (
                f"Action '{name}' failed: {type(exc).__name__}: {str(exc).splitlines()[0]}. "
                "The page may have changed — call get_page_snapshot and retry."
            )
            return ToolOutcome(summary=msg, is_error=True, content_blocks=[_text(msg)])
        except Exception as exc:  # noqa: BLE001 - tools must never crash the loop
            msg = f"Action '{name}' raised {type(exc).__name__}: {exc}"
            return ToolOutcome(summary=msg, is_error=True, content_blocks=[_text(msg)])

        # Attach a stable locator descriptor for any ref-targeted action.
        ref = tool_input.get("ref")
        if ref and outcome.locator_hint is None:
            el = self._elements_by_ref.get(ref)
            if el is not None:
                outcome.locator_hint = el.locator_hint()
        return outcome

    # -- guardrail --------------------------------------------------------------

    def _check_domain(self, url: str) -> None:
        if not self.allowed_domains:
            return
        host = (urlparse(url).hostname or "").lower()
        if not any(host == d or host.endswith("." + d) for d in self.allowed_domains):
            raise DomainGuardError(
                f"Refused: '{host}' is outside the allowed domains {self.allowed_domains}. "
                "Stay on the site under test."
            )

    def _locator(self, ref: str):
        return self.page.locator(ref_selector(ref))

    async def _save_screenshot(self, full_page: bool = False) -> str:
        self._shot_counter += 1
        path = self.screenshot_dir / f"shot_{self._shot_counter:03d}.png"
        await self.page.screenshot(path=str(path), full_page=full_page)
        return str(path)

    # -- tools ------------------------------------------------------------------

    async def _tool_navigate(self, args: dict[str, Any]) -> ToolOutcome:
        url = args["url"]
        self._check_domain(url)
        await self.page.goto(url, wait_until="domcontentloaded")
        title = await self.page.title()
        return ToolOutcome(
            summary=f"Navigated to {self.page.url} ({title!r})",
            content_blocks=[_text(f"Loaded {self.page.url}\nTitle: {title}")],
        )

    async def _tool_get_page_snapshot(self, args: dict[str, Any]) -> ToolOutcome:
        outline, elements = await take_snapshot(self.page)
        self._elements_by_ref = {el.ref: el for el in elements}
        blocks = [_text(outline)]
        shot_path = None
        if args.get("include_screenshot") and self.include_screenshots:
            shot_path = await self._save_screenshot()
            blocks.append(_image(shot_path))
        return ToolOutcome(summary="Snapshot taken", content_blocks=blocks,
                           screenshot_path=shot_path)

    async def _tool_click(self, args: dict[str, Any]) -> ToolOutcome:
        await self._locator(args["ref"]).click()
        return ToolOutcome(summary=f"Clicked {args['ref']}",
                           content_blocks=[_text(f"Clicked element {args['ref']}.")])

    async def _tool_type_text(self, args: dict[str, Any]) -> ToolOutcome:
        loc = self._locator(args["ref"])
        await loc.fill(args["text"])
        if args.get("submit"):
            await loc.press("Enter")
        return ToolOutcome(
            summary=f"Typed into {args['ref']}",
            content_blocks=[_text(f"Filled {args['ref']} with {args['text']!r}"
                                  + (" and pressed Enter." if args.get("submit") else "."))],
        )

    async def _tool_select_option(self, args: dict[str, Any]) -> ToolOutcome:
        loc = self._locator(args["ref"])
        try:
            await loc.select_option(value=args["value"])
        except PlaywrightError:
            await loc.select_option(label=args["value"])
        return ToolOutcome(summary=f"Selected {args['value']!r} in {args['ref']}",
                           content_blocks=[_text(f"Selected {args['value']!r}.")])

    async def _tool_press_key(self, args: dict[str, Any]) -> ToolOutcome:
        await self.page.keyboard.press(args["key"])
        return ToolOutcome(summary=f"Pressed {args['key']}",
                           content_blocks=[_text(f"Pressed key {args['key']}.")])

    async def _tool_wait_for(self, args: dict[str, Any]) -> ToolOutcome:
        timeout = args.get("timeout_ms", 10000)
        state = args["state"]
        if state in ("visible", "hidden"):
            await self._locator(args["ref"]).wait_for(state=state, timeout=timeout)
            msg = f"Element {args['ref']} is {state}."
        else:  # text
            value = args["value"]
            await self.page.get_by_text(value).first.wait_for(timeout=timeout)
            msg = f"Text {value!r} appeared."
        return ToolOutcome(summary=msg, content_blocks=[_text(msg)])

    async def _tool_screenshot(self, args: dict[str, Any]) -> ToolOutcome:
        path = await self._save_screenshot(full_page=args.get("full_page", False))
        blocks = [_text("Screenshot captured.")]
        if self.include_screenshots:
            blocks.append(_image(path))
        return ToolOutcome(summary="Screenshot captured", content_blocks=blocks,
                           screenshot_path=path)

    async def _tool_get_text(self, args: dict[str, Any]) -> ToolOutcome:
        if args.get("ref"):
            text = (await self._locator(args["ref"]).inner_text())[:2000]
        else:
            text = (await self.page.inner_text("body"))[:2000]
        return ToolOutcome(summary="Read text", content_blocks=[_text(text)])

    async def _tool_assert_that(self, args: dict[str, Any]) -> ToolOutcome:
        condition = args["condition"]
        expected = args.get("expected")
        passed, actual = await self._evaluate_assertion(condition, args.get("ref"), expected)
        shot = await self._save_screenshot()
        result = AssertionResult(
            description=args["description"],
            condition=condition,
            expected=expected,
            actual=actual,
            passed=passed,
            screenshot_path=shot,
        )
        verdict = "PASSED" if passed else "FAILED"
        msg = f"Assertion {verdict}: {args['description']} (actual={actual!r})"
        return ToolOutcome(summary=msg, content_blocks=[_text(msg)], assertion=result,
                           screenshot_path=shot)

    async def _tool_finish_test(self, args: dict[str, Any]) -> ToolOutcome:
        return ToolOutcome(
            summary=f"Test finished: {args['verdict']}",
            content_blocks=[_text(f"Recorded verdict: {args['verdict']}.")],
            finished=True,
            verdict=args["verdict"],
        )

    # -- assertion evaluation ---------------------------------------------------

    async def _evaluate_assertion(
        self, condition: str, ref: str | None, expected: str | None
    ) -> tuple[bool, str]:
        if condition == "visible":
            if not ref:
                return False, "no ref provided"
            visible = await self._locator(ref).is_visible()
            return visible, "visible" if visible else "not visible"
        if condition == "url_contains":
            url = self.page.url
            return (expected or "") in url, url
        if condition == "text_contains":
            if ref:
                actual = await self._locator(ref).inner_text()
            else:
                actual = await self.page.inner_text("body")
            actual = actual.strip()
            return (expected or "") in actual, actual[:500]
        if condition == "value_equals":
            if not ref:
                return False, "no ref provided"
            actual = await self._locator(ref).input_value()
            return actual == (expected or ""), actual
        return False, f"unknown condition {condition}"


def _text(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def _image(path: str) -> dict[str, Any]:
    data = base64.standard_b64encode(Path(path).read_bytes()).decode("ascii")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": data},
    }
