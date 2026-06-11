"""Generate a standalone Playwright script from a completed aiwebtest report.

The script replays the recorded run without an AI model. Each action targets its
element through a *stable* locator derived from the descriptor captured at run time
(id, data-testid, name, role+accessible-name, or text) — falling back to the
ephemeral ``data-aiwebtest-ref`` only when nothing stable is available. This makes
replay robust across multi-page flows, where the ordinal ref ids are not stable.
"""

from __future__ import annotations

import json
from typing import Any

from ..agent.schemas import StepKind, TestReport
from ..browser.snapshot import _SNAPSHOT_JS
from ..config import BrowserConfig

# Runtime helpers embedded verbatim into the generated script.
_HELPERS = f'''
SNAPSHOT_JS = {json.dumps(_SNAPSHOT_JS)}

# Map input types to ARIA roles for get_by_role fallback.
ROLE_MAP = {{
    "text": "textbox", "email": "textbox", "password": "textbox", "search": "textbox",
    "tel": "textbox", "url": "textbox", "number": "spinbutton",
    "checkbox": "checkbox", "radio": "radio",
}}
ROLE_OK = {{"button", "link", "heading", "textbox", "checkbox", "radio", "tab", "menuitem"}}


async def snapshot(page):
    elements = await page.evaluate(SNAPSHOT_JS)
    print(f"Snapshot: {{len(elements)}} elements at {{page.url}}")
    return elements


async def tag(page):
    # Re-apply ref attributes so the ref fallback in resolve() can still match.
    await page.evaluate(SNAPSHOT_JS)


async def settle(page):
    # Give the page a moment to finish a navigation / re-render before the next action.
    try:
        await page.wait_for_load_state("domcontentloaded", timeout=2000)
    except Exception:
        pass
    await page.wait_for_timeout(150)


def _candidates(page, hint):
    """Ordered locator candidates for a recorded element descriptor."""
    out = []
    if hint.get("testid"):
        out.append(page.get_by_test_id(hint["testid"]))
    if hint.get("id"):
        out.append(page.locator(f'[id="{{hint["id"]}}"]'))
    if hint.get("attr_name") and hint.get("tag") in ("input", "select", "textarea"):
        out.append(page.locator(f'[name="{{hint["attr_name"]}}"]'))
    role = ROLE_MAP.get(hint.get("role"), hint.get("role"))
    name = hint.get("name")
    if role in ROLE_OK and name:
        out.append(page.get_by_role(role, name=name, exact=False).first)
    if name and hint.get("tag"):
        # The recorded role can disagree with the live ARIA role (e.g. an <a>
        # without href is not a "link"), so tag + visible text is tried too.
        safe = name.replace('"', '\\\\"')
        out.append(page.locator(f'{{hint["tag"]}}:has-text("{{safe}}")').first)
    if name:
        out.append(page.get_by_text(name, exact=False).first)
    if hint.get("ref"):
        out.append(page.locator(f'[data-aiwebtest-ref="{{hint["ref"]}}"]'))
    return out


async def resolve(page, hint):
    """Return the first candidate locator that matches something on the page.

    The recorded descriptor is only a hint: probing the candidates instead of
    trusting the strongest one blindly keeps replay working when a recorded
    field does not match the live page (stale id, ARIA role mismatch, ...).
    """
    candidates = _candidates(page, hint or {{}})
    if not candidates:
        raise RuntimeError(f"cannot resolve element from hint {{hint!r}}")
    deadline = time.monotonic() + ACTION_TIMEOUT_MS / 1000
    while True:
        for loc in candidates:
            try:
                if await loc.count() > 0:
                    return loc
            except Exception:
                continue
        if time.monotonic() >= deadline:
            # Let the strongest candidate raise the actionable timeout error.
            return candidates[0]
        await page.wait_for_timeout(200)


async def assert_that(page, condition, target=None, expected=None, description=""):
    await tag(page)
    loc = await resolve(page, target) if target else None
    if condition == "visible":
        if loc is None:
            raise AssertionError(f"{{description}}: no target provided")
        if not await loc.is_visible():
            raise AssertionError(f"{{description}}: expected visible")
    elif condition == "url_contains":
        if (expected or "") not in page.url:
            raise AssertionError(f"{{description}}: {{expected!r}} not in {{page.url!r}}")
    elif condition == "text_contains":
        actual = await (loc.inner_text() if loc else page.inner_text("body"))
        if (expected or "") not in actual:
            raise AssertionError(f"{{description}}: {{expected!r}} not found in {{actual[:500]!r}}")
    elif condition == "value_equals":
        if loc is None:
            raise AssertionError(f"{{description}}: no target provided")
        actual = await loc.input_value()
        if actual != (expected or ""):
            raise AssertionError(f"{{description}}: expected {{expected!r}}, got {{actual!r}}")
    else:
        raise AssertionError(f"{{description}}: unknown condition {{condition!r}}")
    print(f"PASS: {{description or condition}}")
'''


def generate_playwright_script(report: TestReport, browser: BrowserConfig | None = None) -> str:
    """Return Python code that replays the recorded browser tool calls.

    ``browser`` is the configuration the live run used; the generated script
    launches the same browser channel (e.g. installed Chrome) and falls back to
    Playwright's bundled Chromium, mirroring ``BrowserSession``.
    """
    cfg = browser or BrowserConfig()
    viewport = f"{{'width': {cfg.viewport.width}, 'height': {cfg.viewport.height}}}"
    lines = [
        '"""Generated by aiwebtest. Replays one completed run without an AI model."""',
        "",
        "from __future__ import annotations",
        "",
        "import asyncio",
        "import os",
        "import time",
        "from pathlib import Path",
        "",
        "from playwright.async_api import async_playwright",
        "",
        "# Settings of the recorded run.",
        f"CHANNEL = {cfg.channel!r}  # override with AIWEBTEST_REPLAY_CHANNEL",
        f"ACTION_TIMEOUT_MS = {cfg.action_timeout_ms}",
        "",
        _HELPERS.strip("\n"),
        "",
        "",
        "async def launch_browser(pw, headless):",
        '    """Launch the browser like the recorded run: the configured channel first',
        '    (e.g. installed Chrome), then Playwright\'s bundled Chromium."""',
        "    args = ['--disable-dev-shm-usage', '--disable-extensions']",
        "    if headless:",
        "        args.append('--no-sandbox')",
        "    channel = os.environ.get('AIWEBTEST_REPLAY_CHANNEL', CHANNEL or '')",
        "    attempts = [{'channel': channel}] if channel and channel != 'chromium' else []",
        "    attempts.append({})",
        "    last_error = None",
        "    for extra in attempts:",
        "        try:",
        "            return await pw.chromium.launch(headless=headless, args=args, **extra)",
        "        except Exception as exc:",
        "            last_error = exc",
        "    raise RuntimeError(",
        "        'Could not launch a browser for replay. Install Google Chrome, or run '",
        "        '\"playwright install chromium\" to download the bundled browser. '",
        "        f'Last error: {last_error}'",
        "    )",
        "",
        "",
        "async def main():",
        "    async with async_playwright() as pw:",
        "        # Headful by default; set AIWEBTEST_REPLAY_HEADLESS=1 to run headless.",
        "        headless = os.environ.get('AIWEBTEST_REPLAY_HEADLESS', '') == '1'",
        "        browser = await launch_browser(pw, headless)",
        f"        context = await browser.new_context(viewport={viewport})",
        f"        context.set_default_timeout({cfg.action_timeout_ms})",
        f"        context.set_default_navigation_timeout({cfg.nav_timeout_ms})",
        "        page = await context.new_page()",
        "        screenshots_dir = Path('generated_screenshots')",
        "        screenshots_dir.mkdir(exist_ok=True)",
        "        try:",
    ]

    body = _render_steps(report)
    if body:
        lines.extend("            " + line if line else "" for line in body)
    else:
        lines.append("            print('No replayable tool calls were recorded.')")

    lines.extend(
        [
            "        finally:",
            "            await context.close()",
            "            await browser.close()",
            "",
            "",
            "if __name__ == '__main__':",
            "    asyncio.run(main())",
            "",
        ]
    )
    return "\n".join(lines)


def _hint_for(step, ref: str) -> dict[str, Any]:
    return {**(step.locator_hint or {}), "ref": ref}


def _render_steps(report: TestReport) -> list[str]:
    lines: list[str] = []
    shot_index = 0
    for step in report.steps:
        if step.kind != StepKind.TOOL_CALL or not step.tool_name:
            continue
        args = step.tool_input or {}
        tool = step.tool_name
        ref = args.get("ref")
        hint = _hint_for(step, ref) if ref else None
        lines.append(f"print('tool: {tool}')")

        if tool == "navigate":
            lines.append(f"await page.goto({_py(args.get('url', ''))}, wait_until='load')")
            lines.append("await settle(page)")
        elif tool == "get_page_snapshot":
            lines.append("await snapshot(page)")
        elif tool == "click":
            lines.append("await tag(page)")
            lines.append(f"loc = await resolve(page, {_py(hint)})")
            lines.append("await loc.click()")
            # A click often navigates / re-renders; let the page settle before the next step.
            lines.append("await settle(page)")
        elif tool == "type_text":
            lines.append("await tag(page)")
            lines.append(f"loc = await resolve(page, {_py(hint)})")
            lines.append(f"await loc.fill({_py(args.get('text', ''))})")
            if args.get("submit"):
                lines.append("await loc.press('Enter')")
                lines.append("await settle(page)")
        elif tool == "select_option":
            lines.append("await tag(page)")
            lines.extend(_select_option_lines(hint, args))
        elif tool == "press_key":
            lines.append(f"await page.keyboard.press({_py(args.get('key', ''))})")
        elif tool == "wait_for":
            lines.extend(_wait_for_lines(hint, args))
        elif tool == "screenshot":
            shot_index += 1
            lines.append(
                "await page.screenshot("
                f"path=str(screenshots_dir / 'shot_{shot_index:03d}.png'), "
                f"full_page={bool(args.get('full_page', False))})"
            )
        elif tool == "get_text":
            if ref:
                lines.append("await tag(page)")
                lines.append(f"loc = await resolve(page, {_py(hint)})")
                lines.append("print(await loc.inner_text())")
            else:
                lines.append("print(await page.inner_text('body'))")
        elif tool == "assert_that":
            target = _py(hint) if ref else "None"
            lines.append(
                "await assert_that(page, "
                f"condition={_py(args.get('condition', ''))}, "
                f"target={target}, "
                f"expected={_py(args.get('expected'))}, "
                f"description={_py(args.get('description', ''))})"
            )
        elif tool == "finish_test":
            lines.append(f"print({_py('Finished: ' + str(args.get('summary', '')))})")
        else:
            lines.append(f"print('Skipped unsupported tool: {tool}')")
    return lines


def _select_option_lines(hint: dict[str, Any] | None, args: dict[str, Any]) -> list[str]:
    value = _py(args.get("value", ""))
    return [
        f"loc = await resolve(page, {_py(hint)})",
        "try:",
        f"    await loc.select_option(value={value})",
        "except Exception:",
        f"    await loc.select_option(label={value})",
    ]


def _wait_for_lines(hint: dict[str, Any] | None, args: dict[str, Any]) -> list[str]:
    state = args.get("state")
    timeout = int(args.get("timeout_ms", 10000))
    if state in {"visible", "hidden"}:
        return [
            "await tag(page)",
            f"loc = await resolve(page, {_py(hint)})",
            f"await loc.wait_for(state={_py(state)}, timeout={timeout})",
        ]
    return [
        f"await page.get_by_text({_py(args.get('value', ''))}, exact=False)"
        f".first.wait_for(timeout={timeout})"
    ]


def _py(value: Any) -> str:
    return repr(value)
