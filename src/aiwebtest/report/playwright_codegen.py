"""Generate a standalone Playwright script from a completed aiwebtest report.

The script replays the recorded run without an AI model. Each action targets its
element through a *stable* locator derived from the descriptor captured at run time
(id, data-testid, name, role+accessible-name, or text) — falling back to the
ephemeral ``data-aiwebtest-ref`` only when nothing stable is available. This makes
replay robust across multi-page flows, where the ordinal ref ids are not stable.

The replay also produces the same artifacts as a live run — ``report.json`` and
``report.html`` (plus assertion screenshots) — by recording every step into a
``ReportBuilder`` from the locally installed aiwebtest package. When the package
is not importable the replay still works and only prints to stdout.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..agent.schemas import StepKind, TestReport
from ..browser.snapshot import _SNAPSHOT_JS
from ..config import BrowserConfig

# Runtime helpers embedded verbatim into the generated script. Only the first
# line interpolates anything; the body is a plain string so braces stay single.
_HELPERS_PREFIX = f"SNAPSHOT_JS = {json.dumps(_SNAPSHOT_JS)}"

_HELPERS_BODY = '''
# Map input types to ARIA roles for get_by_role fallback.
ROLE_MAP = {
    "text": "textbox", "email": "textbox", "password": "textbox", "search": "textbox",
    "tel": "textbox", "url": "textbox", "number": "spinbutton",
    "checkbox": "checkbox", "radio": "radio",
}
ROLE_OK = {"button", "link", "heading", "textbox", "checkbox", "radio", "tab", "menuitem"}

PASSED_ASSERTIONS = 0     # incremented by assert_that; reported in the final summary
FAILED_ASSERTIONS = []    # descriptions of failed assertions; forces the fail verdict
REC = None                # Recorder instance, created in main()
SHOT_DIR = Path('screenshots')
_shot_counter = 0


def note(message):
    print(message, flush=True)
    return message


def next_shot_path():
    global _shot_counter
    _shot_counter += 1
    SHOT_DIR.mkdir(parents=True, exist_ok=True)
    return str(SHOT_DIR / f'shot_{_shot_counter:03d}.png')


class Recorder:
    """Writes the same report.json / report.html artifacts as a live run.

    Best effort: uses the locally installed aiwebtest package; when it is not
    importable the replay still runs and only prints to stdout.
    """

    def __init__(self):
        self._builder = None
        self._schemas = None
        self._last_tool = None
        out = Path.cwd()
        if (out / 'report.json').exists():
            out = out / 'replay'  # never overwrite the original run's artifacts
        self.output_dir = out
        try:
            from aiwebtest.agent import schemas
            from aiwebtest.report.builder import ReportBuilder
        except ImportError:
            print('note: aiwebtest is not importable; skipping report.json/report.html')
            return
        self._schemas = schemas
        # No model is called during a replay; naming the recording model keeps the
        # provenance of the steps visible in the replay's own report.
        self._builder = ReportBuilder(
            run_id=out.name, instruction=INSTRUCTION,
            model=f'replay (recorded with {RECORDED_MODEL})',
            target_url=TARGET_URL, output_dir=out,
        )

    def tool_call(self, name, args, hint=None):
        self._last_tool = name
        if self._builder:
            step = self._builder.add_tool_call(name, dict(args))
            step.locator_hint = hint

    def tool_result(self, summary, screenshot_path=None):
        if self._builder and self._last_tool:
            self._builder.add_tool_result(self._last_tool, summary,
                                          screenshot_path=screenshot_path)

    def fail(self, error):
        if self._builder:
            self._builder.add_tool_result(self._last_tool or 'replay',
                                          'Step failed', error=error)

    def assertion(self, description, condition, expected, actual, passed, shot):
        if self._builder:
            self._builder.add_assertion(self._schemas.AssertionResult(
                description=description, condition=condition, expected=expected,
                actual=str(actual)[:500], passed=passed, screenshot_path=shot,
            ))

    def finalize(self, verdict, summary):
        """Persist report.json/report.html; returns their paths (or None)."""
        if not self._builder:
            return None
        self._builder.finalize(self._schemas.Verdict(verdict), summary)
        return self._builder.persist()


async def snapshot(page):
    elements = await page.evaluate(SNAPSHOT_JS)
    print(f"Snapshot: {len(elements)} elements at {page.url}")
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
        out.append(page.locator(f'[id="{hint["id"]}"]'))
    if hint.get("attr_name") and hint.get("tag") in ("input", "select", "textarea"):
        out.append(page.locator(f'[name="{hint["attr_name"]}"]'))
    role = ROLE_MAP.get(hint.get("role"), hint.get("role"))
    name = hint.get("name")
    if role in ROLE_OK and name:
        out.append(page.get_by_role(role, name=name, exact=False).first)
    if name and hint.get("tag"):
        # The recorded role can disagree with the live ARIA role (e.g. an <a>
        # without href is not a "link"), so tag + visible text is tried too.
        safe = name.replace('"', '\\\\"')
        out.append(page.locator(f'{hint["tag"]}:has-text("{safe}")').first)
    if name:
        out.append(page.get_by_text(name, exact=False).first)
    if hint.get("ref"):
        # .first: stale refs from older runs could still match more than one node.
        out.append(page.locator(f'[data-aiwebtest-ref="{hint["ref"]}"]').first)
    return out


async def resolve(page, hint):
    """Return the first candidate locator that matches something on the page.

    The recorded descriptor is only a hint: probing the candidates instead of
    trusting the strongest one blindly keeps replay working when a recorded
    field does not match the live page (stale id, ARIA role mismatch, ...).
    """
    candidates = _candidates(page, hint or {})
    if not candidates:
        raise RuntimeError(f"cannot resolve element from hint {hint!r}")
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


async def _evaluate_assertion(page, loc, condition, expected):
    """Mirror of the live run's assertion evaluation: returns (passed, actual)."""
    if condition == "visible":
        if loc is None:
            return False, "no target provided"
        visible = await loc.is_visible()
        return visible, "visible" if visible else "not visible"
    if condition == "url_contains":
        return (expected or "") in page.url, page.url
    if condition == "text_contains":
        actual = await (loc.inner_text() if loc else page.inner_text("body"))
        actual = actual.strip()
        return (expected or "") in actual, actual[:500]
    if condition == "value_equals":
        if loc is None:
            return False, "no target provided"
        actual = await loc.input_value()
        return actual == (expected or ""), actual
    return False, f"unknown condition {condition!r}"


async def assert_that(page, condition, target=None, expected=None, description=""):
    # Like the live agent run, a failed assertion does NOT abort the replay: it is
    # recorded, the remaining steps still execute, and the final verdict is "fail".
    global PASSED_ASSERTIONS
    await tag(page)
    try:
        loc = await resolve(page, target) if target else None
        passed, actual = await _evaluate_assertion(page, loc, condition, expected)
    except Exception as exc:
        # A bad/recorded assertion (e.g. value_equals on a non-input, or an unresolvable
        # target) must fail softly and let the replay continue — never abort the run.
        passed, actual = False, f'{type(exc).__name__}: {exc}'
    shot = None
    try:
        shot = next_shot_path()
        await page.screenshot(path=shot)
    except Exception:
        shot = None
    if REC:
        REC.assertion(description, condition, expected, actual, passed, shot)
    if passed:
        PASSED_ASSERTIONS += 1
        msg = f"PASS: {description or condition}"
    else:
        FAILED_ASSERTIONS.append(description or condition)
        msg = (f"FAIL: {description or condition}: {condition} failed "
               f"(expected {expected!r}, actual {str(actual)[:300]!r}) - continuing")
    print(msg)
    if REC:
        REC.tool_result(msg, screenshot_path=shot)
'''


def generate_playwright_script(report: TestReport, browser: BrowserConfig | None = None) -> str:
    """Return Python code that replays the recorded browser tool calls.

    ``browser`` is the configuration the live run used; the generated script
    launches the same browser channel (e.g. installed Chrome) and falls back to
    Playwright's bundled Chromium, mirroring ``BrowserSession``.
    """
    cfg = browser or BrowserConfig()
    viewport = f"{{'width': {cfg.viewport.width}, 'height': {cfg.viewport.height}}}"
    total_steps = sum(
        1 for s in report.steps if s.kind == StepKind.TOOL_CALL and s.tool_name
    )
    lines = [
        '"""Generated by aiwebtest. Replays one completed run without an AI model."""',
        "",
        "from __future__ import annotations",
        "",
        "import asyncio",
        "import os",
        "import sys",
        "import time",
        "import traceback",
        "from pathlib import Path",
        "",
        "from playwright.async_api import async_playwright",
        "",
        "# Output may go through a pipe (e.g. the /runner page); force UTF-8 so prints",
        "# never crash on a cp1252 Windows console.",
        "for _stream in (sys.stdout, sys.stderr):",
        "    try:",
        "        _stream.reconfigure(encoding='utf-8', errors='replace')",
        "    except Exception:",
        "        pass",
        "",
        "# Settings of the recorded run.",
        f"CHANNEL = {cfg.channel!r}  # override with AIWEBTEST_REPLAY_CHANNEL",
        f"ACTION_TIMEOUT_MS = {cfg.action_timeout_ms}",
        f"INSTRUCTION = {report.instruction!r}",
        f"TARGET_URL = {report.target_url!r}",
        f"RECORDED_MODEL = {report.model!r}  # model that recorded these steps",
        f"TOTAL_STEPS = {total_steps}",
        "",
        _HELPERS_PREFIX,
        _HELPERS_BODY.rstrip("\n"),
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
        "    global REC, SHOT_DIR",
        "    failure = None",
        "    current = 'setup'",
        "    REC = Recorder()",
        "    SHOT_DIR = REC.output_dir / 'screenshots'",
        "    async with async_playwright() as pw:",
        "        browser = context = None",
        "        try:",
        "            # Headful by default; set AIWEBTEST_REPLAY_HEADLESS=1 to run headless.",
        "            headless = os.environ.get('AIWEBTEST_REPLAY_HEADLESS', '') == '1'",
        "            browser = await launch_browser(pw, headless)",
        f"            context = await browser.new_context(viewport={viewport})",
        f"            context.set_default_timeout({cfg.action_timeout_ms})",
        f"            context.set_default_navigation_timeout({cfg.nav_timeout_ms})",
        "            page = await context.new_page()",
    ]

    body = _render_steps(report)
    if body:
        lines.extend("            " + line if line else "" for line in body)
    else:
        lines.append("            print('No replayable tool calls were recorded.')")

    lines.extend(
        [
            "        except Exception as exc:",
            "            failure = exc",
            "            traceback.print_exc()",
            "            REC.fail(f'{type(exc).__name__}: {exc}')",
            "        finally:",
            "            for closer in (context, browser):",
            "                try:",
            "                    if closer is not None:",
            "                        await closer.close()",
            "                except Exception:",
            "                    pass",
            "",
            "    if failure is not None:",
            "        verdict = 'error'",
            "        outcome = f'ERROR at {current} - {type(failure).__name__}: {failure}'",
            "    elif FAILED_ASSERTIONS:",
            "        verdict = 'fail'",
            "        n_total = PASSED_ASSERTIONS + len(FAILED_ASSERTIONS)",
            "        outcome = (f'FAIL - {len(FAILED_ASSERTIONS)} of {n_total} '",
            "                   'assertion(s) failed: ' + '; '.join(FAILED_ASSERTIONS))",
            "    else:",
            "        verdict = 'pass'",
            "        outcome = (f'PASS - {TOTAL_STEPS} step(s) replayed, '",
            "                   f'{PASSED_ASSERTIONS} assertion(s) verified')",
            "    artifacts = REC.finalize(verdict, outcome)",
            "",
            "    # Always end with a visible verdict; the callers below turn a non-pass",
            "    # verdict into a failing exit code (CLI) or a failing test (pytest).",
            "    print()",
            "    print('=' * 60)",
            "    print(f'REPLAY RESULT: {outcome}')",
            "    if artifacts:",
            "        print(f\"Report: {artifacts['html']}\")",
            "    print('=' * 60)",
            "    return verdict, outcome",
            "",
            "",
            "def test_replay():",
            '    """Pytest entrypoint: `pytest playwright_test.py` replays the run as a test."""',
            "    verdict, outcome = asyncio.run(main())",
            "    assert verdict == 'pass', outcome",
            "",
            "",
            "if __name__ == '__main__':",
            "    verdict, _outcome = asyncio.run(main())",
            "    if verdict != 'pass':",
            "        raise SystemExit(1)",
            "",
        ]
    )
    return "\n".join(lines)


def _hint_for(step, ref: str) -> dict[str, Any]:
    return {**(step.locator_hint or {}), "ref": ref}


_CSS_IDENT = re.compile(r"^[A-Za-z_][\w-]*$")

# Mirror of the ROLE_MAP / ROLE_OK baked into the generated helper body, used to
# render the idiomatic get_by_role locator shown in the step comments.
_ROLE_MAP = {
    "text": "textbox", "email": "textbox", "password": "textbox", "search": "textbox",
    "tel": "textbox", "url": "textbox", "number": "spinbutton",
    "checkbox": "checkbox", "radio": "radio",
}
_ROLE_OK = {"button", "link", "heading", "textbox", "checkbox", "radio", "tab", "menuitem"}


def _dq(value: str) -> str:
    """A double-quoted Python string literal (nicer than repr for comments)."""
    return json.dumps(value)


def _idiomatic_locator(hint: dict[str, Any] | None) -> str | None:
    """Best-guess *idiomatic* Playwright locator for a recorded descriptor.

    Documentation only — emitted as a comment above each step so the generated
    script reads like hand-written Playwright and can be adopted into a
    maintained suite. The runtime still resolves through ``resolve()``, which
    probes the full candidate chain; this shows the single strongest locator a
    developer would likely write by hand. Mirrors ``_candidates`` priority.
    """
    hint = hint or {}
    if hint.get("testid"):
        return f"page.get_by_test_id({_dq(hint['testid'])})"
    el_id = hint.get("id")
    if el_id:
        selector = f"#{el_id}" if _CSS_IDENT.match(el_id) else f'[id="{el_id}"]'
        return f"page.locator({_dq(selector)})"
    if hint.get("attr_name") and hint.get("tag") in ("input", "select", "textarea"):
        selector = f'[name="{hint["attr_name"]}"]'
        return f"page.locator({_dq(selector)})"
    role = _ROLE_MAP.get(hint.get("role"), hint.get("role"))
    name = hint.get("name")
    if role in _ROLE_OK and name:
        return f"page.get_by_role({_dq(role)}, name={_dq(name)}, exact=False)"
    if name:
        return f"page.get_by_text({_dq(name)}, exact=False)"
    return None


def _render_steps(report: TestReport) -> list[str]:
    lines: list[str] = []
    replayable = [s for s in report.steps if s.kind == StepKind.TOOL_CALL and s.tool_name]
    total = len(replayable)
    for step_no, step in enumerate(replayable, start=1):
        args = step.tool_input or {}
        tool = step.tool_name
        ref = args.get("ref")
        hint = _hint_for(step, ref) if ref else None
        # "current" feeds the final FAIL summary with the step that broke; the
        # Recorder mirrors each call into report.json like the live run.
        lines.append(f"current = note('step {step_no}/{total}: {tool}')")
        lines.append(f"REC.tool_call({_py(tool)}, {_py(args)}, hint={_py(step.locator_hint)})")
        # Show the idiomatic hand-written locator so the script reads as adoptable
        # Playwright; the resolve() call below still probes the robust fallback chain.
        idiomatic = _idiomatic_locator(hint) if hint else None
        if idiomatic:
            lines.append(f"# locator: {idiomatic}")

        if tool == "navigate":
            lines.append(f"await page.goto({_py(args.get('url', ''))}, wait_until='load')")
            lines.append("await settle(page)")
            lines.append("REC.tool_result(f'Navigated to {page.url}')")
        elif tool == "get_page_snapshot":
            lines.append("await snapshot(page)")
            lines.append("REC.tool_result('Snapshot taken')")
        elif tool == "click":
            lines.append("await tag(page)")
            lines.append(f"loc = await resolve(page, {_py(hint)})")
            lines.append("await loc.click()")
            # A click often navigates / re-renders; let the page settle before the next step.
            lines.append("await settle(page)")
            lines.append(f"REC.tool_result({_py(f'Clicked {ref}')})")
        elif tool == "type_text":
            lines.append("await tag(page)")
            lines.append(f"loc = await resolve(page, {_py(hint)})")
            lines.append(f"await loc.fill({_py(args.get('text', ''))})")
            if args.get("submit"):
                lines.append("await loc.press('Enter')")
                lines.append("await settle(page)")
            lines.append(f"REC.tool_result({_py(f'Typed into {ref}')})")
        elif tool == "select_option":
            lines.append("await tag(page)")
            lines.extend(_select_option_lines(hint, args))
            selected = args.get("value", "")
            lines.append(f"REC.tool_result({_py(f'Selected {selected!r} in {ref}')})")
        elif tool == "press_key":
            key = args.get("key", "")
            lines.append(f"await page.keyboard.press({_py(key)})")
            lines.append(f"REC.tool_result({_py(f'Pressed {key}')})")
        elif tool == "wait_for":
            lines.extend(_wait_for_lines(hint, args))
            lines.append("REC.tool_result('Wait satisfied')")
        elif tool == "screenshot":
            lines.append("shot = next_shot_path()")
            lines.append(
                f"await page.screenshot(path=shot, full_page={bool(args.get('full_page', False))})"
            )
            lines.append("REC.tool_result('Screenshot captured', screenshot_path=shot)")
        elif tool == "get_text":
            if ref:
                lines.append("await tag(page)")
                lines.append(f"loc = await resolve(page, {_py(hint)})")
                lines.append("print(await loc.inner_text())")
            else:
                lines.append("print(await page.inner_text('body'))")
            lines.append("REC.tool_result('Read text')")
        elif tool == "assert_that":
            target = _py(hint) if ref else "None"
            # assert_that records the AssertionResult (with screenshot) itself.
            lines.append(
                "await assert_that(page, "
                f"condition={_py(args.get('condition', ''))}, "
                f"target={target}, "
                f"expected={_py(args.get('expected'))}, "
                f"description={_py(args.get('description', ''))})"
            )
        elif tool == "finish_test":
            lines.append(f"print({_py('Finished: ' + str(args.get('summary', '')))})")
            lines.append(
                f"REC.tool_result({_py('Recorded verdict: ' + str(args.get('verdict', '')))})"
            )
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
