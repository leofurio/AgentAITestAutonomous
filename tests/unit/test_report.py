"""ReportBuilder + HTML rendering, including the failing-assertion forces-fail rule."""

from __future__ import annotations

import json
from pathlib import Path

from aiwebtest.agent.schemas import AssertionResult, Verdict
from aiwebtest.report.builder import ReportBuilder
from aiwebtest.report.playwright_codegen import generate_playwright_script


def _builder(tmp_path: Path) -> ReportBuilder:
    return ReportBuilder(
        run_id="abc123", instruction="do a thing", model="claude-opus-4-8",
        target_url="https://example.com", output_dir=tmp_path / "run",
    )


def test_persist_writes_json_and_html(tmp_path: Path):
    b = _builder(tmp_path)
    b.add_reasoning("thinking about it")
    b.add_tool_call("navigate", {"url": "https://example.com"})
    b.add_tool_result("navigate", "Navigated", screenshot_path=None)
    b.add_assertion(AssertionResult(description="title", condition="text_contains",
                                    expected="Demo", actual="Demo App", passed=True))
    report = b.finalize(Verdict.PASS, "all good")
    paths = b.persist()

    assert report.verdict == Verdict.PASS
    data = json.loads(paths["json"].read_text())
    assert data["verdict"] == "pass"
    assert len(data["steps"]) == 3
    html = paths["html"].read_text()
    assert "Demo App" in html and "pass" in html
    playwright = paths["playwright"].read_text()
    assert "async_playwright" in playwright
    assert "https://example.com" in playwright


def test_failing_assertion_forces_fail(tmp_path: Path):
    b = _builder(tmp_path)
    b.add_assertion(AssertionResult(description="x", condition="visible",
                                    expected=None, actual="not visible", passed=False))
    report = b.finalize(Verdict.PASS, "agent thought it passed")  # declared pass
    assert report.verdict == Verdict.FAIL  # overridden by the failed assertion


def test_playwright_replay_uses_stable_locators_and_settles(tmp_path: Path):
    b = _builder(tmp_path)
    # A click with a captured descriptor must replay via a stable locator, not the
    # ephemeral ordinal ref; clicks settle the page before the next step.
    step = b.add_tool_call("click", {"ref": "e6"})
    step.locator_hint = {
        "id": "submit", "tag": "button", "role": "button",
        "name": "Submit", "attr_name": "", "testid": "",
    }
    b.add_tool_call("click", {"ref": "e3"})  # no hint → ref fallback inside resolve()
    report = b.finalize(Verdict.PASS, "clicked through")

    script = generate_playwright_script(report)

    compile(script, "generated_replay.py", "exec")
    assert "async def resolve(page, hint):" in script
    assert "'id': 'submit'" in script          # stable descriptor carried into the script
    assert "await settle(page)" in script      # settle after clicks
    assert "'ref': 'e3'" in script             # ref kept only as a fallback


def test_playwright_replay_probes_locator_candidates(tmp_path: Path):
    # Recorded descriptors are hints, not ground truth: e.g. the snapshot labels
    # every <a> as role "link", but an <a> without href has no ARIA link role, so
    # get_by_role would never match. The replay must probe a candidate chain
    # (testid/id/name/role/tag+text/text/ref) and use the first that exists.
    b = _builder(tmp_path)
    step = b.add_tool_call("click", {"ref": "e2"})
    step.locator_hint = {
        "id": "", "tag": "a", "role": "link", "name": "1", "attr_name": "", "testid": "",
    }
    report = b.finalize(Verdict.PASS, "ok")

    script = generate_playwright_script(report)

    compile(script, "generated_replay.py", "exec")
    assert "def _candidates(page, hint):" in script
    assert ":has-text(" in script              # tag + visible-text fallback
    assert "await loc.count() > 0" in script   # candidates are probed for existence
    assert "loc = await resolve(page," in script


def test_playwright_replay_launches_like_the_live_run(tmp_path: Path):
    # The replay must use the same browser channel as the recorded run (e.g. the
    # installed Chrome) and fall back to the bundled Chromium, mirroring
    # BrowserSession. Otherwise replay fails on machines that never ran
    # `playwright install` with "Executable doesn't exist".
    from aiwebtest.config import BrowserConfig

    b = _builder(tmp_path)
    b.add_tool_call("navigate", {"url": "https://example.com"})
    report = b.finalize(Verdict.PASS, "ok")

    script = generate_playwright_script(report, browser=BrowserConfig(channel="msedge"))

    compile(script, "generated_replay.py", "exec")
    assert "CHANNEL = 'msedge'" in script
    assert "async def launch_browser(pw, headless):" in script
    assert "playwright install chromium" in script   # actionable error message
    assert "context.set_default_timeout(10000)" in script

    # Without an explicit config the default channel ("chrome") is baked in.
    default_script = generate_playwright_script(report)
    assert "CHANNEL = 'chrome'" in default_script


def test_playwright_replay_always_prints_a_verdict(tmp_path: Path):
    # The replay must end with a visible REPLAY RESULT banner (and a matching
    # exit code) whether it passes, fails mid-flow, or breaks during setup —
    # otherwise a replay that stops early gives no outcome at all.
    b = _builder(tmp_path)
    b.add_tool_call("navigate", {"url": "https://example.com"})
    step = b.add_tool_call("click", {"ref": "e1"})
    step.locator_hint = {"id": "go", "tag": "button", "role": "button",
                         "name": "Go", "attr_name": "", "testid": ""}
    report = b.finalize(Verdict.PASS, "ok")

    script = generate_playwright_script(report)

    compile(script, "generated_replay.py", "exec")
    assert "current = note('step 1/2: navigate')" in script   # numbered progress
    assert "current = note('step 2/2: click')" in script
    assert "TOTAL_STEPS = 2" in script
    assert "REPLAY RESULT: {outcome}" in script
    assert "outcome = f'FAIL at {current} - {failure}'" in script
    assert "raise SystemExit(1)" in script                     # exit code mirrors verdict
    assert "traceback.print_exc()" in script


def test_playwright_replay_records_a_report_like_the_live_run(tmp_path: Path):
    # The replay must produce the same artifacts as the autonomous run
    # (report.json + report.html + assertion screenshots) by recording every
    # step into the locally installed aiwebtest ReportBuilder.
    b = _builder(tmp_path)
    b.add_tool_call("navigate", {"url": "https://example.com"})
    b.add_tool_call("assert_that", {"condition": "url_contains", "expected": "example",
                                    "description": "landed"})
    report = b.finalize(Verdict.PASS, "ok")

    script = generate_playwright_script(report)

    compile(script, "generated_replay.py", "exec")
    assert "class Recorder:" in script
    assert "from aiwebtest.report.builder import ReportBuilder" in script
    assert "REC.tool_call('navigate'" in script        # steps mirrored into the report
    assert "REC.assertion(" in script                  # assertions recorded with screenshot
    assert "artifacts = REC.finalize(verdict, outcome)" in script
    assert "never overwrite the original run's artifacts" in script
