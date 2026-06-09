"""ReportBuilder + HTML rendering, including the failing-assertion forces-fail rule."""

from __future__ import annotations

import json
from pathlib import Path

from aiwebtest.agent.schemas import AssertionResult, Verdict
from aiwebtest.report.builder import ReportBuilder


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
