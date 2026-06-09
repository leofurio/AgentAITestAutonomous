"""ReportBuilder: accumulate steps/assertions and persist the final TestReport."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from ..agent.schemas import AssertionResult, Step, StepKind, TestReport, Verdict
from .html import render_html


class ReportBuilder:
    def __init__(self, run_id: str, instruction: str, model: str,
                 target_url: str | None, output_dir: Path) -> None:
        self.report = TestReport(
            run_id=run_id, instruction=instruction, model=model, target_url=target_url
        )
        self.output_dir = output_dir
        self._step_index = 0

    def _next_index(self) -> int:
        idx = self._step_index
        self._step_index += 1
        return idx

    def add_reasoning(self, text: str) -> Step:
        step = Step(index=self._next_index(), kind=StepKind.REASONING, text=text)
        self.report.steps.append(step)
        return step

    def add_tool_call(self, tool_name: str, tool_input: dict) -> Step:
        step = Step(index=self._next_index(), kind=StepKind.TOOL_CALL,
                    tool_name=tool_name, tool_input=tool_input)
        self.report.steps.append(step)
        return step

    def add_tool_result(self, tool_name: str, summary: str, *,
                        screenshot_path: str | None = None, error: str | None = None) -> Step:
        step = Step(index=self._next_index(), kind=StepKind.TOOL_RESULT, tool_name=tool_name,
                    result_summary=summary, screenshot_path=screenshot_path, error=error)
        self.report.steps.append(step)
        return step

    def add_assertion(self, result: AssertionResult) -> None:
        self.report.assertions.append(result)

    def finalize(self, verdict: Verdict | None, summary: str) -> TestReport:
        # A failing assertion forces a fail, regardless of the declared verdict.
        if any(not a.passed for a in self.report.assertions):
            verdict = Verdict.FAIL
        self.report.verdict = verdict or Verdict.ERROR
        self.report.summary = summary
        self.report.finished_at = datetime.now(UTC)
        return self.report

    def persist(self) -> dict[str, Path]:
        """Write report.json and report.html into the run's output dir."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        json_path = self.output_dir / "report.json"
        html_path = self.output_dir / "report.html"
        json_path.write_text(self.report.model_dump_json(indent=2), encoding="utf-8")
        html_path.write_text(render_html(self.report), encoding="utf-8")
        return {"json": json_path, "html": html_path}
