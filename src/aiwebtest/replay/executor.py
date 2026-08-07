"""LocalizedReplayer: re-run a recorded run in-process, healing broken steps in place.

Unlike the standalone ``playwright_test.py`` (deterministic, model-free), this executor
keeps a live browser page in hand, so at the step where a recorded locator no longer
resolves it can show the agent the current page and adopt the element the step meant —
then continue. It reuses the same ``BrowserToolset`` the live agent drives and records
into the same ``ReportBuilder``, so its output directory is a normal recording (a fresh
``report.json`` / ``report.html`` / ``playwright_test.py`` carrying the corrected
locators). A passing heal therefore becomes the suite's new deterministic recording.

Assertions stay soft, exactly like the live run and the generated replay: a failed
assertion does not abort the flow and forces a ``fail`` verdict (a real regression the
heal must surface, never mask). Only a step that cannot be resolved or dispatched — even
after repair — yields ``error``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from ..agent.schemas import StepKind, TestReport, Verdict
from ..config import BrowserConfig
from ..logging_config import get_logger
from . import REPLAY_LOG_NAME
from .matcher import match_ref

logger = get_logger("replay.executor")

# Given the recorded step, the live outline and the live elements, return the ref of the
# element that fulfils the step's intent, or None. Invoked only when the deterministic
# match fails, and only for steps that target an element.
RepairCallback = Callable[[object, str, list], Awaitable[str | None]]


@dataclass
class HealOutcome:
    """Result of an in-process localized-repair run."""

    run_id: str
    verdict: str  # "pass" | "fail" | "error"
    summary: str
    repaired: bool  # True if the AI repair pass re-pointed at least one step


class LocalizedReplayer:
    def __init__(
        self,
        report: TestReport,
        browser: BrowserConfig,
        output_dir: Path,
        run_id: str,
        repair: RepairCallback | None = None,
        include_screenshots: bool = True,
    ) -> None:
        self.report = report
        self.browser = browser
        self.output_dir = output_dir
        self.run_id = run_id
        self.repair = repair
        self.include_screenshots = include_screenshots

    async def run(self) -> HealOutcome:
        # Heavy browser deps are imported lazily so importing the suite stays cheap.
        from ..browser.session import BrowserSession
        from ..browser.snapshot import take_snapshot
        from ..browser.tools import BrowserToolset
        from ..report.builder import ReportBuilder

        builder = ReportBuilder(
            run_id=self.run_id,
            instruction=self.report.instruction,
            model="heal",
            target_url=self.report.target_url,
            output_dir=self.output_dir,
            browser=self.browser,
        )
        replayable = [
            s for s in self.report.steps if s.kind == StepKind.TOOL_CALL and s.tool_name
        ]
        failure: str | None = None
        repaired = False
        steps_done = 0
        # Narrates every step (and every repair) so an AI-free run leaves the same
        # human-readable trail as the generated replay script's stdout.
        log: list[str] = [f"Localized repair of {self.report.instruction!r}", ""]
        total = len(replayable)

        try:
            async with BrowserSession(self.browser, self.output_dir / "screenshots") as session:
                toolset = BrowserToolset(
                    page=session.page,
                    screenshot_dir=self.output_dir / "screenshots",
                    allowed_domains=[],  # replaying a trusted recording: don't re-gate hosts
                    include_screenshots=self.include_screenshots,
                )
                for step_no, step in enumerate(replayable, start=1):
                    tool = step.tool_name
                    args = dict(step.tool_input or {})
                    is_assert = tool == "assert_that"
                    log.append(f"step {step_no}/{total}: {tool}")

                    # A recorded tool_input carrying a ref targets an element: re-resolve it
                    # against the live page (its ordinal ref is meaningless now) and heal it
                    # if the recorded descriptor no longer matches anything.
                    if "ref" in args:
                        outline, elements = await take_snapshot(session.page)
                        toolset._elements_by_ref = {el.ref: el for el in elements}
                        ref = match_ref(step.locator_hint, elements)
                        # Only *actions* are AI-repaired: re-pointing an assertion's target
                        # could turn a real "the checked element vanished" regression into a
                        # false pass. An assertion whose target is gone soft-fails instead,
                        # exactly as the deterministic replay does — never a hard error.
                        if ref is None and not is_assert and self.repair is not None:
                            log.append("  ! recorded locator no longer matches — "
                                       "asking the AI to re-point it")
                            ref = await self.repair(step, outline, elements)
                            if ref is not None:
                                repaired = True
                                log.append(f"  ✓ repaired: now targeting {ref}")
                                logger.info("repaired step %r -> live ref %s", tool, ref)
                        if ref is None:
                            if is_assert:
                                log.append("  ✗ assertion target not found — recorded as FAIL")
                                self._record_failed_assertion(builder, args)
                                continue
                            log.append("  ✗ could not resolve the element — stopping")
                            builder.add_tool_call(tool, args)
                            builder.add_tool_result(
                                tool, f"Could not resolve target for {tool}",
                                error="unresolved element",
                            )
                            failure = (
                                f"Could not resolve the element for '{tool}' "
                                f"(recorded hint: {step.locator_hint})"
                            )
                            break
                        args["ref"] = ref

                    call = builder.add_tool_call(tool, args)
                    outcome = await toolset.dispatch(tool, args)
                    if outcome.locator_hint:
                        call.locator_hint = outcome.locator_hint
                    builder.add_tool_result(
                        tool, outcome.summary,
                        screenshot_path=outcome.screenshot_path,
                        error=outcome.summary if outcome.is_error else None,
                    )
                    if outcome.assertion is not None:
                        builder.add_assertion(outcome.assertion)
                    log.append(f"  {'✗' if outcome.is_error else '→'} {outcome.summary}")
                    steps_done += 1
                    if outcome.is_error:
                        failure = outcome.summary
                        break
                    if outcome.finished:
                        break
        except Exception as exc:  # noqa: BLE001 - a heal crash must degrade, not propagate
            failure = f"{type(exc).__name__}: {exc}"
            log.append(f"  ✗ {failure}")
            logger.warning("localized repair aborted: %s", failure)

        summary = self._summary(failure, repaired, steps_done)
        # PASS unless a step could not be executed; finalize() downgrades to FAIL when an
        # assertion failed (a real regression). An execution failure dominates either way.
        builder.finalize(Verdict.PASS if failure is None else Verdict.ERROR, summary)
        if failure is not None:
            builder.report.verdict = Verdict.ERROR
        builder.persist()
        verdict = builder.report.verdict.value
        log += ["", f"RESULT: {verdict.upper()} - {builder.report.summary}"]
        self._write_log(log)
        logger.info("localized repair %s -> %s (repaired=%s)", self.run_id, verdict, repaired)
        return HealOutcome(self.run_id, verdict, builder.report.summary, repaired)

    def _write_log(self, lines: list[str]) -> None:
        """Persist the step narration; a logging failure must never fail the run."""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / REPLAY_LOG_NAME).write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )
        except Exception as exc:  # noqa: BLE001 - the log is diagnostics, not the result
            logger.warning("could not write the heal log in %s: %s", self.output_dir, exc)

    @staticmethod
    def _record_failed_assertion(builder, args: dict) -> None:
        """Record an assertion whose target vanished as a soft failure (mirrors replay)."""
        from ..agent.schemas import AssertionResult

        description = args.get("description", "")
        builder.add_tool_call("assert_that", args)
        builder.add_assertion(AssertionResult(
            description=description, condition=args.get("condition", ""),
            expected=args.get("expected"), actual="target element not found", passed=False,
        ))
        builder.add_tool_result(
            "assert_that", f"Assertion FAILED: {description or 'target element not found'}"
        )

    @staticmethod
    def _summary(failure: str | None, repaired: bool, steps_done: int) -> str:
        if failure is not None:
            return f"Localized repair could not complete: {failure}"
        if repaired:
            return (
                f"Healed in-process: re-pointed broken locator(s), "
                f"replayed {steps_done} step(s)."
            )
        return f"Replayed {steps_done} step(s) in-process; no locator repair was needed."
