"""AgentLoop: the tool-use orchestration that drives the browser.

The loop is intentionally *manual* (rather than the SDK tool-runner) so it can stream
each step to the UI via the EventBus, enforce a max-steps guard, and flush a partial
report on any failure.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ..config import Settings
from ..logging_config import get_logger
from .events import EventBus
from .prompts import SYSTEM_PROMPT, build_user_instruction
from .providers import ensure_agent_client
from .schemas import Verdict

logger = get_logger("loop")


class AgentLoop:
    def __init__(
        self,
        client: Any,
        settings: Settings,
        run_id: str,
        instruction: str,
        target_url: str | None,
        data: dict[str, Any],
        bus: EventBus,
        run_dir: Path,
        normalizer_factory: Callable[[], Any] | None = None,
        normalizer_settings: Settings | None = None,
    ) -> None:
        self.client = ensure_agent_client(client, settings)
        self.settings = settings
        self.run_id = run_id
        self.instruction = instruction
        self.target_url = target_url
        self.data = data
        self.bus = bus
        self.run_dir = run_dir
        # A factory minting a *separate* client for the normalizer pass, plus the Settings
        # it runs under (provider/model may differ from the loop's). When absent, the
        # normalizer pass is skipped regardless of the config flag.
        self.normalizer_factory = normalizer_factory
        self.normalizer_settings = normalizer_settings or settings

    def _allowed_domains(self) -> list[str]:
        configured = list(self.settings.agent.allowed_domains)
        if configured:
            return configured
        if self.target_url:
            host = urlparse(self.target_url).hostname
            if host:
                return [host]
        return []  # empty = no restriction

    async def run(self) -> TestReport:  # noqa: F821 (forward ref for readability)
        # Imports here keep schema/browser deps out of module import time for tests.
        from ..browser.session import BrowserSession
        from ..browser.tools import TOOL_SCHEMAS, BrowserToolset
        from ..report.builder import ReportBuilder

        builder = ReportBuilder(
            run_id=self.run_id,
            instruction=self.instruction,
            model=self.settings.model,
            target_url=self.target_url,
            output_dir=self.run_dir,
            browser=self.settings.browser,
        )
        self.bus.publish("status", state="starting")

        instruction = await self._maybe_normalize(builder)

        try:
            async with BrowserSession(
                self.settings.browser, self.run_dir / "screenshots"
            ) as session:
                toolset = BrowserToolset(
                    page=session.page,
                    screenshot_dir=self.run_dir / "screenshots",
                    allowed_domains=self._allowed_domains(),
                    include_screenshots=self.settings.agent.include_screenshots,
                )
                verdict, summary = await self._drive(
                    builder, toolset, TOOL_SCHEMAS, instruction
                )
        except Exception as exc:  # noqa: BLE001 - top-level safety net
            verdict, summary = Verdict.ERROR, f"Run aborted: {type(exc).__name__}: {exc}"
            self.bus.publish("error", message=summary)

        report = builder.finalize(verdict, summary)
        builder.persist()
        self.bus.publish(
            "report",
            verdict=report.verdict.value,
            summary=report.summary,
            report_url=f"/api/runs/{self.run_id}/report.html",
            playwright_url=f"/api/runs/{self.run_id}/playwright_test.py",
            runner_url="/runner",
        )
        self.bus.publish("status", state="done")
        self.bus.close()
        return report

    async def _maybe_normalize(self, builder) -> str:
        """Rewrite the instruction into a canonical form for a more deterministic run.

        Returns the instruction the loop should drive with — the normalized rewrite when
        the pass is enabled and succeeds, otherwise the original. Normalization is
        best-effort: any failure falls back to the raw instruction so a run is never lost
        to a normalizer error.
        """
        if not self.settings.agent.normalize_instruction or self.normalizer_factory is None:
            logger.debug("normalization disabled or unavailable; using raw instruction")
            return self.instruction

        from .normalizer import InstructionNormalizer

        self.bus.publish("status", state="normalizing")
        logger.debug("normalizing instruction: %r", self.instruction)
        try:
            client = ensure_agent_client(self.normalizer_factory(), self.normalizer_settings)
            normalizer = InstructionNormalizer(client, self.normalizer_settings)
            normalized = await normalizer.normalize(
                self.instruction, self.target_url, self.data
            )
        except Exception as exc:  # noqa: BLE001 - normalization must never abort a run
            logger.warning("normalization skipped: %s", exc)
            self.bus.publish("warning", message=f"Instruction normalization skipped: {exc}")
            return self.instruction

        if normalized and normalized.strip() != self.instruction.strip():
            builder.set_normalized_instruction(normalized)
            self.bus.publish("normalized", text=normalized)
            logger.info("instruction normalized to: %s", normalized)
            return normalized
        logger.debug("normalization produced no change; using raw instruction")
        return self.instruction

    async def _drive(self, builder, toolset, tool_schemas, instruction) -> tuple[Verdict, str]:
        self.bus.publish("status", state="running")
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": build_user_instruction(instruction, self.target_url, self.data),
            }
        ]
        max_steps = self.settings.agent.max_steps

        for step_no in range(max_steps):
            logger.debug("agent turn %d/%d", step_no + 1, max_steps)
            message = await self.client.complete(messages, tool_schemas, SYSTEM_PROMPT)

            text = _collect_text(message)
            if text:
                builder.add_reasoning(text)
                self.bus.publish("reasoning", text=text)

            messages.append(message)
            tool_uses = [b for b in message["content"] if b.get("type") == "tool_use"]

            if not tool_uses:
                # Agent stopped without calling finish_test → treat its text as the summary.
                return Verdict.PASS, text or "Agent ended without an explicit verdict."

            tool_result_blocks = []
            for block in tool_uses:
                tool_input = dict(block["input"])
                step = builder.add_tool_call(block["name"], tool_input)
                self.bus.publish("step", index=step.index, tool=block["name"], input=tool_input)
                logger.debug("tool call: %s(%s)", block["name"], tool_input)

                outcome = await toolset.dispatch(block["name"], tool_input)
                logger.debug(
                    "tool result: %s -> %s%s", block["name"], outcome.summary,
                    " [error]" if outcome.is_error else "",
                )

                if outcome.locator_hint:
                    step.locator_hint = outcome.locator_hint
                builder.add_tool_result(
                    block["name"], outcome.summary,
                    screenshot_path=outcome.screenshot_path,
                    error=outcome.summary if outcome.is_error else None,
                )
                if outcome.screenshot_path:
                    self.bus.publish(
                        "screenshot",
                        url=f"/api/runs/{self.run_id}/screenshots/"
                        f"{Path(outcome.screenshot_path).name}",
                    )
                if outcome.assertion is not None:
                    builder.add_assertion(outcome.assertion)
                    self.bus.publish(
                        "assertion",
                        description=outcome.assertion.description,
                        passed=outcome.assertion.passed,
                        expected=outcome.assertion.expected,
                        actual=outcome.assertion.actual,
                    )

                tool_result_blocks.append({
                    "type": "tool_result",
                    "tool_use_id": block["id"],
                    "content": outcome.content_blocks,
                    "is_error": outcome.is_error,
                })

                if outcome.finished:
                    verdict = Verdict.PASS if outcome.verdict == "pass" else Verdict.FAIL
                    return verdict, _finish_summary(tool_input)

            messages.append({"role": "user", "content": tool_result_blocks})

        return Verdict.ERROR, f"Reached max_steps ({max_steps}) without finishing the test."

def _collect_text(message: dict[str, Any]) -> str:
    parts = [b.get("text", "") for b in message["content"] if b.get("type") == "text"]
    return "\n".join(p for p in parts if p).strip()


def _finish_summary(tool_input: Any) -> str:
    try:
        return dict(tool_input).get("summary", "")
    except Exception:  # noqa: BLE001
        return ""
