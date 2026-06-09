"""AgentLoop: the Claude tool-use orchestration that drives the browser.

The loop is intentionally *manual* (rather than the SDK tool-runner) so it can stream
each step to the UI via the EventBus, enforce a max-steps guard, and flush a partial
report on any failure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ..config import Settings
from .events import EventBus
from .prompts import SYSTEM_PROMPT, build_user_instruction
from .schemas import Verdict


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
    ) -> None:
        self.client = client
        self.settings = settings
        self.run_id = run_id
        self.instruction = instruction
        self.target_url = target_url
        self.data = data
        self.bus = bus
        self.run_dir = run_dir

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
        )
        self.bus.publish("status", state="starting")

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
                verdict, summary = await self._drive(builder, toolset, TOOL_SCHEMAS)
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
        )
        self.bus.publish("status", state="done")
        self.bus.close()
        return report

    async def _drive(self, builder, toolset, tool_schemas) -> tuple[Verdict, str]:
        self.bus.publish("status", state="running")
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": build_user_instruction(self.instruction, self.target_url, self.data),
            }
        ]
        max_steps = self.settings.agent.max_steps

        for _ in range(max_steps):
            message = await self._call_model(messages, tool_schemas)

            text = _collect_text(message)
            if text:
                builder.add_reasoning(text)
                self.bus.publish("reasoning", text=text)

            messages.append({"role": "assistant", "content": message.content})
            tool_uses = [b for b in message.content if getattr(b, "type", None) == "tool_use"]

            if not tool_uses:
                # Agent stopped without calling finish_test → treat its text as the summary.
                return Verdict.PASS, text or "Agent ended without an explicit verdict."

            tool_result_blocks = []
            for block in tool_uses:
                step = builder.add_tool_call(block.name, dict(block.input))
                self.bus.publish("step", index=step.index, tool=block.name, input=block.input)

                outcome = await toolset.dispatch(block.name, dict(block.input))

                builder.add_tool_result(
                    block.name, outcome.summary,
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
                    "tool_use_id": block.id,
                    "content": outcome.content_blocks,
                    "is_error": outcome.is_error,
                })

                if outcome.finished:
                    verdict = Verdict.PASS if outcome.verdict == "pass" else Verdict.FAIL
                    return verdict, _finish_summary(block.input)

            messages.append({"role": "user", "content": tool_result_blocks})

        return Verdict.ERROR, f"Reached max_steps ({max_steps}) without finishing the test."

    async def _call_model(self, messages: list[dict[str, Any]], tool_schemas) -> Any:
        """Stream one assistant turn, publishing text deltas, return the final message."""
        kwargs: dict[str, Any] = {
            "model": self.settings.model,
            "max_tokens": self.settings.max_tokens,
            "system": SYSTEM_PROMPT,
            "tools": tool_schemas,
            "messages": messages,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": self.settings.effort},
        }
        async with self.client.messages.stream(**kwargs) as stream:
            async for _ in stream.text_stream:
                # Text deltas are surfaced from the final message; iterate to drive the stream.
                pass
            return await stream.get_final_message()


def _collect_text(message: Any) -> str:
    parts = [b.text for b in message.content if getattr(b, "type", None) == "text"]
    return "\n".join(p for p in parts if p).strip()


def _finish_summary(tool_input: Any) -> str:
    try:
        return dict(tool_input).get("summary", "")
    except Exception:  # noqa: BLE001
        return ""
