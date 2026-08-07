"""RunManager: creates runs, owns their EventBus and background task."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..agent.events import EventBus
from ..agent.loop import AgentLoop
from ..config import Settings


@dataclass
class Run:
    run_id: str
    bus: EventBus
    run_dir: Path
    task: asyncio.Task | None = field(default=None)


class RunManager:
    def __init__(
        self,
        settings: Settings,
        client_factory: Callable[[], Any],
        normalizer_factory: Callable[[], Any] | None = None,
        normalizer_settings: Settings | None = None,
        client_factory_for: Callable[[Settings], Any] | None = None,
    ) -> None:
        self.settings = settings
        self.client_factory = client_factory
        # Builds a client for *derived* settings, which is how a run overrides the
        # configured model (the OpenAI/OpenRouter adapters bake settings in, so the
        # client has to be rebuilt rather than reused). Falls back to the plain
        # factory, which is what test/custom factories want.
        self.client_factory_for = client_factory_for or (lambda _settings: client_factory())
        self.normalizer_factory = normalizer_factory
        self.normalizer_settings = normalizer_settings or settings
        self._runs: dict[str, Run] = {}

    def run_settings(self, model: str | None = None, provider: str | None = None) -> Settings:
        """Settings for one run, with the model/provider overridden when given."""
        update: dict[str, Any] = {}
        if model:
            update["model"] = model
        if provider:
            update["agent_provider"] = provider
        return self.settings.model_copy(update=update) if update else self.settings

    def create_run(
        self,
        instruction: str,
        target_url: str | None,
        data: dict[str, Any] | None,
        *,
        model: str | None = None,
        provider: str | None = None,
        normalized_instruction: str | None = None,
    ) -> str:
        run_id = uuid.uuid4().hex[:12]
        run_dir = self.settings.output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        bus = EventBus()
        run = Run(run_id=run_id, bus=bus, run_dir=run_dir)
        self._runs[run_id] = run

        settings = self.run_settings(model, provider)
        # Only rebuild the client when this run overrides the configured model/provider.
        client = (
            self.client_factory()
            if settings is self.settings
            else self.client_factory_for(settings)
        )
        merged_data = {**self.settings.data, **(data or {})}
        loop = AgentLoop(
            client=client,
            settings=settings,
            run_id=run_id,
            instruction=instruction,
            target_url=target_url,
            data=merged_data,
            bus=bus,
            run_dir=run_dir,
            normalizer_factory=self.normalizer_factory,
            normalizer_settings=self.normalizer_settings,
            normalized_instruction=normalized_instruction,
        )
        run.task = asyncio.create_task(self._guarded_run(loop, bus))
        return run_id

    @staticmethod
    async def _guarded_run(loop: AgentLoop, bus: EventBus) -> None:
        # AgentLoop.run already has a top-level try/except, but guard the task too so a
        # failure can never leave the WebSocket consumer hanging on an unclosed bus.
        try:
            await loop.run()
        except Exception as exc:  # noqa: BLE001
            bus.publish("error", message=f"Run task crashed: {exc}")
            bus.close()

    def get(self, run_id: str) -> Run | None:
        return self._runs.get(run_id)
