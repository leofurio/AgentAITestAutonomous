"""Model comparison: run one test against several models at once and diff the outcomes.

Choosing a model for a test suite is otherwise guesswork — a cheaper model may well
drive a given flow just as reliably as the flagship, and the only way to find out is to
watch them do the same job. A comparison starts one ordinary run per model, in parallel,
and collects each one's verdict, steps and token spend side by side.

Every contender is compared on the *whole pipeline*, not just the browser-driving step.
Each run normalizes the instruction with its own model and then drives the browser with
its own canonical spec, so the comparison answers the question actually being asked —
"how does this test go if I configure this model?" — and reports the full token cost of
that answer. It mirrors an ordinary run, where ``normalizer_model`` defaults to empty and
therefore reuses the run's own model. How each contender chose to read the request is
itself a result worth seeing, so the specs are surfaced side by side.

Each run is a normal run: it writes its own ``report.json`` / ``report.html`` /
``playwright_test.py`` under ``runs/<run_id>/`` and streams over the usual WebSocket. The
comparison itself is a thin index — ``runs/<comparison_id>/comparison.json`` — so results
survive a restart and can be re-read from disk at any time.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .logging_config import get_logger

logger = get_logger("compare")

# Two is the smallest set that compares anything; the cap keeps a single click from
# launching an unbounded number of concurrent browsers.
MIN_MODELS = 2
MAX_MODELS = 6


class ModelSpec(BaseModel):
    """One contender: a model id, optionally on a provider other than the configured one."""

    model: str
    provider: str | None = None


class ComparisonRun(BaseModel):
    """The run started for one contender."""

    run_id: str
    model: str
    provider: str


class Comparison(BaseModel):
    """The index of one comparison; the runs themselves live in their own directories."""

    comparison_id: str
    instruction: str
    target_url: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    runs: list[ComparisonRun] = Field(default_factory=list)


class ComparisonRunner:
    """Starts the parallel runs and reads their reports back."""

    def __init__(self, manager: Any) -> None:
        self.manager = manager

    async def start(
        self,
        instruction: str,
        target_url: str | None,
        data: dict[str, Any] | None,
        models: list[ModelSpec],
    ) -> Comparison:
        """Launch one run per model concurrently, each on its own model end to end."""
        comparison_id = f"cmp_{uuid.uuid4().hex[:12]}"

        runs = []
        for spec in models:
            provider = spec.provider or self.manager.settings.agent_provider
            run_id = self.manager.create_run(
                instruction, target_url, data,
                model=spec.model, provider=provider,
            )
            runs.append(ComparisonRun(run_id=run_id, model=spec.model, provider=provider))

        comparison = Comparison(
            comparison_id=comparison_id, instruction=instruction,
            target_url=target_url, runs=runs,
        )
        self._persist(comparison)
        logger.info(
            "comparison %s started: %s", comparison_id,
            ", ".join(f"{r.provider}/{r.model}" for r in runs),
        )
        return comparison

    async def run_to_completion(self, comparison: Comparison) -> dict[str, Any]:
        """Await every run of a comparison, then return its results (batch/CLI use)."""
        tasks = [
            run.task for run in (self.manager.get(r.run_id) for r in comparison.runs)
            if run is not None and run.task is not None
        ]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        return self.results(comparison.comparison_id)

    def results(self, comparison_id: str) -> dict[str, Any] | None:
        """Comparison index plus each run's outcome, or None if there is no such comparison.

        Read from disk on every call: a run in flight simply has no report yet and comes
        back as ``pending``, so the same endpoint serves progress and final results.
        """
        comparison = self._load(comparison_id)
        if comparison is None:
            return None
        results = [self._run_result(entry) for entry in comparison.runs]
        payload = comparison.model_dump(mode="json")
        payload["results"] = results
        payload["pending"] = sum(1 for r in results if r["status"] == "pending")
        return payload

    def _run_result(self, entry: ComparisonRun) -> dict[str, Any]:
        """One contender's outcome, flattened for a comparison table."""
        result: dict[str, Any] = {
            "run_id": entry.run_id, "model": entry.model, "provider": entry.provider,
            "status": "pending", "verdict": None, "summary": "",
            "normalized_instruction": None,
            "steps": 0, "assertions": 0, "assertions_passed": 0,
            "calls": 0, "input_tokens": 0, "output_tokens": 0,
            "cache_read_tokens": 0, "cache_write_tokens": 0, "duration_seconds": None,
        }
        report = self._read_report(entry.run_id)
        if report is None:
            return result

        assertions = report.get("assertions") or []
        steps = report.get("steps") or []
        result.update(
            status="done",
            verdict=report.get("verdict"),
            summary=report.get("summary", ""),
            # How this contender chose to read the request — a result in its own right.
            normalized_instruction=report.get("normalized_instruction"),
            steps=sum(1 for s in steps if s.get("kind") == "tool_call"),
            assertions=len(assertions),
            assertions_passed=sum(1 for a in assertions if a.get("passed")),
            duration_seconds=_duration(report),
        )
        # Every role the run used, normalizer included: each contender normalizes with
        # its own model, so that call is part of what choosing this model costs.
        for usage in report.get("models") or []:
            for key in ("calls", "input_tokens", "output_tokens",
                        "cache_read_tokens", "cache_write_tokens"):
                result[key] += usage.get(key, 0) or 0
        return result

    def _dir(self, comparison_id: str) -> Path:
        return self.manager.settings.output_dir / comparison_id

    def _persist(self, comparison: Comparison) -> None:
        path = self._dir(comparison.comparison_id)
        path.mkdir(parents=True, exist_ok=True)
        (path / "comparison.json").write_text(
            comparison.model_dump_json(indent=2), encoding="utf-8"
        )

    def _load(self, comparison_id: str) -> Comparison | None:
        path = self._dir(comparison_id) / "comparison.json"
        if not path.exists():
            return None
        try:
            return Comparison.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - a corrupt index must not 500
            logger.warning("unreadable comparison %s: %s", comparison_id, exc)
            return None

    def _read_report(self, run_id: str) -> dict[str, Any] | None:
        path = self.manager.settings.output_dir / run_id / "report.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("unreadable report for run %s: %s", run_id, exc)
            return None


def _duration(report: dict[str, Any]) -> float | None:
    started, finished = report.get("started_at"), report.get("finished_at")
    if not started or not finished:
        return None
    try:
        delta = datetime.fromisoformat(finished) - datetime.fromisoformat(started)
    except ValueError:
        return None
    return round(delta.total_seconds(), 1)
