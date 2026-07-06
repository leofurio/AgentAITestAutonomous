"""Test suite: saved tests, batch runs, per-test history, and self-healing re-runs.

A suite test is a saved instruction (plus target URL / data) with a pointer to its
latest *recording* — the run whose deterministic ``playwright_test.py`` replays it
without an AI model. Re-running a suite test therefore has three modes:

- ``replay``: execute the recorded script (fast, free, deterministic).
- ``agent``: run the full agentic loop again (re-records the test).
- ``auto`` (default): replay first; when the replay **errors** (broken locator,
  crash — the *test* broke, not the app) fall back to an agent run that re-records
  the script. A replay that *fails* its assertions is reported as a genuine fail —
  healing must never mask a real regression.

Every executed run is appended to the test's history, so the suite shows whether a
test passed yesterday and whether it needed healing.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .logging_config import get_logger

logger = get_logger("suite")

RUN_MODES = ("auto", "replay", "agent")

# History is a diagnostic trail, not a database: keep the most recent entries only.
_HISTORY_LIMIT = 50


class RunRecord(BaseModel):
    """One executed run of a suite test (replay or agent)."""

    run_id: str
    mode: str  # "replay" | "agent"
    verdict: str  # "pass" | "fail" | "error"
    summary: str = ""
    # True on the agent run that re-recorded a test after its replay errored.
    healed: bool = False
    finished_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SuiteTest(BaseModel):
    """A saved, re-runnable test definition."""

    test_id: str
    name: str
    instruction: str
    target_url: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    # The run whose playwright_test.py is the current recording (replayed by
    # replay/auto). Updated whenever an agent run passes.
    source_run_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    history: list[RunRecord] = Field(default_factory=list)


class SuiteStore:
    """JSON-file persistence for the suite (single-process, atomic writes)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._tests: dict[str, SuiteTest] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            for item in raw.get("tests", []):
                test = SuiteTest.model_validate(item)
                self._tests[test.test_id] = test
        except Exception as exc:  # noqa: BLE001 - a corrupt file must not brick the app
            logger.warning("could not load suite file %s: %s", self.path, exc)

    def _save(self) -> None:
        payload = {"tests": [t.model_dump(mode="json") for t in self._tests.values()]}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def add(
        self,
        name: str,
        instruction: str,
        target_url: str | None = None,
        data: dict[str, Any] | None = None,
        source_run_id: str | None = None,
    ) -> SuiteTest:
        test = SuiteTest(
            test_id=uuid.uuid4().hex[:12],
            name=name,
            instruction=instruction,
            target_url=target_url,
            data=data or {},
            source_run_id=source_run_id,
        )
        self._tests[test.test_id] = test
        self._save()
        return test

    def get(self, test_id: str) -> SuiteTest | None:
        return self._tests.get(test_id)

    def list(self) -> list[SuiteTest]:
        return list(self._tests.values())

    def remove(self, test_id: str) -> bool:
        if test_id not in self._tests:
            return False
        del self._tests[test_id]
        self._save()
        return True

    def record_run(self, test_id: str, record: RunRecord) -> None:
        test = self._tests[test_id]
        test.history.append(record)
        del test.history[:-_HISTORY_LIMIT]
        # A passing agent run is the freshest good recording: replay it next time.
        if record.mode == "agent" and record.verdict == "pass":
            test.source_run_id = record.run_id
        self._save()


class SuiteRunner:
    """Executes suite tests through the RunManager (agent) or a subprocess (replay)."""

    def __init__(self, manager: Any, store: SuiteStore, replay_timeout: int = 600) -> None:
        self.manager = manager
        self.store = store
        self.replay_timeout = replay_timeout

    async def run_test(self, test_id: str, mode: str = "auto") -> RunRecord:
        test = self.store.get(test_id)
        if test is None:
            raise KeyError(test_id)
        if mode not in RUN_MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {RUN_MODES}")

        script = self._recording(test)
        if mode == "replay" and script is None:
            record = RunRecord(
                run_id="", mode="replay", verdict="error",
                summary="No recording to replay: run the test in agent mode first.",
            )
            self.store.record_run(test_id, record)
            return record

        if mode in ("replay", "auto") and script is not None:
            record = await self._replay(script)
            self.store.record_run(test_id, record)
            # Self-healing: only an *error* means the test broke (stale locators,
            # crash) — re-record it. A fail is the app regressing: report it.
            if not (mode == "auto" and record.verdict == "error"):
                return record
            logger.info("replay of %s errored; healing with an agent re-run", test.name)

        healed = mode == "auto" and script is not None
        record = await self._agent_run(test, healed=healed)
        self.store.record_run(test_id, record)
        return record

    async def run_all(self, mode: str = "auto") -> list[RunRecord]:
        """Run every suite test sequentially (batch/regression mode)."""
        records = []
        for test in self.store.list():
            records.append(await self.run_test(test.test_id, mode=mode))
        return records

    def _recording(self, test: SuiteTest) -> Path | None:
        if not test.source_run_id:
            return None
        script = self.manager.settings.output_dir / test.source_run_id / "playwright_test.py"
        return script if script.exists() else None

    async def _replay(self, script: Path) -> RunRecord:
        """Execute the recorded script in its own run dir; read back its report."""
        run_id = f"replay_{uuid.uuid4().hex[:12]}"
        work_dir = self.manager.settings.output_dir / run_id
        work_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(script, work_dir / "playwright_test.py")

        env = dict(os.environ)
        if self.manager.settings.browser.headless:
            env["AIWEBTEST_REPLAY_HEADLESS"] = "1"

        proc = await asyncio.create_subprocess_exec(
            sys.executable, "playwright_test.py",
            cwd=str(work_dir), env=env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=self.replay_timeout)
        except TimeoutError:
            proc.kill()
            await proc.communicate()
            return RunRecord(
                run_id=run_id, mode="replay", verdict="error",
                summary=f"Replay timed out after {self.replay_timeout}s",
            )

        verdict, summary = self._replay_outcome(
            work_dir, proc.returncode, stdout.decode("utf-8", errors="replace")
        )
        return RunRecord(run_id=run_id, mode="replay", verdict=verdict, summary=summary)

    @staticmethod
    def _replay_outcome(work_dir: Path, returncode: int | None, stdout: str) -> tuple[str, str]:
        # The replay's Recorder writes the same report.json as a live run; trust it
        # first, and fall back to exit code + output when it is missing (e.g. the
        # script crashed before finalize, or aiwebtest was not importable).
        report = work_dir / "report.json"
        if report.exists():
            try:
                data = json.loads(report.read_text(encoding="utf-8"))
                return data.get("verdict", "error"), data.get("summary", "")
            except Exception as exc:  # noqa: BLE001
                logger.warning("unreadable replay report %s: %s", report, exc)
        if returncode == 0:
            return "pass", "Replay exited 0 (no report.json found)"
        tail = "\n".join(stdout.strip().splitlines()[-5:])
        verdict = "fail" if "FAIL - " in stdout else "error"
        return verdict, f"Replay exited {returncode}: {tail[-500:]}"

    async def _agent_run(self, test: SuiteTest, healed: bool = False) -> RunRecord:
        run_id = self.manager.create_run(test.instruction, test.target_url, test.data)
        run = self.manager.get(run_id)
        await run.task  # the loop persists report.json even on failure
        verdict, summary = self._read_report(run.run_dir)
        return RunRecord(
            run_id=run_id, mode="agent", verdict=verdict, summary=summary, healed=healed
        )

    @staticmethod
    def _read_report(run_dir: Path) -> tuple[str, str]:
        report = run_dir / "report.json"
        try:
            data = json.loads(report.read_text(encoding="utf-8"))
            return data.get("verdict", "error"), data.get("summary", "")
        except Exception as exc:  # noqa: BLE001
            return "error", f"Run produced no readable report: {exc}"
