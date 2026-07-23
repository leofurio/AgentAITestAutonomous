"""Suite layer: store persistence, replay/heal runner semantics, API routes.

The runner tests use stub recordings (plain Python scripts that mimic the generated
replay's observable behavior: report.json + exit code) so no browser or LLM is needed.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from aiwebtest.suite import RunRecord, SuiteRunner, SuiteStore
from aiwebtest.web.app import create_app

# --- helpers -----------------------------------------------------------------------


def _store(tmp_path: Path) -> SuiteStore:
    return SuiteStore(tmp_path / "suite.json")


def _stub_recording(runs_dir: Path, run_id: str, body: str) -> None:
    """Write a fake recorded run whose playwright_test.py is a plain Python stub."""
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "playwright_test.py").write_text(body, encoding="utf-8")


_PASSING_REPLAY = """\
import json, pathlib
pathlib.Path('report.json').write_text(
    json.dumps({'verdict': 'pass', 'summary': 'replayed fine'}))
print('REPLAY RESULT: PASS - stub')
"""

_FAILING_REPLAY = """\
import json, pathlib
pathlib.Path('report.json').write_text(
    json.dumps({'verdict': 'fail', 'summary': '1 assertion failed'}))
print('REPLAY RESULT: FAIL - stub')
raise SystemExit(1)
"""

# Crashes before writing any report: the "recording broke" case that must heal.
_BROKEN_REPLAY = """\
print('boom: locator not found')
raise SystemExit(3)
"""


@dataclass
class FakeRun:
    run_id: str
    run_dir: Path
    task: asyncio.Task | None = None


@dataclass
class FakeManager:
    """RunManager stand-in: 'agent runs' immediately write a canned report.json."""

    settings: object
    agent_verdict: str = "pass"
    created: list[str] = field(default_factory=list)
    _runs: dict = field(default_factory=dict)

    def create_run(self, instruction, target_url, data):
        run_id = f"agent{len(self.created)}"
        self.created.append(run_id)
        run_dir = self.settings.output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "report.json").write_text(
            json.dumps({"verdict": self.agent_verdict, "summary": "agent ran"})
        )
        (run_dir / "playwright_test.py").write_text("# regenerated recording")

        async def _noop():
            return None

        self._runs[run_id] = FakeRun(run_id, run_dir, asyncio.create_task(_noop()))
        return run_id

    def get(self, run_id):
        return self._runs.get(run_id)


# --- store -------------------------------------------------------------------------


def test_store_roundtrip(tmp_path: Path):
    store = _store(tmp_path)
    test = store.add("login", "log in and check dashboard",
                     target_url="https://ex.com", data={"user": "bob"})
    assert store.get(test.test_id).name == "login"

    # A new instance reads the same file back: persistence survives restarts.
    reloaded = _store(tmp_path)
    loaded = reloaded.get(test.test_id)
    assert loaded is not None
    assert loaded.instruction == "log in and check dashboard"
    assert loaded.data == {"user": "bob"}

    assert reloaded.remove(test.test_id)
    assert not reloaded.remove(test.test_id)
    assert _store(tmp_path).list() == []


def test_store_records_history_and_promotes_passing_agent_runs(tmp_path: Path):
    store = _store(tmp_path)
    test = store.add("t", "do it", source_run_id="orig")

    store.record_run(test.test_id, RunRecord(run_id="r1", mode="replay", verdict="fail"))
    assert store.get(test.test_id).source_run_id == "orig"  # replay never re-records

    store.record_run(test.test_id, RunRecord(run_id="a1", mode="agent", verdict="error"))
    assert store.get(test.test_id).source_run_id == "orig"  # a bad agent run neither

    store.record_run(test.test_id, RunRecord(run_id="a2", mode="agent", verdict="pass"))
    assert store.get(test.test_id).source_run_id == "a2"    # fresh good recording wins
    assert [r.run_id for r in store.get(test.test_id).history] == ["r1", "a1", "a2"]


def test_store_survives_a_corrupt_file(tmp_path: Path):
    (tmp_path / "suite.json").write_text("{not json", encoding="utf-8")
    assert _store(tmp_path).list() == []  # degraded, not crashed


# --- runner ------------------------------------------------------------------------


async def test_replay_mode_runs_the_recording(tmp_path: Path, settings):
    manager = FakeManager(settings=settings)
    _stub_recording(settings.output_dir, "rec1", _PASSING_REPLAY)
    store = _store(tmp_path)
    test = store.add("t", "do it", source_run_id="rec1")

    record = await SuiteRunner(manager, store).run_test(test.test_id, mode="replay")

    assert (record.mode, record.verdict, record.healed) == ("replay", "pass", False)
    assert manager.created == []                    # no agent, no LLM
    assert store.get(test.test_id).history[-1].run_id == record.run_id
    # The replay ran in its own run dir and left the usual artifacts there.
    assert (settings.output_dir / record.run_id / "report.json").exists()


async def test_auto_reports_a_replay_fail_without_healing(tmp_path: Path, settings):
    # A failing assertion means the APP regressed: healing would mask the regression.
    manager = FakeManager(settings=settings)
    _stub_recording(settings.output_dir, "rec1", _FAILING_REPLAY)
    store = _store(tmp_path)
    test = store.add("t", "do it", source_run_id="rec1")

    record = await SuiteRunner(manager, store).run_test(test.test_id, mode="auto")

    assert (record.mode, record.verdict) == ("replay", "fail")
    assert manager.created == []                    # no heal on fail
    assert store.get(test.test_id).source_run_id == "rec1"


async def test_auto_heals_a_broken_recording_via_agent_rerun(tmp_path: Path, settings):
    # An ERROR means the TEST broke (stale locator, crash): re-record via the agent.
    manager = FakeManager(settings=settings, agent_verdict="pass")
    _stub_recording(settings.output_dir, "rec1", _BROKEN_REPLAY)
    store = _store(tmp_path)
    test = store.add("t", "do it", source_run_id="rec1")

    record = await SuiteRunner(manager, store).run_test(test.test_id, mode="auto")

    assert (record.mode, record.verdict, record.healed) == ("agent", "pass", True)
    assert manager.created == [record.run_id]
    # History keeps the honest trail: the errored replay AND the healing run.
    history = store.get(test.test_id).history
    assert [(r.mode, r.verdict) for r in history] == [("replay", "error"), ("agent", "pass")]
    # The healing run becomes the recording replayed next time.
    assert store.get(test.test_id).source_run_id == record.run_id


def _healable_recording(settings, run_id: str) -> None:
    """A broken replay stub PLUS a readable report.json the heal tier re-executes."""
    from aiwebtest.agent.schemas import TestReport

    _stub_recording(settings.output_dir, run_id, _BROKEN_REPLAY)
    (settings.output_dir / run_id / "report.json").write_text(
        TestReport(run_id=run_id, instruction="do it", model="rec").model_dump_json(),
        encoding="utf-8",
    )


class _FakeReplayer:
    """Stands in for LocalizedReplayer: returns a scripted outcome, no browser/LLM."""

    outcome_verdict = "pass"
    outcome_repaired = True

    def __init__(self, **kwargs):
        self.run_id = kwargs["run_id"]

    async def run(self):
        from aiwebtest.replay.executor import HealOutcome

        return HealOutcome(self.run_id, self.outcome_verdict, "in-process heal",
                           self.outcome_repaired)


async def test_auto_localized_repair_heals_before_the_agent(tmp_path, settings, monkeypatch):
    # An errored replay is first healed in-process (Tier 1); the expensive agent re-run
    # (Tier 2) is never reached, and the passing heal becomes the new recording.
    _FakeReplayer.outcome_verdict = "pass"
    monkeypatch.setattr("aiwebtest.replay.executor.LocalizedReplayer", _FakeReplayer)
    manager = FakeManager(settings=settings)
    manager.client_factory = lambda: object()  # enables the heal tier
    _healable_recording(settings, "rec1")
    store = _store(tmp_path)
    test = store.add("t", "do it", source_run_id="rec1")

    record = await SuiteRunner(manager, store).run_test(test.test_id, mode="auto")

    assert (record.mode, record.verdict, record.healed) == ("heal", "pass", True)
    assert manager.created == []  # the agent tier was NOT reached
    history = [(r.mode, r.verdict) for r in store.get(test.test_id).history]
    assert history == [("replay", "error"), ("heal", "pass")]
    assert store.get(test.test_id).source_run_id == record.run_id  # heal is the recording


async def test_auto_localized_repair_error_falls_back_to_agent(tmp_path, settings, monkeypatch):
    # If the in-process heal itself errors (site changed structurally), escalate to the
    # full agent re-run — the heal attempt is still recorded in the honest history.
    _FakeReplayer.outcome_verdict = "error"
    monkeypatch.setattr("aiwebtest.replay.executor.LocalizedReplayer", _FakeReplayer)
    manager = FakeManager(settings=settings, agent_verdict="pass")
    manager.client_factory = lambda: object()
    _healable_recording(settings, "rec1")
    store = _store(tmp_path)
    test = store.add("t", "do it", source_run_id="rec1")

    record = await SuiteRunner(manager, store).run_test(test.test_id, mode="auto")

    assert (record.mode, record.verdict) == ("agent", "pass")
    assert manager.created == [record.run_id]
    history = [(r.mode, r.verdict) for r in store.get(test.test_id).history]
    assert history == [("replay", "error"), ("heal", "error"), ("agent", "pass")]


async def test_auto_localized_repair_can_be_disabled(tmp_path, settings, monkeypatch):
    # With the flag off, an errored replay skips straight to the agent re-run (the prior
    # behavior) — the in-process replayer must never even be constructed.
    settings.agent.localized_repair = False
    monkeypatch.setattr(
        "aiwebtest.replay.executor.LocalizedReplayer",
        lambda **kw: (_ for _ in ()).throw(AssertionError("heal tier should be skipped")),
    )
    manager = FakeManager(settings=settings, agent_verdict="pass")
    manager.client_factory = lambda: object()
    _healable_recording(settings, "rec1")
    store = _store(tmp_path)
    test = store.add("t", "do it", source_run_id="rec1")

    record = await SuiteRunner(manager, store).run_test(test.test_id, mode="auto")

    assert record.mode == "agent"
    history = [(r.mode, r.verdict) for r in store.get(test.test_id).history]
    assert history == [("replay", "error"), ("agent", "pass")]


async def test_auto_without_recording_runs_the_agent(tmp_path: Path, settings):
    manager = FakeManager(settings=settings)
    store = _store(tmp_path)
    test = store.add("t", "do it")  # never recorded

    record = await SuiteRunner(manager, store).run_test(test.test_id, mode="auto")

    assert (record.mode, record.verdict) == ("agent", "pass")
    assert record.healed is False   # first recording, nothing to heal


async def test_replay_mode_without_recording_errors(tmp_path: Path, settings):
    store = _store(tmp_path)
    test = store.add("t", "do it")

    record = await SuiteRunner(FakeManager(settings=settings), store).run_test(
        test.test_id, mode="replay"
    )

    assert record.verdict == "error"
    assert "agent mode first" in record.summary


async def test_run_all_returns_a_record_per_test(tmp_path: Path, settings):
    manager = FakeManager(settings=settings)
    _stub_recording(settings.output_dir, "rec1", _PASSING_REPLAY)
    _stub_recording(settings.output_dir, "rec2", _FAILING_REPLAY)
    store = _store(tmp_path)
    store.add("a", "one", source_run_id="rec1")
    store.add("b", "two", source_run_id="rec2")

    records = await SuiteRunner(manager, store).run_all(mode="replay")

    assert sorted(r.verdict for r in records) == ["fail", "pass"]


async def test_unknown_test_raises(tmp_path: Path, settings):
    with pytest.raises(KeyError):
        await SuiteRunner(FakeManager(settings=settings), _store(tmp_path)).run_test("nope")


# --- routes ------------------------------------------------------------------------


def _client(settings) -> TestClient:
    app = create_app(settings=settings, client_factory=lambda: None)
    return TestClient(app)


def test_suite_crud_over_http(settings):
    with _client(settings) as client:
        resp = client.post("/api/suite", json={"name": "login", "instruction": "log in"})
        assert resp.status_code == 200
        test_id = resp.json()["test_id"]

        listed = client.get("/api/suite").json()["tests"]
        assert [t["name"] for t in listed] == ["login"]

        assert client.delete(f"/api/suite/{test_id}").status_code == 200
        assert client.delete(f"/api/suite/{test_id}").status_code == 404
        assert client.get("/api/suite").json()["tests"] == []


def test_suite_save_validation(settings):
    with _client(settings) as client:
        assert client.post("/api/suite", json={"name": " ", "instruction": "x"}).status_code == 422
        assert client.post("/api/suite", json={"name": "x", "instruction": " "}).status_code == 422
        resp = client.post(
            "/api/suite",
            json={"name": "x", "instruction": "y", "source_run_id": "../evil"},
        )
        assert resp.status_code == 422


def test_suite_run_replay_over_http(settings):
    with _client(settings) as client:
        # Recording is a stub script: exercises the full subprocess replay path.
        run_dir = settings.output_dir / "recA"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "playwright_test.py").write_text(_PASSING_REPLAY, encoding="utf-8")

        test_id = client.post(
            "/api/suite",
            json={"name": "t", "instruction": "do", "source_run_id": "recA"},
        ).json()["test_id"]

        record = client.post(f"/api/suite/{test_id}/run", json={"mode": "replay"}).json()
        assert (record["mode"], record["verdict"]) == ("replay", "pass")

        bad = client.post(f"/api/suite/{test_id}/run", json={"mode": "yolo"})
        assert bad.status_code == 422
        missing = client.post("/api/suite/nope/run", json={"mode": "replay"})
        assert missing.status_code == 404


def test_suite_run_all_over_http(settings):
    with _client(settings) as client:
        run_dir = settings.output_dir / "recB"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "playwright_test.py").write_text(_PASSING_REPLAY, encoding="utf-8")
        client.post(
            "/api/suite",
            json={"name": "t", "instruction": "do", "source_run_id": "recB"},
        )

        res = client.post("/api/suite/run_all", json={"mode": "replay"}).json()
        assert (res["total"], res["passed"]) == (1, 1)
