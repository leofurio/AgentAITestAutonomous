"""REST routes: start a run and fetch its artifacts."""

from __future__ import annotations

import asyncio
import re
import sys
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..replay import REPLAY_LOG_NAME

router = APIRouter(prefix="/api")

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
# Hosts considered local callers; "testclient" is Starlette's in-process TestClient.
_LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost", "testclient"}


class RunRequest(BaseModel):
    instruction: str
    target_url: str | None = None
    data: dict | None = None


class RunResponse(BaseModel):
    run_id: str


class CodeRunRequest(BaseModel):
    code: str
    timeout_seconds: int = 120


class SuiteSaveRequest(BaseModel):
    name: str
    instruction: str
    target_url: str | None = None
    data: dict | None = None
    # The completed run whose playwright_test.py becomes the test's recording.
    source_run_id: str | None = None


class SuiteRunModeRequest(BaseModel):
    mode: str = "auto"  # auto | replay | agent


class SuiteRenameRequest(BaseModel):
    name: str


class CodeRunResponse(BaseModel):
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool
    work_dir: str


class HealRequest(BaseModel):
    # The completed run whose recording (report.json) should be healed in-process.
    run_id: str
    timeout_seconds: int = 120


class HealResponse(BaseModel):
    verdict: str  # "pass" | "fail" | "error"
    repaired: bool  # True when the AI re-pointed at least one broken locator
    summary: str
    heal_run_id: str
    # The re-recorded, deterministic script + report of the heal run (present on success).
    script_url: str | None = None
    report_url: str | None = None


@router.post("/runs", response_model=RunResponse)
async def create_run(req: RunRequest, request: Request) -> RunResponse:
    if not req.instruction.strip():
        raise HTTPException(status_code=422, detail="instruction must not be empty")
    manager = request.app.state.manager
    run_id = manager.create_run(req.instruction, req.target_url, req.data)
    return RunResponse(run_id=run_id)


def _artifact(request: Request, run_id: str, name: str) -> Path:
    # Reject run ids that could traverse out of the runs directory (e.g. "..").
    if not _RUN_ID_RE.fullmatch(run_id):
        raise HTTPException(status_code=404, detail="artifact not found")
    manager = request.app.state.manager
    run = manager.get(run_id)
    base = (run.run_dir if run else manager.settings.output_dir / run_id).resolve()
    path = (base / name).resolve()
    # Prevent path traversal: the resolved path must stay inside the run dir.
    if not path.is_relative_to(base) or not path.exists():
        raise HTTPException(status_code=404, detail="artifact not found")
    return path


@router.get("/runs/{run_id}/report.json")
async def get_report_json(run_id: str, request: Request) -> FileResponse:
    return FileResponse(_artifact(request, run_id, "report.json"), media_type="application/json")


@router.get("/runs/{run_id}/report.html")
async def get_report_html(run_id: str, request: Request) -> FileResponse:
    return FileResponse(_artifact(request, run_id, "report.html"), media_type="text/html")


@router.get("/runs/{run_id}/" + REPLAY_LOG_NAME)
async def get_replay_log(run_id: str, request: Request) -> FileResponse:
    # The step-by-step narration of a model-free run: what it clicked, typed, asserted,
    # and (for a heal) which locator it had to re-point.
    return FileResponse(
        _artifact(request, run_id, REPLAY_LOG_NAME), media_type="text/plain"
    )


@router.get("/runs/{run_id}/playwright_test.py")
async def get_playwright_script(run_id: str, request: Request) -> FileResponse:
    return FileResponse(
        _artifact(request, run_id, "playwright_test.py"),
        media_type="text/x-python",
    )


@router.get("/runs/{run_id}/screenshots/{filename}")
async def get_screenshot(run_id: str, filename: str, request: Request) -> FileResponse:
    safe = Path(filename).name  # strip any directory components
    return FileResponse(_artifact(request, run_id, f"screenshots/{safe}"), media_type="image/png")


# --- Model comparison: the same test against several models in parallel -----------


@router.get("/config")
async def get_config(request: Request) -> dict:
    """The configured provider/model, so the UI can prefill the comparison form.

    Deliberately narrow: never expose API keys or the rest of the settings object.
    """
    settings = request.app.state.settings
    return {
        "agent_provider": settings.agent_provider,
        "model": settings.model,
        "min_models": MIN_MODELS,
        "max_models": MAX_MODELS,
    }


@router.post("/compare")
async def start_comparison(req: CompareRequest, request: Request) -> dict:
    if not req.instruction.strip():
        raise HTTPException(status_code=422, detail="instruction must not be empty")

    models = [m for m in req.models if m.model.strip()]
    for spec in models:
        spec.model = spec.model.strip()
        spec.provider = (spec.provider or "").strip() or None
    if not MIN_MODELS <= len(models) <= MAX_MODELS:
        raise HTTPException(
            status_code=422,
            detail=f"give between {MIN_MODELS} and {MAX_MODELS} models to compare",
        )

    runner = request.app.state.comparison_runner
    # Returns as soon as the runs are started: they stream over /ws/runs/{run_id} and
    # their outcomes are polled from GET /api/compare/{comparison_id}.
    comparison = await runner.start(req.instruction, req.target_url, req.data, models)
    return comparison.model_dump(mode="json")


@router.get("/compare/{comparison_id}")
async def get_comparison(comparison_id: str, request: Request) -> dict:
    if not _RUN_ID_RE.fullmatch(comparison_id):
        raise HTTPException(status_code=404, detail="comparison not found")
    results = request.app.state.comparison_runner.results(comparison_id)
    if results is None:
        raise HTTPException(status_code=404, detail="comparison not found")
    return results


# --- Suite: saved tests, history, replay / self-healing re-runs -------------------


def _with_artifact_urls(test: dict, output_dir: Path) -> dict:
    """Annotate each history entry with the artifacts that actually exist on disk.

    Computed server-side (rather than stored on the record) so the links can never go
    stale when a run directory is pruned, and the UI needs no extra probing requests.
    """
    for record in test.get("history", []):
        run_id = record.get("run_id") or ""
        if not _RUN_ID_RE.fullmatch(run_id):
            continue
        run_dir = output_dir / run_id
        if (run_dir / REPLAY_LOG_NAME).exists():
            record["log_url"] = f"/api/runs/{run_id}/{REPLAY_LOG_NAME}"
        if (run_dir / "report.html").exists():
            record["report_url"] = f"/api/runs/{run_id}/report.html"
    return test


@router.get("/suite")
async def list_suite(request: Request) -> dict:
    store = request.app.state.suite_store
    output_dir = request.app.state.settings.output_dir
    return {
        "tests": [
            _with_artifact_urls(t.model_dump(mode="json"), output_dir) for t in store.list()
        ]
    }


@router.patch("/suite/{test_id}")
async def rename_suite_test(test_id: str, req: SuiteRenameRequest, request: Request) -> dict:
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="name must not be empty")
    test = request.app.state.suite_store.rename(test_id, name)
    if test is None:
        raise HTTPException(status_code=404, detail="suite test not found")
    return test.model_dump(mode="json")


@router.post("/suite")
async def save_suite_test(req: SuiteSaveRequest, request: Request) -> dict:
    if not req.name.strip():
        raise HTTPException(status_code=422, detail="name must not be empty")
    if not req.instruction.strip():
        raise HTTPException(status_code=422, detail="instruction must not be empty")
    source = req.source_run_id
    if source is not None and not _RUN_ID_RE.fullmatch(source):
        raise HTTPException(status_code=422, detail="invalid source_run_id")
    store = request.app.state.suite_store
    test = store.add(
        name=req.name.strip(), instruction=req.instruction,
        target_url=req.target_url, data=req.data, source_run_id=source,
    )
    return test.model_dump(mode="json")


@router.delete("/suite/{test_id}")
async def delete_suite_test(test_id: str, request: Request) -> dict:
    if not request.app.state.suite_store.remove(test_id):
        raise HTTPException(status_code=404, detail="suite test not found")
    return {"deleted": test_id}


@router.post("/suite/{test_id}/run")
async def run_suite_test(test_id: str, req: SuiteRunModeRequest, request: Request) -> dict:
    from ..suite import RUN_MODES

    if req.mode not in RUN_MODES:
        raise HTTPException(status_code=422, detail=f"mode must be one of {RUN_MODES}")
    runner = request.app.state.suite_runner
    try:
        # Awaited to completion: replays are quick; agent runs can take minutes,
        # which is acceptable for a local tool (the UI disables the button).
        record = await runner.run_test(test_id, mode=req.mode)
    except KeyError:
        raise HTTPException(status_code=404, detail="suite test not found") from None
    return record.model_dump(mode="json")


@router.post("/suite/run_all")
async def run_suite(req: SuiteRunModeRequest, request: Request) -> dict:
    from ..suite import RUN_MODES

    if req.mode not in RUN_MODES:
        raise HTTPException(status_code=422, detail=f"mode must be one of {RUN_MODES}")
    records = await request.app.state.suite_runner.run_all(mode=req.mode)
    passed = sum(1 for r in records if r.verdict == "pass")
    return {
        "total": len(records),
        "passed": passed,
        "records": [r.model_dump(mode="json") for r in records],
    }


@router.post("/playwright/execute", response_model=CodeRunResponse)
async def execute_playwright_code(req: CodeRunRequest, request: Request) -> CodeRunResponse:
    # This endpoint runs arbitrary Python on the host. Refuse unless enabled, and by
    # default only accept calls from the local machine.
    settings = request.app.state.settings
    if not settings.code_runner_enabled:
        raise HTTPException(status_code=403, detail="code runner is disabled")
    client_host = request.client.host if request.client else None
    if not settings.code_runner_allow_remote and client_host not in _LOCAL_HOSTS:
        raise HTTPException(
            status_code=403,
            detail="code runner only accepts local requests "
            "(set code_runner_allow_remote to override)",
        )

    code = req.code.strip()
    if not code:
        raise HTTPException(status_code=422, detail="code must not be empty")

    timeout = max(1, min(req.timeout_seconds, 300))
    manager = request.app.state.manager
    run_id = f"manual_{uuid.uuid4().hex[:12]}"
    work_dir = manager.settings.output_dir / run_id
    work_dir.mkdir(parents=True, exist_ok=True)
    script_path = work_dir / "pasted_playwright.py"
    script_path.write_text(code, encoding="utf-8")

    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        str(script_path),
        cwd=str(work_dir),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        timed_out = False
    except TimeoutError:
        proc.kill()
        stdout, stderr = await proc.communicate()
        timed_out = True

    return CodeRunResponse(
        exit_code=proc.returncode,
        stdout=stdout.decode("utf-8", errors="replace"),
        stderr=stderr.decode("utf-8", errors="replace"),
        timed_out=timed_out,
        work_dir=str(work_dir),
    )


@router.post("/playwright/heal", response_model=HealResponse)
async def heal_playwright_recording(req: HealRequest, request: Request) -> HealResponse:
    # Localized self-healing for the runner: re-run a *recording* in-process and let the
    # agent re-point the step whose locator broke. Same host/enable gate as the code
    # runner (this launches a browser and may call the model), keyed by a run id — arbitrary
    # pasted code has no recorded step intent to repair, so it must come from a real run.
    settings = request.app.state.settings
    if not settings.code_runner_enabled:
        raise HTTPException(status_code=403, detail="code runner is disabled")
    client_host = request.client.host if request.client else None
    if not settings.code_runner_allow_remote and client_host not in _LOCAL_HOSTS:
        raise HTTPException(
            status_code=403,
            detail="code runner only accepts local requests "
            "(set code_runner_allow_remote to override)",
        )
    if not _RUN_ID_RE.fullmatch(req.run_id):
        raise HTTPException(status_code=404, detail="recording not found")

    manager = request.app.state.manager
    base = manager.settings.output_dir.resolve()
    report_path = (base / req.run_id / "report.json").resolve()
    if not report_path.is_relative_to(base) or not report_path.exists():
        raise HTTPException(status_code=404, detail="recording not found")

    from ..agent.providers import ensure_agent_client
    from ..agent.schemas import TestReport
    from ..replay.executor import LocalizedReplayer
    from ..replay.repair import RepairAgent

    try:
        report = TestReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - a malformed recording is a client-visible 422
        raise HTTPException(status_code=422, detail="recording is not readable") from None

    heal_run_id = f"heal_{uuid.uuid4().hex[:12]}"
    client = ensure_agent_client(manager.client_factory(), settings)
    replayer = LocalizedReplayer(
        report=report,
        browser=settings.browser,
        output_dir=settings.output_dir / heal_run_id,
        run_id=heal_run_id,
        repair=RepairAgent(client, settings).repair,
        include_screenshots=settings.agent.include_screenshots,
    )
    timeout = max(1, min(req.timeout_seconds, 600))
    try:
        outcome = await asyncio.wait_for(replayer.run(), timeout=timeout)
    except TimeoutError:
        raise HTTPException(
            status_code=504, detail=f"heal timed out after {timeout}s"
        ) from None

    heal_dir = settings.output_dir / heal_run_id
    return HealResponse(
        verdict=outcome.verdict,
        repaired=outcome.repaired,
        summary=outcome.summary,
        heal_run_id=heal_run_id,
        script_url=(f"/api/runs/{heal_run_id}/playwright_test.py"
                    if (heal_dir / "playwright_test.py").exists() else None),
        report_url=(f"/api/runs/{heal_run_id}/report.html"
                    if (heal_dir / "report.html").exists() else None),
    )
