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


class CodeRunResponse(BaseModel):
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool
    work_dir: str


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
