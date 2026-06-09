"""REST routes: start a run and fetch its artifacts."""

from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

router = APIRouter(prefix="/api")


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
    manager = request.app.state.manager
    run = manager.get(run_id)
    base = run.run_dir if run else manager.settings.output_dir / run_id
    path = (base / name).resolve()
    # Prevent path traversal: the resolved path must stay inside the run dir.
    if not str(path).startswith(str(base.resolve())) or not path.exists():
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
