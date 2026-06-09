"""Web layer: request validation and artifact 404s (no browser launched)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from aiwebtest.web.app import create_app


def _client(settings):
    # client_factory is never invoked for these endpoints (no run is started).
    app = create_app(settings=settings, client_factory=lambda: None)
    return TestClient(app)


def test_empty_instruction_rejected(settings):
    with _client(settings) as client:
        resp = client.post("/api/runs", json={"instruction": "   "})
    assert resp.status_code == 422


def test_unknown_report_is_404(settings):
    with _client(settings) as client:
        resp = client.get("/api/runs/doesnotexist/report.json")
    assert resp.status_code == 404


def test_index_served(settings):
    with _client(settings) as client:
        resp = client.get("/")
    assert resp.status_code == 200
    assert "aiwebtest" in resp.text


def test_runner_served(settings):
    with _client(settings) as client:
        resp = client.get("/runner")
    assert resp.status_code == 200
    assert "Playwright runner" in resp.text


def test_execute_playwright_code_runs_python(settings):
    with _client(settings) as client:
        resp = client.post("/api/playwright/execute", json={"code": "print('hello')"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["exit_code"] == 0
    assert body["stdout"].strip() == "hello"
