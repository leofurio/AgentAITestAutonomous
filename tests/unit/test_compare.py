"""Model comparison: one instruction, several models in parallel, one diffable result."""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from aiwebtest.compare import ComparisonRunner, ModelSpec
from aiwebtest.web.app import create_app
from tests.conftest import FakeAnthropicClient, text_turn, tool_turn


def _finishing_turns():
    return [tool_turn("finish_test", {"verdict": "pass", "summary": "done"})]


def _no_op_normalizer():
    # Output with no GOAL line: the pass finds no canonical spec and the runs fall
    # back to the raw instruction, which is what tests not about normalization want.
    return FakeAnthropicClient([text_turn("(nothing canonical here)")])


def _app(settings, **kwargs):
    return create_app(settings=settings, **kwargs)


def _runner(settings, **kwargs):
    app = _app(settings, **kwargs)
    return app.state.comparison_runner, app.state.manager


@pytest.mark.asyncio
async def test_comparison_runs_every_model_on_the_same_instruction(settings):
    # Each contender gets its own run, driven by its own model, from one instruction.
    runner, _ = _runner(
        settings,
        client_factory=lambda: FakeAnthropicClient(_finishing_turns()),
        normalizer_factory=_no_op_normalizer,
    )

    comparison = await runner.start(
        "log in", "https://example.test", None,
        [ModelSpec(model="model-a"), ModelSpec(model="model-b", provider="openai")],
    )

    assert [r.model for r in comparison.runs] == ["model-a", "model-b"]
    # An unset provider inherits the configured one; an explicit one overrides it.
    assert [r.provider for r in comparison.runs] == [settings.agent_provider, "openai"]
    assert len(set(r.run_id for r in comparison.runs)) == 2  # independent runs

    results = await runner.run_to_completion(comparison)
    assert results["pending"] == 0
    assert [r["verdict"] for r in results["results"]] == ["pass", "pass"]

    # Each run reports the model it was actually given, not the configured default.
    for entry in comparison.runs:
        report = json.loads(
            (settings.output_dir / entry.run_id / "report.json").read_text()
        )
        assert report["model"] == entry.model
        assert report["models"][0]["model"] == entry.model


@pytest.mark.asyncio
async def test_comparison_normalizes_once_for_every_model(settings):
    # Fairness rule: the pre-pass runs once and every model is handed the identical
    # spec. Re-normalizing per model would change the input between contenders.
    canonical = "GOAL: log in\nSTEPS:\n1. open the login page"
    normalizer_calls = []

    def normalizer_factory():
        normalizer_calls.append(1)
        return FakeAnthropicClient([text_turn(canonical)])

    runner, _ = _runner(
        settings,
        client_factory=lambda: FakeAnthropicClient(_finishing_turns()),
        normalizer_factory=normalizer_factory,
    )

    comparison = await runner.start(
        "log in pls", "https://example.test", None,
        [ModelSpec(model="model-a"), ModelSpec(model="model-b"),
         ModelSpec(model="model-c")],
    )
    await runner.run_to_completion(comparison)

    assert len(normalizer_calls) == 1                     # one pass, not one per model
    assert comparison.normalized_instruction == canonical
    assert comparison.normalizer_usage.calls == 1

    for entry in comparison.runs:
        report = json.loads(
            (settings.output_dir / entry.run_id / "report.json").read_text()
        )
        assert report["normalized_instruction"] == canonical
        assert report["instruction"] == "log in pls"      # original preserved
        # The shared pre-pass is billed to the comparison, never to a contender.
        assert [m["role"] for m in report["models"]] == ["agent"]


@pytest.mark.asyncio
async def test_comparison_survives_a_normalizer_failure(settings):
    # The pre-pass is best-effort: every model still runs, all on the raw instruction.
    def exploding_normalizer():
        raise RuntimeError("no normalizer key")

    runner, _ = _runner(
        settings,
        client_factory=lambda: FakeAnthropicClient(_finishing_turns()),
        normalizer_factory=exploding_normalizer,
    )

    comparison = await runner.start(
        "log in", None, None,
        [ModelSpec(model="model-a"), ModelSpec(model="model-b")],
    )
    results = await runner.run_to_completion(comparison)

    assert comparison.normalized_instruction is None
    assert [r["verdict"] for r in results["results"]] == ["pass", "pass"]


@pytest.mark.asyncio
async def test_comparison_results_are_readable_before_the_runs_finish(settings):
    # The same endpoint serves progress and final results: a run with no report yet
    # is "pending" rather than an error, so the UI can poll one shape throughout.
    runner, _ = _runner(
        settings,
        client_factory=lambda: FakeAnthropicClient(_finishing_turns()),
        normalizer_factory=_no_op_normalizer,
    )
    comparison = await runner.start(
        "log in", None, None,
        [ModelSpec(model="model-a"), ModelSpec(model="model-b")],
    )

    results = runner.results(comparison.comparison_id)
    assert results["pending"] == len(results["results"])
    assert all(r["status"] == "pending" and r["verdict"] is None for r in results["results"])

    await runner.run_to_completion(comparison)
    assert runner.results(comparison.comparison_id)["pending"] == 0


@pytest.mark.asyncio
async def test_comparison_index_is_reread_from_disk(settings):
    # The index lives in runs/<comparison_id>/comparison.json, so results survive a
    # restart: a fresh runner over the same output dir still finds the comparison.
    runner, manager = _runner(
        settings,
        client_factory=lambda: FakeAnthropicClient(_finishing_turns()),
        normalizer_factory=_no_op_normalizer,
    )
    comparison = await runner.start(
        "log in", None, None,
        [ModelSpec(model="model-a"), ModelSpec(model="model-b")],
    )
    await runner.run_to_completion(comparison)

    assert (settings.output_dir / comparison.comparison_id / "comparison.json").exists()
    reread = ComparisonRunner(manager).results(comparison.comparison_id)
    assert [r["model"] for r in reread["results"]] == ["model-a", "model-b"]
    assert reread["pending"] == 0


def test_unknown_comparison_is_404(settings):
    with TestClient(_app(settings, client_factory=lambda: None)) as client:
        assert client.get("/api/compare/doesnotexist").status_code == 404
        # A traversal attempt is rejected by the same id guard as run artifacts.
        assert client.get("/api/compare/%2e%2e").status_code == 404


def _rejection(response) -> str:
    """The reason a request was rejected, insisting it came from our own validation.

    FastAPI also answers 422 when it cannot resolve the endpoint's body model at all —
    it then treats the parameter as a missing *query* field and never parses the body.
    That failure mode once made a completely dead endpoint look validated, so assert
    the shape: only our HTTPException produces a plain-string detail.
    """
    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert isinstance(detail, str), f"request body was never parsed: {detail}"
    return detail


def test_compare_endpoint_validates_the_model_list(settings):
    with TestClient(_app(settings, client_factory=lambda: None)) as client:
        too_few = client.post("/api/compare", json={
            "instruction": "log in", "models": [{"model": "only-one"}],
        })
        assert "between 2 and 6 models" in _rejection(too_few)

        # Blank entries are dropped before counting, so three empty rows are "too few".
        blanks = client.post("/api/compare", json={
            "instruction": "log in",
            "models": [{"model": "a"}, {"model": "  "}, {"model": ""}],
        })
        assert "between 2 and 6 models" in _rejection(blanks)

        empty_instruction = client.post("/api/compare", json={
            "instruction": "  ", "models": [{"model": "a"}, {"model": "b"}],
        })
        assert "instruction must not be empty" in _rejection(empty_instruction)


@pytest.mark.asyncio
async def test_compare_endpoint_starts_runs_from_the_request_body(settings):
    # The happy path over real HTTP. Without it, only rejection paths were exercised,
    # and a broken endpoint that never reads its body still looked like it validated.
    app = _app(
        settings,
        client_factory=lambda: FakeAnthropicClient(_finishing_turns()),
        normalizer_factory=_no_op_normalizer,
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/compare", json={
            "instruction": "log in",
            "target_url": "https://example.test",
            "models": [{"model": "model-a"}, {"model": "model-b", "provider": "openai"}],
        })

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["comparison_id"].startswith("cmp_")
    assert [r["model"] for r in body["runs"]] == ["model-a", "model-b"]
    assert [r["provider"] for r in body["runs"]] == [settings.agent_provider, "openai"]

    for entry in body["runs"]:
        await app.state.manager.get(entry["run_id"]).task

    results = app.state.comparison_runner.results(body["comparison_id"])
    assert results["pending"] == 0
    assert [r["verdict"] for r in results["results"]] == ["pass", "pass"]


def test_config_endpoint_exposes_only_provider_and_model(settings):
    # The UI prefills from this; it must never leak keys or the whole settings object.
    with TestClient(_app(settings, client_factory=lambda: None)) as client:
        body = client.get("/api/config").json()
    assert body["model"] == settings.model
    assert body["agent_provider"] == settings.agent_provider
    assert set(body) == {"agent_provider", "model", "min_models", "max_models"}
