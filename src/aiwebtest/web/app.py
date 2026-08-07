"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ..compare import ComparisonRunner
from ..config import Settings, load_settings, normalizer_settings
from ..logging_config import configure_logging
from ..suite import SuiteRunner, SuiteStore
from .manager import RunManager
from .routes import router as api_router
from .ws import router as ws_router

_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def build_client(settings: Settings) -> Any:
    """Build the agent client for ``settings.agent_provider`` / ``settings.model``."""
    if settings.agent_provider == "openai":
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "OpenAI provider selected but the openai package is not installed. "
                'Install with: pip install -e ".[openai]"'
            ) from exc

        from ..agent.providers import OpenAIAgentClient

        kwargs = {}
        if settings.openai_api_key:
            kwargs["api_key"] = settings.openai_api_key
        return OpenAIAgentClient(AsyncOpenAI(**kwargs), settings)

    if settings.agent_provider == "openrouter":
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "OpenRouter provider selected but the openai package is not installed. "
                'Install with: pip install -e ".[openai]"'
            ) from exc

        from ..agent.providers import OpenRouterAgentClient

        default_headers = {}
        if settings.openrouter_http_referer:
            default_headers["HTTP-Referer"] = settings.openrouter_http_referer
        if settings.openrouter_app_title:
            default_headers["X-OpenRouter-Title"] = settings.openrouter_app_title

        kwargs = {"base_url": settings.openrouter_base_url}
        if settings.openrouter_api_key:
            kwargs["api_key"] = settings.openrouter_api_key
        if default_headers:
            kwargs["default_headers"] = default_headers
        return OpenRouterAgentClient(AsyncOpenAI(**kwargs), settings)

    if settings.agent_provider == "anthropic":
        from anthropic import AsyncAnthropic

        kwargs = {}
        if settings.anthropic_api_key:
            kwargs["api_key"] = settings.anthropic_api_key
        return AsyncAnthropic(**kwargs)

    raise ValueError(f"Unsupported agent provider: {settings.agent_provider}")


def _default_client_factory(settings: Settings) -> Callable[[], Any]:
    return lambda: build_client(settings)


def create_app(
    settings: Settings | None = None,
    client_factory: Callable[[], Any] | None = None,
    normalizer_factory: Callable[[], Any] | None = None,
    client_factory_for: Callable[[Settings], Any] | None = None,
) -> FastAPI:
    settings = settings or load_settings()
    configure_logging(settings.log_level)
    n_settings = normalizer_settings(settings)
    # A model comparison runs each contender on derived settings, so it needs a factory
    # that takes them. A caller supplying its own client_factory (tests, custom wiring)
    # gets that client for every model unless it says otherwise.
    if client_factory_for is None:
        client_factory_for = (
            (lambda _settings: client_factory()) if client_factory else build_client
        )
    app = FastAPI(title="aiwebtest", version="0.1.0")
    app.state.settings = settings
    app.state.manager = RunManager(
        settings,
        client_factory or _default_client_factory(settings),
        normalizer_factory=normalizer_factory or (lambda: build_client(n_settings)),
        normalizer_settings=n_settings,
        client_factory_for=client_factory_for,
    )
    app.state.suite_store = SuiteStore(settings.output_dir / "suite.json")
    app.state.suite_runner = SuiteRunner(app.state.manager, app.state.suite_store)
    app.state.comparison_runner = ComparisonRunner(app.state.manager)

    app.include_router(api_router)
    app.include_router(ws_router)

    @app.middleware("http")
    async def _revalidate_frontend(request, call_next):
        # The UI (HTML + bundled JS/CSS) ships with the app and changes on every update.
        # Force revalidation so a browser never runs a stale runner.js/app.js against a
        # newer server — otherwise features (e.g. the runner's upload) silently do nothing
        # until a hard refresh. "no-cache" still allows 304s, so it stays cheap.
        response = await call_next(request)
        path = request.url.path
        if path in ("/", "/runner") or path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html")

    @app.get("/runner")
    async def runner() -> FileResponse:
        return FileResponse(_STATIC_DIR / "runner.html")

    if _STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    return app
