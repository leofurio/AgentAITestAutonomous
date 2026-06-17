"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ..config import Settings, load_settings, normalizer_settings
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
) -> FastAPI:
    settings = settings or load_settings()
    n_settings = normalizer_settings(settings)
    app = FastAPI(title="aiwebtest", version="0.1.0")
    app.state.settings = settings
    app.state.manager = RunManager(
        settings,
        client_factory or _default_client_factory(settings),
        normalizer_factory=normalizer_factory or (lambda: build_client(n_settings)),
        normalizer_settings=n_settings,
    )

    app.include_router(api_router)
    app.include_router(ws_router)

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html")

    @app.get("/runner")
    async def runner() -> FileResponse:
        return FileResponse(_STATIC_DIR / "runner.html")

    if _STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    return app
