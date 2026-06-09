"""Uvicorn entrypoint. ``app`` is what ``uvicorn aiwebtest.main:app`` loads."""

from __future__ import annotations

from .web.app import create_app

app = create_app()


def run() -> None:
    """Console-script entrypoint: ``aiwebtest``."""
    import uvicorn

    uvicorn.run("aiwebtest.main:app", host="127.0.0.1", port=8000, reload=False)
