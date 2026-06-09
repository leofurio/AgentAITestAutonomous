"""Uvicorn entrypoint. ``app`` is what ``uvicorn aiwebtest.main:app`` loads."""

from __future__ import annotations

from .asyncio_compat import configure_windows_event_loop_policy
from .web.app import create_app

configure_windows_event_loop_policy()

app = create_app()


def run() -> None:
    """Console-script entrypoint: ``aiwebtest``."""
    import uvicorn

    configure_windows_event_loop_policy()
    uvicorn.run("aiwebtest.main:app", host="127.0.0.1", port=8000, reload=False)
