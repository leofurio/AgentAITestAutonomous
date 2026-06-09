"""Asyncio compatibility helpers for Windows subprocess users such as Playwright."""

from __future__ import annotations

import asyncio
import sys
import warnings


def configure_windows_event_loop_policy() -> None:
    """Use a Windows event loop policy that supports asyncio subprocesses."""
    if sys.platform != "win32":
        return

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        proactor_policy = getattr(asyncio, "WindowsProactorEventLoopPolicy", None)
        if proactor_policy is None:
            return
        if not isinstance(asyncio.get_event_loop_policy(), proactor_policy):
            asyncio.set_event_loop_policy(proactor_policy())


def ensure_subprocess_event_loop() -> None:
    """Fail early with a useful message when the active Windows loop cannot spawn processes."""
    if sys.platform != "win32":
        return

    loop = asyncio.get_running_loop()
    if "Selector" not in type(loop).__name__:
        return

    raise RuntimeError(
        "Playwright requires a Windows Proactor event loop because it starts a subprocess. "
        "Start aiwebtest through `aiwebtest` or without uvicorn reload/workers so the "
        "application can install WindowsProactorEventLoopPolicy before the loop is created."
    )
