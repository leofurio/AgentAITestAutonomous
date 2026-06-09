"""BrowserSession: owns the Playwright lifecycle for one test run."""

from __future__ import annotations

from pathlib import Path

from playwright.async_api import Browser, BrowserContext, Page, async_playwright

from ..config import BrowserConfig


class BrowserSession:
    """Async context manager wrapping a Playwright browser/context/page.

    Usage::

        async with BrowserSession(cfg, screenshot_dir) as session:
            await session.page.goto(url)
    """

    def __init__(self, cfg: BrowserConfig, screenshot_dir: Path) -> None:
        self._cfg = cfg
        self._screenshot_dir = screenshot_dir
        self._pw = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self.page: Page | None = None

    async def __aenter__(self) -> BrowserSession:
        self._pw = await async_playwright().start()
        launch_kwargs: dict = {"headless": self._cfg.headless}
        if self._cfg.channel:
            launch_kwargs["channel"] = self._cfg.channel
        try:
            self._browser = await self._pw.chromium.launch(**launch_kwargs)
        except Exception:
            # Fall back to the bundled Chromium if the requested channel is absent.
            launch_kwargs.pop("channel", None)
            self._browser = await self._pw.chromium.launch(**launch_kwargs)

        self._context = await self._browser.new_context(
            viewport={"width": self._cfg.viewport.width, "height": self._cfg.viewport.height}
        )
        self._context.set_default_timeout(self._cfg.action_timeout_ms)
        self._context.set_default_navigation_timeout(self._cfg.nav_timeout_ms)
        self.page = await self._context.new_page()
        self._screenshot_dir.mkdir(parents=True, exist_ok=True)
        return self

    async def __aexit__(self, *exc) -> None:
        # Best-effort teardown; never raise out of cleanup.
        for closer in (self._context, self._browser):
            try:
                if closer is not None:
                    await closer.close()
            except Exception:
                pass
        try:
            if self._pw is not None:
                await self._pw.stop()
        except Exception:
            pass
