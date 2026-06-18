"""Logging setup for aiwebtest.

A single package logger (``aiwebtest``) with a console handler. Set the level to ``DEBUG``
(via ``AIWEBTEST_LOG_LEVEL`` or ``Settings.log_level``) to trace the normalization rewrite
and every agent (LLM) call. At DEBUG, request/response payloads — which can include data
typed into the page — are written to the console, so enable it only for local debugging.
"""

from __future__ import annotations

import logging

LOGGER_NAME = "aiwebtest"


def configure_logging(level: str | int = "INFO") -> logging.Logger:
    """Configure and return the package logger. Idempotent (won't add duplicate handlers)."""
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    first_setup = not logger.handlers
    if first_setup:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logger.addHandler(handler)
    # Don't double-emit through the root logger's handlers.
    logger.propagate = False
    if first_setup:
        # A visible confirmation of the effective level so it's easy to tell whether
        # DEBUG actually took effect (e.g. the env var was set with the right syntax).
        logger.info("aiwebtest logging configured at level %s", logging.getLevelName(level))
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child of the package logger, e.g. get_logger("normalizer")."""
    return logging.getLogger(f"{LOGGER_NAME}.{name}")
