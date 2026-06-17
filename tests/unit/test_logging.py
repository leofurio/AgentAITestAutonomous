"""Logging setup and the debug traces emitted by the normalizer."""

from __future__ import annotations

import logging

import pytest

from aiwebtest.agent.normalizer import InstructionNormalizer
from aiwebtest.config import Settings
from aiwebtest.logging_config import configure_logging, get_logger


class _FakeClient:
    def __init__(self, text: str) -> None:
        self.text = text

    async def complete(self, messages, tools, system_prompt):
        return {"role": "assistant", "content": [{"type": "text", "text": self.text}]}


def test_configure_logging_sets_level_and_single_handler():
    logger = configure_logging("DEBUG")
    assert logger.level == logging.DEBUG
    before = len(logger.handlers)
    # Idempotent: a second call must not stack another handler.
    configure_logging("INFO")
    assert len(logger.handlers) == before
    assert logger.level == logging.INFO


def test_get_logger_is_child_of_package_logger():
    assert get_logger("normalizer").name == "aiwebtest.normalizer"


@pytest.mark.asyncio
async def test_normalizer_emits_debug_trace(caplog):
    client = _FakeClient('{"objective":"x","steps":["go"]}')
    normalizer = InstructionNormalizer(client, Settings(model="m", max_tokens=256))

    with caplog.at_level(logging.DEBUG, logger="aiwebtest.normalizer"):
        out = await normalizer.normalize("do x")

    assert out == '{"objective":"x","steps":["go"],"expected_results":[]}'
    messages = "\n".join(r.getMessage() for r in caplog.records)
    assert "normalize request" in messages
    assert "canonical JSON" in messages
