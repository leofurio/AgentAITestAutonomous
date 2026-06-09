"""Async event bus: the agent loop publishes events, the WebSocket handler drains them."""

from __future__ import annotations

import asyncio
from typing import Any

# Sentinel pushed onto the queue to signal that no more events will arrive.
_SENTINEL = object()


class EventBus:
    """A single-run, multi-producer/single-consumer async event stream.

    The agent loop calls :meth:`publish` for each event and :meth:`close` when done.
    The WebSocket handler iterates with ``async for event in bus``.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        self._closed = False

    def publish(self, event_type: str, **data: Any) -> None:
        if self._closed:
            return
        self._queue.put_nowait({"type": event_type, "data": data})

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._queue.put_nowait(_SENTINEL)

    def __aiter__(self) -> EventBus:
        return self

    async def __anext__(self) -> dict[str, Any]:
        item = await self._queue.get()
        if item is _SENTINEL:
            raise StopAsyncIteration
        return item
