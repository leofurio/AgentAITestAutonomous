"""WebSocket endpoint: stream a run's events to the browser as JSON."""

from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()


@router.websocket("/ws/runs/{run_id}")
async def run_events(websocket: WebSocket, run_id: str) -> None:
    await websocket.accept()
    manager = websocket.app.state.manager
    run = manager.get(run_id)
    if run is None:
        await websocket.send_json({"type": "error", "data": {"message": "unknown run_id"}})
        await websocket.close()
        return

    try:
        async for event in run.bus:
            await websocket.send_json(event)
            # The 'report' and 'error' events are terminal for the UI.
            if event["type"] in ("report", "error"):
                # keep draining until the bus closes, but nothing more is expected
                pass
    except WebSocketDisconnect:
        return
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
