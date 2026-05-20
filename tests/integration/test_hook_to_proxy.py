import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import httpx
import pytest


@pytest.mark.asyncio
async def test_set_query_updates_state():
    from src.proxy.control import start_control_server

    class FakeState:
        def __init__(self):
            self.query = None
            self.history = []

    state = FakeState()
    task = asyncio.create_task(start_control_server(state, 17801))
    await asyncio.sleep(0.5)
    try:
        async with httpx.AsyncClient() as client:
            r = await client.post("http://127.0.0.1:17801/query", json={"text": "list files in /tmp"})
            assert r.status_code == 200
        assert state.query == "list files in /tmp"
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
