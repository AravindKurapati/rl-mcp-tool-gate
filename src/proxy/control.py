"""HTTP control channel for hook -> proxy IPC.

Endpoints:
  POST /query  body={"text": "..."} -> sets state.query for the next tools/list call
  GET  /health -> 200 OK
"""
from __future__ import annotations
import uvicorn
from fastapi import FastAPI


def _build_app(state) -> FastAPI:
    app = FastAPI()

    @app.post("/query")
    async def set_query(payload: dict):
        text = (payload or {}).get("text", "").strip()
        if text:
            state.query = text
        return {"ok": True}

    @app.get("/health")
    async def health():
        return {"ok": True}

    return app


async def start_control_server(state, port: int) -> None:
    app = _build_app(state)
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    await server.serve()
