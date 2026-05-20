"""MCP server that proxies multiple upstreams and applies the tool gate.

B1 signal: accumulated tool-call history (proxy can't see user queries directly).
B2 signal: user query, fed via control_channel.

Re-emits notifications/tools/list_changed after each tools/call to encourage clients
to re-list with a fresh gate decision.
"""
from __future__ import annotations
import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
import mcp.types as mt

from src.proxy.config import load_config
from src.proxy.multiplex import Multiplexer, NamespacedTool
from src.gate.encoder import GateEncoder
from src.gate.select import select_tools


class GateState:
    def __init__(self):
        self.history: list[str] = []
        self.query: str | None = None

    def signal(self) -> str:
        if self.query:
            return self.query
        if self.history:
            return " ".join(self.history)
        return "general purpose agent task"


def build_catalog_from_tools(tools: list[NamespacedTool]) -> list[dict[str, Any]]:
    catalog = []
    for t in tools:
        embed_text = f"{t.full_name}: {t.description}"
        catalog.append({
            "name": t.full_name,
            "server": t.server,
            "tool": t.tool,
            "description": t.description,
            "embed_text": embed_text,
            "input_schema": t.input_schema,
        })
    return catalog


async def build_server(config_path: Path):
    cfg = load_config(config_path)
    mux = Multiplexer(cfg.upstreams)
    await mux.start()
    print(f"Connected {len(cfg.upstreams)} upstreams, {len(mux.all_tools())} total tools", file=sys.stderr, flush=True)

    ckpt_path = Path(cfg.gate_checkpoint) if cfg.gate_checkpoint else None
    encoder = GateEncoder(lora_adapter_path=ckpt_path if (ckpt_path and ckpt_path.exists()) else None)
    catalog = build_catalog_from_tools(mux.all_tools())
    encoder.precompute_catalog(catalog)

    state = GateState()
    server = Server("rl-mcp-tool-gate")

    @server.list_tools()
    async def list_tools() -> list[mt.Tool]:
        subset = select_tools(
            signal=state.signal(),
            catalog=catalog,
            encoder=encoder,
            top_k=cfg.top_k,
            budget_tokens=cfg.budget_tokens,
        )
        return [
            mt.Tool(name=t["name"], description=t["description"], inputSchema=t["input_schema"])
            for t in subset
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[mt.TextContent]:
        state.history.append(name)
        state.query = None  # query is a one-turn signal
        result = await mux.call(name, arguments)
        out_text = ""
        if hasattr(result, "content"):
            for c in result.content:
                if hasattr(c, "text"):
                    out_text += c.text
        else:
            out_text = json.dumps(result, default=str)
        try:
            await server.request_context.session.send_notification(
                mt.ServerNotification(
                    root=mt.ToolListChangedNotification(method="notifications/tools/list_changed")
                )
            )
        except Exception:
            pass
        return [mt.TextContent(type="text", text=out_text)]

    return server, state, mux, cfg


async def main_async(config_path: Path):
    server, state, mux, cfg = await build_server(config_path)
    from src.proxy.control import start_control_server
    control_task = asyncio.create_task(start_control_server(state, cfg.control_channel_port))
    try:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())
    finally:
        control_task.cancel()
        await mux.stop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("upstreams.toml"))
    args = ap.parse_args()
    asyncio.run(main_async(args.config))


if __name__ == "__main__":
    main()
