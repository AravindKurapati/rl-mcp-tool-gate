"""Multiplex multiple upstream MCP servers via stdio."""
from __future__ import annotations
import os
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from src.proxy.config import Upstream


@dataclass
class NamespacedTool:
    server: str
    tool: str
    description: str
    input_schema: dict[str, Any]

    @property
    def full_name(self) -> str:
        return f"{self.server}.{self.tool}"

    @staticmethod
    def split(full: str) -> tuple[str, str]:
        srv, _, tool = full.partition(".")
        return srv, tool


class UpstreamProxy:
    def __init__(self, spec: Upstream):
        self.spec = spec
        self.session: ClientSession | None = None
        self.tools: list[NamespacedTool] = []
        self._stack = AsyncExitStack()

    async def connect(self) -> None:
        env = {**os.environ, **self.spec.env}
        params = StdioServerParameters(
            command=self.spec.command[0],
            args=list(self.spec.command[1:]),
            env=env,
        )
        read, write = await self._stack.enter_async_context(stdio_client(params))
        self.session = await self._stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()
        resp = await self.session.list_tools()
        self.tools = [
            NamespacedTool(
                server=self.spec.name,
                tool=t.name,
                description=t.description or "",
                input_schema=t.inputSchema or {},
            )
            for t in resp.tools
        ]

    async def call(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        if self.session is None:
            raise RuntimeError(f"upstream {self.spec.name} not connected")
        return await self.session.call_tool(tool_name, arguments)

    async def close(self) -> None:
        await self._stack.aclose()


class Multiplexer:
    def __init__(self, upstreams: list[Upstream]):
        self.upstreams = [UpstreamProxy(u) for u in upstreams]
        self._by_name: dict[str, UpstreamProxy] = {}
        # Maps an advertised tool name (dotted "server.tool" AND its sanitized
        # "server_tool" form, since some clients replace "." in tool names) to the
        # owning upstream and the upstream's original tool name.
        self._tool_index: dict[str, tuple[UpstreamProxy, str]] = {}
        self._stack = AsyncExitStack()

    async def start(self) -> None:
        # Own each upstream's lifecycle in a single shared stack entered in THIS task,
        # so all anyio cancel scopes are exited LIFO in the same task on teardown.
        for u in self.upstreams:
            await self._stack.enter_async_context(_managed(u))
            self._by_name[u.spec.name] = u
            for t in u.tools:
                self._tool_index[t.full_name] = (u, t.tool)
                self._tool_index[t.full_name.replace(".", "_")] = (u, t.tool)

    def all_tools(self) -> list[NamespacedTool]:
        return [t for u in self.upstreams for t in u.tools]

    async def call(self, full_name: str, arguments: dict[str, Any]) -> Any:
        entry = self._tool_index.get(full_name)
        if entry is None:
            # Fall back to prefix split (handles tool names that themselves contain ".").
            server, tool = NamespacedTool.split(full_name)
            up = self._by_name.get(server)
            if up is None:
                raise ValueError(f"unknown tool: {full_name}")
            return await up.call(tool, arguments)
        up, tool = entry
        return await up.call(tool, arguments)

    async def stop(self) -> None:
        await self._stack.aclose()

    async def __aenter__(self) -> "Multiplexer":
        await self.start()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.stop()


from contextlib import asynccontextmanager


@asynccontextmanager
async def _managed(up: "UpstreamProxy"):
    await up.connect()
    try:
        yield up
    finally:
        await up.close()
