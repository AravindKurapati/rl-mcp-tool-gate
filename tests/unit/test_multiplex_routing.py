"""Tool-call routing for the live MCP proxy (Multiplexer.start + Multiplexer.call).

The proxy advertises every upstream tool as "server.tool", but some MCP clients
sanitize "." to "_" before issuing the call (see commit "Route tool calls via name
index tolerant of '.' -> '_' sanitization"). Multiplexer therefore indexes BOTH the
dotted and the sanitized form of each name, returns the upstream's ORIGINAL tool name
when it dispatches, and falls back to a first-"." split for names that are not in the
index. Before this file only NamespacedTool (the dataclass) was covered; the routing
that actually dispatches a call to the right upstream — the proxy's production hot
path — was not.

These tests drive the real Multiplexer.start() so the index-building code runs, but
swap UpstreamProxy.connect() for a fake that fabricates a toolset instead of spawning
a stdio subprocess. CPU-only, no network, no model download.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.proxy.config import Upstream
from src.proxy.multiplex import Multiplexer, NamespacedTool, UpstreamProxy


class FakeSession:
    """Stand-in for an mcp ClientSession; records the (tool, args) it was asked to call."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, tool, arguments):
        self.calls.append((tool, arguments))
        return {"tool": tool, "arguments": arguments}


class FakeProxy(UpstreamProxy):
    """An UpstreamProxy whose connect() fakes a stdio MCP server with a fixed toolset,
    so Multiplexer.start() builds its real index without spawning a subprocess."""

    def __init__(self, name: str, tool_names: list[str]):
        super().__init__(Upstream(name=name, command=["unused"], env={}))
        self._tool_names = list(tool_names)

    async def connect(self) -> None:
        self.session = FakeSession()
        self.tools = [
            NamespacedTool(server=self.spec.name, tool=t, description=f"{t} desc", input_schema={})
            for t in self._tool_names
        ]


def _mux(*proxies: FakeProxy) -> Multiplexer:
    mux = Multiplexer([])
    mux.upstreams = list(proxies)
    return mux


@pytest.mark.asyncio
async def test_routes_dotted_name_to_correct_upstream():
    gh = FakeProxy("github", ["list_issues", "get_issue"])
    fs = FakeProxy("fs", ["read_file"])
    mux = _mux(gh, fs)
    async with mux:
        await mux.call("github.get_issue", {"id": 1})
        await mux.call("fs.read_file", {"path": "/x"})
    # Each call lands on its owning upstream under the upstream's ORIGINAL tool name.
    assert gh.session.calls == [("get_issue", {"id": 1})]
    assert fs.session.calls == [("read_file", {"path": "/x"})]


@pytest.mark.asyncio
async def test_routes_sanitized_underscore_name():
    # Client replaced "." with "_": "github_list_issues" must still reach github.list_issues
    # and be dispatched under the un-sanitized upstream tool name "list_issues".
    gh = FakeProxy("github", ["list_issues"])
    mux = _mux(gh)
    async with mux:
        await mux.call("github_list_issues", {"state": "open"})
    assert gh.session.calls == [("list_issues", {"state": "open"})]


@pytest.mark.asyncio
async def test_routes_sanitized_name_for_underscored_server():
    # Trickier real case: the server name itself contains "_". The dotted form is
    # "aws_s3.put_object"; its sanitized form "aws_s3_put_object" must still resolve to
    # server "aws_s3", tool "put_object".
    aws = FakeProxy("aws_s3", ["put_object"])
    mux = _mux(aws)
    async with mux:
        await mux.call("aws_s3_put_object", {"key": "k"})
    assert aws.session.calls == [("put_object", {"key": "k"})]


@pytest.mark.asyncio
async def test_index_has_both_dotted_and_sanitized_keys():
    gh = FakeProxy("github", ["list_issues"])
    mux = _mux(gh)
    async with mux:
        assert "github.list_issues" in mux._tool_index
        assert "github_list_issues" in mux._tool_index


@pytest.mark.asyncio
async def test_fallback_split_when_name_not_indexed():
    # A name that was never advertised (so it is absent from the index) but whose prefix
    # is a known server falls back to a first-"." split and dispatches the remainder.
    aws = FakeProxy("aws", ["describe"])
    mux = _mux(aws)
    async with mux:
        await mux.call("aws.run_instances", {"n": 1})
    assert aws.session.calls == [("run_instances", {"n": 1})]


@pytest.mark.asyncio
async def test_fallback_split_preserves_dots_in_tool_remainder():
    # The fallback splits on the FIRST "." only, so a tool name that itself contains a
    # "." is dispatched whole to the upstream.
    srv = FakeProxy("srv", ["seed"])  # 'seed' just to have a connected upstream
    mux = _mux(srv)
    async with mux:
        await mux.call("srv.s3.put_object", {})
    assert srv.session.calls == [("s3.put_object", {})]


@pytest.mark.asyncio
async def test_unknown_server_raises_value_error():
    gh = FakeProxy("github", ["list_issues"])
    mux = _mux(gh)
    with pytest.raises(ValueError):
        async with mux:
            await mux.call("nope.whatever", {})


@pytest.mark.asyncio
async def test_all_tools_aggregates_across_upstreams():
    gh = FakeProxy("github", ["list_issues", "get_issue"])
    fs = FakeProxy("fs", ["read_file"])
    mux = _mux(gh, fs)
    async with mux:
        names = sorted(t.full_name for t in mux.all_tools())
    assert names == ["fs.read_file", "github.get_issue", "github.list_issues"]
