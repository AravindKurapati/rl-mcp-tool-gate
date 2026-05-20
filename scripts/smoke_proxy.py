"""Standalone smoke: connect to real upstream MCP servers, run the gate, print results.

Verifies the multiplex + gate path end-to-end WITHOUT needing a full MCP client.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.proxy.config import load_config
from src.proxy.multiplex import Multiplexer
from src.proxy.server import build_catalog_from_tools
from src.gate.encoder import GateEncoder
from src.gate.select import select_tools


async def main():
    cfg = load_config(Path("upstreams.toml"))
    mux = Multiplexer(cfg.upstreams)
    print("Connecting to upstreams...", flush=True)
    await mux.start()
    tools = mux.all_tools()
    print(f"Connected. Total tools across upstreams: {len(tools)}")
    for t in tools:
        print(f"  - {t.full_name}")

    catalog = build_catalog_from_tools(tools)
    ckpt = Path(cfg.gate_checkpoint) if cfg.gate_checkpoint else None
    enc = GateEncoder(lora_adapter_path=ckpt if (ckpt and ckpt.exists()) else None)
    enc.precompute_catalog(catalog)

    for query in [
        "read the file notes.txt",
        "remember that my favorite color is blue",
        "list the files in my home directory",
    ]:
        subset = select_tools(signal=query, catalog=catalog, encoder=enc, top_k=cfg.top_k)
        print(f"\nQUERY: {query}")
        print(f"  gated -> {[t['name'] for t in subset]}")

    await mux.stop()
    print("\nOK: multiplex + gate path works against live upstreams.")


if __name__ == "__main__":
    asyncio.run(main())
