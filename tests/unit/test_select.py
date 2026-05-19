import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.gate.encoder import GateEncoder
from src.gate.select import select_tools


def test_select_tools_returns_subset():
    catalog = json.loads(Path("data/synthetic/catalog.json").read_text(encoding="utf-8"))["tools"]
    enc = GateEncoder()
    enc.precompute_catalog(catalog)
    chosen = select_tools(
        signal="post a message to slack",
        catalog=catalog,
        encoder=enc,
        top_k=5,
    )
    assert 1 <= len(chosen) <= 5
    chosen_names = {t["name"] for t in chosen}
    assert "slack.post_message" in chosen_names


def test_select_tools_deterministic():
    catalog = json.loads(Path("data/synthetic/catalog.json").read_text(encoding="utf-8"))["tools"]
    enc = GateEncoder()
    enc.precompute_catalog(catalog)
    a = select_tools(signal="list files in /tmp", catalog=catalog, encoder=enc, top_k=3)
    b = select_tools(signal="list files in /tmp", catalog=catalog, encoder=enc, top_k=3)
    assert [t["name"] for t in a] == [t["name"] for t in b]
