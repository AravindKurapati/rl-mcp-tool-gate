"""Build v2 training data: unify the synthetic catalog with real-format MCP tools
and add realistic real-tool queries so the gate stops being OOD on real traffic.

Outputs (all local, no Modal):
  data/synthetic/catalog_v2.json   synthetic 99 + real-format MCP tools
  data/synthetic/train_v2.jsonl    v1 train + real seeds (+ light local augmentation)
  data/synthetic/heldout_v2.jsonl  v1 heldout + held-out real-tool queries
"""
from __future__ import annotations
import json
from pathlib import Path

from src.data_gen.real_tools import real_tool_catalog, REAL_SEEDS, REAL_HELDOUT

DATA = Path("data/synthetic")

# Light, deterministic paraphrase prefixes to expand the real-tool training signal
# without an LLM. ground_truth is unchanged.
_PREFIXES = ["", "can you ", "please ", "i need to ", "help me "]


def _augment(seed: dict) -> list[dict]:
    out = []
    q = seed["query"]
    for p in _PREFIXES:
        text = (p + q) if p else q
        out.append({
            "query": text,
            "ground_truth": seed["ground_truth"],
            "min_k": seed["min_k"],
            "category": "real_mcp",
            "source": "real_seed" if p == "" else "real_aug",
        })
    return out


def build():
    catalog = json.loads((DATA / "catalog.json").read_text(encoding="utf-8"))["tools"]
    existing = {t["name"] for t in catalog}
    for t in real_tool_catalog():
        if t["name"] not in existing:
            catalog.append(t)
    (DATA / "catalog_v2.json").write_text(
        json.dumps({"tools": catalog, "count": len(catalog)}, indent=2), encoding="utf-8"
    )

    train = [json.loads(l) for l in (DATA / "train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    for seed in REAL_SEEDS:
        train.extend(_augment(seed))
    with (DATA / "train_v2.jsonl").open("w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r) + "\n")

    heldout = [json.loads(l) for l in (DATA / "heldout.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    heldout.extend(REAL_HELDOUT)
    with (DATA / "heldout_v2.jsonl").open("w", encoding="utf-8") as f:
        for r in heldout:
            f.write(json.dumps(r) + "\n")

    print(f"catalog_v2: {len(catalog)} tools (+{len(real_tool_catalog())} real-format)")
    print(f"train_v2:   {len(train)} queries (+{len(REAL_SEEDS)} real seeds x{len(_PREFIXES)} aug)")
    print(f"heldout_v2: {len(heldout)} queries (+{len(REAL_HELDOUT)} real held-out)")


if __name__ == "__main__":
    build()
