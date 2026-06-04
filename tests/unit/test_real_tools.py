"""Real-format tool definitions and the v2 data build (src/data_gen).

The v2 retrain's whole purpose was to stop the gate being OOD on real traffic by
teaching it the real-format mcp__* tool names (see docs: v1 overfit synthetic names).
That only works if the real seeds/heldout are internally consistent with the real
catalog — every ground_truth name must be a tool the catalog actually contains, or
eval recall is silently computed against tools that can never be retrieved. None of
this was covered. All CPU-only / pure-Python: static data + file IO into a tmp dir.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import src.data_gen.build_v2 as build_v2
from src.data_gen.real_tools import real_tool_catalog, REAL_SEEDS, REAL_HELDOUT


# --- real_tools static-data invariants -------------------------------------------------

def test_catalog_names_unique():
    names = [t["name"] for t in real_tool_catalog()]
    assert len(names) == len(set(names))


def test_catalog_entries_well_formed():
    for t in real_tool_catalog():
        # mcp__<server>__<tool> -> server is the 2nd "__" field, tool the last.
        assert t["name"].startswith("mcp__")
        parts = t["name"].split("__")
        assert t["server"] == parts[1]
        assert t["tool"] == parts[-1]
        # embed_text is what the encoder actually sees; must carry name + description.
        assert t["embed_text"] == f"{t['name']}: {t['description']}"
        assert t["description"]


def test_seed_ground_truth_tools_exist_in_catalog():
    catalog_names = {t["name"] for t in real_tool_catalog()}
    for seed in REAL_SEEDS:
        for gt in seed["ground_truth"]:
            assert gt in catalog_names, f"seed gt {gt!r} not in real catalog"


def test_heldout_ground_truth_tools_exist_in_catalog():
    catalog_names = {t["name"] for t in real_tool_catalog()}
    for row in REAL_HELDOUT:
        for gt in row["ground_truth"]:
            assert gt in catalog_names, f"heldout gt {gt!r} not in real catalog"


def test_min_k_matches_ground_truth_size():
    # min_k is the number of tools a query genuinely needs; keep it in sync with gt so
    # downstream budget/recall reasoning is honest.
    for row in REAL_SEEDS + REAL_HELDOUT:
        assert row["min_k"] == len(row["ground_truth"])


def test_heldout_rows_tagged_real_mcp():
    for row in REAL_HELDOUT:
        assert row["category"] == "real_mcp"


# --- build_v2 pipeline -----------------------------------------------------------------

def _seed_inputs(data_dir: Path) -> None:
    """Minimal catalog/train/heldout inputs. The synthetic catalog deliberately includes
    one entry whose name collides with a real tool, to exercise the merge's dedup."""
    collide = real_tool_catalog()[0]["name"]
    catalog = {"tools": [
        {"name": "synthetic.only", "server": "synthetic", "tool": "only",
         "description": "d", "args": "", "embed_text": "synthetic.only: d"},
        {"name": collide, "server": "x", "tool": "y",
         "description": "preexisting", "args": "", "embed_text": "pre"},
    ]}
    (data_dir / "catalog.json").write_text(json.dumps(catalog), encoding="utf-8")
    (data_dir / "train.jsonl").write_text(
        json.dumps({"query": "synth q", "ground_truth": ["synthetic.only"], "min_k": 1}) + "\n",
        encoding="utf-8",
    )
    (data_dir / "heldout.jsonl").write_text(
        json.dumps({"query": "synth heldout", "ground_truth": ["synthetic.only"], "min_k": 1}) + "\n",
        encoding="utf-8",
    )


def test_build_v2_merges_and_dedups_catalog(tmp_path, monkeypatch):
    monkeypatch.setattr(build_v2, "DATA", tmp_path)
    _seed_inputs(tmp_path)
    build_v2.build()

    out = json.loads((tmp_path / "catalog_v2.json").read_text(encoding="utf-8"))
    names = [t["name"] for t in out["tools"]]
    # No duplicate names after merge, and the declared count matches.
    assert len(names) == len(set(names))
    assert out["count"] == len(out["tools"])
    # 2 synthetic (one of which collides with a real name) + the non-colliding real tools.
    assert len(out["tools"]) == 2 + (len(real_tool_catalog()) - 1)
    # The pre-existing colliding entry is kept (dedup skips the real one), not overwritten.
    collide = real_tool_catalog()[0]["name"]
    kept = next(t for t in out["tools"] if t["name"] == collide)
    assert kept["description"] == "preexisting"


def test_build_v2_train_expands_seeds_by_prefixes(tmp_path, monkeypatch):
    monkeypatch.setattr(build_v2, "DATA", tmp_path)
    _seed_inputs(tmp_path)
    build_v2.build()

    rows = [json.loads(l) for l in (tmp_path / "train_v2.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    # 1 original synthetic train row + every REAL_SEED expanded by each prefix.
    assert len(rows) == 1 + len(REAL_SEEDS) * len(build_v2._PREFIXES)
    aug = [r for r in rows if r.get("category") == "real_mcp"]
    assert len(aug) == len(REAL_SEEDS) * len(build_v2._PREFIXES)
    # The empty-prefix variant is the verbatim seed, tagged real_seed; others real_aug.
    seeds_kept = [r for r in aug if r["source"] == "real_seed"]
    assert {r["query"] for r in seeds_kept} == {s["query"] for s in REAL_SEEDS}
    assert all(r["source"] == "real_aug" for r in aug if r not in seeds_kept)


def test_build_v2_heldout_appends_real_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(build_v2, "DATA", tmp_path)
    _seed_inputs(tmp_path)
    build_v2.build()

    rows = [json.loads(l) for l in (tmp_path / "heldout_v2.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) == 1 + len(REAL_HELDOUT)
    # Every real heldout query is present and the original synthetic one is preserved.
    queries = {r["query"] for r in rows}
    assert "synth heldout" in queries
    assert {r["query"] for r in REAL_HELDOUT} <= queries


def test_build_v2_preserves_ground_truth_under_augmentation(tmp_path, monkeypatch):
    monkeypatch.setattr(build_v2, "DATA", tmp_path)
    _seed_inputs(tmp_path)
    build_v2.build()

    rows = [json.loads(l) for l in (tmp_path / "train_v2.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    by_seed = {s["query"]: s for s in REAL_SEEDS}
    for r in rows:
        if r.get("source") == "real_seed":
            assert r["ground_truth"] == by_seed[r["query"]]["ground_truth"]
            assert r["min_k"] == by_seed[r["query"]]["min_k"]
