import json
from pathlib import Path
import sys
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def test_heldout_count_and_distribution():
    held = [json.loads(l) for l in Path("data/synthetic/heldout.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(held) == 100
    by_cat = Counter(h["category"] for h in held)
    assert len(by_cat) == 10, f"Expected 10 categories, got {list(by_cat)}"
    for cat, n in by_cat.items():
        assert n == 10, f"Category {cat} has {n} entries, expected 10"


def test_heldout_tools_in_catalog():
    cat = json.loads(Path("data/synthetic/catalog.json").read_text(encoding="utf-8"))
    tool_names = {t["name"] for t in cat["tools"]}
    held = [json.loads(l) for l in Path("data/synthetic/heldout.jsonl").read_text(encoding="utf-8").splitlines()]
    missing = []
    for h in held:
        for gt in h["ground_truth"]:
            if gt not in tool_names:
                missing.append((gt, h["query"]))
    assert not missing, f"Missing tools: {missing}"
