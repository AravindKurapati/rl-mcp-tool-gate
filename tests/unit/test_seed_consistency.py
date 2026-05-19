import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def test_all_seed_tools_exist_in_catalog():
    cat = json.loads(Path("data/synthetic/catalog.json").read_text(encoding="utf-8"))
    tool_names = {t["name"] for t in cat["tools"]}
    seeds = [json.loads(l) for l in Path("data/synthetic/seeds.jsonl").read_text(encoding="utf-8").splitlines()]
    missing = []
    for s in seeds:
        for gt in s["ground_truth"]:
            if gt not in tool_names:
                missing.append((gt, s["query"]))
    assert not missing, f"Missing tools: {missing}"
