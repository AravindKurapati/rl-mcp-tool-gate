import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def test_train_tools_in_catalog():
    cat = json.loads(Path("data/synthetic/catalog.json").read_text(encoding="utf-8"))
    tool_names = {t["name"] for t in cat["tools"]}
    train = [json.loads(l) for l in Path("data/synthetic/train.jsonl").read_text(encoding="utf-8").splitlines()]
    for t in train:
        for gt in t["ground_truth"]:
            assert gt in tool_names, f"Train references missing tool: {gt}"
    assert len(train) >= 300, f"Only {len(train)} train queries, want >=300"
