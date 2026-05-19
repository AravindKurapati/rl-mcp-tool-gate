import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.gate.budget import select_under_budget, estimate_tokens


def _t(name: str, text: str) -> dict:
    return {"name": name, "embed_text": text}


def test_estimate_tokens_nonzero():
    assert estimate_tokens(_t("a", "hello world")) >= 1


def test_budget_respects_top_k():
    tools = [_t(f"t{i}", "x" * 40) for i in range(10)]
    out = select_under_budget(tools, budget_tokens=None, top_k=3)
    assert len(out) == 3


def test_budget_respects_tokens():
    tools = [_t(f"t{i}", "x" * 40) for i in range(10)]
    out = select_under_budget(tools, budget_tokens=25, top_k=None)
    assert len(out) <= 3


def test_budget_skips_expensive_takes_cheap():
    tools = [
        _t("big1", "x" * 200),
        _t("small", "x" * 8),
    ]
    out = select_under_budget(tools, budget_tokens=10, top_k=None)
    assert [t["name"] for t in out] == ["small"]
