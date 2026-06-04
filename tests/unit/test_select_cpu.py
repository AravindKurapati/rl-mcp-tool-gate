"""CPU-only coverage for the gate's public API src/gate/select.select_tools.

select_tools is the gate's entry point, but its only existing tests (test_select.py)
construct a real GateEncoder and therefore download BAAI/bge-small-en-v1.5 — they
fail offline. Here we inject a deterministic fake encoder so the ranking-by-score and
budget/top_k integration logic is exercised without any network or model download.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.gate.select import select_tools


class FakeEncoder:
    """Returns a fixed score per tool name, regardless of the signal."""

    def __init__(self, scores_by_name: dict[str, float]):
        self._scores = scores_by_name
        self._names: list[str] | None = None

    def precompute_catalog(self, catalog):
        self._names = [t["name"] for t in catalog]

    def score(self, signal: str) -> np.ndarray:
        return np.array([self._scores[n] for n in self._names], dtype=float)


def _catalog(names_to_text: dict[str, str]):
    return [{"name": n, "embed_text": txt} for n, txt in names_to_text.items()]


def test_select_ranks_by_descending_score():
    cat = _catalog({"a": "x" * 8, "b": "x" * 8, "c": "x" * 8})
    enc = FakeEncoder({"a": 0.1, "b": 0.9, "c": 0.5})
    enc.precompute_catalog(cat)
    out = select_tools(signal="q", catalog=cat, encoder=enc, top_k=3)
    assert [t["name"] for t in out] == ["b", "c", "a"]


def test_select_respects_top_k():
    cat = _catalog({"a": "x" * 8, "b": "x" * 8, "c": "x" * 8})
    enc = FakeEncoder({"a": 0.1, "b": 0.9, "c": 0.5})
    enc.precompute_catalog(cat)
    out = select_tools(signal="q", catalog=cat, encoder=enc, top_k=2)
    assert [t["name"] for t in out] == ["b", "c"]


def test_select_budget_drops_lower_ranked_when_over_budget():
    # Each tool costs len(embed_text)//4 tokens. With 12-char text -> 3 tokens each,
    # a 7-token budget admits the two highest-scored tools and rejects the third.
    cat = _catalog({"a": "x" * 12, "b": "x" * 12, "c": "x" * 12})
    enc = FakeEncoder({"a": 0.9, "b": 0.5, "c": 0.1})
    enc.precompute_catalog(cat)
    out = select_tools(signal="q", catalog=cat, encoder=enc, top_k=None, budget_tokens=7)
    assert [t["name"] for t in out] == ["a", "b"]


def test_select_budget_skips_expensive_but_keeps_later_cheap():
    # The budget loop is greedy-by-rank but skips (not stops at) an over-budget item,
    # so a cheap lower-ranked tool can still be admitted after an expensive one is skipped.
    cat = _catalog({"big": "x" * 80, "small": "x" * 8})
    enc = FakeEncoder({"big": 0.9, "small": 0.5})
    enc.precompute_catalog(cat)
    out = select_tools(signal="q", catalog=cat, encoder=enc, top_k=None, budget_tokens=5)
    assert [t["name"] for t in out] == ["small"]


def test_select_top_k_and_budget_apply_together():
    cat = _catalog({"a": "x" * 8, "b": "x" * 8, "c": "x" * 8, "d": "x" * 8})
    enc = FakeEncoder({"a": 0.9, "b": 0.8, "c": 0.7, "d": 0.6})
    enc.precompute_catalog(cat)
    # top_k caps at 3; budget is generous, so the cap is the binding constraint.
    out = select_tools(signal="q", catalog=cat, encoder=enc, top_k=3, budget_tokens=10_000)
    assert [t["name"] for t in out] == ["a", "b", "c"]
