import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.eval.dashboard import _percentiles, _measure


class FakeEncoder:
    def __init__(self):
        self._names = None

    def precompute_catalog(self, catalog):
        self._names = [t["name"] for t in catalog]

    def score(self, signal):
        return np.array([1.0 if n in signal else 0.0 for n in self._names])


def test_percentiles():
    p50, p95 = _percentiles([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert 5 <= p50 <= 6
    assert p95 >= 9
    assert _percentiles([]) == (0.0, 0.0)


def test_measure_reports_savings_and_latency():
    catalog = [{"name": n, "server": "s", "tool": n, "description": n,
                "args": "", "embed_text": f"{n}: a fairly long description here"} for n in
               [f"t{i}" for i in range(30)] + ["Read"]]
    sessions = [{"user_goal": "Read", "tools_called": ["Read"], "n_tool_calls": 1}] * 5
    enc = FakeEncoder()
    enc.precompute_catalog(catalog)
    m = _measure(enc, sessions, catalog, top_k=5)
    assert m["full_catalog_tokens"] > m["mean_gated_tokens"]
    assert m["mean_tokens_saved_per_turn"] > 0
    assert m["latency_p50_ms"] >= 0
