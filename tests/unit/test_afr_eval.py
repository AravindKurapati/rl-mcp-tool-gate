import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.eval.afr_eval import _eval_encoder, _observed_tools


class FakeEncoder:
    """Scores a catalog by exact-substring match of tool name in the signal."""
    def __init__(self):
        self._names = None

    def precompute_catalog(self, catalog):
        self._names = [t["name"] for t in catalog]

    def score(self, signal: str) -> np.ndarray:
        return np.array([1.0 if n in signal else 0.0 for n in self._names])


def _catalog(names):
    return [{"name": n, "server": "s", "tool": n, "description": n,
             "args": "", "embed_text": f"{n}: desc"} for n in names]


def test_non_breaking_rate_and_recall():
    catalog = _catalog(["Read", "Write", "Bash", "Glob", "Grep", "Edit"])
    sessions = [
        # signal names the needed tools -> fully covered at top_k>=2
        {"user_goal": "use Read and Write", "tools_called": ["Read", "Write"], "n_tool_calls": 2},
        # only Bash named; Glob needed but not in signal -> dropped at small k
        {"user_goal": "run Bash", "tools_called": ["Bash", "Glob"], "n_tool_calls": 2},
    ]
    enc = FakeEncoder()
    enc.precompute_catalog(catalog)
    out = _eval_encoder("fake", enc, sessions, catalog, [2])
    m = out["by_k"][2]
    # run 1 fully covered, run 2 drops Glob -> non_breaking 0.5
    assert m["non_breaking_rate"] == 0.5
    assert m["catastrophic_rate"] == 0.5
    assert m["n_runs"] == 2
    # recall: run1 = 1.0, run2 = 0.5 (got Bash, missed Glob) -> mean 0.75
    assert abs(m["recall"] - 0.75) < 1e-9


def test_token_savings_positive():
    catalog = _catalog([f"t{i}" for i in range(20)] + ["Read"])
    sessions = [{"user_goal": "Read", "tools_called": ["Read"], "n_tool_calls": 1}]
    enc = FakeEncoder()
    enc.precompute_catalog(catalog)
    out = _eval_encoder("fake", enc, sessions, catalog, [3])
    assert out["by_k"][3]["mean_token_savings"] > 0


def test_observed_tools_dedup_order():
    sessions = [
        {"tools_called": ["Read", "Bash"]},
        {"tools_called": ["Bash", "Edit"]},
    ]
    assert _observed_tools(sessions) == ["Read", "Bash", "Edit"]
