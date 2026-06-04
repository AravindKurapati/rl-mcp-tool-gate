"""CPU-only tests for the synthetic eval harness (src/eval/baselines.py and
src/eval/pareto.py) — the code that produces the README's synthetic recall tables.

These cover the classical baselines (no_gate, random, BM25) and the metric
aggregation in pareto.evaluate / eval_qwen_from_file, none of which require the
BGE model download (so they run offline, unlike test_select.py). The BGE encoder
path is exercised separately; here we lock in the harness plumbing and the
"reproduces bit-identically across runs" claim from the README.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.eval import baselines
from src.eval.baselines import baseline_no_gate, baseline_random, baseline_bm25
from src.eval.pareto import evaluate, eval_qwen_from_file


def _catalog():
    return [
        {"name": "slack.post", "embed_text": "slack post message send chat channel"},
        {"name": "gh.issues", "embed_text": "github list issues repository tracker"},
        {"name": "fs.read", "embed_text": "read file from local filesystem path"},
        {"name": "fs.write", "embed_text": "write file to local filesystem path"},
        {"name": "web.search", "embed_text": "search the web for query results"},
    ]


# --- baseline_no_gate -------------------------------------------------------

def test_no_gate_returns_full_catalog_unchanged():
    cat = _catalog()
    out = baseline_no_gate("anything", cat)
    assert out is cat
    assert len(out) == 5


# --- baseline_random --------------------------------------------------------

def test_random_deterministic_with_explicit_seed():
    cat = _catalog()
    a = baseline_random("q", cat, top_k=3, seed=42)
    b = baseline_random("q", cat, top_k=3, seed=42)
    assert [t["name"] for t in a] == [t["name"] for t in b]


def test_random_respects_top_k_and_is_subset():
    cat = _catalog()
    out = baseline_random("q", cat, top_k=3, seed=1)
    assert len(out) == 3
    names = {t["name"] for t in out}
    assert names.issubset({t["name"] for t in cat})
    # sample() draws without replacement -> no duplicates
    assert len(names) == 3


def test_random_top_k_larger_than_catalog_is_clamped():
    cat = _catalog()
    out = baseline_random("q", cat, top_k=99, seed=1)
    assert len(out) == len(cat)


def test_random_without_seed_is_query_deterministic():
    # Falls back to hashing the query, so the same query reproduces.
    cat = _catalog()
    a = baseline_random("same query", cat, top_k=2)
    b = baseline_random("same query", cat, top_k=2)
    assert [t["name"] for t in a] == [t["name"] for t in b]


# --- baseline_bm25 ----------------------------------------------------------

def test_bm25_ranks_lexical_match_first():
    baselines._bm25_cache = None  # avoid inheriting a stale cache from another test
    cat = _catalog()
    out = baseline_bm25("post a message to a slack channel", cat, top_k=1)
    assert out[0]["name"] == "slack.post"


def test_bm25_respects_top_k():
    baselines._bm25_cache = None
    cat = _catalog()
    out = baseline_bm25("read a file", cat, top_k=2)
    assert len(out) == 2


def test_bm25_cache_follows_the_passed_catalog():
    # The module caches the BM25 index by id(catalog); a different catalog object
    # must rebuild the index rather than reuse the previous corpus. Both catalogs
    # are held alive here so their ids stay distinct for the duration of the test.
    baselines._bm25_cache = None
    cat1 = [{"name": "only.alpha", "embed_text": "alpha keyword unique token"}]
    cat2 = [{"name": "only.beta", "embed_text": "beta keyword unique token"}]
    out1 = baseline_bm25("alpha", cat1, top_k=1)
    out2 = baseline_bm25("beta", cat2, top_k=1)
    assert out1[0]["name"] == "only.alpha"
    assert out2[0]["name"] == "only.beta"


# --- pareto.evaluate --------------------------------------------------------

def _queries():
    return [
        {"query": "post a message to a slack channel", "ground_truth": ["slack.post"]},
        {"query": "read a file from the local filesystem", "ground_truth": ["fs.read"]},
    ]


def test_evaluate_metric_aggregation_matches_hand_computation():
    baselines._bm25_cache = None
    cat = _catalog()
    res = evaluate("bm25", lambda q, c, top_k: baseline_bm25(q, c, top_k=top_k),
                   _queries(), cat, [1, 2])
    assert res["method"] == "bm25"
    by_k = res["by_k"]

    # At k=1 each query surfaces exactly its single ground-truth tool.
    assert by_k[1]["recall"] == 1.0
    assert by_k[1]["precision"] == 1.0
    assert by_k[1]["catastrophic_rate"] == 0.0
    assert by_k[1]["mean_size"] == 1.0

    # At k=2 recall stays 1.0 (GT still surfaced) but precision halves: 1 of 2.
    assert by_k[2]["recall"] == 1.0
    assert by_k[2]["precision"] == 0.5
    assert by_k[2]["catastrophic_rate"] == 0.0
    assert by_k[2]["mean_size"] == 2.0


def test_evaluate_reports_catastrophic_when_gt_dropped():
    # A query whose ground truth is not lexically recoverable by BM25 at k=1.
    baselines._bm25_cache = None
    cat = _catalog()
    queries = [{"query": "post a message to a slack channel",
                "ground_truth": ["fs.write"]}]  # GT unrelated to the query terms
    res = evaluate("bm25", lambda q, c, top_k: baseline_bm25(q, c, top_k=top_k),
                   queries, cat, [1])
    m = res["by_k"][1]
    assert m["recall"] == 0.0
    assert m["catastrophic_rate"] == 1.0


def test_evaluate_is_bit_identical_across_runs():
    # README claims the harness reproduces baselines bit-identically across runs.
    # random is the only stochastic baseline; pin the seed and assert exact equality.
    cat = _catalog()

    def fn(q, c, top_k):
        return baseline_random(q, c, top_k=top_k, seed=42)

    a = evaluate("random", fn, _queries(), cat, [3, 5])
    b = evaluate("random", fn, _queries(), cat, [3, 5])
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


# --- pareto.eval_qwen_from_file --------------------------------------------

def test_eval_qwen_from_file_parses_and_scores(tmp_path: Path):
    preds = tmp_path / "qwen.jsonl"
    preds.write_text(
        json.dumps({"query": "post a message to a slack channel",
                    "predicted": ["slack.post", "gh.issues"]}) + "\n"
        + json.dumps({"query": "read a file from the local filesystem",
                      "predicted": ["fs.read"]}) + "\n",
        encoding="utf-8",
    )
    res = eval_qwen_from_file(preds, _queries())
    assert res["method"] == "qwen2.5-1.5b"
    m = res["by_k"]["N/A"]
    # both GTs surfaced -> recall 1.0
    assert m["recall"] == 1.0
    # precision: q1 = 1/2, q2 = 1/1 -> mean 0.75
    assert abs(m["precision"] - 0.75) < 1e-9
    assert m["catastrophic_rate"] == 0.0
    # mean predicted size: (2 + 1) / 2
    assert m["mean_size"] == 1.5


def test_eval_qwen_from_file_missing_query_counts_as_empty(tmp_path: Path):
    preds = tmp_path / "qwen.jsonl"
    # Only one of the two eval queries has a prediction; the other defaults to empty.
    preds.write_text(
        json.dumps({"query": "post a message to a slack channel",
                    "predicted": ["slack.post"]}) + "\n",
        encoding="utf-8",
    )
    res = eval_qwen_from_file(preds, _queries())
    m = res["by_k"]["N/A"]
    # q1 recall 1.0, q2 (no prediction) recall 0.0 -> mean 0.5
    assert m["recall"] == 0.5
    # q2 with empty prediction is a catastrophic drop
    assert m["catastrophic_rate"] == 0.5
