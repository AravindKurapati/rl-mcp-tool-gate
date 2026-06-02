"""Tests for the BM25 eval baseline and its catalog-content-keyed index cache.

BM25 is the classical, CPU-only baseline that appears in the headline tables, so its
ranking and (cached) reuse behaviour need to be pinned down. None of this touches the
network or a real model — only rank_bm25 over in-memory catalogs.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from src.eval import baselines
from src.eval.baselines import baseline_bm25, _catalog_fingerprint


def _t(name: str, text: str) -> dict:
    return {"name": name, "embed_text": text}


@pytest.fixture(autouse=True)
def _clear_cache():
    """Each test starts from an empty index cache so ordering can't leak state."""
    baselines._bm25_cache.clear()
    yield
    baselines._bm25_cache.clear()


def _catalog_a() -> list[dict]:
    return [
        _t("slack.post_message", "slack post message send a chat message to a channel"),
        _t("github.create_issue", "github create issue open a bug report in a repository"),
        _t("fs.read_file", "filesystem read file contents from a path on disk"),
    ]


def _catalog_b() -> list[dict]:
    return [
        _t("calendar.create_event", "calendar create event schedule a meeting on a date"),
        _t("email.send", "email send compose and deliver a message to a recipient"),
        _t("weather.forecast", "weather forecast temperature and rain for a location"),
    ]


def test_ranks_relevant_tool_first():
    out = baseline_bm25("send a message to a slack channel", _catalog_a(), top_k=1)
    assert [t["name"] for t in out] == ["slack.post_message"]


def test_respects_top_k():
    out = baseline_bm25("read file from disk", _catalog_a(), top_k=2)
    assert len(out) == 2


def test_top_k_larger_than_catalog_returns_all():
    cat = _catalog_a()
    out = baseline_bm25("anything", cat, top_k=99)
    assert len(out) == len(cat)


def test_no_token_overlap_still_returns_top_k():
    # A query that shares no tokens with any tool still yields a deterministic subset
    # (all BM25 scores zero -> stable sort keeps catalog order), never a crash.
    out = baseline_bm25("zzzz qqqq", _catalog_a(), top_k=2)
    assert len(out) == 2


def test_same_catalog_twice_is_consistent():
    cat = _catalog_a()
    a = baseline_bm25("create a github issue", cat, top_k=2)
    b = baseline_bm25("create a github issue", cat, top_k=2)
    assert [t["name"] for t in a] == [t["name"] for t in b]


def test_fingerprint_distinguishes_content():
    # The invariant the fix relies on: distinct catalog content -> distinct cache key,
    # identical content -> identical key (regardless of object identity).
    assert _catalog_fingerprint(_catalog_a()) != _catalog_fingerprint(_catalog_b())
    assert _catalog_fingerprint(_catalog_a()) == _catalog_fingerprint(_catalog_a())


def test_distinct_catalogs_do_not_share_index():
    # Regression for the id(catalog) cache bug: two different catalogs evaluated in
    # sequence must each be scored against their *own* corpus. With id-keyed caching,
    # GC id-reuse could make the second call reuse the first catalog's index.
    out_a = baseline_bm25("send a slack message", _catalog_a(), top_k=1)
    out_b = baseline_bm25("schedule a calendar event", _catalog_b(), top_k=1)
    assert out_a[0]["name"] == "slack.post_message"
    assert out_b[0]["name"] == "calendar.create_event"
    # Both indexes coexist in the cache, keyed independently.
    assert len(baselines._bm25_cache) == 2


def test_index_is_actually_cached_and_reused():
    cat = _catalog_a()
    baseline_bm25("read file", cat, top_k=1)
    fp = _catalog_fingerprint(cat)
    first_index = baselines._bm25_cache[fp]
    baseline_bm25("github issue", cat, top_k=1)
    # Same fingerprint -> the expensive BM25Okapi object is reused, not rebuilt.
    assert baselines._bm25_cache[fp] is first_index


def test_cache_is_bounded():
    # Distinct catalogs beyond the cap evict the oldest rather than growing unbounded.
    for i in range(baselines._BM25_CACHE_MAXSIZE + 5):
        cat = [_t(f"tool_{i}", f"unique embed text number {i} alpha beta")]
        baseline_bm25("query", cat, top_k=1)
    assert len(baselines._bm25_cache) <= baselines._BM25_CACHE_MAXSIZE


def test_in_place_mutation_is_not_stale():
    # A reused catalog object whose contents change must not return the stale index.
    cat = _catalog_a()
    out1 = baseline_bm25("send a slack message", cat, top_k=1)
    assert out1[0]["name"] == "slack.post_message"
    cat[:] = _catalog_b()  # same list object, new content
    out2 = baseline_bm25("schedule a calendar event", cat, top_k=1)
    assert out2[0]["name"] == "calendar.create_event"
