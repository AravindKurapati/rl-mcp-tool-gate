"""Baseline tool selectors for comparison vs RL-tuned BGE."""
from __future__ import annotations
import random
from collections import OrderedDict
from typing import Any
from rank_bm25 import BM25Okapi

from src.gate.encoder import GateEncoder
from src.gate.select import select_tools


def baseline_no_gate(query: str, catalog: list[dict[str, Any]], **_) -> list[dict[str, Any]]:
    return catalog


def baseline_random(query: str, catalog: list[dict[str, Any]], top_k: int = 10, seed: int | None = None) -> list[dict[str, Any]]:
    rng = random.Random(seed if seed is not None else hash(query) & 0xFFFFFFFF)
    return rng.sample(catalog, min(top_k, len(catalog)))


# Cache the (expensive) BM25 index keyed on catalog *content*, not id(catalog).
# id() is only unique among live objects: once a catalog list is garbage-collected,
# a later catalog can be allocated at the same address and produce a cache hit that
# returns an index built on the wrong corpus. Evaluating BM25 over more than one
# catalog in a single process (e.g. synthetic vs v2 vs afr) would then silently
# corrupt the baseline numbers. Keying on a content fingerprint is GC-safe and also
# robust to in-place mutation of a reused catalog object.
_BM25_CACHE_MAXSIZE = 8
_bm25_cache: "OrderedDict[int, BM25Okapi]" = OrderedDict()


def _catalog_fingerprint(catalog: list[dict[str, Any]]) -> int:
    return hash(tuple(t["embed_text"] for t in catalog))


def baseline_bm25(query: str, catalog: list[dict[str, Any]], top_k: int = 10) -> list[dict[str, Any]]:
    fp = _catalog_fingerprint(catalog)
    bm25 = _bm25_cache.get(fp)
    if bm25 is None:
        corpus = [t["embed_text"].lower().split() for t in catalog]
        bm25 = BM25Okapi(corpus)
        _bm25_cache[fp] = bm25
        if len(_bm25_cache) > _BM25_CACHE_MAXSIZE:
            _bm25_cache.popitem(last=False)
    else:
        _bm25_cache.move_to_end(fp)
    scores = bm25.get_scores(query.lower().split())
    ranked = sorted(zip(scores, catalog), key=lambda x: -x[0])
    return [t for _, t in ranked[:top_k]]


def baseline_bge(query: str, catalog: list[dict[str, Any]], encoder: GateEncoder, top_k: int = 10) -> list[dict[str, Any]]:
    return select_tools(signal=query, catalog=catalog, encoder=encoder, top_k=top_k)
