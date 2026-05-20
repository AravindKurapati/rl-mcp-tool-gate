"""Baseline tool selectors for comparison vs RL-tuned BGE."""
from __future__ import annotations
import random
from typing import Any
from rank_bm25 import BM25Okapi

from src.gate.encoder import GateEncoder
from src.gate.select import select_tools


def baseline_no_gate(query: str, catalog: list[dict[str, Any]], **_) -> list[dict[str, Any]]:
    return catalog


def baseline_random(query: str, catalog: list[dict[str, Any]], top_k: int = 10, seed: int | None = None) -> list[dict[str, Any]]:
    rng = random.Random(seed if seed is not None else hash(query) & 0xFFFFFFFF)
    return rng.sample(catalog, min(top_k, len(catalog)))


_bm25_cache: tuple[BM25Okapi, int] | None = None


def baseline_bm25(query: str, catalog: list[dict[str, Any]], top_k: int = 10) -> list[dict[str, Any]]:
    global _bm25_cache
    if _bm25_cache is None or _bm25_cache[1] != id(catalog):
        corpus = [t["embed_text"].lower().split() for t in catalog]
        _bm25_cache = (BM25Okapi(corpus), id(catalog))
    bm25 = _bm25_cache[0]
    scores = bm25.get_scores(query.lower().split())
    ranked = sorted(zip(scores, catalog), key=lambda x: -x[0])
    return [t for _, t in ranked[:top_k]]


def baseline_bge(query: str, catalog: list[dict[str, Any]], encoder: GateEncoder, top_k: int = 10) -> list[dict[str, Any]]:
    return select_tools(signal=query, catalog=catalog, encoder=encoder, top_k=top_k)
