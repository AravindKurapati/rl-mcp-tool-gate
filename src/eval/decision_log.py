"""Offline reader for the proxy's decision log.

Loads ~/.tool_gate/decisions.db into in-memory dicts so analysis scripts can
pair gate decisions with downstream tool calls without depending on the proxy.
"""
from __future__ import annotations
import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from src.proxy.decision_log import default_db_path


@dataclass
class Decision:
    session_id: str
    decision_idx: int
    timestamp: float
    signal_kind: str
    signal_text: str
    catalog_hash: str
    catalog_size: int
    top_k: int
    budget_tokens: int
    selected: list[dict]
    encoder_ckpt: str | None
    latency_ms: float


@dataclass
class Call:
    session_id: str
    decision_idx: int | None
    timestamp: float
    tool_name: str
    was_surfaced: int
    status: str | None
    duration_ms: int | None


def load_decisions(db_path: Path | None = None) -> list[Decision]:
    db_path = Path(db_path) if db_path else default_db_path()
    if not db_path.exists():
        return []
    with sqlite3.connect(str(db_path)) as con:
        rows = con.execute(
            """SELECT session_id, decision_idx, timestamp, signal_kind, signal_text,
                      catalog_hash, catalog_size, top_k, budget_tokens, selected_json,
                      encoder_ckpt, latency_ms
               FROM gate_decisions ORDER BY session_id, decision_idx"""
        ).fetchall()
    return [
        Decision(
            session_id=r[0], decision_idx=r[1], timestamp=r[2], signal_kind=r[3],
            signal_text=r[4], catalog_hash=r[5], catalog_size=r[6], top_k=r[7],
            budget_tokens=r[8], selected=json.loads(r[9]), encoder_ckpt=r[10],
            latency_ms=r[11],
        )
        for r in rows
    ]


def load_calls(db_path: Path | None = None) -> list[Call]:
    db_path = Path(db_path) if db_path else default_db_path()
    if not db_path.exists():
        return []
    with sqlite3.connect(str(db_path)) as con:
        rows = con.execute(
            """SELECT session_id, decision_idx, timestamp, tool_name, was_surfaced,
                      status, duration_ms
               FROM tool_calls ORDER BY session_id, timestamp"""
        ).fetchall()
    return [
        Call(
            session_id=r[0], decision_idx=r[1], timestamp=r[2], tool_name=r[3],
            was_surfaced=r[4], status=r[5], duration_ms=r[6],
        )
        for r in rows
    ]


def bandit_dataset(db_path: Path | None = None) -> list[dict]:
    """One row per decision with the calls that landed under it.

    Schema per row:
      session_id, decision_idx, signal_text, signal_kind, catalog_size,
      top_k, surfaced (list[str]), called (list[str]),
      called_and_surfaced (list[str]), called_but_dropped (list[str]),
      latency_ms, encoder_ckpt
    """
    decisions = load_decisions(db_path)
    calls = load_calls(db_path)
    calls_by_decision: dict[tuple[str, int], list[Call]] = defaultdict(list)
    for c in calls:
        if c.decision_idx is None:
            continue
        calls_by_decision[(c.session_id, c.decision_idx)].append(c)

    rows: list[dict] = []
    for d in decisions:
        surfaced = [t["name"] for t in d.selected]
        surfaced_set = set(surfaced)
        called = [c.tool_name for c in calls_by_decision.get((d.session_id, d.decision_idx), [])]
        called_set = set(called)
        rows.append({
            "session_id": d.session_id,
            "decision_idx": d.decision_idx,
            "signal_kind": d.signal_kind,
            "signal_text": d.signal_text,
            "catalog_size": d.catalog_size,
            "top_k": d.top_k,
            "surfaced": surfaced,
            "called": called,
            "called_and_surfaced": sorted(called_set & surfaced_set),
            "called_but_dropped": sorted(called_set - surfaced_set),
            "latency_ms": d.latency_ms,
            "encoder_ckpt": d.encoder_ckpt,
        })
    return rows
