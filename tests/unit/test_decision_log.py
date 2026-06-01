"""Tests for the gate decision logger."""
from __future__ import annotations
import asyncio
import sqlite3
from pathlib import Path

import pytest

from src.proxy.decision_log import (
    CallRow,
    DecisionLog,
    DecisionRow,
    catalog_hash,
    mint_session_id,
)


def _make_decision(name="foo", **overrides) -> DecisionRow:
    base = dict(
        signal_kind="query",
        signal_text="search the web for tutorials",
        catalog_hash="abc123",
        catalog_size=99,
        top_k=5,
        budget_tokens=4000,
        selected=[{"name": name}],
        encoder_ckpt="checkpoints/sft",
        latency_ms=27.3,
    )
    base.update(overrides)
    return DecisionRow(**base)


def test_schema_round_trip(tmp_path: Path):
    db = tmp_path / "x.db"
    log = DecisionLog(db_path=db, session_id="s1")
    idx = log.log_decision(_make_decision())
    log.log_call(CallRow(tool_name="foo", was_surfaced=1, status="ok", duration_ms=12))

    with sqlite3.connect(str(db)) as con:
        decisions = con.execute("SELECT session_id, decision_idx, signal_text, selected_json FROM gate_decisions").fetchall()
        calls = con.execute("SELECT session_id, decision_idx, tool_name, was_surfaced, status FROM tool_calls").fetchall()

    assert idx == 0
    assert decisions == [("s1", 0, "search the web for tutorials", '[{"name": "foo"}]')]
    assert calls == [("s1", 0, "foo", 1, "ok")]


def test_was_surfaced_uses_most_recent_decision(tmp_path: Path):
    log = DecisionLog(db_path=tmp_path / "x.db", session_id="s2")
    log.log_decision(_make_decision(name="foo"))
    assert log.was_surfaced("foo") == 1
    assert log.was_surfaced("bar") == 0
    log.log_decision(_make_decision(name="bar"))
    assert log.was_surfaced("foo") == 0
    assert log.was_surfaced("bar") == 1


def test_decision_idx_monotonic_per_log(tmp_path: Path):
    log = DecisionLog(db_path=tmp_path / "x.db", session_id="s3")
    assert log.log_decision(_make_decision()) == 0
    assert log.log_decision(_make_decision()) == 1
    assert log.log_decision(_make_decision()) == 2


def test_session_id_stable_across_decisions(tmp_path: Path):
    log = DecisionLog(db_path=tmp_path / "x.db", session_id="stable")
    log.log_decision(_make_decision())
    log.log_decision(_make_decision())
    with sqlite3.connect(str(log.db_path)) as con:
        ids = {row[0] for row in con.execute("SELECT session_id FROM gate_decisions")}
    assert ids == {"stable"}


def test_session_id_changes_across_boots():
    a = mint_session_id()
    b = mint_session_id()
    assert a != b
    assert len(a) == 16


def test_catalog_hash_is_order_independent():
    a = catalog_hash([{"name": "a"}, {"name": "b"}, {"name": "c"}])
    b = catalog_hash([{"name": "c"}, {"name": "a"}, {"name": "b"}])
    assert a == b


def test_call_inserts_null_decision_idx_when_no_decisions(tmp_path: Path):
    log = DecisionLog(db_path=tmp_path / "x.db", session_id="s4")
    log.log_call(CallRow(tool_name="orphan", was_surfaced=0))
    with sqlite3.connect(str(log.db_path)) as con:
        rows = con.execute("SELECT decision_idx, tool_name FROM tool_calls").fetchall()
    assert rows == [(None, "orphan")]


def test_sync_write_failure_does_not_propagate(tmp_path: Path, monkeypatch):
    log = DecisionLog(db_path=tmp_path / "x.db", session_id="s5")

    def boom(*a, **kw):
        raise sqlite3.OperationalError("disk full")
    monkeypatch.setattr(log, "_write_decision_sync", boom)
    log.log_decision(_make_decision())  # should not raise


def test_async_write_path(tmp_path: Path):
    async def run():
        log = DecisionLog(db_path=tmp_path / "x.db", session_id="s6")
        await log.start()
        log.log_decision(_make_decision())
        log.log_call(CallRow(tool_name="foo", was_surfaced=1, status="ok", duration_ms=5))
        await log.stop()
        with sqlite3.connect(str(log.db_path)) as con:
            d = con.execute("SELECT COUNT(*) FROM gate_decisions").fetchone()[0]
            c = con.execute("SELECT COUNT(*) FROM tool_calls").fetchone()[0]
        assert d == 1
        assert c == 1
    asyncio.run(run())
