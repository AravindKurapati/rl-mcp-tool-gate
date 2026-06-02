"""Tests for the offline decision log reader."""
from __future__ import annotations
from pathlib import Path

from src.proxy.decision_log import CallRow, DecisionLog, DecisionRow
from src.eval.decision_log import bandit_dataset, load_calls, load_decisions


def _decision(name="foo"):
    return DecisionRow(
        signal_kind="query",
        signal_text="do the thing",
        catalog_hash="h",
        catalog_size=10,
        top_k=3,
        budget_tokens=2000,
        selected=[{"name": name}, {"name": "bar"}],
        encoder_ckpt="off_the_shelf",
        latency_ms=15.0,
    )


def test_load_returns_empty_when_db_missing(tmp_path: Path):
    assert load_decisions(tmp_path / "missing.db") == []
    assert load_calls(tmp_path / "missing.db") == []
    assert bandit_dataset(tmp_path / "missing.db") == []


def test_bandit_dataset_pairs_calls_with_decisions(tmp_path: Path):
    log = DecisionLog(db_path=tmp_path / "x.db", session_id="s1")
    log.log_decision(_decision(name="foo"))
    log.log_call(CallRow(tool_name="foo", was_surfaced=1))
    log.log_call(CallRow(tool_name="quux", was_surfaced=0))
    log.log_decision(_decision(name="bar"))
    log.log_call(CallRow(tool_name="bar", was_surfaced=1))

    rows = bandit_dataset(log.db_path)
    assert len(rows) == 2

    r0 = rows[0]
    assert r0["surfaced"] == ["foo", "bar"]
    assert sorted(r0["called"]) == ["foo", "quux"]
    assert r0["called_and_surfaced"] == ["foo"]
    assert r0["called_but_dropped"] == ["quux"]

    r1 = rows[1]
    assert r1["called"] == ["bar"]
    assert r1["called_and_surfaced"] == ["bar"]
    assert r1["called_but_dropped"] == []
