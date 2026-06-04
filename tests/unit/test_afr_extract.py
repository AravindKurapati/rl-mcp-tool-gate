"""Tests for src.eval.afr_extract.extract — the SQLite entry point of the real-traffic
eval pipeline.

The whole "test against 26 real MCP runs" story (the thing the README says should have
been built first) starts here: extract() turns the agent-flight-recorder DB into the
sessions.jsonl that bootstrap_afr and afr_eval consume. Its filtering and normalization
rules are load-bearing and were untested:

  - drop runs whose goal is missing or shorter than MIN_GOAL_CHARS
  - drop runs with no tool calls (and runs whose every tool name is empty)
  - dedup tools_called while preserving first-call order
  - n_tool_calls counts every call row (including empty-named ones); tools_called does not
  - n_errors counts status == "error"; total_duration_ms sums duration_ms with null -> 0
  - null cost/tokens/outcome normalize to 0.0 / 0 / "untagged"
  - runs are ordered started_at DESC and capped at `limit`

CPU-only, builds a throwaway SQLite DB; no model, no network.
"""
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.eval.afr_extract import extract, MIN_GOAL_CHARS


def _make_db(path: Path, runs, calls) -> None:
    """runs: list of (id, started_at, user_goal, outcome, cost_usd, tokens_in, tokens_out).
    calls: list of (run_id, tool_name, status, duration_ms, timestamp)."""
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE runs (id TEXT, started_at TEXT, user_goal TEXT, outcome TEXT, "
        "cost_usd REAL, tokens_in INTEGER, tokens_out INTEGER)"
    )
    conn.execute(
        "CREATE TABLE tool_calls (run_id TEXT, tool_name TEXT, status TEXT, "
        "duration_ms INTEGER, timestamp TEXT)"
    )
    conn.executemany("INSERT INTO runs VALUES (?,?,?,?,?,?,?)", runs)
    conn.executemany("INSERT INTO tool_calls VALUES (?,?,?,?,?)", calls)
    conn.commit()
    conn.close()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_happy_path_dedup_errors_duration_and_nulls(tmp_path: Path):
    db = tmp_path / "afr.db"
    out = tmp_path / "sessions.jsonl"
    _make_db(
        db,
        runs=[("r1", "2026-05-01", "Refactor the auth module thoroughly", None, None, None, None)],
        calls=[
            ("r1", "Read", "ok", 100, "t1"),
            ("r1", "Bash", "error", 50, "t2"),
            ("r1", "Read", "ok", None, "t3"),  # duplicate name, null duration
        ],
    )
    n = extract(db, out, limit=500)
    assert n == 1
    [row] = _read_jsonl(out)
    assert row["run_id"] == "r1"
    # dedup preserves first-call order; duplicate Read collapses
    assert row["tools_called"] == ["Read", "Bash"]
    assert row["n_tool_calls"] == 3  # all call rows counted
    assert row["n_errors"] == 1
    assert row["total_duration_ms"] == 150  # 100 + 50 + (null->0)
    # null cost/tokens/outcome normalized
    assert row["cost_usd"] == 0.0
    assert row["tokens_in"] == 0 and row["tokens_out"] == 0
    assert row["outcome"] == "untagged"


def test_short_or_missing_goal_is_dropped(tmp_path: Path):
    db = tmp_path / "afr.db"
    out = tmp_path / "sessions.jsonl"
    short_goal = "x" * (MIN_GOAL_CHARS - 1)
    _make_db(
        db,
        runs=[
            ("short", "2026-05-01", short_goal, "ok", 0.0, 0, 0),
            ("none", "2026-05-02", None, "ok", 0.0, 0, 0),
            ("good", "2026-05-03", "A sufficiently long goal", "ok", 0.0, 0, 0),
        ],
        calls=[
            ("short", "Read", "ok", 1, "t"),
            ("none", "Read", "ok", 1, "t"),
            ("good", "Read", "ok", 1, "t"),
        ],
    )
    n = extract(db, out, limit=500)
    rows = _read_jsonl(out)
    assert n == 1
    assert [r["run_id"] for r in rows] == ["good"]


def test_runs_without_usable_tool_calls_are_dropped(tmp_path: Path):
    db = tmp_path / "afr.db"
    out = tmp_path / "sessions.jsonl"
    _make_db(
        db,
        runs=[
            ("nocalls", "2026-05-01", "Goal with no tool calls at all", "ok", 0.0, 0, 0),
            ("emptynames", "2026-05-02", "Goal whose calls have empty names", "ok", 0.0, 0, 0),
            ("ok", "2026-05-03", "Goal with a real tool call", "ok", 0.0, 0, 0),
        ],
        calls=[
            # nocalls: none
            ("emptynames", "", "ok", 1, "t"),
            ("emptynames", None, "ok", 1, "t"),
            ("ok", "Bash", "ok", 1, "t"),
        ],
    )
    n = extract(db, out, limit=500)
    rows = _read_jsonl(out)
    assert n == 1
    assert [r["run_id"] for r in rows] == ["ok"]


def test_empty_named_call_counts_toward_n_tool_calls_but_not_tools_called(tmp_path: Path):
    db = tmp_path / "afr.db"
    out = tmp_path / "sessions.jsonl"
    _make_db(
        db,
        runs=[("r", "2026-05-01", "Mixed empty and real tool calls", "ok", 0.0, 0, 0)],
        calls=[
            ("r", "Read", "ok", 10, "t1"),
            ("r", "", "ok", 10, "t2"),
        ],
    )
    extract(db, out, limit=500)
    [row] = _read_jsonl(out)
    assert row["tools_called"] == ["Read"]
    assert row["n_tool_calls"] == 2


def test_ordering_started_at_desc_and_limit(tmp_path: Path):
    db = tmp_path / "afr.db"
    out = tmp_path / "sessions.jsonl"
    _make_db(
        db,
        runs=[
            ("old", "2026-01-01", "Oldest run with a goal", "ok", 0.0, 0, 0),
            ("mid", "2026-03-01", "Middle run with a goal", "ok", 0.0, 0, 0),
            ("new", "2026-06-01", "Newest run with a goal", "ok", 0.0, 0, 0),
        ],
        calls=[
            ("old", "Read", "ok", 1, "t"),
            ("mid", "Read", "ok", 1, "t"),
            ("new", "Read", "ok", 1, "t"),
        ],
    )
    n = extract(db, out, limit=2)
    rows = _read_jsonl(out)
    # LIMIT 2 applied to started_at DESC -> the two newest runs, newest first
    assert n == 2
    assert [r["run_id"] for r in rows] == ["new", "mid"]


def test_returns_zero_and_writes_empty_file_when_no_usable_runs(tmp_path: Path):
    db = tmp_path / "afr.db"
    out = tmp_path / "nested" / "sessions.jsonl"  # also checks parent mkdir
    _make_db(
        db,
        runs=[("r", "2026-05-01", "tiny", "ok", 0.0, 0, 0)],  # goal too short
        calls=[("r", "Read", "ok", 1, "t")],
    )
    n = extract(db, out, limit=500)
    assert n == 0
    assert out.exists()
    assert _read_jsonl(out) == []
