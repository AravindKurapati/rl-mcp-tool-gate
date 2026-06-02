"""Tests for the heuristic ground-truth filters in afr_extract."""
from __future__ import annotations
import json
import sqlite3
from pathlib import Path

from src.eval.afr_extract import _parse_ts, extract, filter_calls


def _make_call(name: str, status: str = "ok", duration_ms: int = 10, ts: float = 0.0):
    return (name, status, duration_ms, ts)


def test_drop_failed_removes_error_rows():
    calls = [
        _make_call("foo", "ok"),
        _make_call("bar", "error"),
        _make_call("baz", "ok"),
    ]
    out = filter_calls(calls, drop_failed=True)
    assert [c[0] for c in out] == ["foo", "baz"]


def test_drop_failed_off_keeps_errors():
    calls = [_make_call("foo", "error"), _make_call("bar", "ok")]
    out = filter_calls(calls, drop_failed=False)
    assert [c[0] for c in out] == ["foo", "bar"]


def test_dedup_window_collapses_close_retries():
    calls = [
        _make_call("foo", ts=0.0),
        _make_call("foo", ts=5.0),     # within window: drop
        _make_call("bar", ts=10.0),
        _make_call("foo", ts=60.0),    # outside window: keep
    ]
    out = filter_calls(calls, dedup_window_s=30.0)
    assert [(c[0], c[3]) for c in out] == [("foo", 0.0), ("bar", 10.0), ("foo", 60.0)]


def test_dedup_window_zero_disables():
    calls = [_make_call("foo", ts=0.0), _make_call("foo", ts=1.0)]
    out = filter_calls(calls, dedup_window_s=0.0)
    assert len(out) == 2


def test_dedup_window_preserves_distinct_tools():
    calls = [
        _make_call("foo", ts=0.0),
        _make_call("bar", ts=1.0),
        _make_call("baz", ts=2.0),
    ]
    out = filter_calls(calls, dedup_window_s=30.0)
    assert [c[0] for c in out] == ["foo", "bar", "baz"]


def test_filter_calls_skips_empty_names():
    calls = [_make_call("", "ok"), _make_call("foo", "ok")]
    out = filter_calls(calls)
    assert [c[0] for c in out] == ["foo"]


def test_parse_ts_handles_seconds_ms_and_iso():
    assert _parse_ts(1700000000.0) == 1700000000.0
    assert _parse_ts(1700000000000) == 1700000000.0   # ms
    assert _parse_ts("1700000000") == 1700000000.0
    assert _parse_ts("2023-11-14T22:13:20") > 0
    assert _parse_ts(None) == 0.0
    assert _parse_ts("not-a-time") == 0.0


def _seed_afr(db_path: Path) -> None:
    """Build a tiny afr-shaped DB with two runs to exercise extract end to end."""
    con = sqlite3.connect(str(db_path))
    con.executescript("""
        CREATE TABLE runs (
            id INTEGER PRIMARY KEY, source TEXT, project_path TEXT,
            started_at REAL, ended_at REAL, user_goal TEXT,
            final_summary TEXT, outcome TEXT,
            cost_usd REAL, tokens_in INTEGER, tokens_out INTEGER,
            cache_read INTEGER, cache_write INTEGER
        );
        CREATE TABLE tool_calls (
            id INTEGER PRIMARY KEY, run_id INTEGER, tool_name TEXT,
            input_summary TEXT, output_summary TEXT, status TEXT,
            duration_ms INTEGER, timestamp REAL, raw_json TEXT
        );
    """)
    con.execute(
        "INSERT INTO runs VALUES (1, 'cc', '/p', 0, 100, 'find a github file and show me its contents', NULL, 'success', 0.01, 100, 50, 0, 0)"
    )
    con.executemany(
        "INSERT INTO tool_calls (run_id, tool_name, status, duration_ms, timestamp) VALUES (?, ?, ?, ?, ?)",
        [
            (1, "mcp__github__search_code", "error", 50, 1.0),
            (1, "mcp__github__search_code", "ok", 80, 5.0),   # retry within window
            (1, "mcp__github__get_file_contents", "ok", 40, 30.0),
        ],
    )
    con.execute(
        "INSERT INTO runs VALUES (2, 'cc', '/p', 200, 300, 'too short', NULL, 'success', 0.0, 10, 5, 0, 0)"
    )
    con.commit()
    con.close()


def test_extract_unfiltered_keeps_everything(tmp_path: Path):
    db = tmp_path / "afr.db"
    out = tmp_path / "sessions.jsonl"
    _seed_afr(db)
    summary = extract(db, out, limit=10)
    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert summary["n_written"] == 1
    assert rows[0]["tools_called"] == [
        "mcp__github__search_code",
        "mcp__github__get_file_contents",
    ]
    assert rows[0]["n_errors"] == 1


def test_extract_with_drop_failed_and_dedup(tmp_path: Path):
    db = tmp_path / "afr.db"
    out = tmp_path / "sessions_filtered.jsonl"
    _seed_afr(db)
    summary = extract(db, out, limit=10, drop_failed=True, dedup_window_s=30.0)
    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert summary["n_written"] == 1
    # The error retry is dropped by status=ok filter, and the second search_code
    # would have been deduped anyway. Only the two distinct successful tools remain.
    assert rows[0]["tools_called"] == [
        "mcp__github__search_code",
        "mcp__github__get_file_contents",
    ]
    assert rows[0]["n_tool_calls"] == 2
    assert rows[0]["n_tool_calls_raw"] == 3
    assert summary["filters"]["drop_failed"] is True
    assert summary["filters"]["dedup_window_s"] == 30.0
