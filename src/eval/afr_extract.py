"""Extract labeled session data from agent-flight-recorder SQLite.

Real afr schema (verified 2026-05-20):
  runs(id, source, project_path, started_at, ended_at, user_goal, final_summary,
       outcome, cost_usd, tokens_in, tokens_out, cache_read, cache_write)
  tool_calls(id, run_id, tool_name, input_summary, output_summary, status,
             duration_ms, timestamp, raw_json)

Emits one JSONL row per usable run (has a goal and >=1 tool call):
  run_id, user_goal, tools_called (distinct, call-ordered), n_tool_calls,
  n_errors, cost_usd, tokens_in, tokens_out, total_duration_ms, outcome

Filters (ground-truth v2 Pass 1):
- drop_failed: skip tool_calls rows whose status is 'error' before deduping.
- dedup_window_s: a second call to the same tool within N seconds counts as
  one call (collapses retries). Set 0 to disable.
- min_tools: drop runs whose filtered tools_called list is empty.

Defaults preserve the original behavior (noisy ground truth) so existing
results reproduce. Pass --filtered to apply the recommended filter set.
"""
from __future__ import annotations
import argparse
import json
import sqlite3
from pathlib import Path

DEFAULT_DB = Path("~/.afr/afr.db").expanduser()
MIN_GOAL_CHARS = 10

_RUNS_SQL = (
    "SELECT id, user_goal, outcome, cost_usd, tokens_in, tokens_out "
    "FROM runs ORDER BY started_at DESC LIMIT ?"
)
_CALLS_SQL = (
    "SELECT tool_name, status, duration_ms, timestamp FROM tool_calls "
    "WHERE run_id = ? ORDER BY timestamp ASC"
)


def _parse_ts(ts) -> float:
    """afr.tool_calls.timestamp is sometimes a number (unix s or ms), sometimes a
    string. Return seconds since epoch, or 0 if unparseable."""
    if ts is None:
        return 0.0
    if isinstance(ts, (int, float)):
        v = float(ts)
        return v / 1000.0 if v > 1e12 else v  # detect ms
    s = str(ts).strip()
    if not s:
        return 0.0
    try:
        v = float(s)
        return v / 1000.0 if v > 1e12 else v
    except ValueError:
        pass
    try:
        from datetime import datetime
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0


def filter_calls(
    calls: list[tuple],
    drop_failed: bool = False,
    dedup_window_s: float = 0.0,
) -> list[tuple]:
    """Apply heuristic filters to a sequence of (tool_name, status, duration_ms, timestamp)
    rows. Preserves call order."""
    out: list[tuple] = []
    last_seen: dict[str, float] = {}
    for name, status, dur, ts in calls:
        if not name:
            continue
        if drop_failed and (status or "") == "error":
            continue
        if dedup_window_s > 0:
            t = _parse_ts(ts)
            prev = last_seen.get(name)
            if prev is not None and (t - prev) <= dedup_window_s:
                continue
            last_seen[name] = t
        out.append((name, status, dur, ts))
    return out


def extract(
    db_path: Path,
    out_path: Path,
    limit: int = 500,
    drop_failed: bool = False,
    dedup_window_s: float = 0.0,
    min_tools: int = 1,
) -> dict:
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    runs = cur.execute(_RUNS_SQL, (limit,)).fetchall()
    rows = []
    n_runs_seen = 0
    n_dropped_no_goal = 0
    n_dropped_no_calls = 0
    n_dropped_min_tools = 0
    for rid, goal, outcome, cost, tin, tout in runs:
        n_runs_seen += 1
        goal = (goal or "").strip()
        if len(goal) < MIN_GOAL_CHARS:
            n_dropped_no_goal += 1
            continue
        calls_raw = cur.execute(_CALLS_SQL, (rid,)).fetchall()
        if not calls_raw:
            n_dropped_no_calls += 1
            continue
        calls = filter_calls(calls_raw, drop_failed=drop_failed, dedup_window_s=dedup_window_s)
        tools_called = list(dict.fromkeys(c[0] for c in calls))
        if len(tools_called) < min_tools:
            n_dropped_min_tools += 1
            continue
        n_errors = sum(1 for c in calls_raw if (c[1] or "") == "error")
        total_ms = sum(int(c[2] or 0) for c in calls)
        rows.append({
            "run_id": rid,
            "user_goal": goal,
            "tools_called": tools_called,
            "n_tool_calls": len(calls),
            "n_tool_calls_raw": len(calls_raw),
            "n_errors": n_errors,
            "cost_usd": float(cost or 0.0),
            "tokens_in": int(tin or 0),
            "tokens_out": int(tout or 0),
            "total_duration_ms": total_ms,
            "outcome": outcome or "untagged",
        })
    conn.close()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return {
        "n_written": len(rows),
        "n_runs_seen": n_runs_seen,
        "n_dropped_no_goal": n_dropped_no_goal,
        "n_dropped_no_calls": n_dropped_no_calls,
        "n_dropped_min_tools": n_dropped_min_tools,
        "filters": {
            "drop_failed": drop_failed,
            "dedup_window_s": dedup_window_s,
            "min_tools": min_tools,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--out", default="data/afr_replay/sessions.jsonl")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--drop-failed", action="store_true",
                    help="Drop tool_calls rows with status='error' before counting.")
    ap.add_argument("--dedup-window-s", type=float, default=0.0,
                    help="Collapse same-tool calls within N seconds. 0 disables.")
    ap.add_argument("--min-tools", type=int, default=1,
                    help="Drop runs whose tools_called list is shorter than this.")
    ap.add_argument("--filtered", action="store_true",
                    help="Shortcut: enable the recommended filter set "
                         "(drop_failed=True, dedup_window_s=30, min_tools=1) and write "
                         "to sessions_filtered.jsonl by default.")
    args = ap.parse_args()

    if args.filtered:
        if args.out == "data/afr_replay/sessions.jsonl":
            args.out = "data/afr_replay/sessions_filtered.jsonl"
        args.drop_failed = True
        if args.dedup_window_s == 0.0:
            args.dedup_window_s = 30.0

    summary = extract(
        Path(args.db).expanduser(),
        Path(args.out),
        limit=args.limit,
        drop_failed=args.drop_failed,
        dedup_window_s=args.dedup_window_s,
        min_tools=args.min_tools,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
