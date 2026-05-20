"""Extract labeled session data from agent-flight-recorder SQLite.

Real afr schema (verified 2026-05-20):
  runs(id, source, project_path, started_at, ended_at, user_goal, final_summary,
       outcome, cost_usd, tokens_in, tokens_out, cache_read, cache_write)
  tool_calls(id, run_id, tool_name, input_summary, output_summary, status,
             duration_ms, timestamp, raw_json)

Emits one JSONL row per usable run (has a goal and >=1 tool call):
  run_id, user_goal, tools_called (distinct, call-ordered), n_tool_calls,
  n_errors, cost_usd, tokens_in, tokens_out, total_duration_ms, outcome
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
    "SELECT tool_name, status, duration_ms FROM tool_calls "
    "WHERE run_id = ? ORDER BY timestamp ASC"
)


def extract(db_path: Path, out_path: Path, limit: int = 500) -> int:
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    runs = cur.execute(_RUNS_SQL, (limit,)).fetchall()
    rows = []
    for rid, goal, outcome, cost, tin, tout in runs:
        goal = (goal or "").strip()
        if len(goal) < MIN_GOAL_CHARS:
            continue
        calls = cur.execute(_CALLS_SQL, (rid,)).fetchall()
        if not calls:
            continue
        tools_called = list(dict.fromkeys(c[0] for c in calls if c[0]))
        if not tools_called:
            continue
        n_errors = sum(1 for c in calls if (c[1] or "") == "error")
        total_ms = sum(int(c[2] or 0) for c in calls)
        rows.append({
            "run_id": rid,
            "user_goal": goal,
            "tools_called": tools_called,
            "n_tool_calls": len(calls),
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
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--out", default="data/afr_replay/sessions.jsonl")
    ap.add_argument("--limit", type=int, default=500)
    args = ap.parse_args()
    n = extract(Path(args.db).expanduser(), Path(args.out), limit=args.limit)
    print(f"Extracted {n} usable runs to {args.out}")


if __name__ == "__main__":
    main()
