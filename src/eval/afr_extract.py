"""Extract labeled session data from agent-flight-recorder SQLite.

If your afr install has a different schema, adapt the SQL constants below.
"""
from __future__ import annotations
import argparse
import json
import sqlite3
from pathlib import Path

_SESSION_SQL = "SELECT id, started_at, ended_at FROM sessions ORDER BY started_at DESC LIMIT ?"
_EVENTS_SQL = "SELECT ts, kind, payload_json FROM events WHERE session_id = ? ORDER BY ts ASC"


def extract(db_path: Path, out_path: Path, limit: int = 100) -> int:
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    sessions = cur.execute(_SESSION_SQL, (limit,)).fetchall()
    rows = []
    for sid, started, ended in sessions:
        events = cur.execute(_EVENTS_SQL, (sid,)).fetchall()
        user_prompts, tool_calls = [], []
        for ts, kind, payload in events:
            try:
                pl = json.loads(payload) if payload else {}
            except Exception:
                pl = {}
            if kind == "user_prompt":
                user_prompts.append(pl.get("text", ""))
            elif kind == "tool_call":
                name = pl.get("tool_name") or pl.get("name")
                if name:
                    tool_calls.append(name)
        if not user_prompts or not tool_calls:
            continue
        outcome = "needed_followup" if len(user_prompts) > 1 else "ok"
        rows.append({
            "session_id": sid,
            "initial_prompt": user_prompts[0],
            "all_prompts": user_prompts,
            "tools_called": list(dict.fromkeys(tool_calls)),
            "n_tool_calls": len(tool_calls),
            "outcome": outcome,
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", default="data/afr_replay/sessions.jsonl")
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()
    n = extract(Path(args.db).expanduser(), Path(args.out), limit=args.limit)
    print(f"Extracted {n} sessions to {args.out}")


if __name__ == "__main__":
    main()
