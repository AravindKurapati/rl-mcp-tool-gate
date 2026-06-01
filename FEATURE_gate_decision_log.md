# FEATURE: Gate decision logger

Status: draft
Author: Aravind
Date: 2026-06-01

## Why

Today the proxy gates the catalog but does not record what it did. We can see
`tools_called` later in afr.db, but we cannot answer:

- What did the gate surface for that query?
- Was the tool the agent actually called even in the surfaced set, or did the
  agent fall back to a built-in because the right MCP tool was pruned?
- How often does the gate prune a tool that the agent later needed in a
  follow-up turn?

Without logging, every future eval improvement (Item 3 ground-truth v2, the
bandit pivot, latency tuning) starts from "we have to instrument first." This
ships that instrumentation once.

## Scope

In: log every `tools/list` interception and every `tools/call`, link them by
session, persist locally as SQLite, and add an offline pairing query that joins
decisions to calls.

Out: any change to selection behavior, any UI, any cloud sink.

## Storage

New file: `~/.tool_gate/decisions.db` (gitignored). Kept separate from
`~/.afr/afr.db` so afr stays single-purpose and the two can be joined at eval
time on `(source, session_id)`.

### Schema

```sql
CREATE TABLE gate_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    decision_idx INTEGER NOT NULL,    -- 0-based per session
    timestamp REAL NOT NULL,           -- unix seconds, monotonic per session
    signal_kind TEXT NOT NULL,         -- 'query' (B2 hook) or 'history' (B1)
    signal_text TEXT NOT NULL,
    catalog_hash TEXT NOT NULL,        -- sha1 of sorted full catalog names
    catalog_size INTEGER NOT NULL,
    top_k INTEGER NOT NULL,
    budget_tokens INTEGER NOT NULL,
    selected_json TEXT NOT NULL,       -- JSON list of {name, score, tokens}
    encoder_ckpt TEXT,                 -- path or 'off_the_shelf'
    latency_ms REAL NOT NULL,
    UNIQUE(session_id, decision_idx)
);

CREATE TABLE tool_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    decision_idx INTEGER,              -- nullable: most recent decision at call time
    timestamp REAL NOT NULL,
    tool_name TEXT NOT NULL,
    was_surfaced INTEGER NOT NULL,     -- 0/1, did the most recent decision include this tool
    status TEXT,                        -- ok | error | unknown
    duration_ms INTEGER
);

CREATE INDEX idx_decisions_session ON gate_decisions(session_id);
CREATE INDEX idx_calls_session ON tool_calls(session_id);
```

`was_surfaced` is computed at write time from the most recent decision in the
same session. It is the single most useful column for the bandit reward.

### Session ID

The MCP stdio transport gives no stable client-side ID. We derive one as:

```
session_id = sha1(server_boot_uuid + "::" + first_request_timestamp)[:16]
```

A new session ID is minted on the first `tools/list` after every proxy boot.
Inside a single proxy lifetime, all requests belong to one session. This is
correct for Claude Code (one proxy process per Code window) and good enough for
Codex.

## Wire-up

`src/proxy/server.py` gets a `DecisionLog` instance constructed in
`build_server`. Two call sites:

1. `list_tools`: before returning, append a row with the signal, selected
   subset, and timing. Increment `decision_idx`.
2. `call_tool`: at entry, look up the most recent decision in this session,
   set `was_surfaced`, then write the row. After `mux.call` returns, update
   the row with `status` and `duration_ms`.

All writes go through an async queue so the hot path stays non-blocking. On
exceptions the logger swallows and logs to stderr; it must never break the
proxy.

## Offline pairing

New module `src/eval/decision_log.py`:

- `load_decisions(db) -> DataFrame` of `gate_decisions`
- `load_calls(db) -> DataFrame` of `tool_calls`
- `pair_with_afr(decisions, afr_db) -> DataFrame` joins on `session_id` when
  the proxy's session corresponds to an afr run (afr writes its own `run_id`
  per Claude Code window; we link via the proxy boot timestamp landing inside
  a run's `started_at..ended_at`)
- `bandit_dataset(decisions, calls) -> list[dict]` materializes one row per
  decision with `(signal, candidates, surfaced, called, was_surfaced)` for
  any future bandit training

## Privacy

`signal_text` and `selected_json` are personal. The DB stays local, gitignored
at `~/.tool_gate/`. Add `.tool_gate/` to repo `.gitignore` if it ever lives
inside the project (it does not by default).

## Tests

`tests/unit/test_decision_log.py`:
- Schema round-trip: insert + select
- `was_surfaced` correctness across multiple decisions in one session
- Session ID stability within a proxy lifetime, change across boots
- Async write does not block on a stalled DB (mock with a sleeping connection)
- Exception in logger does not propagate to caller

`tests/integration/test_proxy_logging.py`:
- Spin up the proxy against a fake upstream, fire 1 `tools/list` + 2
  `tools/call`, assert 1 decision row + 2 call rows with correct
  `was_surfaced` flags

## Schema impact

This is the project's first SQLite store. Adds `SCHEMA.md` at the repo root
documenting both tables. No changes to afr.db. No migration needed (fresh DB
on first run).

## Rollout

The logger writes to a new DB nobody depends on yet. Safe to ship behind a
config flag `enable_decision_log` (default on for local dev, easy to disable).

## Risks

- **Hot path latency.** Async write means the `list_tools` response does not
  wait on disk. Worst case the queue fills and we drop a row, which is
  acceptable.
- **Session linkage to afr is best-effort.** If a Claude Code window spans
  multiple proxy boots (unlikely), pairing falls back to "most recent boot
  inside the afr run window."
- **DB grows.** Estimate ~1 KB per decision + ~0.5 KB per call. At 100
  decisions/day that is ~50 MB/year. Add a `--prune-older-than` CLI later
  if needed.

## Followups

- Wire `bandit_dataset` into the future bandit trainer
- Add `--session-id` flag to the proxy for tests
- Expose a tiny `tool_gate inspect <session_id>` CLI for debugging
