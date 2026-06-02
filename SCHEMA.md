# Schema

## `~/.tool_gate/decisions.db`

Local SQLite store written by the MCP proxy. One row per gate decision and one
row per downstream tool call. Kept separate from `~/.afr/afr.db` so afr stays
single-purpose.

### `gate_decisions`

One row per `tools/list` interception.

| Column | Type | Notes |
|--------|------|-------|
| `id` | INTEGER PRIMARY KEY | autoincrement |
| `session_id` | TEXT | derived from proxy boot UUID + first request time |
| `decision_idx` | INTEGER | 0-based per session |
| `timestamp` | REAL | unix seconds |
| `signal_kind` | TEXT | `query` (B2 hook) or `history` (B1) |
| `signal_text` | TEXT | the user goal or accumulated history that drove selection |
| `catalog_hash` | TEXT | sha1[:12] of sorted catalog tool names |
| `catalog_size` | INTEGER | full catalog size before pruning |
| `top_k` | INTEGER | budget passed to the selector |
| `budget_tokens` | INTEGER | token budget passed to the selector |
| `selected_json` | TEXT | JSON list of `{name}` for the surfaced subset |
| `encoder_ckpt` | TEXT | LoRA path or `off_the_shelf` |
| `latency_ms` | REAL | gate selection latency |

UNIQUE: `(session_id, decision_idx)`.

### `tool_calls`

One row per `tools/call` the proxy handled.

| Column | Type | Notes |
|--------|------|-------|
| `id` | INTEGER PRIMARY KEY | autoincrement |
| `session_id` | TEXT | links back to `gate_decisions.session_id` |
| `decision_idx` | INTEGER | the most recent decision in this session at call time; NULL if no decision yet |
| `timestamp` | REAL | unix seconds |
| `tool_name` | TEXT | the tool the agent invoked |
| `was_surfaced` | INTEGER | 0 or 1: did the most recent decision include this tool |
| `status` | TEXT | `ok` or `error` |
| `duration_ms` | INTEGER | upstream call duration |

### Privacy

`signal_text` is the user's prompt or recent tool-call history. It is personal.
The DB lives under `~/.tool_gate/` and is never written to the repo tree.

### Join with afr.db

The proxy's `session_id` is independent of afr's `run_id`. To pair:

1. For each gate `session_id`, take the earliest `timestamp` (first decision).
2. Find the afr `runs` row whose `[started_at, ended_at]` contains that
   timestamp.
3. That afr run is the agent session the proxy was serving.

See `src/eval/decision_log.py` for the readers and the `bandit_dataset`
materializer.

## `data/afr_replay/sessions.jsonl`

Extracted from `~/.afr/afr.db` by `src/eval/afr_extract.py`. One JSON object per
usable afr run.

| Field | Notes |
|-------|-------|
| `run_id` | from afr `runs.id` |
| `user_goal` | from afr `runs.user_goal`, the eval signal |
| `tools_called` | deduped list of `tool_calls.tool_name` in call order |
| `n_tool_calls` | raw count before dedup |
| `n_errors` | rows with `status == 'error'` |
| `cost_usd`, `tokens_in`, `tokens_out` | from afr |
| `total_duration_ms` | sum of per-call durations |
| `outcome` | `success`, `failure`, or `untagged` |

Ground truth for the gate's real-traffic eval is `tools_called`. Ground truth
v2 (see `FEATURE_ground_truth_v2.md`) filters this set and adds a hand-rubric
gold subset.
