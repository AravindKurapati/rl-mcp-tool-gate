# rl-mcp-tool-gate — End-to-End Pipeline

This document explains the whole system: the problem it solves, the architecture,
how data flows through it at training time and at serving time, every component,
and how to run and use it. The top-level `README.md` is the elevator pitch; this
is the deep dive.

---

## 1. The problem

Agents (Claude Code, Cursor, etc.) connect to MCP servers, and each server exposes
a set of tools. Once you wire up a handful of servers — filesystem, memory, github,
a database, a search API — the agent is staring at **30–200 tool definitions** on
*every single turn*.

Two things go wrong past ~20 tools:

1. **Context bloat.** Every tool's name, description, and JSON input schema is
   re-sent to the model each turn. That is tokens spent before the user's request
   is even considered.
2. **Tool confusion.** With many similar tools (`read_file`, `read_text_file`,
   `read_multiple_files`, …) the model picks the wrong one or invents arguments.
   Selection accuracy measurably degrades.

**The fix:** put a gate *in front of* the tool catalog that, given the current
query, forwards only the handful of tools that are actually relevant. The agent
never sees the rest.

The hard part is doing the selection **well** and **fast**:
- *Well* = never drop a tool the agent actually needed (a "catastrophic failure").
- *Fast* = it runs on every turn, so it must be ~tens of milliseconds, not the
  ~400 ms an LLM-as-router would cost.

This project trains a tiny bi-encoder with reinforcement learning to do exactly
that, and ships it as a drop-in MCP proxy.

---

## 2. Architecture at a glance

```
                          ┌─────────────────────────────────────────────┐
   user query             │   rl-mcp-tool-gate (MCP proxy)               │
       │                  │                                              │
       │  (B2 hook)        │   ┌────────────┐      ┌──────────────────┐  │
       └─────────────────▶│──▶│  control   │      │   gate/          │  │
         POST /query      │   │  channel   │─────▶│  BGE-small (LoRA) │  │
                          │   │ (FastAPI)  │ sig  │  + budget knapsack│  │
   Claude Code            │   └────────────┘      └────────┬─────────┘  │
       │                  │                                │ subset      │
       │  tools/list      │                                ▼             │
       │─────────────────▶│   ┌──────────────────────────────────────┐  │
       │  (pruned list)   │   │  proxy/  multiplexer                  │  │
       │◀─────────────────│   │  spawns + speaks MCP to upstreams     │  │
       │                  │   └───────┬───────────────┬──────────────┘  │
       │  tools/call      │           │               │                 │
       │─────────────────▶│──────────▶│               │                 │
       └──────────────────┘     ┌─────▼─────┐   ┌─────▼─────┐           │
                                │filesystem │   │  memory   │  …upstream │
                                │ MCP server│   │ MCP server│   servers  │
                                └───────────┘   └───────────┘            │
```

Source layout:

| Package | Role |
|---------|------|
| `src/gate/` | The inference path: BGE-small bi-encoder (optionally LoRA-tuned) + token-budget knapsack. Pure, ~30 ms. |
| `src/train/` | The custom GRPO loop, Gumbel-top-k sampler, reward function, Modal training entrypoint. |
| `src/data_gen/` | Builds the tool catalog and the query datasets (seeds, augmented train set, held-out eval set). |
| `src/eval/` | Metrics, baselines (random, BM25, off-the-shelf BGE, Qwen2.5-1.5B router), Pareto plot, agent-flight-recorder replay. |
| `src/proxy/` | The MCP server: multiplexes upstream MCP servers and applies the gate to `tools/list`. |
| `hooks/` | The Claude Code `UserPromptSubmit` hook (B2 mode — feeds the real user query to the gate). |

---

## 3. The core idea: why a bi-encoder + RL

**Why a bi-encoder, not an LLM router?**
A generative model that reads the catalog and outputs "use tools X, Y, Z" works,
but costs ~400 ms/turn on a GPU. A bi-encoder embeds the query once and the
catalog once (cached), then selection is a dot-product + sort — ~30 ms on CPU.
That latency budget is what makes per-turn gating viable.

**Why RL (GRPO), not just off-the-shelf embeddings?**
An off-the-shelf BGE-small already does decent retrieval (recall@5 0.673). But the
objective we care about is *not* generic semantic similarity — it is "select a
**subset** under a budget that contains every required tool." That is a
combinatorial, asymmetric objective:
- Including an extra irrelevant tool is mildly wasteful.
- Dropping a required tool makes the agent fail outright.

You cannot express that asymmetry with a contrastive embedding loss. You *can*
express it as a **reward** over sampled subsets, and optimize it with policy
gradient. That is what lifts recall@5 from 0.673 → **0.810** and nearly halves
catastrophic failures (0.36 → **0.19**).

---

## 4. Training pipeline (offline, one-time, ~$6 on Modal)

### 4.1 Build the tool catalog — `src/data_gen/pull_schemas.py`
Assembles a **99-tool catalog** spanning 21 MCP servers (filesystem, github,
slack, postgres, …). Each entry is a dict:

```python
{
  "name": "filesystem.read_file",     # server.tool
  "server": "filesystem",
  "tool": "read_file",
  "description": "...",
  "args": {...},
  "embed_text": "filesystem.read_file: Read the complete contents of a file ...",
}
```

`embed_text` is what the encoder actually embeds — name + description.

### 4.2 Build the query datasets
Three datasets, with **strict train/eval separation to prevent leakage**:

- `seed_queries.py` — 50 hand-written seed queries, each labelled with its
  ground-truth set of required tools.
- `bootstrap.py` (runs on Modal) — uses **Qwen2.5-1.5B** to paraphrase/augment the
  50 seeds into **400 training queries**, keeping the seed's ground-truth labels.
- `heldout.py` — **100 hand-written held-out queries** (10 per category), with **no
  augmentation**, used only for evaluation. These never touch training.

### 4.3 The RL objective

**Reward** (`src/train/reward.py`), computed for a selected subset `S` against
ground-truth `GT`:

```
reward(S) =  1.0 · recall(S, GT)               # reward covering required tools
           − 0.5 · (|S| / |catalog|)           # penalize context cost
           − 1.0 · 1[GT ⊄ S]                    # HARD penalty if any required tool dropped
           + 0.1 · 1[|S| ≤ k_target]           # small bonus for staying within budget
```

The hard `−1.0` missing-critical term is what makes the reward **asymmetric** and
encodes "a dropped tool is catastrophic." It also prevents the classic
subset-selection reward hack where the policy collapses to the empty set (empty set
→ recall 0, and the hard penalty fires).

**Sampling** (`src/train/sampler.py`) — `gumbel_top_k_sample`. To do policy
gradient over *subsets*, we need (a) stochastic subsets and (b) their
log-probabilities. Gumbel-top-k gives both: add Gumbel noise to the scores, take
top-k, and compute the exact log-prob under sampling-without-replacement
(sequential softmax-without-replacement). This makes the subset policy
differentiable.

**Update** (`src/train/grpo.py`) — `grpo_step`. Group Relative Policy Optimization:
for one query, sample `n_samples` subsets, score each with the reward, then use the
**group-normalized advantage** `(r − mean) / (std + ε)` — no learned value head.
The policy-gradient loss is `−(advantage.detach() · log_prob).mean()`, plus an
optional KL term to a frozen reference encoder to keep the tuned model from
drifting too far.

### 4.4 Two non-obvious lessons baked into the code

These were the difference between a flat reward curve and a working model:

1. **Score temperature controls exploration** (`score_scale`, default `10.0` in
   `grpo.py`). Cosine sims are tiny (≈ −1..1); they must be scaled before Gumbel
   noise. Too sharp (×20) → the top tools are *always* sampled → zero reward
   variance → no gradient signal. Too flat (×5) → the right tool is *never*
   sampled. **×10** puts the ground-truth tool in ~75 % of samples — maximum
   learning signal. (Found empirically; see commit history.)

2. **Encode the catalog *with* gradient every step.** A naive implementation
   embeds the catalog once under `no_grad` and only back-props through the query
   side. That only trains half the bi-encoder and the reward curve stays flat.
   Re-encoding the catalog *with* gradient each step (in `modal_train.py`) was what
   moved the reward EMA from **−0.65 → +0.70**.

### 4.5 Where it runs — `src/train/modal_train.py`
Training runs on a **Modal A10G GPU**. The base model is `BAAI/bge-small-en-v1.5`
(33M params) with a **LoRA adapter** (r=16, α=32, on the attention `query/key/value`
projections) via PEFT — so we only train ~1–2 MB of adapter weights. A frozen copy
of the base model is the KL reference. Output (the LoRA adapter) is written to a
Modal volume `rl-mcp-gate-ckpts` under `run1/`.

- `smoke` entrypoint: short run to confirm the reward trends up (~$0.30).
- `full` entrypoint: the real run (~$3).

### 4.6 Evaluation — `src/eval/`
Held-out 100-query eval against four baselines, all open-weights/classical (no paid
APIs):

| Method | recall@5 | recall@12 | catastrophic-fail@8 | latency/turn |
|--------|:--------:|:---------:|:-------------------:|:------------:|
| Random top-k | 0.045 | 0.113 | 0.92 | <1 ms |
| BM25 | 0.484 | 0.537 | 0.56 | <5 ms |
| Qwen2.5-1.5B as router | 0.559 | — | 0.52 | ~400 ms (GPU) |
| BGE-small (off-the-shelf) | 0.673 | 0.793 | 0.36 | ~30 ms |
| **BGE-small + GRPO (ours)** | **0.810** | **0.887** | **0.19** | ~30 ms |

`pareto.py` produces the recall-vs-context Pareto curve (`results/pareto.png`); the
RL gate dominates the off-the-shelf encoder at every budget **on the synthetic
held-out set** (in-distribution).

### 4.7 Real-traffic eval — and a generalization gap (important, honest)

`afr_extract.py` + `afr_eval.py` evaluate the gate on **real agent-flight-recorder
traffic** (90 usable Claude Code runs). Method: signal = the run's `user_goal`,
ground truth = the tools the run actually called, catalog = the synthetic catalog
**augmented** with the real tools observed in traffic (distractors). The gate's real
domain is MCP tools (built-ins like Bash/Read are always available and never gated),
so the primary block restricts to the 26 runs that used ≥1 MCP tool.

**Finding: the RL-tuned encoder is *worse* than off-the-shelf BGE on real traffic** —
the reverse of the synthetic result. Recall@8: off-the-shelf **0.30** vs RL **0.07**;
non-breaking rate@12: off-the-shelf **0.27** vs RL **0.15**. This is a classic
**overfitting / distribution-shift** result: the LoRA specialized to the synthetic
catalog's phrasing and embed-text format and lost generality the base BGE retains.
Real goals are long and messy and the real MCP tools were never in training.

This is reported as-is, not hidden. The headline synthetic numbers in `README.md`
are valid **in-distribution**; they do not transfer to this user's real traffic. The
real-traffic eval exists precisely to catch this. Fix paths (not yet applied): train
on real-format tool names + messier queries, lower LoRA rank / stronger KL, and
re-examine whether the synthetic eval is inflated by shared train/eval distribution.

`results/afr_eval.json` carries both the MCP-domain block and an `all_tools_session_level`
block (a deliberately harsh, transparent baseline that conflates a whole multi-turn
session's tools — including ungated built-ins — into one selection).

### 4.8 Cost / latency dashboard

`dashboard.py` measures, on the same real traffic: per-turn gate latency (p50/p95),
tokens saved/turn (full catalog vs gated), and a $ projection grounded in afr's real
`cost_usd`/`tokens_in`. Writes `results/dashboard.md` + `.png`.

---

## 5. Serving pipeline (online, every turn, ~30 ms)

This is what runs when the proxy is registered in Claude Code.

### 5.1 The inference path — `src/gate/`

`select_tools(...)` in `src/gate/select.py` is a pure function:

```
signal (query) ──▶ encoder.score(signal) ──▶ cosine sims vs cached catalog embeds
                                          ──▶ argsort (rank tools)
                                          ──▶ select_under_budget(ranked, top_k, budget_tokens)
                                          ──▶ pruned subset
```

- `GateEncoder` (`encoder.py`) wraps BGE-small. On load, if a LoRA checkpoint path
  is given, it wraps the base model with `PeftModel` — that is the RL-tuned variant.
  The catalog is embedded **once** at startup (`precompute_catalog`) and cached, so
  per-query work is just one query embed + a dot product.
- `select_under_budget` (`budget.py`) walks the ranked list and greedily admits
  tools until it hits either `top_k` (count cap) or `budget_tokens` (a token
  knapsack, where each tool's cost ≈ `len(embed_text)/4`).

### 5.2 The proxy — `src/proxy/`

`server.py` builds an MCP `Server` named `rl-mcp-tool-gate` and:

1. **`build_server`** loads `upstreams.toml`, starts the **multiplexer**, loads the
   encoder + LoRA checkpoint, embeds the catalog. (Prints status to **stderr** —
   stdout is reserved for the JSON-RPC stream.)
2. **`@server.list_tools()`** — on every `tools/list`, computes the current signal
   and returns only the gated subset (as MCP `Tool` objects). *This is the gate.*
3. **`@server.call_tool()`** — forwards the call to the right upstream via the
   multiplexer, records the tool in history, clears the one-turn query signal, and
   re-emits `notifications/tools/list_changed` so the client re-lists with a fresh
   gate decision for the next step.

`multiplex.py` — `Multiplexer` spawns each upstream MCP server as a stdio
subprocess, speaks MCP to it, and namespaces its tools as `server.tool`. Lifecycle
is owned by a single `AsyncExitStack` entered/exited in the same task (so anyio
cancel scopes tear down cleanly). Call routing uses a name index that tolerates both
`server.tool` and the `server_tool` form (some clients sanitize `.` → `_`).

`config.py` — parses `upstreams.toml`. Resolves a relative `gate_checkpoint`
against the **config file's directory** (not the process cwd), because Claude Code
spawns the server from the project root, not the server's folder.

### 5.3 The signal: B1 vs B2

The gate needs to know *what the user is trying to do*. There are two signal modes
(`GateState.signal()` in `server.py`):

- **B1 (default).** An MCP server can't see the user's prompt — only tool calls. So
  the signal is the accumulated **tool-call history** for the turn. Works with zero
  setup, but the first list of a turn has weak signal.
- **B2 (crisp signal).** The Claude Code `UserPromptSubmit` hook
  (`hooks/user_prompt_submit.py`) fires on every prompt and POSTs the prompt text
  to the proxy's localhost **control channel** (`control.py`, FastAPI on
  `127.0.0.1:17800`). The next `tools/list` then gates against the *actual user
  query*. The hook is fire-and-forget with a 0.5 s timeout: if the proxy is down it
  silently no-ops and the system degrades to B1 — it never blocks your prompt.

> **Timing note (by design):** the gate is *pre-flight*. The signal used for a
> turn's first `tools/list` is whatever was known at prompt-submit time. After each
> tool call, the proxy re-emits `tools/list_changed`, so the set refines *between*
> steps within a turn.

### 5.4 B2 for clients other than Claude Code

The control channel (`POST /query`) is plain HTTP, so any client can feed the
signal. `hooks/user_prompt_submit.py` is Claude-Code-specific (it parses Claude
Code's hook JSON). For everything else use the generic shim
`hooks/generic_signal.py`, which accepts the query from a flag, positional args, or
stdin (raw text or JSON with a `prompt`/`query`/`text`/`message` key):

```bash
python -m hooks.generic_signal "refactor the auth middleware"
echo '{"prompt":"list my files"}' | python -m hooks.generic_signal
```

**Cursor / Continue / shell wrappers:** call the shim from whatever prompt hook or
wrapper the client exposes, passing the user's text. It is fire-and-forget (0.5 s
timeout, exits 0, no-ops if the proxy is down), so it is safe on any prompt path.
This is a generic shim plus integration notes — not turnkey Cursor parity.

---

## 6. How to run it

### 6.1 Reproduce training + eval (optional, ~$6 Modal)

```bash
pip install -e ".[dev]"
python -m src.data_gen.pull_schemas     # 99-tool catalog
python -m src.data_gen.seed_queries     # 50 seeds
modal run src/data_gen/bootstrap.py     # ~$0.20  -> 400 train queries
python -m src.data_gen.heldout          # 100 held-out queries
bash scripts/train.sh smoke             # ~$0.30  verify reward trends up
bash scripts/train.sh full              # ~$3     full GRPO run
modal volume get rl-mcp-gate-ckpts run1 ./checkpoints/run1
bash scripts/eval.sh                    # ~$2     Pareto + baselines
```

A trained adapter already lives in `checkpoints/run1/` (`adapter_model.safetensors`
+ `adapter_config.json`), so you can skip straight to serving.

### 6.2 Run the proxy standalone (no agent needed)

```bash
python scripts/smoke_proxy.py
```

Connects to the upstreams in `upstreams.toml`, lists all their tools, then prints
the gated subset for a few sample queries. This is the fastest way to confirm the
multiplex + gate path works end-to-end.

### 6.3 Use it inside Claude Code

**`upstreams.toml`** (the live config) declares which MCP servers to wrap:

```toml
budget_tokens = 4000
top_k = 8
gate_checkpoint = "checkpoints/run1"     # resolved relative to THIS file
control_channel_port = 17800

[[upstreams]]
name = "filesystem"
command = ["npx", "-y", "@modelcontextprotocol/server-filesystem", "C:/Users/user"]

[[upstreams]]
name = "memory"
command = ["npx", "-y", "@modelcontextprotocol/server-memory"]
```

**Register the proxy** in `~/.claude.json`. It can be global (`mcpServers` at the
top level) or scoped to one project (`projects["<path>"].mcpServers`). Use an
**absolute** `--config` path — Claude Code spawns the server from the project root,
not the server's folder, so a relative path won't resolve:

```json
"tool-gate": {
  "type": "stdio",
  "command": "python",
  "args": ["-m", "src.proxy.server", "--config", "D:/Aru/NYU/rl-mcp-tool-gate/upstreams.toml"],
  "env": { "PYTHONIOENCODING": "utf-8" }
}
```

> Register **only** the gate proxy — not the upstream servers directly. The whole
> point is that the gate fronts them; exposing them raw would defeat it.

**Register the B2 hook** (optional but recommended) in `~/.claude/settings.json`,
appended alongside any existing `UserPromptSubmit` hooks:

```json
"UserPromptSubmit": [
  {
    "matcher": "*",
    "hooks": [
      { "type": "command", "command": "python \"D:\\Aru\\NYU\\rl-mcp-tool-gate\\hooks\\user_prompt_submit.py\"", "timeout": 5 }
    ]
  }
]
```

**Then:** restart Claude Code, run `/mcp` → `tool-gate` should be connected, and
expanding it shows ~8 gated tools instead of the full upstream catalog. Ask
"read the file notes.txt" vs "remember my favorite color is blue" and watch the
surfaced set shift toward `filesystem.*` vs `memory.*`.

---

## 7. Gotchas worth knowing (learned the hard way)

| Symptom | Cause | Fix (already in the code) |
|---------|-------|---------------------------|
| `/mcp` shows `tool-gate` failed, `-32000` | A status line was printed to **stdout**, which is the JSON-RPC channel | All logging goes to **stderr** (`server.py`) |
| `FileNotFoundError: upstreams.toml` | Claude Code spawns from the project root, ignoring the config `cwd` field | Pass an **absolute** `--config` path |
| Gate behaves like off-the-shelf (no RL benefit) | Relative `gate_checkpoint` didn't resolve from the spawn cwd, so the LoRA silently didn't load | `config.py` resolves it relative to the config file |
| Tool *call* fails after the list works | Client sanitizes `server.tool` → `server_tool` on call | Multiplexer name index covers both forms |
| Teardown `RuntimeError: exit cancel scope in a different task` | anyio cancel scopes from `stdio_client` torn down across tasks | Single `AsyncExitStack` entered/exited in one task |
| `peft` load raises `AttributeError` on `torch.distributed.tensor.DTensor` | peft ≥0.19 accesses the lazy submodule without importing it (torch ≥2.10) | Force-import `torch.distributed.tensor` before `PeftModel.from_pretrained` |

---

## 8. Test suite

```bash
python -m pytest -q       # 32 tests
```

Covers the reward function, the Gumbel-top-k sampler (incl. log-prob correctness),
the budget knapsack, the encoder, the GRPO step, config parsing, the multiplexer,
and a hook→proxy control-channel integration test.
