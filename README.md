# rl-mcp-tool-gate

A pre-flight gate that prunes MCP tool catalogs to the handful a query actually needs, shipped as an MCP proxy. The project started as an RL experiment and ended up as an honest comparison between GRPO, off-the-shelf retrieval, and a plain supervised baseline.

## The problem

Claude Code and Cursor users routinely load 30 to 200 MCP tools into context. Past about 20 tools, agent accuracy drops from context bloat and tool-name confusion. The gate sits in front of the agent and prunes the tool list per query so the model only sees what it needs.

## The story behind the numbers

I tried three things on the same BGE-small bi-encoder, in this order:

1. **v1 GRPO on synthetic data.** Reported recall@5 of 0.81 against an off-the-shelf BGE baseline of 0.67. Looked like a clean win.
2. **Real-traffic eval against `afr.db`** (a session-recording system I built for Claude Code and Codex). Once I tested against 26 real MCP-using runs, v1 RL fell apart. Recall@8 on real traffic was 0.07, versus 0.30 for off-the-shelf. The RL model had overfit the synthetic distribution.
3. **v2 RL with distribution-matched data and stronger regularization.** Recall@8 on real traffic recovered to 0.228, roughly tied with off-the-shelf at 0.223.
4. **SFT contrastive baseline.** Same model, same data, same split as v2, only the loss changed (full-catalog multi-positive InfoNCE). It beat both RL and off-the-shelf decisively, on synthetic and on real traffic.

The conclusion was the opposite of what I started with: for static per-query tool selection, plain supervised retrieval extracts more signal than GRPO. The reward shaping that made v1 RL win on synthetic data was the same thing that made it overfit, and the regularization needed to fix v1 left v2 underfitting.

## Headline numbers

**Real traffic (26 MCP-using runs from `afr.db`, 114-tool catalog)**

| k | Off-the-shelf BGE | RL v2 | SFT |
|---|:-----------------:|:-----:|:---:|
| 3 | 0.038 | 0.071 | **0.163** |
| 5 | 0.111 | 0.157 | **0.279** |
| 8 | 0.223 | 0.228 | **0.455** |
| 12 | 0.313 | 0.237 | **0.545** |
| 20 | 0.377 | 0.372 | **0.631** |

Non-breaking rate at k=8 (fraction of runs where every needed tool was surfaced) went from 0.077 off-the-shelf to 0.346 with SFT, a 4.5x improvement on the metric that matters for production.

**Synthetic eval (heldout_v2, 114 tools)**

| k | Off-the-shelf | RL v2 | SFT |
|---|:-------------:|:-----:|:---:|
| 5 | 0.670 | 0.717 | **0.804** |
| 8 | 0.720 | 0.785 | **0.870** |
| 12 | 0.756 | 0.853 | **0.903** |

Equal subset budgets per k, harness reproduces the off-the-shelf baseline bit-identically across runs. Full sanity checks in `docs/RESULT_sft_baseline.md`.

## Architecture

The proxy multiplexes upstream MCP servers and applies the gate to `tools/list`. The gate itself is a pure inference path: encode the query, score every tool in the catalog, run a token-budget knapsack to pick the subset.

```
Claude Code or Cursor
       |
       v
MCP Tool Gate proxy  (multiplexes upstream MCPs)
       |
       +-- gate/    BGE-small encoder, knapsack, ~30 ms
       +-- proxy/   prunes tools/list per query
       +-- hooks/   optional UserPromptSubmit hook for crisp query signal
       |
       v
Upstream MCP servers (Exa, GitHub, filesystem, etc.)
```

- `src/gate/` bi-encoder plus token-budget knapsack
- `src/train/` GRPO trainer, Gumbel top-k sampler, reward; SFT contrastive loss and trainer
- `src/eval/` baselines (random, BM25, off-the-shelf BGE, Qwen2.5-1.5B router), synthetic Pareto, real-traffic eval against afr, paired cluster bootstrap for CIs
- `src/proxy/` MCP proxy that applies the gate
- `hooks/` Claude Code UserPromptSubmit hook

## What I would do differently

- Start with the SFT baseline, not the RL run. I picked GRPO first because I had GRPO experience from prior projects, which is the wrong reason to pick a method.
- Build the real-traffic eval before any training. I built it as a sanity check after v1, and that is what exposed the overfit. It should have been the first thing.
- Wire the gate as MCP middleware from day one instead of a standalone proxy. Less infrastructure to maintain.
- Stop chasing RL on a problem that is supervised-solvable. The only framing where RL would be load-bearing here is an online bandit on downstream usage signal (gate decision led to a successful agent action), not on static per-query selection. That pivot is specced out in `docs/FUTURE_bandit_pivot.md` but is on hold until the static side runs out of room.

## Quickstart

```bash
pip install -e ".[dev]"
python -m src.data_gen.pull_schemas
python -m src.data_gen.seed_queries
modal run src/data_gen/bootstrap.py
python -m src.data_gen.heldout

# SFT (the one that won)
bash scripts/train.sh sft

# or RL, for reproduction
bash scripts/train.sh full

modal volume get rl-mcp-gate-ckpts sft ./checkpoints/sft

# Synthetic Pareto
bash scripts/eval.sh

# Real-traffic eval against agent-flight-recorder DB
python -m src.eval.afr_eval --ckpt checkpoints/sft
python -m src.eval.bootstrap_afr
```

## Use it as an MCP proxy

1. Copy `upstreams.toml.example` to `upstreams.toml` and list the MCP servers you want to wrap.
2. Add to Claude Code's `~/.claude.json`:
   ```json
   {"mcpServers": {"tool-gate": {"command": "python", "args": ["-m", "src.proxy.server", "--config", "upstreams.toml"], "cwd": "/path/to/rl-mcp-tool-gate"}}}
   ```
3. Optional crisp-signal mode: register `hooks/user_prompt_submit.py` as a `UserPromptSubmit` hook in `~/.claude/settings.json`.
4. Restart Claude Code. `/mcp` should show `tool-gate` connected.

## Reproducibility

All baselines are open-weights or classical. Total Modal spend to reproduce end to end is around $5: roughly $2 for the SFT run, $3 for the GRPO run, plus a few cents for the synthetic data bootstrap.

## Tests

```bash
python -m pytest -q   # 51 tests
```

## More

- `SCHEMA.md` documents the local SQLite store written by the proxy
- Decision-log reader and bandit-dataset materializer in `src/eval/decision_log.py`
