# rl-mcp-tool-gate

**An RL-trained pre-flight gate that prunes MCP tool catalogs to a query-relevant subset, shipped as an MCP proxy.**

Claude Code / Cursor users routinely load 30–200 MCP tools into context. Beyond ~20 tools, agent accuracy drops from context bloat and tool-name confusion. This project trains a tiny bi-encoder with GRPO to pick the relevant handful of tools per query, *before* the agent sees them.

## Headline numbers (100 held-out queries, 99-tool catalog)

| Method | recall@5 | recall@12 | catastrophic-fail@8 | latency/turn |
|--------|:--------:|:---------:|:-------------------:|:------------:|
| Random top-k | 0.045 | 0.113 | 0.92 | <1 ms |
| BM25 | 0.484 | 0.537 | 0.56 | <5 ms |
| Qwen2.5-1.5B as router | 0.559 | — | 0.52 | ~400 ms (GPU) |
| BGE-small (off-the-shelf) | 0.673 | 0.793 | 0.36 | ~30 ms |
| **BGE-small + GRPO (ours)** | **0.810** | **0.887** | **0.19** | ~30 ms |

The RL-tuned gate beats the off-the-shelf encoder at **every** budget on the recall-vs-context Pareto curve, nearly **halves catastrophic failures** (dropping a tool the agent needed), and beats the open-weights LLM router by 25 recall points at ~1/13th the latency. Training cost: **~$3** on a Modal A10G. See `results/pareto.png`.

> "Catastrophic failure" = the gate dropped a tool the query actually required. This is the metric that matters: an over-inclusive gate is mildly wasteful, but a gate that drops a needed tool makes the agent fail outright. The reward is shaped asymmetrically to reflect this.

## Architecture

```
User query ─▶ MCP Tool Gate (proxy) ─▶ pruned tool list ─▶ Agent (Claude Code / Cursor / …)
                     │
                     ├── gate/   BGE-small (RL-tuned) + token-budget knapsack — pure, <30 ms
                     ├── proxy/  multiplexes upstream MCP servers, applies the gate to tools/list
                     └── (B2) optional Claude Code UserPromptSubmit hook feeds the user query
                              over a localhost control channel for crisp signal
```

- **`src/gate/`** — bi-encoder + token-budget knapsack. The pure inference path.
- **`src/train/`** — custom GRPO loop, Gumbel-top-k sampler, reward function.
- **`src/eval/`** — metrics, baselines (random, BM25, off-the-shelf BGE, Qwen2.5-1.5B), `agent-flight-recorder` replay.
- **`src/proxy/`** — MCP server that multiplexes upstream MCPs and applies the gate.
- **`hooks/`** — Claude Code `UserPromptSubmit` hook (B2 mode).

## Method

- **Bi-encoder (BGE-small) + GRPO**, not a generative LLM selector — inference must be fast enough to run on every turn (~30 ms vs ~400 ms for an LLM router).
- **Gumbel-top-k sampling** gives proper log-probabilities for sampling-without-replacement, so the subset policy is differentiable.
- **Group-normalized advantage (GRPO)** over N stochastic subsets per query — no learned value head.
- **Score temperature matters for exploration.** Cosine scores are scaled by ~10 before adding Gumbel noise: too sharp (×20) and the ground-truth tool is always sampled (no reward variance, no signal); too flat (×5) and it's never sampled. ×10 puts GT in ~75% of samples, maximizing the learning signal.
- **Full bi-encoder gradient.** Encoding the catalog with gradient every step (not just the query side) was the difference between a flat reward curve and one that climbed from −0.65 to +0.70.
- **Asymmetric reward**: recall-dominant + a hard penalty when a critical tool is dropped. The hard penalty prevents the canonical subset-selection reward hack (collapse to the empty set).
- **Synthetic for training, real for eval.** Training uses 400 synthetic queries (50 hand-written seeds × Qwen-augmented variations); held-out eval uses 100 hand-written queries with no augmentation (leakage prevention). `agent-flight-recorder` session replay is wired in as a real-world eval for when sessions accumulate.

Full design + decisions: `docs/superpowers/specs/2026-05-19-rl-mcp-tool-gate-design.md` (parent workspace).

## Quickstart

```bash
pip install -e ".[dev]"
python -m src.data_gen.pull_schemas        # build the 99-tool catalog
python -m src.data_gen.seed_queries        # 50 seeds
modal run src/data_gen/bootstrap.py        # ~$0.20 -> 400 train queries
python -m src.data_gen.heldout             # 100 held-out queries
bash scripts/train.sh smoke                # verify reward trends up (~$0.30)
bash scripts/train.sh full                 # ~$3, full GRPO run
modal volume get rl-mcp-gate-ckpts run1 ./checkpoints/run1
bash scripts/eval.sh                       # ~$2, Pareto + baselines
```

## Use it as an MCP proxy

1. Copy `upstreams.toml.example` to `upstreams.toml`, list the MCP servers you want to wrap.
2. Add to Claude Code's `~/.claude.json`:
   ```json
   {"mcpServers": {"tool-gate": {"command": "python", "args": ["-m", "src.proxy.server", "--config", "upstreams.toml"], "cwd": "/path/to/rl-mcp-tool-gate"}}}
   ```
3. Optional (B2 crisp-signal mode): register `hooks/user_prompt_submit.py` as a `UserPromptSubmit` hook in `~/.claude/settings.json`.
4. Restart Claude Code; `/mcp` should show `tool-gate` connected.

## Reproducibility

All baselines are open-weights or classical (no paid APIs). The LLM-router baseline (Qwen2.5-1.5B-Instruct) runs on the same Modal GPU as training. Total spend to reproduce end-to-end: **~$6** of Modal compute.

## Tests

```bash
python -m pytest -q     # 32 tests
```
