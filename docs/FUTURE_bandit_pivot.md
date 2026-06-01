# Handoff: tool-gate RL pivot → online contextual bandit

**Written 2026-05-20. Read this first in a new session.** Self-contained; assumes no
prior conversation context.

---

## 0. TL;DR / why this doc exists

`rl-mcp-tool-gate` is an MCP proxy that prunes a large tool catalog to a query-relevant
subset before an agent sees it. We trained a BGE-small bi-encoder with GRPO. After
building a **real-traffic eval** from agent-flight-recorder, we found the RL model
**overfit** and only *matched* off-the-shelf BGE on real traffic. The honest strategic
conclusion:

> **The static per-query tool-selection problem is fundamentally supervised retrieval.
> RL (GRPO on labeled query→tool pairs) is over-engineered for it — supervised
> contrastive fine-tuning (SFT) would likely match it. RL only becomes load-bearing if
> the reward comes from a real action→feedback loop you cannot express as a label.**

The pivot that makes RL genuinely justified: **reframe tool-gating as an online
contextual bandit that learns from real agent-flight-recorder feedback** (which tools
the agent actually used). That is the work for the next session.

---

## 1. Current state (what already exists)

- **Live**: the proxy is registered in the user's Claude Code, project-scoped to
  `D:/Aru/NYU` (config in `~/.claude.json` → projects → `D:/Aru/NYU` → mcpServers →
  `tool-gate`; B2 hooks in `~/.claude/settings.json`). It currently uses
  `checkpoints/run1` per `upstreams.toml` (could switch to `run2`).
- **Trained checkpoints**: `checkpoints/run1` (v1, r=16) and `checkpoints/run2` (v2,
  r=8, the better-generalizing one). Both are LoRA adapters on `BAAI/bge-small-en-v1.5`.
- **Tests**: 46 passing (`python -m pytest -q`).
- **Eval results** (`results/`):
  - Synthetic in-distribution (v2): RL beats off-the-shelf at every budget
    (recall@8 0.785 vs 0.720, @12 0.853 vs 0.756).
  - Real afr traffic (26 MCP-using runs): v2 *matches* off-the-shelf
    (recall@8 0.228 vs 0.223), after v1 had *lost* badly (0.071). Does not decisively
    beat BGE — real goals avg ~1.8 tools (mostly retrieval, BGE's strength).
  - Dashboard: gate latency p50 ~80–110 ms, ~2,400 tokens saved/turn.

### Key files
- Gate (inference): `src/gate/{encoder,select,budget}.py`
- Training (GRPO, Modal): `src/train/{grpo,sampler,reward,modal_train}.py`
- Data: `src/data_gen/{pull_schemas,seed_queries,heldout,bootstrap,real_tools,build_v2}.py`
- Eval: `src/eval/{metrics,baselines,pareto,qwen_baseline,afr_extract,afr_catalog,afr_eval,dashboard}.py`
- Proxy: `src/proxy/{config,multiplex,server,control}.py`
- Hooks: `hooks/{user_prompt_submit,generic_signal}.py`
- Docs: `docs/PIPELINE.md` (full pipeline + honest v1→v2 story), this file.
- Specs (parent workspace): `D:/Aru/NYU/docs/superpowers/specs/2026-05-19-...md` and
  `2026-05-20-afr-eval-dashboard-tasksuccess-b2.md`.

### agent-flight-recorder DB
- Path: `C:/Users/user/.afr/afr.db` (real data, grows over time via a Stop hook).
- Schema (verified 2026-05-20):
  - `runs(id, source, project_path, started_at, ended_at, user_goal, final_summary,
    outcome, cost_usd, tokens_in, tokens_out, cache_read, cache_write)`
  - `tool_calls(id, run_id, tool_name, input_summary, output_summary, status,
    duration_ms, timestamp, raw_json)`
  - also `shell_commands`, `files`, `errors`, `lessons`.
- As of 2026-05-20: 99 runs, 3,421 tool calls, 35 distinct tools (mostly built-in;
  MCP tools = `mcp__exa__*`, `mcp__github__*`). 26 runs used ≥1 MCP tool.

---

## 2. The decision: SFT baseline vs the bandit pivot

Two separate, both-valid moves. Decide which to do (or both, in order):

### Move A — SFT baseline (cheap, ~half a day, do first)
Add a supervised contrastive fine-tune of the same BGE bi-encoder (InfoNCE/triplet:
pull query toward its ground-truth tools, push others away) and evaluate it on the
SAME synthetic + afr evals. **Purpose: honestly answer "did RL's reward-shaping buy
anything over copying the labels?"** If SFT ≈ RL, the honest headline becomes "for
static selection, labels were enough; RL's machinery was unnecessary" — which is a
*more* sophisticated result than asserting RL was needed. If RL still wins, the
asymmetric reward (catastrophic-drop penalty + budget) is proven to matter.
- New file: `src/train/sft_train.py` (Modal, contrastive loss, same catalog/data).
- Reuse `eval/pareto.py` + `eval/afr_eval.py` with `--ckpt checkpoints/sft`.

### Move B — Online contextual bandit (the real RL project)
This is the headline pivot. Make RL load-bearing by optimizing a **downstream usage
reward** that no label can express.

**Formalization:**
- **State / context**: the query signal embedding + the candidate tool catalog.
- **Action**: which subset of tools to surface (top-k under budget).
- **Reward**: from real usage — e.g.
  `+1` for each surfaced tool the agent actually called, `−penalty` for each tool the
  agent needed but the gate dropped (catastrophic), small `−cost` per surfaced tool.
- **Why it's genuinely RL**: the reward is the agent's *behavior*, not a label; it's a
  repeated decision with feedback; a bandit/policy-gradient learner is the natural fit
  and SFT cannot do it.

**The gap to close first (critical):** afr logs what the agent *called*, but NOT what
the gate *surfaced*. For a true (on-policy) bandit you must log the gate's own
decisions. Plan:
1. **Instrument the proxy** to append every gate decision to a local JSONL:
   `{ts, signal, candidate_tool_names, surfaced_subset, k, budget}`. Add this in
   `src/proxy/server.py` (`@server.list_tools()` and/or `call_tool`).
2. **Join** gate-decision logs with afr `tool_calls` by time/session to compute the
   reward per decision (surfaced-and-used vs needed-but-dropped).
3. **Offline/counterfactual warm-start**: until on-policy logs accumulate, bootstrap
   from historical afr observationally (this is ~what `afr_eval.py` already measures as
   non-breaking rate). Treat v2 as the cold-start policy.
4. **Online learner**: start simple — LinUCB / contextual ε-greedy over the bi-encoder
   embeddings, or a lightweight policy-gradient bandit. Reward = the usage signal above.
   Evaluate with replay / off-policy estimators (IPS/doubly-robust) before going live.

**First concrete tasks for the new session (Move B):**
- [ ] Verify reward feasibility: can we join gate decisions ↔ afr calls? (afr has
  `timestamp` on tool_calls and `started_at/ended_at` on runs; the proxy decision log
  needs matching timestamps + ideally the run/session id.)
- [ ] Implement gate-decision logging in the proxy (behind a config flag).
- [ ] Write `src/bandit/` : reward computation from (decision log ⋈ afr), a LinUCB or
  policy-gradient bandit, and an off-policy replay evaluator.
- [ ] Spec it first: `docs/superpowers/specs/2026-05-2x-tool-gate-bandit.md`
  (the user's workflow requires a FEATURE/spec MD before non-trivial coding, with two
  approaches + a recommendation, and approval before implementing).

---

## 3. User working preferences (carry into new session)
- Use **Exa MCP** for all search/research; do NOT use built-in web search.
- **Before non-trivial features**: plan, propose two approaches + recommend one, write a
  FEATURE/TWEAK spec MD, wait for approval. Bug → write a failing test first.
- **Ask before** destructive/irreversible actions and before **Modal spend** (user has
  ~$30 credits; each train run ~$1–3).
- Be honest and concise; the user values rigor over flattering numbers (the whole
  "RL overfit / RL over-engineered" thread came from this).
- Never read/print `.env` or secrets. afr `user_goal` text is personal — keep
  `data/afr_replay/` and `results/` gitignored (already are).
