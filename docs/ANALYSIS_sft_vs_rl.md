# Analysis: reading the SFT-vs-RL results

**Date:** 2026-05-21
**Inputs:** `docs/RESULT_sft_baseline.md`, `results/pareto_sft.json`, `results/afr_eval_sft.json`
**Purpose:** a worksheet for *interpreting* the numbers — what they mean, what they
don't, and which follow-up questions are worth (and not worth) running. This is the
"so what" layer on top of the raw result table.

---

## 1. The one-paragraph read

A plain supervised contrastive fine-tune of BGE-small beat both the GRPO model and
off-the-shelf BGE everywhere we measured — synthetic in-distribution **and** real
out-of-distribution afr traffic. On real traffic SFT roughly **doubles** recall@8
(0.455 vs 0.223 off-shelf, 0.228 RL) and quadruples the non-breaking rate@8
(0.346 vs 0.077). Because everything but the loss was held constant, the only
explanation is the **objective**: dense per-step cross-entropy toward every label
extracts more signal than GRPO's noisy advantage on sampled subsets, especially once
v2's anti-overfit regularization (LoRA r=8, KL 0.05, early stop) is layered on.

---

## 2. Three claims the data supports, and how strongly

| Claim | Evidence | Strength |
|-------|----------|----------|
| SFT > RL > off-shelf on **synthetic** | recall@8 0.870 / 0.785 / 0.720, monotone at every k | **Strong** — large, consistent gaps across all 5 budgets |
| SFT > {RL, off-shelf} on **real OOD** | recall@8 0.455 vs 0.228 / 0.223; non-breaking@8 0.346 vs 0.077 | **Moderate** — big effect but n=26 runs (wide CI) |
| RL ≈ off-shelf on **real OOD** (the prior finding) | recall@8 0.228 vs 0.223; RL even *dips* @12 (0.237 vs 0.313) | **Strong** — bit-identical baseline confirms harness, gap is ~0 |

The synthetic claim is rock-solid. The real-traffic claim is **directionally strong
but statistically thin** — 26 runs is small, so treat "≈2×" as the headline and don't
over-precision the decimals.

---

## 3. The number that matters most, and why

**Real-traffic non-breaking rate@8: 0.077 → 0.346.**

Recall@k averages partial credit (surfaced 2 of 3 needed tools = 0.67). But an agent
that's missing *any* required tool is broken for that task. Non-breaking rate is the
fraction of runs where **all** needed tools survived the gate — the metric that maps
to "did the gate silently break the agent." Off-shelf and RL both leave ~92% of real
runs broken at k=8; SFT cuts that to ~65%. Still not great in absolute terms (the gate
is hard on real traffic), but it's the only method that moves the needle.

> When presenting this, lead with non-breaking rate, not recall. Recall is the
> academic metric; non-breaking rate is the product metric.

---

## 4. Why did RL lose? (mechanism, not just "it overfit")

Walk the causal chain — this is the part an interviewer will probe:

1. **Sparse, noisy gradient.** GRPO scores the full catalog, *samples* a top-k subset
   (Gumbel), and weights the policy-gradient by a scalar advantage. One scalar reward
   per sampled subset → high-variance, indirect signal about *which* tool belonged.
2. **SFT's gradient is dense and direct.** Cross-entropy pushes *up* every ground-truth
   logit and *down* all others, every step. No sampling variance, no credit-assignment
   gap. For a problem where the labels ARE the target, this is strictly more informative.
3. **Regularization compounded it.** v2 added LoRA r=8 + KL 0.05 + early stop
   *specifically* to stop v1's overfitting. Against an already weak gradient, that
   regularization tipped from "anti-overfit" into "underfit" — it held the model near
   the off-the-shelf init, which is exactly what the real-traffic tie shows.
4. **Net:** RL paid the full variance/credit-assignment cost of policy gradient and got
   none of its benefit, because there's no non-differentiable reward here to justify it.
   The reward *was* the label.

**The honest framing:** RL isn't broken; it's mis-applied. Policy gradient earns its
keep when the reward can't be expressed as a per-example label (downstream agent
success, human preference, multi-step credit). Static per-query tool selection is none
of those — it's supervised retrieval wearing an RL costume.

---

## 5. What this does NOT prove (guard against over-claiming)

- **Not** "RL is worse than SFT in general." Only for *this* objective (static,
  per-query, label-defined selection) on *this* data scale.
- **Not** "the asymmetric catastrophic-drop reward was useless." We proved a *symmetric*
  cross-entropy beats it here; that means the asymmetry didn't help *on this data*, not
  that it's worthless under genuine subset-selection pressure.
- **Not** "SFT generalizes well." It generalizes *better than the alternatives*, but
  0.455 recall / 0.346 non-breaking on real traffic means the gate is still wrong on a
  third-to-half of real runs. The win is relative, not absolute.
- **Not** validated at scale. 114-tool catalog, 26 real runs. The story holds at this
  scale; a 500-tool catalog with richer traffic is an open question (and the only place
  Move B could earn a mandate).

---

## 5b. Bootstrap CIs on the 26 real-traffic runs (Q1, resolved 2026-05-21)

Paired cluster bootstrap (B=10,000) over the 26 MCP runs — resample run indices,
recompute both methods on the *same* draw each iteration, read off the gap CI and a
one-sided bootstrap p-value `P(SFT ≤ off-shelf)`. Script: `src/eval/bootstrap_afr.py`,
output `results/afr_bootstrap.json`. Off-the-shelf reproduces the documented 0.223@8
exactly (used `catalog_v2.json` base — see caveat below), so the harness is validated.

**SFT vs off-the-shelf — recall@k [95% CI], gap, p**

| k | off-shelf | SFT | gap [95% CI] | p(SFT≤off) |
|---|-----------|-----|--------------|------------|
| 3 | 0.038 [0.00, 0.10] | 0.163 [0.06, 0.29] | +0.125 [+0.04, +0.23] | 0.0015 |
| 5 | 0.111 [0.03, 0.21] | 0.279 [0.13, 0.44] | +0.168 [+0.03, +0.33] | 0.0096 |
| **8** | 0.223 [0.11, 0.35] | 0.455 [0.28, 0.63] | **+0.232 [+0.06, +0.42]** | **0.0032** |
| 12 | 0.313 [0.18, 0.46] | 0.545 [0.37, 0.71] | +0.232 [+0.06, +0.42] | 0.0035 |
| 20 | 0.377 [0.24, 0.53] | 0.631 [0.46, 0.79] | +0.254 [+0.12, +0.40] | <0.0001 |

**SFT vs off-the-shelf — non-breaking rate@k**

| k | off-shelf | SFT | gap [95% CI] | p(SFT≤off) |
|---|-----------|-----|--------------|------------|
| 3 | 0.000 [0.00, 0.00] | 0.077 [0.00, 0.19] | +0.077 [+0.00, +0.19] | 0.124 |
| 5 | 0.038 [0.00, 0.12] | 0.231 [0.08, 0.39] | +0.192 [+0.04, +0.35] | 0.0039 |
| **8** | 0.077 [0.00, 0.19] | 0.346 [0.15, 0.54] | **+0.269 [+0.12, +0.46]** | **0.0003** |
| 12 | 0.154 [0.04, 0.31] | 0.423 [0.23, 0.62] | +0.269 [+0.12, +0.46] | 0.0004 |
| 20 | 0.192 [0.04, 0.35] | 0.500 [0.31, 0.69] | +0.308 [+0.15, +0.50] | <0.0001 |

**Verdict: the headline survives.** Even at n=26, the SFT gain over off-the-shelf is
statistically significant at every k≥5 for both metrics (p ≤ 0.01), and the recall@8
gap CI [+0.06, +0.42] excludes zero comfortably. The "~2× recall@8" claim (0.223→0.455)
is real, not noise.

**The one cell that's NOT significant:** non-breaking@3 (gap +0.077, CI touches 0,
p=0.12). That's a floor effect — at k=3 off-shelf gets 0/26 clean runs and SFT only 2/26,
too few successes to separate. Don't claim a k=3 non-breaking win; it's underpowered.

**Honest caveats to state alongside:**
- The per-method CIs are *wide* (recall@8 SFT spans [0.28, 0.63]) — n=26 means the
  *point* estimates are soft even though the *gap* is significant (pairing cancels the
  shared run-difficulty variance, which is why the diff CI is tighter than the marginals).
- The bootstrap p is one-sided and resamples runs i.i.d.; it doesn't model any
  correlation between runs from the same session/day. Fine as a sanity gate, not a
  publication-grade test.
- **Catalog caveat:** reproducing the documented 0.223 baseline required the
  `catalog_v2.json` base (114 tools), not the script default `catalog.json` (99). With
  the 99-tool base + 15 observed MCP tools the catalog also augments to 114 but with
  different contents, and off-shelf recall@8 comes out 0.300. Always pass
  `--catalog data/synthetic/catalog_v2.json` to match the documented eval.

## 6. Open questions, ranked by value-per-Modal-dollar

| # | Question | How to test | Worth it? |
|---|----------|-------------|-----------|
| 1 | ~~Is the real-traffic win inside the noise?~~ | ~~Bootstrap 95% CI~~ | **DONE (§5b)** — gap significant at k≥5, p≤0.01. Win is real. |
| 2 | Is RL's loss purely the regularization, or the objective? | Re-run GRPO at LoRA r=16, KL=0 (v1-style reg) and re-eval real | Maybe — isolates "objective vs reg," ~$2 |
| 3 | Does SFT keep winning at a bigger catalog? | Synthesize a 300–500 tool catalog, retrain both | Only if pitching Move B; ~$4 |
| 4 | Does score_scale (10 vs 20) change SFT? | Sweep, re-eval | **No** — already converged (val 0.956); diminishing |

**Recommended next action: Q1 only.** It's free, it's the credibility gate for the
headline, and 26 runs is small enough that the CI genuinely matters. Everything else is
optional and gated on whether you actually pursue Move B.

---

## 7. Decision checkpoint (carry-over from the spec)

The locked criterion fired: **SFT beats RL on real traffic** → "RL overfit *and* was
unnecessary; strongest honest-negative result." Consequences already recorded:

- **Move B (online bandit) is weakened, not killed.** It needs (a) a reward no label can
  express — real downstream agent success, not GT membership — and (b) an environment
  with genuine selection pressure (larger catalog, richer traffic). On current afr data,
  SFT already wins, so a bandit has no headroom to demonstrate.
- **If Move B proceeds, the baseline to beat is SFT (0.455 recall@8 real), not
  off-the-shelf.** Beating off-shelf is no longer interesting.

---

## 8. Portfolio framing (the punchline)

> "I trained an RL gate (GRPO) for MCP tool selection, built a real-traffic eval
> harness, found the RL model overfit and only tied off-the-shelf on real traffic, then
> proved a plain supervised baseline **doubled** real-traffic recall. The result is a
> demonstration of judgment about *when RL is the wrong tool* — and a clean methodology
> for catching it (held-constant comparison, locked decision criterion, OOD eval)."

This is stronger than a marginal RL win because it shows: (1) you can implement RL,
(2) you build honest evals that can falsify your own approach, and (3) you know the
boundary where RL stops paying for itself. Frame the project as a **comparative
methodology study (RL vs SFT)** where SFT is the answer and RL is the instructive foil —
not as "an SFT project," which throws away the RL implementation and the negative-result
narrative that's the actual differentiator.

---

## 9. Artifacts to attach when presenting

- Pareto plot: `results/pareto_sft.png` (synthetic, recall vs subset size)
- Real-traffic bars: `results/afr_eval_sft.png`
- The two tables in `docs/RESULT_sft_baseline.md` §Numbers
- (After Q1) the bootstrap CI on the 26-run real-traffic metrics
