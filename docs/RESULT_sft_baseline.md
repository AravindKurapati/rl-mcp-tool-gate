# Result: SFT contrastive baseline beats GRPO (and off-the-shelf)

**Date:** 2026-05-20
**Spec:** `docs/superpowers/specs/2026-05-20-tool-gate-sft-baseline.md`
**TL;DR:** A plain supervised contrastive fine-tune (SFT) of the same BGE-small
bi-encoder **decisively beats** both the GRPO model (`run2`) and off-the-shelf BGE —
on synthetic *and* real out-of-distribution traffic. This inverts the project's prior
conclusion: GRPO's reward-shaping wasn't just unnecessary, it was *worse* than copying
the labels.

---

## What was run

- **Loss:** full-catalog multi-positive InfoNCE (`src/train/sft_loss.py`):
  `loss = -(1/|GT|) Σ_{t∈GT} log_softmax(scale · q·Cᵀ)[t]`. Negatives defined by the
  *labels*, not batch co-occurrence (avoids false negatives on the small shared catalog).
- **Trainer:** `src/train/sft_train.py` (Modal A10G) — identical model, LoRA (`r=8,
  α=16`), data (`catalog_v2`/`train_v2`), split (seed 7), `val_recall` early-stop, and
  best-on-val checkpoint as the GRPO `v2` run. Only the objective differs.
- **Training:** 800 steps, loss 3.59 → 1.32, best `val_recall` **0.956** @ step 750.
- **Eval:** `pareto.py` (synthetic heldout_v2) + `afr_eval.py` (26 real MCP-using afr
  runs), both with `--ckpt checkpoints/sft`, 114-tool catalog.
- **Cost:** ~$2 Modal (smoke ~$0.20 + full ~$1.8).

## Numbers

**Synthetic (heldout_v2, 114 tools) — recall@k**

| k | off-shelf | RL (run2) | SFT |
|---|-----------|-----------|-----|
| 3 | 0.577 | 0.647 | **0.743** |
| 5 | 0.670 | 0.717 | **0.804** |
| 8 | 0.720 | 0.785 | **0.870** |
| 12 | 0.756 | 0.853 | **0.903** |
| 20 | 0.823 | 0.908 | **0.946** |

**Real traffic (afr, 26 MCP runs) — recall@k**

| k | off-shelf | RL (run2) | SFT |
|---|-----------|-----------|-----|
| 3 | 0.038 | 0.071 | **0.163** |
| 5 | 0.111 | 0.157 | **0.279** |
| 8 | 0.223 | 0.228 | **0.455** |
| 12 | 0.313 | 0.237 | **0.545** |
| 20 | 0.377 | 0.372 | **0.631** |

**Real traffic — non-breaking rate@k** (fraction of runs where *all* needed tools surfaced)

| k | off-shelf | RL (run2) | SFT |
|---|-----------|-----------|-----|
| 8 | 0.077 | 0.077 | **0.346** |
| 12 | 0.154 | 0.077 | **0.423** |
| 20 | 0.192 | 0.231 | **0.500** |

## Why this is trustworthy (sanity checks, all passed)

1. **Equal budgets** — mean subset size is identical across methods at each k (3/5/8/12/20),
   so SFT's higher recall is not from surfacing more tools.
2. **Harness reproduces documented baseline** — fresh off-the-shelf synthetic recall@8 =
   0.720, @12 = 0.756, matching the prior v2 numbers exactly.
3. **afr off-the-shelf bit-identical** between the SFT-run and the v2-run (0.2234@8,
   0.3132@12) — same baseline, same data, same 26 runs / 114 tools.
4. **Real-traffic gain is OOD** — SFT never trained on afr data, so the ~2× recall gain
   is generalization, not eval leakage.

## Interpretation

- Static per-query tool-selection is **fundamentally supervised retrieval**. Direct
  cross-entropy on the labels gives a clean per-step gradient toward every ground-truth
  tool; GRPO's noisy sampled-subset advantage — plus the heavy anti-overfit
  regularization added in v2 (LoRA `r=8`, KL `0.05`, early stop) — *underfit* and left
  signal on the table.
- The honest headline: **"I trained an RL gate, built a real-traffic eval, found it
  overfit, then proved a simpler supervised baseline doubled real-traffic recall."**
  That demonstrates judgment about *when RL is the wrong tool* — a stronger result than
  a marginal RL win.

## Implication for the bandit pivot (Move B)

This **weakens** the mandate for the online contextual bandit. Static selection is now
clearly and strongly supervised-solvable, so RL is justified *only* if the reward comes
from a signal no label can express (real downstream agent usage) **and** in an
environment with genuine subset-selection pressure (large catalog, richer traffic) —
not on the current afr data, where SFT already wins. If Move B proceeds, the baseline to
beat is now **SFT**, not off-the-shelf BGE.

## Artifacts

- `checkpoints/sft/` (LoRA adapter), `results/pareto_sft.{json,png}`,
  `results/afr_eval_sft.{json,png}`.
- Tests: `tests/unit/test_sft_loss.py` (5 tests); suite 51 green.
