# Result: heuristic filters confirm the SFT win is robust

**Date:** 2026-06-01
**Spec:** `FEATURE_ground_truth_v2.md` (Pass 1)
**TL;DR:** Adding `drop_failed=True` and a 30-second retry dedup to the afr
ground-truth pipeline barely moves the headline numbers. SFT still beats
off-the-shelf at every k on real Claude Code traffic. The noise was not
carrying the result.

## Setup

- afr DB grown from ~99 usable runs (May) to **138 usable runs**, **32 of
  them MCP-using**. SFT checkpoint (`checkpoints/sft`) unchanged.
- Two extracts compared: `sessions.jsonl` (noisy, original behavior) vs
  `sessions_filtered.jsonl` (`--filtered`: `drop_failed=True`,
  `dedup_window_s=30`, `min_tools=1`).
- Filter footprint: 33 runs had at least one tool-call row removed. Total
  tool calls dropped from 654 to 638. MCP-only tool counts dropped from 56
  to 55. 32 runs still have at least one MCP call in both extracts.

## Numbers

**MCP-domain recall@k (32 runs)**

| k | Off-the-shelf (noisy) | Off-the-shelf (filtered) | SFT (noisy) | SFT (filtered) |
|---|:----:|:----:|:----:|:----:|
| 3 | 0.102 | 0.117 | 0.234 | 0.219 |
| 5 | 0.184 | 0.169 | 0.281 | 0.273 |
| 8 | 0.291 | 0.307 | 0.464 | 0.448 |
| 12 | 0.442 | 0.437 | 0.589 | 0.573 |
| 20 | 0.517 | 0.512 | 0.724 | 0.740 |

**MCP-domain non-breaking rate@k**

| k | Off-the-shelf (noisy) | Off-the-shelf (filtered) | SFT (noisy) | SFT (filtered) |
|---|:----:|:----:|:----:|:----:|
| 8 | 0.188 | 0.219 | 0.344 | 0.344 |
| 12 | 0.312 | 0.312 | 0.500 | 0.500 |
| 20 | 0.344 | 0.375 | 0.625 | 0.656 |

## Read

1. **SFT still wins.** Recall@8 of 0.448 to 0.464 vs off-the-shelf 0.291 to
   0.307. SFT keeps a roughly 14-point lead in absolute recall at k=8 and
   the non-breaking-rate gap holds at ~12 points.
2. **Filters move things by 1 to 2 percentage points in either direction.**
   Within bootstrap CI width on n=32. The filters cleaned the data, they
   did not reshape the comparison.
3. **Headline numbers shifted from the May snapshot.** Old SFT recall@8 on
   n=26 was 0.455; new on n=32 noisy is 0.464. Old off-the-shelf was 0.223;
   new is 0.291. The off-the-shelf baseline improved meaningfully on the
   larger sample, which means SFT's *lead* is smaller than reported in May
   (16-17 points vs 23 points). The win is real, the multiplier is closer
   to 1.5x than 2x at this n.
4. **The retry-inflation hypothesis was wrong.** Most retries in the trace
   are built-ins (Read, Bash, Edit), not MCP tools. The MCP catalog is
   small enough and clean enough that the filter does not change which
   tools were "needed."

## What this means for Pass 2

Pass 1 was the cheap defense against the v1-style "the numbers are noise"
failure mode. It cleared. The remaining gain from ground-truth v2 would
have to come from cases where `tools_called` itself disagrees with what
the goal actually required, which is what the LLM-labeled gold set
targets.

Decision: Pass 2 is still worth running because the per-tool MCP catalog
is growing and at n=32 the CIs are wide. But it is no longer a
correctness blocker for the existing SFT result; it is an upgrade to
eval rigor for the next round of model work.

## Artifacts

- `results/afr_eval_sft_noisy.json` and `.png`
- `results/afr_eval_sft_filtered.json` and `.png`
- Extraction summaries embedded as the last block of stdout from
  `python -m src.eval.afr_extract` (with and without `--filtered`).

## Reproduce

```bash
python -m src.eval.afr_extract --out data/afr_replay/sessions.jsonl
python -m src.eval.afr_extract --filtered

python -m src.eval.afr_eval \
  --sessions data/afr_replay/sessions.jsonl \
  --ckpt checkpoints/sft \
  --out results/afr_eval_sft_noisy.json \
  --plot results/afr_eval_sft_noisy.png

python -m src.eval.afr_eval \
  --sessions data/afr_replay/sessions_filtered.jsonl \
  --ckpt checkpoints/sft \
  --out results/afr_eval_sft_filtered.json \
  --plot results/afr_eval_sft_filtered.png
```
