# FEATURE: Ground truth v2 for real-traffic eval

Status: draft
Author: Aravind
Date: 2026-06-01

## Why

The current real-traffic eval uses `tools_called` from an afr session as
ground truth: every tool the agent invoked counts as "needed." That is noisy
in three ways that all bias the result:

1. **Failed calls count as needed.** A `WebFetch` that errored and was
   immediately retried with `WebSearch` shows up as two "needed" tools when
   the goal only ever needed one.
2. **Retries inflate the set.** Same tool called five times appears once
   after dedup, but a tool called once after four failed alternatives
   shouldn't count those alternatives.
3. **Wrong tool picked counts as right.** If the agent reached for
   `mcp__github__search_code` when the right answer was
   `mcp__github__get_file_contents`, the trace says both were "needed."

This affects v1, v2, and SFT numbers equally, so the comparison between them
is roughly preserved. But absolute numbers (recall@8 = 0.455) overstate or
understate the gate's real value, and the cleaner the eval, the easier it is
to spot the next overfit before publishing it.

## Scope

In: heuristic filter pass on existing afr extraction, plus a hand-rubric LLM
labeling pass that produces a 100-query "gold" eval set. Report both
noisy-afr and gold numbers side by side.

Out: rewriting the synthetic eval, changing the SFT trainer, retraining
anything. The gold set only changes how we measure, not what we train.

## Approach

### Pass 1: heuristic filters (cheap, deterministic)

Modify `src/eval/afr_extract.py` to apply the filters at extraction time, so
every downstream consumer benefits. Filters are explicit flags so we can
keep noisy-afr as a baseline.

| Filter | What it catches |
|--------|----------------|
| `drop_failed=True` | `status == 'error'` rows in `tool_calls` |
| `dedup_within_window=N` | Same `tool_name` called within N seconds counts once |
| `min_calls=1` | Drop runs with no successful tool calls left after filtering |
| `mcp_only=True` (already exists) | Drop built-ins like `Bash`, `Read` |

Sanity: report `n_runs` before/after each filter. If a filter drops more
than half the runs, the filter is too aggressive.

### Pass 2: LLM-labeled gold set (high-trust, n=100)

Take 100 runs from afr, hand each to an LLM with a strict rubric, get back
the *minimal* set of tools the goal actually required.

**Rubric (single prompt, used for all 100 queries):**

```
You will see:
- A user goal (one paragraph)
- A trace of tool calls the agent made, with status and short input/output summaries

Your job: list the MINIMAL set of MCP tool names that were genuinely required
to complete the goal.

Rules:
- A tool counts as required only if the goal cannot be completed without it
- Retries of a failed call: count the tool once, not once per retry
- A tool that was called but turned out to be the wrong choice does not count
- Built-in tools (Bash, Read, Edit, etc.) never count
- If the goal is ambiguous or the trace is too short to judge, output {"skip": true}

Return JSON: {"required_tools": ["mcp__server__tool", ...], "confidence": 0.0-1.0, "skip": false}
```

**Inter-rater check.** Label a 20-run subset twice with two different
models (claude-sonnet-4-6 and claude-opus-4-8). Compute Jaccard agreement
on the required-tools set. Report agreement; if it is below 0.7, the
rubric is too loose and we tighten before labeling the rest.

**Skip handling.** Any run flagged `skip:true` by either model is excluded
from the gold set. Document how many were skipped.

### Pass 3: report side by side

`src/eval/afr_eval.py` gets a `--ground-truth {noisy,filtered,gold}` flag.
Default stays `noisy` so old numbers reproduce exactly. The bootstrap eval
gets the same flag.

Add a section to `docs/RESULT_sft_baseline.md`:

| Ground truth | Off-the-shelf @8 | RL v2 @8 | SFT @8 |
|--------------|:----:|:----:|:----:|
| Noisy (current) | 0.223 | 0.228 | 0.455 |
| Filtered | TBD | TBD | TBD |
| Gold (n=100) | TBD | TBD | TBD |

If SFT's lead survives the gold set, the result is bulletproof. If it
shrinks, that is a real and publishable finding too: SFT is partly
benefiting from the noise pattern in the labels.

## Storage

- Filtered extraction: still `data/afr_replay/sessions.jsonl`, with a
  `_filtered.jsonl` sibling so both can coexist.
- Gold set: `data/afr_replay/gold_v1.jsonl`, one row per labeled run with
  `run_id`, `user_goal`, `required_tools`, `confidence`, `labeler_model`.
- Agreement report: `results/gold_agreement.json`.

All under existing gitignore (`data/afr_replay/*` is already ignored).

## Cost

- Filters: zero (deterministic).
- Labeling 100 runs with 2 models: ~100 * 2 = 200 calls. Each prompt is
  short (rubric ~400 tok, trace ~1-2k tok, response ~200 tok). Roughly
  500k input + 40k output total. At Opus pricing that is under $10.
- Single-model labeling for the other 80 runs: ~half that. Total under $15.

## Tests

`tests/unit/test_afr_filters.py`:
- `drop_failed` removes error rows but keeps the rest of the run
- `dedup_within_window` collapses repeats but preserves distinct tools
- `min_calls=1` drops emptied runs

`tests/unit/test_gold_labeler.py`:
- Rubric prompt is stable (regression test on the prompt text)
- JSON parser handles `skip:true`, missing fields, malformed responses
- Agreement metric matches Jaccard on toy inputs

`tests/integration/test_eval_gold.py`:
- End-to-end on a 3-row fixture gold set: `afr_eval --ground-truth gold`
  produces a JSON with expected fields and `n_runs=3`

## Schema impact

No new SQLite. Adds two JSONL artifacts under `data/afr_replay/`. Update
`SCHEMA.md` (new in the gate decision log spec) with their formats.

## Risks

- **LLM labels can be wrong too.** Agreement check is the only defense. If
  agreement is low, we either tighten the rubric or fall back to filtered
  numbers as the headline.
- **Gold set is small (n=100, minus skips).** CIs will still be wide.
  Filtered numbers on full afr (~26 runs today, more as the logger
  accumulates) remain the primary view.
- **Selection bias.** If we hand-pick the 100 runs, we bias the gold set.
  Solution: random sample with seed=7 from all afr runs with at least one
  successful MCP call.

## Followups

- After the gate decision logger ships, the gold rubric can also see *what
  the gate surfaced* and rate whether that surfaced set was sufficient,
  not just whether the called set was minimal. That is a stronger eval but
  it depends on the logger landing first.
- Periodic re-labeling as the catalog grows.

## Order of work

1. Pass 1 filters (1-2 hours, no LLM)
2. Re-run all real-traffic numbers with filtered ground truth
3. If the filtered numbers move materially, that alone may be enough to
   publish a follow-up. If not, proceed to Pass 2.
4. Pass 2 gold labeling + agreement check
5. Pass 3 reporting
