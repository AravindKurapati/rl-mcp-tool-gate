"""Latency + cost dashboard (Feature 2).

Measures, on REAL agent-flight-recorder traffic:
  - Gate latency per turn (encode query + select), p50/p95, off-the-shelf vs RL.
  - Context savings: full-catalog tokens vs gated tokens per turn -> tokens saved.
  - Projected $ savings at a configurable input-token price, contextualized against
    real afr usage (mean tokens_in, total cost_usd).

Writes results/dashboard.md (human-readable) + results/dashboard.png.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.gate.encoder import GateEncoder
from src.gate.budget import estimate_tokens
from src.gate.select import select_tools
from src.eval.afr_catalog import build_augmented_catalog


def _percentiles(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    a = np.array(xs)
    return float(np.percentile(a, 50)), float(np.percentile(a, 95))


def _measure(encoder, sessions, catalog, top_k) -> dict:
    full_tokens = sum(estimate_tokens(t) for t in catalog)
    latencies, gated_tokens = [], []
    for s in sessions:
        t0 = time.perf_counter()
        sel = select_tools(signal=s["user_goal"], catalog=catalog, encoder=encoder, top_k=top_k)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        gated_tokens.append(sum(estimate_tokens(t) for t in sel))
    p50, p95 = _percentiles(latencies)
    mean_gated = float(np.mean(gated_tokens))
    return {
        "latency_p50_ms": p50,
        "latency_p95_ms": p95,
        "full_catalog_tokens": full_tokens,
        "mean_gated_tokens": mean_gated,
        "mean_tokens_saved_per_turn": full_tokens - mean_gated,
    }


def build_dashboard(sessions_path: Path, catalog_path: Path, ckpt: Path, top_k: int, price_per_mtok: float) -> dict:
    sessions = [json.loads(l) for l in sessions_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    synthetic = json.loads(catalog_path.read_text(encoding="utf-8"))["tools"]
    observed = list({t for s in sessions for t in s["tools_called"]})
    catalog = build_augmented_catalog(synthetic, observed)

    enc_off = GateEncoder()
    enc_off.precompute_catalog(catalog)
    rows = {"bge_off_the_shelf": _measure(enc_off, sessions, catalog, top_k)}
    if ckpt.exists():
        enc_rl = GateEncoder(lora_adapter_path=ckpt)
        enc_rl.precompute_catalog(catalog)
        rows["bge_rl"] = _measure(enc_rl, sessions, catalog, top_k)

    total_turns = sum(s["n_tool_calls"] for s in sessions)
    real = {
        "n_runs": len(sessions),
        "total_tool_calls": total_turns,
        "mean_tokens_in_per_run": float(np.mean([s["tokens_in"] for s in sessions])),
        "total_cost_usd": float(sum(s["cost_usd"] for s in sessions)),
    }
    saved = rows.get("bge_rl", rows["bge_off_the_shelf"])["mean_tokens_saved_per_turn"]
    projection = {
        "price_per_mtok_usd": price_per_mtok,
        "saved_per_turn_usd": saved / 1e6 * price_per_mtok,
        "saved_per_1k_turns_usd": saved / 1e6 * price_per_mtok * 1000,
    }
    return {"top_k": top_k, "real_traffic": real, "gate": rows, "projection": projection}


def write_markdown(d: dict, out_md: Path):
    real, gate, proj = d["real_traffic"], d["gate"], d["projection"]
    L = []
    L.append("# Tool-gate cost / latency dashboard\n")
    L.append(f"Measured on **{real['n_runs']} real agent-flight-recorder runs** "
             f"({real['total_tool_calls']} tool calls). top_k = {d['top_k']}.\n")
    L.append("## Latency (per turn)\n")
    L.append("| encoder | p50 (ms) | p95 (ms) |")
    L.append("|---------|:--------:|:--------:|")
    for name, r in gate.items():
        L.append(f"| {name} | {r['latency_p50_ms']:.1f} | {r['latency_p95_ms']:.1f} |")
    L.append("\n## Context savings\n")
    any_r = next(iter(gate.values()))
    L.append(f"- Full augmented catalog: **{any_r['full_catalog_tokens']:,} tokens** if all loaded every turn.")
    for name, r in gate.items():
        L.append(f"- {name}: gated to **{r['mean_gated_tokens']:.0f} tokens/turn** "
                 f"→ saves **{r['mean_tokens_saved_per_turn']:.0f} tokens/turn**.")
    L.append("\n## Real usage context\n")
    L.append(f"- Mean input tokens / run: **{real['mean_tokens_in_per_run']:,.0f}**")
    L.append(f"- Total recorded spend across these runs: **${real['total_cost_usd']:.2f}**")
    L.append("\n## Projected savings\n")
    L.append(f"At ${proj['price_per_mtok_usd']:.2f}/Mtok input:")
    L.append(f"- Per turn: ~**${proj['saved_per_turn_usd']:.6f}** saved in tool-definition tokens.")
    L.append(f"- Per **1,000 turns**: ~**${proj['saved_per_1k_turns_usd']:.2f}** saved in tool-definition tokens alone.")
    L.append("\n> Savings count only tool-definition tokens pruned from context; the larger "
             "win is fewer tools → less tool-selection confusion (see afr_eval non-breaking rate).")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(L) + "\n", encoding="utf-8")


def _plot(d: dict, out_png: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    gate = d["gate"]
    names = list(gate.keys())
    saved = [gate[n]["mean_tokens_saved_per_turn"] for n in names]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, saved, color=["#888", "#2a7"][: len(names)])
    ax.set_ylabel("Mean tokens saved / turn")
    ax.set_title(f"Context pruned per turn (top_k={d['top_k']}, real traffic)")
    for i, v in enumerate(saved):
        ax.text(i, v, f"{v:.0f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", default="data/afr_replay/sessions.jsonl")
    ap.add_argument("--catalog", default="data/synthetic/catalog.json")
    ap.add_argument("--ckpt", default="checkpoints/run1")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--price-per-mtok", type=float, default=3.0)
    ap.add_argument("--out-md", default="results/dashboard.md")
    ap.add_argument("--out-png", default="results/dashboard.png")
    args = ap.parse_args()
    sp = Path(args.sessions)
    if not sp.exists():
        print(f"No sessions at {sp}. Run: python -m src.eval.afr_extract")
        return
    d = build_dashboard(sp, Path(args.catalog), Path(args.ckpt), args.top_k, args.price_per_mtok)
    write_markdown(d, Path(args.out_md))
    _plot(d, Path(args.out_png))
    print(json.dumps(d, indent=2))
    print(f"\nWrote {args.out_md} and {args.out_png}")


if __name__ == "__main__":
    main()
