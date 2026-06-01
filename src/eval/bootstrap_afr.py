"""Bootstrap 95% CIs on the real-traffic (afr) metrics.

The headline "SFT ~2x off-the-shelf on real traffic" rests on only 26 MCP runs, so the
point estimates need an interval. This recomputes per-run recall and non-breaking
outcomes (the eval JSON only stores means), then does a paired cluster bootstrap over
runs: resample the 26 run indices with replacement, recompute every method on the SAME
resampled set, and read off (a) per-method CIs and (b) the CI + bootstrap p-value on the
SFT-minus-baseline gap. Pairing on the resample is what makes the gap CI honest — both
methods see identical run draws each iteration.

CPU only (BGE-small), no Modal. Run:
  python -m src.eval.bootstrap_afr --ckpt checkpoints/sft --B 10000
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

from src.gate.encoder import GateEncoder
from src.gate.select import select_tools
from src.eval.afr_catalog import build_augmented_catalog
from src.eval.metrics import recall_at_subset, catastrophic_failure, bootstrap_ci


def _observed_mcp_tools(sessions: list[dict]) -> list[str]:
    seen: dict[str, None] = {}
    for s in sessions:
        for t in s["tools_called"]:
            if t.startswith("mcp__"):
                seen.setdefault(t, None)
    return list(seen)


def per_run_outcomes(name, encoder, sessions, catalog, k_sweep) -> dict:
    """Return per-run vectors {k: {"recall": np.array, "nonbreak": np.array}}."""
    catalog_names = {t["name"] for t in catalog}
    out = {k: {"recall": [], "nonbreak": []} for k in k_sweep}
    for s in sessions:
        gt = {t for t in s["tools_called"] if t.startswith("mcp__")} & catalog_names
        if not gt:
            continue
        for top_k in k_sweep:
            selected = select_tools(signal=s["user_goal"], catalog=catalog, encoder=encoder, top_k=top_k)
            sel = {t["name"] for t in selected}
            out[top_k]["recall"].append(recall_at_subset(sel, gt))
            out[top_k]["nonbreak"].append(0.0 if catastrophic_failure(sel, gt) else 1.0)
    return {name: {k: {m: np.array(v) for m, v in d.items()} for k, d in out.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", default="data/afr_replay/sessions.jsonl")
    ap.add_argument("--catalog", default="data/synthetic/catalog.json")
    ap.add_argument("--ckpt", default="checkpoints/sft")
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="results/afr_bootstrap.json")
    args = ap.parse_args()

    sessions = [json.loads(l) for l in Path(args.sessions).read_text(encoding="utf-8").splitlines() if l.strip()]
    synthetic = json.loads(Path(args.catalog).read_text(encoding="utf-8"))["tools"]
    catalog = build_augmented_catalog(synthetic, _observed_mcp_tools(sessions))
    k_sweep = [3, 5, 8, 12, 20]

    enc_off = GateEncoder()
    enc_off.precompute_catalog(catalog)
    off = per_run_outcomes("off_the_shelf", enc_off, sessions, catalog, k_sweep)

    enc_sft = GateEncoder(lora_adapter_path=Path(args.ckpt))
    enc_sft.precompute_catalog(catalog)
    sft = per_run_outcomes("sft", enc_sft, sessions, catalog, k_sweep)

    n = len(off["off_the_shelf"][k_sweep[0]]["recall"])
    rng = np.random.default_rng(args.seed)

    report = {"n_runs": int(n), "B": args.B, "catalog_size": len(catalog), "by_k": {}}
    print(f"n_runs={n}  catalog={len(catalog)}  B={args.B}\n")
    for k in k_sweep:
        # One shared resample-index matrix per k -> paired across methods and metrics.
        idx = rng.integers(0, n, size=(args.B, n))
        row = {}
        for metric in ("recall", "nonbreak"):
            ov, olo, ohi, odist = bootstrap_ci(off["off_the_shelf"][k][metric], args.B, rng, idx)
            sv, slo, shi, sdist = bootstrap_ci(sft["sft"][k][metric], args.B, rng, idx)
            diff = sdist - odist  # paired difference distribution
            d_point = sv - ov
            d_lo, d_hi = np.percentile(diff, [2.5, 97.5])
            p_le0 = float((diff <= 0).mean())  # bootstrap one-sided p: P(SFT <= off)
            row[metric] = {
                "off": [round(ov, 3), round(olo, 3), round(ohi, 3)],
                "sft": [round(sv, 3), round(slo, 3), round(shi, 3)],
                "diff": [round(d_point, 3), round(float(d_lo), 3), round(float(d_hi), 3)],
                "p_sft_not_better": round(p_le0, 4),
            }
            print(f"k={k:<3} {metric:<8} off={ov:.3f}[{olo:.3f},{ohi:.3f}]  "
                  f"sft={sv:.3f}[{slo:.3f},{shi:.3f}]  "
                  f"diff={d_point:+.3f}[{d_lo:+.3f},{d_hi:+.3f}]  p={p_le0:.4f}")
        report["by_k"][k] = row
        print()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
