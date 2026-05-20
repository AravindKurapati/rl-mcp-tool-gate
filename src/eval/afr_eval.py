"""Real-traffic evaluation on agent-flight-recorder sessions (Features 1 + 3).

Feature 1 (real-traffic eval): gate the AUGMENTED catalog using each run's
user_goal as the signal; ground truth = the tools the run actually called.
Feature 3 (task success): "non-breaking rate" = fraction of runs where the gate
would NOT have dropped any tool the run used (GT subset of selected). A run the
gate keeps whole could not have been broken by gating.

Run:
  python -m src.eval.afr_extract --out data/afr_replay/sessions.jsonl
  python -m src.eval.afr_eval
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

from src.gate.encoder import GateEncoder
from src.gate.budget import estimate_tokens
from src.gate.select import select_tools
from src.eval.afr_catalog import build_augmented_catalog
from src.eval.metrics import recall_at_subset, catastrophic_failure


def _observed_tools(sessions: list[dict], mcp_only: bool = False) -> list[str]:
    seen: dict[str, None] = {}
    for s in sessions:
        for t in s["tools_called"]:
            if mcp_only and not t.startswith("mcp__"):
                continue
            seen.setdefault(t, None)
    return list(seen)


def _ground_truth(session: dict, catalog_names: set[str], mcp_only: bool) -> set[str]:
    tools = session["tools_called"]
    if mcp_only:
        tools = [t for t in tools if t.startswith("mcp__")]
    return set(tools) & catalog_names


def _eval_encoder(name, encoder, sessions, catalog, k_sweep, mcp_only: bool = False) -> dict:
    catalog_names = {t["name"] for t in catalog}
    full_tokens = sum(estimate_tokens(t) for t in catalog)
    by_k = {}
    for top_k in k_sweep:
        recalls, breaks, nonbreak, sizes, savings = [], [], [], [], []
        for s in sessions:
            gt = _ground_truth(s, catalog_names, mcp_only)
            if not gt:
                continue  # gate can't break a run that needed none of its tools
            selected = select_tools(signal=s["user_goal"], catalog=catalog, encoder=encoder, top_k=top_k)
            sel_names = {t["name"] for t in selected}
            recalls.append(recall_at_subset(sel_names, gt))
            broke = catastrophic_failure(sel_names, gt)
            breaks.append(broke)
            nonbreak.append(not broke)
            sizes.append(len(sel_names))
            savings.append(full_tokens - sum(estimate_tokens(t) for t in selected))
        by_k[top_k] = {
            "recall": float(np.mean(recalls)) if recalls else 0.0,
            "non_breaking_rate": float(np.mean(nonbreak)) if nonbreak else 0.0,
            "catastrophic_rate": float(np.mean(breaks)) if breaks else 0.0,
            "mean_size": float(np.mean(sizes)) if sizes else 0.0,
            "mean_token_savings": float(np.mean(savings)) if savings else 0.0,
            "n_runs": len(recalls),
        }
    return {"method": name, "by_k": by_k}


def _block(sessions, synthetic, ckpt, k_sweep, mcp_only: bool) -> dict:
    """One eval block. mcp_only=True restricts the catalog AND ground truth to MCP
    tools (the gate's actual domain — built-ins like Bash/Read are always available
    and never gated)."""
    catalog = build_augmented_catalog(synthetic, _observed_tools(sessions, mcp_only=mcp_only))
    enc_off = GateEncoder()
    enc_off.precompute_catalog(catalog)
    results = [_eval_encoder("bge_off_the_shelf", enc_off, sessions, catalog, k_sweep, mcp_only=mcp_only)]
    if ckpt.exists():
        enc_rl = GateEncoder(lora_adapter_path=ckpt)
        enc_rl.precompute_catalog(catalog)
        results.append(_eval_encoder("bge_rl", enc_rl, sessions, catalog, k_sweep, mcp_only=mcp_only))
    n_eval = results[0]["by_k"][k_sweep[0]]["n_runs"]
    return {"catalog_size": len(catalog), "n_runs_evaluated": n_eval, "results": results}


def run(sessions_path: Path, catalog_path: Path, ckpt: Path, out_json: Path, out_png: Path, k_sweep) -> dict:
    sessions = [json.loads(l) for l in sessions_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    synthetic = json.loads(catalog_path.read_text(encoding="utf-8"))["tools"]

    summary = {
        "n_sessions": len(sessions),
        "synthetic_tools": len(synthetic),
        # Primary: the gate's real domain — does it surface the MCP tools a run needed?
        "mcp_domain": _block(sessions, synthetic, ckpt, k_sweep, mcp_only=True),
        # Secondary (honest baseline): conflates a whole session's tools incl. built-ins,
        # which the gate never manages — expected to look poor; reported for transparency.
        "all_tools_session_level": _block(sessions, synthetic, ckpt, k_sweep, mcp_only=False),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _plot(summary["mcp_domain"]["results"], k_sweep, out_png)
    return summary


def _plot(results, k_sweep, out_png: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    plt.figure(figsize=(8, 5))
    for entry in results:
        ys = [entry["by_k"][k]["non_breaking_rate"] for k in k_sweep]
        plt.plot(list(k_sweep), ys, marker="o", label=entry["method"])
    plt.xlabel("top_k (tools surfaced)")
    plt.ylabel("Non-breaking rate (no needed tool dropped)")
    plt.title("Real Claude Code traffic: gate non-breaking rate vs k")
    plt.ylim(0, 1.02)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", default="data/afr_replay/sessions.jsonl")
    ap.add_argument("--catalog", default="data/synthetic/catalog.json")
    ap.add_argument("--ckpt", default="checkpoints/run1")
    ap.add_argument("--out", default="results/afr_eval.json")
    ap.add_argument("--plot", default="results/afr_eval.png")
    args = ap.parse_args()
    sp = Path(args.sessions)
    if not sp.exists():
        print(f"No sessions at {sp}. Run: python -m src.eval.afr_extract")
        return
    summary = run(sp, Path(args.catalog), Path(args.ckpt), Path(args.out), Path(args.plot), [3, 5, 8, 12, 20])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
