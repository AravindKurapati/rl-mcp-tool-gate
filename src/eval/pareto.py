"""Sweep budgets across baselines, compute metrics, plot Pareto curves."""
from __future__ import annotations
import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.gate.encoder import GateEncoder
from src.eval.metrics import recall_at_subset, precision_at_subset, catastrophic_failure
from src.eval.baselines import baseline_no_gate, baseline_random, baseline_bm25, baseline_bge


def evaluate(method_name, predict_fn, queries, catalog, top_k_sweep) -> dict:
    results = {}
    for top_k in top_k_sweep:
        recalls, precs, cats, sizes = [], [], [], []
        for q in queries:
            selected = predict_fn(q["query"], catalog, top_k=top_k)
            selected_names = {t["name"] for t in selected}
            gt = set(q["ground_truth"])
            recalls.append(recall_at_subset(selected_names, gt))
            precs.append(precision_at_subset(selected_names, gt))
            cats.append(catastrophic_failure(selected_names, gt))
            sizes.append(len(selected_names))
        results[top_k] = {
            "recall": float(np.mean(recalls)),
            "precision": float(np.mean(precs)),
            "catastrophic_rate": float(np.mean(cats)),
            "mean_size": float(np.mean(sizes)),
        }
    return {"method": method_name, "by_k": results}


def eval_qwen_from_file(preds_path: Path, queries: list[dict]) -> dict:
    by_query = {p["query"]: set(p["predicted"]) for p in (json.loads(l) for l in preds_path.read_text(encoding="utf-8").splitlines())}
    recalls, precs, cats, sizes = [], [], [], []
    for q in queries:
        sel = by_query.get(q["query"], set())
        gt = set(q["ground_truth"])
        recalls.append(recall_at_subset(sel, gt))
        precs.append(precision_at_subset(sel, gt))
        cats.append(catastrophic_failure(sel, gt))
        sizes.append(len(sel))
    return {"method": "qwen2.5-1.5b", "by_k": {"N/A": {
        "recall": float(np.mean(recalls)),
        "precision": float(np.mean(precs)),
        "catastrophic_rate": float(np.mean(cats)),
        "mean_size": float(np.mean(sizes)),
    }}}


def afr_coverage(sessions_path: Path, predict_fn, catalog: list[dict], top_k: int = 10) -> dict:
    if not sessions_path.exists():
        return {"method": "afr_replay", "skipped": "no afr db"}
    sessions = [json.loads(l) for l in sessions_path.read_text(encoding="utf-8").splitlines()]
    covered, savings = [], []
    avg_tool_tokens = sum(len(t["embed_text"]) // 4 for t in catalog) / max(1, len(catalog))
    for s in sessions:
        sel = predict_fn(s["initial_prompt"], catalog, top_k=top_k)
        sel_names = {t["name"] for t in sel}
        called = set(s["tools_called"])
        covered.append(called.issubset(sel_names))
        savings.append((len(catalog) - len(sel)) * avg_tool_tokens)
    return {
        "method": "afr_replay",
        "coverage": float(sum(covered) / max(1, len(covered))),
        "mean_token_savings": float(sum(savings) / max(1, len(savings))),
        "n_sessions": len(sessions),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout", default="data/synthetic/heldout.jsonl")
    ap.add_argument("--catalog", default="data/synthetic/catalog.json")
    ap.add_argument("--ckpt", default="checkpoints/run1")
    ap.add_argument("--qwen-preds", default="results/qwen_baseline_preds.jsonl")
    ap.add_argument("--out", default="results/pareto.json")
    ap.add_argument("--plot", default="results/pareto.png")
    args = ap.parse_args()

    queries = [json.loads(l) for l in Path(args.heldout).read_text(encoding="utf-8").splitlines()]
    catalog = json.loads(Path(args.catalog).read_text(encoding="utf-8"))["tools"]
    top_k_sweep = [3, 5, 8, 12, 20]

    enc_off = GateEncoder()
    enc_off.precompute_catalog(catalog)
    ckpt_path = Path(args.ckpt)
    enc_rl = GateEncoder(lora_adapter_path=ckpt_path if ckpt_path.exists() else None)
    enc_rl.precompute_catalog(catalog)

    out_all = []
    out_all.append(evaluate("no_gate", lambda q, c, top_k: baseline_no_gate(q, c), queries, catalog, [len(catalog)]))
    out_all.append(evaluate("random", lambda q, c, top_k: baseline_random(q, c, top_k=top_k, seed=42), queries, catalog, top_k_sweep))
    out_all.append(evaluate("bm25", lambda q, c, top_k: baseline_bm25(q, c, top_k=top_k), queries, catalog, top_k_sweep))
    out_all.append(evaluate("bge_off_the_shelf", lambda q, c, top_k: baseline_bge(q, c, enc_off, top_k=top_k), queries, catalog, top_k_sweep))
    if ckpt_path.exists():
        out_all.append(evaluate("bge_rl", lambda q, c, top_k: baseline_bge(q, c, enc_rl, top_k=top_k), queries, catalog, top_k_sweep))
    if Path(args.qwen_preds).exists():
        out_all.append(eval_qwen_from_file(Path(args.qwen_preds), queries))

    afr_path = Path("data/afr_replay/sessions.jsonl")
    if afr_path.exists() and ckpt_path.exists():
        out_all.append(afr_coverage(afr_path, lambda q, c, top_k: baseline_bge(q, c, enc_rl, top_k=top_k), catalog, top_k=10))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out_all, indent=2), encoding="utf-8")

    plt.figure(figsize=(8, 6))
    for entry in out_all:
        if "by_k" not in entry:
            continue
        xs, ys = [], []
        for _, m in entry["by_k"].items():
            xs.append(m["mean_size"])
            ys.append(m["recall"])
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        plt.plot([xs[i] for i in order], [ys[i] for i in order], marker="o", label=entry["method"])
    plt.xlabel("Mean subset size (lower = less context)")
    plt.ylabel("Recall (higher = fewer dropped tools)")
    plt.title("Tool gate: recall vs context-cost Pareto")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.plot, dpi=120)
    print(f"Wrote {args.out} and {args.plot}")


if __name__ == "__main__":
    main()
