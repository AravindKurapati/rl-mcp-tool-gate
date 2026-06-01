"""Supervised contrastive loss for bi-encoder tool selection.

Full-catalog multi-positive InfoNCE: score a query against the entire (small) tool
catalog, treat the ground-truth tools as positives and every other catalog entry as a
negative, and minimize cross-entropy with uniform target mass over the positives.

This is the SFT counterpart to ``src/train/grpo.py`` — same encoder, same full-catalog
scoring (``q @ catalog.T``), differing ONLY in the objective (supervised cross-entropy
vs policy gradient on sampled subsets). Defining negatives by the *labels* (not by
batch co-occurrence) avoids the false-negative contamination that in-batch negatives
would cause on a small catalog where tools are shared across queries.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def sft_step(
    *,
    query_emb: torch.Tensor,      # (1, D), L2-normalized
    catalog_emb: torch.Tensor,    # (N, D), L2-normalized
    ground_truth_idx: set[int],
    score_scale: float = 10.0,
) -> tuple[torch.Tensor, dict]:
    """Multi-positive InfoNCE over the full catalog for a single query.

    loss = -(1/|GT|) * sum_{t in GT} log_softmax(scores)[t]
    """
    scores = (query_emb @ catalog_emb.T).squeeze(0) * score_scale  # (N,)
    if not ground_truth_idx:
        loss = torch.zeros((), device=scores.device, dtype=scores.dtype)
        return loss, {"loss": 0.0, "mean_gt_logp": 0.0}
    logp = F.log_softmax(scores, dim=0)
    gt_idx = torch.tensor(sorted(ground_truth_idx), device=scores.device, dtype=torch.long)
    gt_logp = logp[gt_idx]
    loss = -gt_logp.mean()
    return loss, {
        "loss": float(loss.detach()),
        "mean_gt_logp": float(gt_logp.mean().detach()),
    }
