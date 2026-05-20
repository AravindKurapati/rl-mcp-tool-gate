"""Custom GRPO loop for bi-encoder tool selection."""
from __future__ import annotations
import torch
import torch.nn.functional as F

from src.train.sampler import gumbel_top_k_sample
from src.train.reward import compute_reward


def grpo_step(
    *,
    query_emb: torch.Tensor,         # (1, D)
    catalog_emb: torch.Tensor,       # (N, D)
    ground_truth_idx: set[int],
    head: torch.nn.Module | None,
    k: int,
    n_samples: int,
    catalog_size: int,
    k_target: int,
    kl_coef: float,
    ref_scores: torch.Tensor | None,  # (N,) or None
    score_scale: float = 10.0,
) -> tuple[torch.Tensor, dict]:
    """One GRPO update step for a single query."""
    # Cosine sim, scaled so softmax has meaningful gradient
    scores = (query_emb @ catalog_emb.T).squeeze(0) * score_scale  # (N,)
    idx, log_probs = gumbel_top_k_sample(scores, k=k, n_samples=n_samples)
    rewards = torch.zeros(n_samples, device=scores.device)
    for i in range(n_samples):
        sel = set(idx[i].tolist())
        r = compute_reward(
            selected={str(j) for j in sel},
            ground_truth={str(j) for j in ground_truth_idx},
            catalog_size=catalog_size,
            k_target=k_target,
        )
        rewards[i] = r
    adv = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
    pg_loss = -(adv.detach() * log_probs).mean()
    kl_loss = torch.tensor(0.0, device=scores.device)
    if ref_scores is not None:
        log_p = F.log_softmax(scores, dim=0)
        log_q = F.log_softmax(ref_scores * score_scale, dim=0)
        p = log_p.exp()
        kl_loss = (p * (log_p - log_q)).sum()
    loss = pg_loss + kl_coef * kl_loss
    return loss, {
        "mean_reward": float(rewards.mean()),
        "max_reward": float(rewards.max()),
        "min_reward": float(rewards.min()),
        "pg_loss": float(pg_loss.detach()),
        "kl_loss": float(kl_loss.detach()) if ref_scores is not None else 0.0,
    }
