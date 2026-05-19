"""Gumbel-top-k sampling with log-probabilities for policy gradient."""
from __future__ import annotations
import torch


def gumbel_top_k_sample(
    scores: torch.Tensor,
    k: int,
    n_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample n_samples subsets of size k from softmax(scores).

    Returns:
        indices: LongTensor of shape (n_samples, k).
        log_probs: Tensor of shape (n_samples,) — log p under sampling-without-replacement.
    """
    n = scores.shape[0]
    assert k <= n, f"k={k} > catalog size {n}"
    u = torch.rand(n_samples, n, device=scores.device).clamp(min=1e-12, max=1 - 1e-12)
    gumbel = -torch.log(-torch.log(u))
    perturbed = scores.unsqueeze(0) + gumbel
    _, idx = perturbed.topk(k, dim=1)

    log_probs = torch.zeros(n_samples, device=scores.device)
    remaining_mask = torch.ones(n_samples, n, device=scores.device, dtype=torch.bool)
    for step in range(k):
        chosen_i = idx[:, step]
        scores_b = scores.unsqueeze(0).expand(n_samples, n).clone()
        scores_b[~remaining_mask] = float("-inf")
        log_denom = torch.logsumexp(scores_b, dim=1)
        log_prob_step = scores[chosen_i] - log_denom
        log_probs = log_probs + log_prob_step
        remaining_mask = remaining_mask.scatter(1, chosen_i.unsqueeze(1), False)

    return idx, log_probs
