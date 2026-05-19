import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from src.train.grpo import grpo_step


def test_grpo_step_returns_finite_loss():
    catalog_emb = torch.randn(5, 16)
    query_emb = torch.randn(1, 16)
    loss, info = grpo_step(
        query_emb=query_emb,
        catalog_emb=catalog_emb,
        ground_truth_idx={0, 1},
        head=None,
        k=3,
        n_samples=4,
        catalog_size=5,
        k_target=2,
        kl_coef=0.02,
        ref_scores=None,
    )
    assert torch.isfinite(loss)
    assert "mean_reward" in info


def test_grpo_step_with_ref():
    catalog_emb = torch.randn(8, 16)
    query_emb = torch.randn(1, 16)
    ref_scores = torch.randn(8)
    loss, info = grpo_step(
        query_emb=query_emb,
        catalog_emb=catalog_emb,
        ground_truth_idx={0},
        head=None,
        k=3,
        n_samples=4,
        catalog_size=8,
        k_target=2,
        kl_coef=0.1,
        ref_scores=ref_scores,
    )
    assert torch.isfinite(loss)
    assert info["kl_loss"] != 0.0
