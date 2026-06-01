import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn.functional as F

from src.train.sft_loss import sft_step


def _norm(x):
    return F.normalize(x, p=2, dim=-1)


def test_loss_lower_when_query_aligned_with_gt():
    torch.manual_seed(0)
    catalog_emb = _norm(torch.randn(6, 16))
    gt = {2}
    # A query pointing at the GT tool should incur lower loss than one pointing elsewhere.
    q_good = catalog_emb[2:3].clone()
    q_bad = catalog_emb[5:6].clone()
    loss_good, _ = sft_step(query_emb=q_good, catalog_emb=catalog_emb, ground_truth_idx=gt)
    loss_bad, _ = sft_step(query_emb=q_bad, catalog_emb=catalog_emb, ground_truth_idx=gt)
    assert torch.isfinite(loss_good) and torch.isfinite(loss_bad)
    assert loss_good < loss_bad


def test_multi_positive_matches_manual_and_is_permutation_invariant():
    torch.manual_seed(1)
    catalog_emb = _norm(torch.randn(8, 16))
    query_emb = _norm(torch.randn(1, 16))
    scale = 10.0

    loss_a, _ = sft_step(query_emb=query_emb, catalog_emb=catalog_emb,
                         ground_truth_idx={0, 3}, score_scale=scale)
    loss_b, _ = sft_step(query_emb=query_emb, catalog_emb=catalog_emb,
                         ground_truth_idx={3, 0}, score_scale=scale)
    # Sets are unordered; result must be identical regardless of iteration order.
    assert torch.allclose(loss_a, loss_b)

    # Matches the closed-form multi-positive InfoNCE: mean of -log_softmax over positives.
    scores = (query_emb @ catalog_emb.T).squeeze(0) * scale
    logp = F.log_softmax(scores, dim=0)
    expected = -(logp[0] + logp[3]) / 2
    assert torch.allclose(loss_a, expected, atol=1e-6)


def test_gradient_pulls_query_toward_gt():
    torch.manual_seed(2)
    catalog_emb = _norm(torch.randn(5, 16))
    query_emb = _norm(torch.randn(1, 16)).requires_grad_(True)
    loss, _ = sft_step(query_emb=query_emb, catalog_emb=catalog_emb, ground_truth_idx={1})
    loss.backward()
    assert query_emb.grad is not None
    assert torch.isfinite(query_emb.grad).all()
    assert query_emb.grad.abs().sum() > 0


def test_single_tool_catalog_is_finite():
    catalog_emb = _norm(torch.randn(1, 16))
    query_emb = _norm(torch.randn(1, 16))
    loss, info = sft_step(query_emb=query_emb, catalog_emb=catalog_emb, ground_truth_idx={0})
    assert torch.isfinite(loss)
    # Only one entry => softmax is degenerate (prob 1) => loss ~ 0.
    assert float(loss) >= 0.0
    assert "loss" in info


def test_empty_ground_truth_returns_zero():
    catalog_emb = _norm(torch.randn(4, 16))
    query_emb = _norm(torch.randn(1, 16))
    loss, info = sft_step(query_emb=query_emb, catalog_emb=catalog_emb, ground_truth_idx=set())
    assert torch.isfinite(loss)
    assert float(loss) == 0.0
