import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from src.train.sampler import gumbel_top_k_sample


def test_returns_correct_size():
    scores = torch.tensor([1.0, 2.0, 3.0, 0.5, -1.0])
    idx, logp = gumbel_top_k_sample(scores, k=3, n_samples=4)
    assert idx.shape == (4, 3)
    assert logp.shape == (4,)


def test_log_prob_finite():
    scores = torch.tensor([1.0, 2.0, 3.0, 0.5, -1.0])
    _, logp = gumbel_top_k_sample(scores, k=3, n_samples=4)
    assert torch.isfinite(logp).all()


def test_indices_unique_per_sample():
    scores = torch.tensor([1.0, 2.0, 3.0, 0.5, -1.0])
    idx, _ = gumbel_top_k_sample(scores, k=3, n_samples=10)
    for s in idx:
        assert len(set(s.tolist())) == 3


def test_high_score_tools_chosen_more_often():
    scores = torch.tensor([10.0, -10.0, -10.0, -10.0])
    idx, _ = gumbel_top_k_sample(scores, k=1, n_samples=100)
    chose_zero = (idx.squeeze(-1) == 0).float().mean()
    assert chose_zero > 0.9
