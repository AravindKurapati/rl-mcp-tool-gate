"""Tests for metrics.bootstrap_ci, the percentile bootstrap behind the real-traffic CIs.

The headline "SFT ~2x off-the-shelf on real traffic" rests on a *paired* cluster
bootstrap (src/eval/bootstrap_afr.py): the same resample-index matrix is fed to both
methods so the per-iteration difference distribution is honestly paired. These tests pin
down both the basic CI behaviour and that pairing invariant. Pure numpy, CPU-only,
no network or model.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from src.eval.metrics import bootstrap_ci


def test_point_estimate_is_the_mean():
    values = np.array([0.0, 0.5, 1.0, 0.25])
    rng = np.random.default_rng(0)
    mean, lo, hi, dist = bootstrap_ci(values, B=1000, rng=rng)
    assert mean == values.mean()


def test_ci_brackets_the_mean():
    rng = np.random.default_rng(1)
    values = rng.random(40)
    mean, lo, hi, dist = bootstrap_ci(values, B=2000, rng=rng)
    assert lo <= mean <= hi


def test_distribution_shape_and_center():
    rng = np.random.default_rng(2)
    values = rng.random(30)
    B = 5000
    mean, lo, hi, dist = bootstrap_ci(values, B=B, rng=rng)
    assert dist.shape == (B,)
    # Bootstrap means concentrate around the sample mean.
    assert abs(dist.mean() - mean) < 0.02


def test_deterministic_under_same_seed():
    values = np.array([0.1, 0.9, 0.3, 0.7, 0.5])
    a = bootstrap_ci(values, B=500, rng=np.random.default_rng(42))
    b = bootstrap_ci(values, B=500, rng=np.random.default_rng(42))
    assert a[0] == b[0] and a[1] == b[1] and a[2] == b[2]
    assert np.array_equal(a[3], b[3])


def test_constant_values_give_degenerate_interval():
    values = np.full(20, 0.42)
    mean, lo, hi, dist = bootstrap_ci(values, B=100, rng=np.random.default_rng(7))
    assert mean == lo == hi
    assert np.isclose(mean, 0.42)


def test_explicit_idx_is_used_verbatim_and_rng_untouched():
    values = np.array([10.0, 20.0, 30.0, 40.0])
    idx = np.array([[0, 0, 0, 0], [3, 3, 3, 3], [0, 1, 2, 3]])
    # rng is None: if idx is honoured, rng must never be consumed.
    mean, lo, hi, dist = bootstrap_ci(values, B=3, rng=None, idx=idx)
    assert np.array_equal(dist, np.array([10.0, 40.0, 25.0]))


def test_paired_resample_invariant():
    # The property bootstrap_afr relies on: feeding the SAME idx to two methods makes
    # their bootstrap distributions paired, so the difference of distributions equals
    # the distribution of the paired per-run difference under that same resample.
    off = np.array([0.2, 0.4, 0.1, 0.6, 0.3])
    sft = np.array([0.5, 0.4, 0.7, 0.6, 0.9])
    rng = np.random.default_rng(2026)
    n = len(off)
    B = 4000
    idx = rng.integers(0, n, size=(B, n))

    _, _, _, off_dist = bootstrap_ci(off, B, rng, idx)
    _, _, _, sft_dist = bootstrap_ci(sft, B, rng, idx)

    paired_diff = (sft - off)[idx].mean(axis=1)
    # Pairing is exact: difference of the two distributions == distribution of the
    # paired difference, draw for draw (not just in expectation).
    assert np.allclose(sft_dist - off_dist, paired_diff)


def test_pairing_is_tighter_than_independent_draws():
    # Sanity check that pairing actually matters: when the two methods are strongly
    # correlated across runs, the paired gap distribution is tighter than pairing two
    # independently-resampled distributions would give.
    off = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8, 0.0])
    sft = off + 0.1  # perfectly correlated, constant gap
    n = len(off)
    B = 5000

    rng = np.random.default_rng(11)
    idx = rng.integers(0, n, size=(B, n))
    _, _, _, off_d = bootstrap_ci(off, B, rng, idx)
    _, _, _, sft_d = bootstrap_ci(sft, B, rng, idx)
    paired_std = (sft_d - off_d).std()

    rng2 = np.random.default_rng(12)
    _, _, _, off_i = bootstrap_ci(off, B, rng2)
    _, _, _, sft_i = bootstrap_ci(sft, B, rng2)
    independent_std = (sft_i - off_i).std()

    # Constant gap -> paired difference has (near) zero variance; independent does not.
    assert paired_std < 1e-9
    assert independent_std > paired_std
