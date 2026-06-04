"""Tests for the paired cluster bootstrap that backs every CI and p-value in the README.

`src.eval.metrics.bootstrap_ci` is the statistical core of `bootstrap_afr.py`: the headline
"SFT ~2x off-the-shelf on real traffic, p=..." rests on it. Two properties make those
numbers honest, and both were previously untested:

  1. Point estimate is the plain sample mean (no resampling bias in the reported value).
  2. *Pairing*: when two methods are bootstrapped with the SAME resample-index matrix
     `idx`, the per-iteration difference of their bootstrap means equals the bootstrap of
     their elementwise difference, exactly. That identity is what lets bootstrap_afr read
     a gap CI and a one-sided p-value off `sdist - odist` without the two methods drifting
     onto different run draws. If pairing ever broke, every gap interval would silently
     widen and the p-values would be wrong.

All CPU-only, pure numpy, no model download.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from src.eval.metrics import bootstrap_ci
from src.eval.bootstrap_afr import _observed_mcp_tools


def test_point_estimate_is_the_plain_sample_mean():
    rng = np.random.default_rng(0)
    v = np.array([0.1, 0.2, 0.3, 0.4])
    mean, _, _, _ = bootstrap_ci(v, B=1000, rng=rng)
    assert mean == v.mean()


def test_distribution_has_length_B_and_ci_brackets_mean():
    rng = np.random.default_rng(1)
    v = np.array([0.0, 1.0, 0.0, 1.0, 1.0, 0.0])
    mean, lo, hi, dist = bootstrap_ci(v, B=2000, rng=rng)
    assert dist.shape == (2000,)
    assert lo <= mean <= hi


def test_ci_endpoints_are_the_2p5_and_97p5_percentiles_of_the_distribution():
    rng = np.random.default_rng(2)
    v = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    _, lo, hi, dist = bootstrap_ci(v, B=5000, rng=rng)
    exp_lo, exp_hi = np.percentile(dist, [2.5, 97.5])
    assert lo == float(exp_lo)
    assert hi == float(exp_hi)


def test_same_seed_is_bit_identical():
    v = np.array([0.0, 1.0, 1.0, 0.0, 1.0])
    m1, lo1, hi1, d1 = bootstrap_ci(v, B=1000, rng=np.random.default_rng(42))
    m2, lo2, hi2, d2 = bootstrap_ci(v, B=1000, rng=np.random.default_rng(42))
    assert (m1, lo1, hi1) == (m2, lo2, hi2)
    assert np.array_equal(d1, d2)


def test_supplied_idx_is_used_verbatim_and_ignores_rng():
    # With idx provided, the result must be a pure function of (values, idx): the rng must
    # not be touched. bootstrap_afr depends on this to share one idx across methods.
    v = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    n = len(v)
    B = 300
    idx = np.random.default_rng(99).integers(0, n, size=(B, n))
    # Two different rngs, same idx -> identical output.
    out_a = bootstrap_ci(v, B=B, rng=np.random.default_rng(1), idx=idx)
    out_b = bootstrap_ci(v, B=B, rng=np.random.default_rng(123456), idx=idx)
    assert np.array_equal(out_a[3], out_b[3])
    # And the distribution is exactly the row-means of values[idx].
    expected = v[idx].mean(axis=1)
    assert np.array_equal(out_a[3], expected)


def test_pairing_diff_of_bootstraps_equals_bootstrap_of_diff():
    # THE load-bearing property for the gap CI / p-value in bootstrap_afr.
    rng = np.random.default_rng(7)
    n, B = 9, 4000
    idx = rng.integers(0, n, size=(B, n))
    off = np.array([0.0, 1, 0, 1, 1, 0, 1, 0, 1], dtype=float)
    sft = np.array([1.0, 1, 1, 0, 1, 1, 1, 0, 1], dtype=float)
    _, _, _, odist = bootstrap_ci(off, B, rng, idx)
    _, _, _, sdist = bootstrap_ci(sft, B, rng, idx)
    _, _, _, ddist = bootstrap_ci(sft - off, B, rng, idx)
    # Paired difference of the two bootstrap distributions is, iteration by iteration,
    # the bootstrap of the elementwise gap. (Difference-of-means and mean-of-differences
    # agree mathematically; they differ only by floating-point summation order, so this is
    # an allclose rather than a bit-exact equality.)
    assert np.allclose(sdist - odist, ddist)


def test_pairing_shift_invariance_gives_zero_variance_gap():
    # If sft = off + c (constant per-run lift), the paired bootstrap gap is exactly c on
    # every iteration -> zero-width gap CI. Unpaired bootstrapping could not guarantee this.
    rng = np.random.default_rng(3)
    n, B = 12, 2000
    idx = rng.integers(0, n, size=(B, n))
    off = rng.random(n)
    c = 0.137
    sft = off + c
    _, _, _, odist = bootstrap_ci(off, B, rng, idx)
    _, _, _, sdist = bootstrap_ci(sft, B, rng, idx)
    gap = sdist - odist
    assert np.allclose(gap, c)
    lo, hi = np.percentile(gap, [2.5, 97.5])
    assert np.isclose(hi - lo, 0.0)


def test_constant_values_yield_zero_width_ci():
    rng = np.random.default_rng(5)
    v = np.full(10, 0.42)
    mean, lo, hi, dist = bootstrap_ci(v, B=500, rng=rng)
    assert mean == v.mean()
    assert lo == hi == mean  # every resample is all-0.42 -> no spread
    assert np.allclose(dist, 0.42)


def test_single_observation_has_no_resample_variance():
    rng = np.random.default_rng(6)
    v = np.array([0.9])
    mean, lo, hi, dist = bootstrap_ci(v, B=100, rng=rng)
    assert mean == lo == hi == 0.9
    assert np.allclose(dist, 0.9)


def test_accepts_python_list_and_binary_nonbreak_metric():
    # nonbreak outcomes are passed as 0/1 floats; values may arrive as plain lists.
    rng = np.random.default_rng(8)
    values = [0.0, 1.0, 1.0, 1.0, 0.0, 1.0]
    mean, lo, hi, dist = bootstrap_ci(values, B=1000, rng=rng)
    assert mean == np.mean(values)
    assert 0.0 <= lo <= mean <= hi <= 1.0


# --- bootstrap_afr.per_run helper -------------------------------------------------------

def test_observed_mcp_tools_keeps_only_mcp_prefixed():
    sessions = [{"tools_called": ["read_file", "mcp__github__search", "bash"]}]
    assert _observed_mcp_tools(sessions) == ["mcp__github__search"]


def test_observed_mcp_tools_dedupes_preserving_first_seen_order():
    sessions = [
        {"tools_called": ["mcp__b", "mcp__a", "mcp__b"]},
        {"tools_called": ["mcp__a", "mcp__c"]},
    ]
    assert _observed_mcp_tools(sessions) == ["mcp__b", "mcp__a", "mcp__c"]


def test_observed_mcp_tools_empty_when_no_mcp_calls():
    sessions = [{"tools_called": ["read", "write"]}, {"tools_called": []}]
    assert _observed_mcp_tools(sessions) == []
