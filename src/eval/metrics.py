"""Eval metrics for tool-selection."""
import numpy as np


def bootstrap_ci(values: np.ndarray, B: int, rng, idx: np.ndarray | None = None) -> tuple:
    """95% percentile CI of the mean. If idx given, use those resample indices (for pairing).

    Canonical home for the bootstrap used by bootstrap_afr.py. Kept here so analysis-only
    scripts do not pull in the torch/transformers import chain behind GateEncoder.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if idx is None:
        idx = rng.integers(0, n, size=(B, n))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(values.mean()), float(lo), float(hi), means


def recall_at_subset(selected: set[str], ground_truth: set[str]) -> float:
    if not ground_truth:
        return 1.0
    return len(selected & ground_truth) / len(ground_truth)


def precision_at_subset(selected: set[str], ground_truth: set[str]) -> float:
    if not selected:
        return 0.0
    return len(selected & ground_truth) / len(selected)


def catastrophic_failure(selected: set[str], ground_truth: set[str]) -> bool:
    return not ground_truth.issubset(selected)
