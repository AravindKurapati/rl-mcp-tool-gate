"""Eval metrics for tool-selection."""


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
