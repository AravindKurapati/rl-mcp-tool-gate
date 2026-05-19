"""Reward function for the tool-selection policy.

reward(S) = 1.0 * recall(S, GT)
          - 0.5 * (|S| / |catalog|)
          - 1.0 * I[GT not subset of S]
          + 0.1 * I[|S| <= k_target]
"""


def compute_reward(
    *,
    selected: set[str],
    ground_truth: set[str],
    catalog_size: int,
    k_target: int,
) -> float:
    if not ground_truth:
        return 1.0 - 0.5 * (len(selected) / max(1, catalog_size))
    hits = len(selected & ground_truth)
    recall = hits / len(ground_truth)
    cost = len(selected) / max(1, catalog_size)
    missing_critical = 0.0 if ground_truth.issubset(selected) else 1.0
    budget_bonus = 0.1 if len(selected) <= k_target else 0.0
    return 1.0 * recall - 0.5 * cost - 1.0 * missing_critical + budget_bonus
