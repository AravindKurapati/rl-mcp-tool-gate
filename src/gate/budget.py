"""Token-budget knapsack helper for tool subset selection."""
from typing import Any


def estimate_tokens(tool: dict[str, Any]) -> int:
    return max(1, len(tool["embed_text"]) // 4)


def select_under_budget(
    ranked_tools: list[dict[str, Any]],
    budget_tokens: int | None,
    top_k: int | None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    used = 0
    for t in ranked_tools:
        if top_k is not None and len(out) >= top_k:
            break
        cost = estimate_tokens(t)
        if budget_tokens is not None and used + cost > budget_tokens:
            continue
        out.append(t)
        used += cost
    return out
