"""Public API: select_tools(...). Stateless, pure function."""
from __future__ import annotations
from typing import Any
import numpy as np

from src.gate.encoder import GateEncoder
from src.gate.budget import select_under_budget


def select_tools(
    *,
    signal: str,
    catalog: list[dict[str, Any]],
    encoder: GateEncoder,
    top_k: int | None = 10,
    budget_tokens: int | None = None,
) -> list[dict[str, Any]]:
    """Select a relevant subset of tools from the catalog given a query signal."""
    scores = encoder.score(signal)
    order = np.argsort(-scores)
    ranked = [catalog[i] for i in order]
    return select_under_budget(ranked, budget_tokens=budget_tokens, top_k=top_k)
