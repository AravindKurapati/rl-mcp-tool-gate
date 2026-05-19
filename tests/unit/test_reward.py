import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.train.reward import compute_reward


def test_perfect_subset_high_reward():
    r = compute_reward(selected={"a", "b"}, ground_truth={"a", "b"}, catalog_size=100, k_target=2)
    assert r > 0.8


def test_empty_subset_punished():
    r = compute_reward(selected=set(), ground_truth={"a"}, catalog_size=100, k_target=5)
    assert r < -0.5


def test_extra_tools_mildly_penalized():
    r_clean = compute_reward(selected={"a"}, ground_truth={"a"}, catalog_size=100, k_target=1)
    r_extra = compute_reward(selected={"a", "b", "c", "d"}, ground_truth={"a"}, catalog_size=100, k_target=1)
    assert r_clean > r_extra
    assert r_extra > 0


def test_missing_critical_dominates():
    r = compute_reward(selected={"x", "y", "z"}, ground_truth={"a", "b"}, catalog_size=100, k_target=2)
    assert r < 0


def test_monotone_in_recall():
    base = {"x", "y"}
    r0 = compute_reward(selected=base, ground_truth={"a", "b", "c"}, catalog_size=50, k_target=3)
    r1 = compute_reward(selected=base | {"a"}, ground_truth={"a", "b", "c"}, catalog_size=50, k_target=3)
    r2 = compute_reward(selected=base | {"a", "b", "c"}, ground_truth={"a", "b", "c"}, catalog_size=50, k_target=3)
    assert r0 <= r1 <= r2
