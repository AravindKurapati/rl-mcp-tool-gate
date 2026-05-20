import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.eval.metrics import recall_at_subset, precision_at_subset, catastrophic_failure


def test_recall_full_match():
    assert recall_at_subset({"a", "b"}, {"a", "b"}) == 1.0


def test_recall_partial():
    assert recall_at_subset({"a"}, {"a", "b"}) == 0.5


def test_precision():
    assert precision_at_subset({"a", "b", "c"}, {"a", "b"}) == 2 / 3


def test_catastrophic_failure_true():
    assert catastrophic_failure({"a"}, {"a", "b"}) is True


def test_catastrophic_failure_false():
    assert catastrophic_failure({"a", "b", "c"}, {"a", "b"}) is False
