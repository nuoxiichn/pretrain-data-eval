from __future__ import annotations

import pytest

from pretrain_data_eval.sampling import sample_documents


def test_head_sampling_and_full_materialization():
    assert sample_documents(iter(range(10)), 3, mode="head") == [0, 1, 2]
    assert sample_documents(iter(range(4)), None) == [0, 1, 2, 3]


def test_reservoir_sampling_is_seeded_and_bounded():
    first = sample_documents(iter(range(100)), 10, mode="random", seed=7)
    second = sample_documents(iter(range(100)), 10, mode="random", seed=7)
    assert first == second
    assert len(first) == 10
    assert len(set(first)) == 10
    assert all(0 <= value < 100 for value in first)


def test_invalid_sampling_mode_fails():
    with pytest.raises(ValueError, match="unknown sample mode"):
        sample_documents([], 1, mode="broken")
