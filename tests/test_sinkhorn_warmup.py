import pytest

from scripts.warmup_sinkhorn import normalize_buckets


def test_sinkhorn_warmup_buckets_are_sorted_and_unique():
    assert normalize_buckets([32, 8, 32, 16]) == (8, 16, 32)


@pytest.mark.parametrize("values", [[], [0, 8], [-1], [4, 0]])
def test_sinkhorn_warmup_rejects_nonpositive_buckets(values):
    with pytest.raises(ValueError):
        normalize_buckets(values)
