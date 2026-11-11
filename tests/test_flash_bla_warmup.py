import pytest

from scripts.warmup_flash_bla import power_of_two_buckets


def test_power_of_two_buckets_cover_requested_range():
    assert power_of_two_buckets(17,100)==[32,64]
    assert power_of_two_buckets(32,32)==[32]


@pytest.mark.parametrize("minimum,maximum",[(0,32),(64,32)])
def test_power_of_two_buckets_reject_invalid_range(minimum,maximum):
    with pytest.raises(ValueError):
        power_of_two_buckets(minimum,maximum)
