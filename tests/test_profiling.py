import torch
from torch.profiler import ProfilerActivity, profile

from brainwm.profiling import enable_profile_ranges, profile_range


def test_brainwm_ranges_are_opt_in_and_visible_to_torch_profiler():
    with profile(activities=[ProfilerActivity.CPU],acc_events=True) as profiler:
        with profile_range("brainwm.disabled"):
            torch.ones(1)
        with enable_profile_ranges():
            with profile_range("brainwm.enabled"):
                torch.ones(1)
    keys = {event.key for event in profiler.key_averages()}
    assert "brainwm.disabled" not in keys
    assert "brainwm.enabled" in keys
