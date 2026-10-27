import numpy as np
import torch

from scripts.run_nanobrain_baselines import _array


def test_array_promotes_bfloat16_tensor_for_numpy_export():
    value = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)

    result = _array(value, np.float32)

    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, np.asarray([1.0, 2.0], dtype=np.float32))
