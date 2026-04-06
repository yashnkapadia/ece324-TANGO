import numpy as np
import pytest

from ece324_tango.asce.env import pad_observation


def test_pad_observation_right_pads_with_zeros():
    src = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out = pad_observation(src, target_dim=5)
    assert out.shape == (5,)
    assert out.dtype == np.float32
    assert out.tolist() == [1.0, 2.0, 3.0, 0.0, 0.0]


def test_pad_observation_rejects_smaller_target():
    src = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    with pytest.raises(ValueError, match="target_dim"):
        pad_observation(src, target_dim=2)
