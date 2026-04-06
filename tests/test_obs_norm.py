import numpy as np
from ece324_tango.asce.obs_norm import ObsRunningNorm


def test_normalize_converges_to_zero_mean_unit_variance():
    rng = np.random.default_rng(42)
    norm = ObsRunningNorm(dim=4)
    for _ in range(1000):
        x = rng.normal(loc=[2.0, -1.0, 0.0, 5.0], scale=[1.0, 2.0, 0.5, 3.0])
        norm.update(x)
    out = norm.normalize(np.array([2.0, -1.0, 0.0, 5.0], dtype=np.float32))
    # Normalizing the mean value should give ~0
    np.testing.assert_allclose(out, 0.0, atol=0.1)


def test_padded_zeros_stay_zero():
    """Dimensions that are always 0 (padded) remain 0 after normalization."""
    norm = ObsRunningNorm(dim=4)
    rng = np.random.default_rng(0)
    for _ in range(500):
        x = np.array([rng.random(), rng.random(), 0.0, 0.0], dtype=np.float32)
        norm.update(x)
    out = norm.normalize(np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32))
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out[2:], 0.0, atol=1e-6)


def test_state_dict_roundtrip():
    norm = ObsRunningNorm(dim=3)
    norm.update(np.array([1.0, 2.0, 3.0]))
    norm.update(np.array([2.0, 4.0, 6.0]))
    state = norm.state_dict()

    norm2 = ObsRunningNorm(dim=3)
    norm2.load_state_dict(state)
    x = np.array([1.5, 3.0, 4.5], dtype=np.float32)
    np.testing.assert_array_equal(norm.normalize(x), norm2.normalize(x))


def test_untrained_normalizer_is_passthrough():
    """Before any updates, normalizer returns input unchanged."""
    norm = ObsRunningNorm(dim=3)
    x = np.array([5.0, -3.0, 1.0], dtype=np.float32)
    out = norm.normalize(x)
    np.testing.assert_array_equal(out, x)
