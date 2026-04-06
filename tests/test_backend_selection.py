import pytest

from ece324_tango.asce.trainers.factory import get_backend
from ece324_tango.asce.trainers.local_mappo_backend import LocalMappoBackend


def test_backend_factory_local():
    backend = get_backend("local_mappo")
    assert isinstance(backend, LocalMappoBackend)


@pytest.mark.parametrize("legacy_backend", ["benchmarl", "xuance", "libsignal"])
def test_backend_factory_rejects_removed_backends(legacy_backend: str):
    with pytest.raises(ValueError, match="local_mappo"):
        get_backend(legacy_backend)


def test_backend_factory_invalid():
    with pytest.raises(ValueError, match="local_mappo"):
        get_backend("not-a-backend")
