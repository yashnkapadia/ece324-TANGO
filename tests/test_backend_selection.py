import pytest

from ece324_tango.asce.trainers.factory import get_backend
from ece324_tango.asce.trainers.local_mappo_backend import LocalMappoBackend
from ece324_tango.asce.trainers.benchmarl_backend import BenchmarlBackend
from ece324_tango.asce.trainers.xuance_backend import XuanceBackend


def test_backend_factory_local():
    backend = get_backend("local_mappo")
    assert isinstance(backend, LocalMappoBackend)


def test_backend_factory_benchmarl():
    backend = get_backend("benchmarl")
    assert isinstance(backend, BenchmarlBackend)


def test_backend_factory_xuance():
    backend = get_backend("xuance")
    assert isinstance(backend, XuanceBackend)


def test_backend_factory_invalid():
    with pytest.raises(ValueError):
        get_backend("not-a-backend")
