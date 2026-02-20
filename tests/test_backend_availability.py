import pytest

from ece324_tango.asce.trainers.benchmarl_backend import BenchmarlBackend
from ece324_tango.asce.trainers.libsignal_backend import LibsignalBackend
from ece324_tango.asce.trainers.xuance_backend import XuanceBackend


def test_benchmarl_backend_availability_check(monkeypatch):
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    with pytest.raises(RuntimeError):
        BenchmarlBackend._ensure_available()


def test_xuance_backend_availability_check(monkeypatch):
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    with pytest.raises(RuntimeError):
        XuanceBackend._ensure_available()


def test_libsignal_backend_is_explicitly_unimplemented():
    with pytest.raises(RuntimeError, match="not yet wired"):
        LibsignalBackend().train(None)  # type: ignore[arg-type]
