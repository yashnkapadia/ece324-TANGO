from ece324_tango.asce.trainers.local_mappo_backend import LocalMappoBackend


def test_resolve_device_explicit_cpu():
    assert LocalMappoBackend._resolve_device("cpu") == "cpu"


def test_resolve_device_explicit_cuda():
    assert LocalMappoBackend._resolve_device("cuda") == "cuda"
