from pathlib import Path

import pytest

from ece324_tango.asce.trainers.base import EvalConfig
from ece324_tango.asce.trainers.local_mappo_backend import LocalMappoBackend


def test_local_eval_checks_model_exists_before_creating_env(monkeypatch, tmp_path: Path):
    def _unexpected_env_create(*args, **kwargs):
        raise AssertionError("create_parallel_env should not run when model file is missing")

    monkeypatch.setattr(
        "ece324_tango.asce.trainers.local_mappo_backend.create_parallel_env",
        _unexpected_env_create,
    )

    cfg = EvalConfig(
        model_path=tmp_path / "missing_model.pt",
        out_csv=tmp_path / "out.csv",
        net_file="dummy.net.xml",
        route_file="dummy.rou.xml",
        seconds=60,
        delta_time=5,
        episodes=1,
        seed=17,
        use_gui=False,
        device="cpu",
        backend_verbose=False,
        reward_mode="objective",
        reward_delay_weight=1.0,
        reward_throughput_weight=1.0,
        reward_fairness_weight=0.25,
    )

    with pytest.raises(FileNotFoundError, match="missing_model.pt"):
        LocalMappoBackend().evaluate(cfg)
