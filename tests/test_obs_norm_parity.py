from pathlib import Path

import pytest
import torch

from ece324_tango.asce.mappo import MAPPOTrainer
from ece324_tango.asce.trainers.base import EvalConfig
from ece324_tango.asce.trainers.local_mappo_backend import LocalMappoBackend


def test_checkpoint_persists_use_obs_norm_flag(tmp_path: Path):
    ckpt = tmp_path / "model.pt"
    trainer = MAPPOTrainer(obs_dim=1, global_obs_dim=1, n_actions=2, use_obs_norm=True)
    trainer.save(str(ckpt))

    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert payload.get("use_obs_norm") is True


def test_local_eval_rejects_obs_norm_mismatch_before_env_creation(
    monkeypatch, tmp_path: Path
):
    ckpt = tmp_path / "model.pt"
    trainer = MAPPOTrainer(obs_dim=1, global_obs_dim=1, n_actions=2, use_obs_norm=False)
    trainer.save(str(ckpt))

    def _unexpected_env_create(*args, **kwargs):
        raise AssertionError("create_parallel_env should not run on obs-norm mismatch")

    monkeypatch.setattr(
        "ece324_tango.asce.trainers.local_mappo_backend.create_parallel_env",
        _unexpected_env_create,
    )

    cfg = EvalConfig(
        model_path=ckpt,
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
        use_obs_norm=True,
    )

    with pytest.raises(RuntimeError, match="use_obs_norm"):
        LocalMappoBackend().evaluate(cfg)
