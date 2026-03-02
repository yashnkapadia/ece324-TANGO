from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from ece324_tango.asce.trainers.base import EvalConfig
from ece324_tango.asce.trainers.benchmarl_backend import BenchmarlBackend
from ece324_tango.asce.trainers.xuance_backend import XuanceBackend


class _FakeBaselineController:
    def __init__(self, *args, **kwargs):
        pass

    def actions(self, obs, env=None):
        return {a: 0 for a in obs.keys()}


class _FakeBaselineEnv:
    def reset(self, seed=None):
        return {"a0": np.asarray([0.0], dtype=np.float32)}

    def action_spaces(self, agent):
        return SimpleNamespace(n=2)

    def step(self, actions):
        obs = {"a0": np.asarray([0.0], dtype=np.float32)}
        rewards = {"a0": 0.0}
        dones = {"a0": True}
        infos = {"a0": {}}
        return obs, rewards, dones, infos

    def close(self):
        return None


class _FakeKPITracker:
    def update(self, env):
        return None

    def summary(self):
        return SimpleNamespace(
            time_loss_s=0.0,
            person_time_loss_s=0.0,
            avg_trip_time_s=0.0,
            arrived_vehicles=0,
        )


def _eval_cfg(tmp_path: Path, model_path: Path) -> EvalConfig:
    return EvalConfig(
        model_path=model_path,
        out_csv=tmp_path / "eval.csv",
        net_file="dummy.net.xml",
        route_file="dummy.rou.xml",
        seconds=30,
        delta_time=5,
        episodes=3,
        seed=17,
        use_gui=False,
        device="cpu",
        backend_verbose=False,
        reward_mode="objective",
        reward_delay_weight=1.0,
        reward_throughput_weight=1.0,
        reward_fairness_weight=0.25,
        use_obs_norm=False,
    )


def test_benchmarl_eval_uses_seed_plus_episode(monkeypatch, tmp_path: Path):
    model_path = tmp_path / "bench.pt"
    torch.save({"backend": "benchmarl", "state_dict": {}}, model_path)
    cfg = _eval_cfg(tmp_path, model_path)

    used_seeds = []

    class _FakeExp:
        def __init__(self, seed):
            self.seed = seed

        def load_state_dict(self, state_dict):
            return None

        def close(self):
            return None

    def _fake_build(self, train_cfg, seed, device, quiet_sumo):
        used_seeds.append(seed)
        return _FakeExp(seed)

    monkeypatch.setattr(
        BenchmarlBackend, "_ensure_available", staticmethod(lambda: None)
    )
    monkeypatch.setattr(BenchmarlBackend, "_build_experiment", _fake_build)
    monkeypatch.setattr(
        BenchmarlBackend,
        "_rollout_episode_stats",
        staticmethod(lambda exp, deterministic=True: ({}, 0.0, {"a0": 0.0}, 1)),
    )
    monkeypatch.setattr(
        BenchmarlBackend,
        "_kpi_from_rollout_replay",
        staticmethod(
            lambda rollout, cfg: SimpleNamespace(
                time_loss_s=0.0,
                person_time_loss_s=0.0,
                avg_trip_time_s=0.0,
                arrived_vehicles=0,
            )
        ),
    )
    monkeypatch.setattr(
        "ece324_tango.asce.trainers.benchmarl_backend.create_parallel_env",
        lambda **kwargs: _FakeBaselineEnv(),
    )
    monkeypatch.setattr(
        "ece324_tango.asce.trainers.benchmarl_backend.FixedTimeController",
        _FakeBaselineController,
    )
    monkeypatch.setattr(
        "ece324_tango.asce.trainers.benchmarl_backend.MaxPressureController",
        _FakeBaselineController,
    )
    monkeypatch.setattr(
        "ece324_tango.asce.trainers.benchmarl_backend.KPITracker", _FakeKPITracker
    )
    monkeypatch.setattr(
        "ece324_tango.asce.trainers.benchmarl_backend.compute_metrics_for_agents",
        lambda **kwargs: {},
    )
    monkeypatch.setattr(
        "ece324_tango.asce.trainers.benchmarl_backend.rewards_from_metrics",
        lambda metrics_by_agent, mode, weights: {},
    )

    BenchmarlBackend().evaluate(cfg)
    assert used_seeds == [cfg.seed + ep for ep in range(cfg.episodes)]


def test_xuance_eval_uses_seed_plus_episode(monkeypatch, tmp_path: Path):
    model_path = tmp_path / "xuance.pt"
    torch.save({"backend": "xuance"}, model_path)
    (tmp_path / "xuance_xuance").mkdir(parents=True, exist_ok=True)
    cfg = _eval_cfg(tmp_path, model_path)

    used_episode_seeds = []

    class _FakeAgents:
        def load_model(self, path: str):
            return None

        def finish(self):
            return None

    class _FakeRunner:
        def __init__(self):
            self.agents = _FakeAgents()
            self.envs = SimpleNamespace(close=lambda: None)

    def _fake_run_episode(cfg, agent, deterministic=False, episode_seed=None):
        used_episode_seeds.append(episode_seed)
        return (
            [],
            0.0,
            {"a0": 0.0},
            1,
            SimpleNamespace(
                time_loss_s=0.0,
                person_time_loss_s=0.0,
                avg_trip_time_s=0.0,
                arrived_vehicles=0,
            ),
        )

    monkeypatch.setattr(XuanceBackend, "_ensure_available", staticmethod(lambda: None))
    monkeypatch.setattr(
        XuanceBackend, "_build_runner", lambda self, cfg, seed: _FakeRunner()
    )
    monkeypatch.setattr(
        "ece324_tango.asce.trainers.xuance_backend.XuanceBackend._run_episode_with_agent",
        staticmethod(_fake_run_episode),
    )
    monkeypatch.setattr(
        "ece324_tango.asce.trainers.xuance_backend.create_parallel_env",
        lambda **kwargs: _FakeBaselineEnv(),
    )
    monkeypatch.setattr(
        "ece324_tango.asce.trainers.xuance_backend.FixedTimeController",
        _FakeBaselineController,
    )
    monkeypatch.setattr(
        "ece324_tango.asce.trainers.xuance_backend.MaxPressureController",
        _FakeBaselineController,
    )
    monkeypatch.setattr(
        "ece324_tango.asce.trainers.xuance_backend.KPITracker", _FakeKPITracker
    )
    monkeypatch.setattr(
        "ece324_tango.asce.trainers.xuance_backend.compute_metrics_for_agents",
        lambda **kwargs: {},
    )
    monkeypatch.setattr(
        "ece324_tango.asce.trainers.xuance_backend.rewards_from_metrics",
        lambda metrics_by_agent, mode, weights: {},
    )

    XuanceBackend().evaluate(cfg)
    assert used_episode_seeds == [cfg.seed + ep for ep in range(cfg.episodes)]
