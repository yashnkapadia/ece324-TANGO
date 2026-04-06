from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ece324_tango.asce.trainers.base import EvalConfig
from ece324_tango.asce.trainers.local_mappo_backend import LocalMappoBackend


class _DummyActionSpace:
    n = 2


class _OneStepEvalEnv:
    def reset(self, seed=None):
        return {"a0": np.asarray([1.0], dtype=np.float32)}

    def action_spaces(self, agent):
        return _DummyActionSpace()

    def step(self, actions):
        return (
            {"a0": np.asarray([2.0], dtype=np.float32)},
            {"a0": 0.0},
            {"a0": True},
            {"a0": {}},
        )

    def close(self):
        return None


class _FakeTrainer:
    @staticmethod
    def checkpoint_use_obs_norm(path: str) -> bool:
        return False

    def __init__(self, *args, **kwargs):
        return None

    def load(self, path: str):
        return None

    def act_batch(self, obs_list, global_obs, n_valid_actions_list):
        return [{"action": 0, "logp": -0.1, "value": 0.0}]


class _TrackingFixedTimeController:
    actions_seen: list[int] = []
    reset_calls = 0

    def __init__(self, *args, **kwargs):
        self.cursor = 0

    def reset(self):
        type(self).reset_calls += 1
        self.cursor = 0

    def actions(self, observations, env=None):
        action = self.cursor
        type(self).actions_seen.append(action)
        self.cursor = (self.cursor + 1) % 2
        return {a: action for a in observations.keys()}


class _FakeMaxPressureController:
    def __init__(self, *args, **kwargs):
        return None

    def actions(self, observations, env=None):
        return {a: 0 for a in observations.keys()}


class _FakeKPITracker:
    def update(self, env):
        return None

    def summary(self):
        return SimpleNamespace(
            time_loss_s=0.0,
            person_time_loss_s=0.0,
            avg_trip_time_s=0.0,
            arrived_vehicles=0,
            vehicle_delay_jain=0.0,
        )


def test_fixed_time_baseline_resets_cursor_each_eval_episode(monkeypatch, tmp_path: Path):
    _TrackingFixedTimeController.actions_seen = []
    _TrackingFixedTimeController.reset_calls = 0

    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"placeholder")
    cfg = EvalConfig(
        model_path=model_path,
        out_csv=tmp_path / "eval.csv",
        net_file="dummy.net.xml",
        route_file="dummy.rou.xml",
        seconds=5,
        delta_time=5,
        episodes=2,
        seed=17,
        use_gui=False,
        device="cpu",
        backend_verbose=False,
        reward_mode="objective",
        reward_delay_weight=1.0,
        reward_throughput_weight=1.0,
        reward_fairness_weight=0.25,
        reward_residual_weight=0.25,
        use_obs_norm=False,
    )

    monkeypatch.setattr(
        "ece324_tango.asce.trainers.local_mappo_backend.create_parallel_env",
        lambda **kwargs: _OneStepEvalEnv(),
    )
    monkeypatch.setattr(
        "ece324_tango.asce.trainers.local_mappo_backend.MAPPOTrainer",
        _FakeTrainer,
    )
    monkeypatch.setattr(
        "ece324_tango.asce.trainers.local_mappo_backend.FixedTimeController",
        _TrackingFixedTimeController,
    )
    monkeypatch.setattr(
        "ece324_tango.asce.trainers.local_mappo_backend.MaxPressureController",
        _FakeMaxPressureController,
    )
    monkeypatch.setattr(
        "ece324_tango.asce.trainers.local_mappo_backend.KPITracker",
        _FakeKPITracker,
    )
    monkeypatch.setattr(
        "ece324_tango.asce.trainers.local_mappo_backend.compute_metrics_for_agents",
        lambda **kwargs: {},
    )
    monkeypatch.setattr(
        "ece324_tango.asce.trainers.local_mappo_backend.rewards_from_metrics",
        lambda metrics_by_agent, mode, weights: {},
    )

    LocalMappoBackend().evaluate(cfg)

    assert _TrackingFixedTimeController.actions_seen == [0, 0]
    assert _TrackingFixedTimeController.reset_calls == cfg.episodes
