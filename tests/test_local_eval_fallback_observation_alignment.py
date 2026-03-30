from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ece324_tango.asce.trainers.base import EvalConfig
from ece324_tango.asce.trainers.local_mappo_backend import LocalMappoBackend


class _DummyActionSpace:
    n = 2


class _DummyEvalEnv:
    def __init__(self):
        self._step_idx = 0

    def reset(self, seed=None):
        self._step_idx = 0
        return {"a0": np.asarray([1.0], dtype=np.float32)}

    def action_spaces(self, agent):
        return _DummyActionSpace()

    def step(self, actions):
        self._step_idx += 1
        if self._step_idx == 1:
            # First step stays alive and updates obs to a distinct value.
            return (
                {"a0": np.asarray([9.0], dtype=np.float32)},
                {"a0": 0.0},
                {"a0": False},
                {"a0": {}},
            )
        return (
            {"a0": np.asarray([8.0], dtype=np.float32)},
            {"a0": 0.0},
            {"a0": True},
            {
                "a0": {},
            },
        )

    def close(self):
        return None


class _FakeTrainer:
    @staticmethod
    def checkpoint_use_obs_norm(path: str) -> bool:
        return False

    def __init__(self, *args, **kwargs):
        return None

    def load(self, in_path: str):
        return None

    def act_batch(self, obs_list, global_obs, n_valid_actions_list):
        return [{"action": 0, "logp": -0.1, "value": 0.0}]


class _FakeController:
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


def test_eval_metrics_fallback_uses_pre_step_observation(monkeypatch, tmp_path: Path):
    observed_values = []

    def _capture_metrics(**kwargs):
        observations = kwargs["observations"]
        observed_values.append(float(np.asarray(observations["a0"]).reshape(-1)[0]))
        return {}

    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"placeholder")
    cfg = EvalConfig(
        model_path=model_path,
        out_csv=tmp_path / "eval.csv",
        net_file="dummy.net.xml",
        route_file="dummy.rou.xml",
        seconds=10,
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
        reward_residual_weight=0.25,
        use_obs_norm=False,
    )

    monkeypatch.setattr(
        "ece324_tango.asce.trainers.local_mappo_backend.create_parallel_env",
        lambda **kwargs: _DummyEvalEnv(),
    )
    monkeypatch.setattr(
        "ece324_tango.asce.trainers.local_mappo_backend.MAPPOTrainer",
        _FakeTrainer,
    )
    monkeypatch.setattr(
        "ece324_tango.asce.trainers.local_mappo_backend.FixedTimeController",
        _FakeController,
    )
    monkeypatch.setattr(
        "ece324_tango.asce.trainers.local_mappo_backend.MaxPressureController",
        _FakeController,
    )
    monkeypatch.setattr(
        "ece324_tango.asce.trainers.local_mappo_backend.KPITracker",
        _FakeKPITracker,
    )
    monkeypatch.setattr(
        "ece324_tango.asce.trainers.local_mappo_backend.compute_metrics_for_agents",
        _capture_metrics,
    )
    monkeypatch.setattr(
        "ece324_tango.asce.trainers.local_mappo_backend.rewards_from_metrics",
        lambda metrics_by_agent, mode, weights, **kwargs: {},
    )

    LocalMappoBackend().evaluate(cfg)
    # One metrics call for each controller (mappo/fixed/max), each should see reset obs (1.0)
    assert observed_values == [1.0, 1.0, 1.0]
