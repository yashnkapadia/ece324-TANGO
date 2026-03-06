from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from ece324_tango.asce.trainers.base import EvalConfig
from ece324_tango.asce.trainers.local_mappo_backend import LocalMappoBackend


class _DummyActionSpace:
    n = 2


class _TwoStepEvalEnv:
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
            return (
                {"a0": np.asarray([2.0], dtype=np.float32)},
                {"a0": 0.0},
                {"a0": False},
                {"a0": {}},
            )
        return (
            {"a0": np.asarray([3.0], dtype=np.float32)},
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


def test_local_eval_records_objective_score_for_all_controllers(
    monkeypatch, tmp_path: Path
):
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"placeholder")
    out_csv = tmp_path / "eval.csv"
    cfg = EvalConfig(
        model_path=model_path,
        out_csv=out_csv,
        net_file="dummy.net.xml",
        route_file="dummy.rou.xml",
        seconds=10,
        delta_time=5,
        episodes=1,
        seed=17,
        use_gui=False,
        device="cpu",
        backend_verbose=False,
        reward_mode="time_loss",
        reward_delay_weight=1.0,
        reward_throughput_weight=1.0,
        reward_fairness_weight=0.25,
        use_obs_norm=False,
    )

    monkeypatch.setattr(
        "ece324_tango.asce.trainers.local_mappo_backend.create_parallel_env",
        lambda **kwargs: _TwoStepEvalEnv(),
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
        lambda **kwargs: {"a0": object()},
    )

    def _rewards_from_metrics(metrics_by_agent, mode, weights):
        if mode == "time_loss":
            return {"a0": 1.5}
        if mode == "objective":
            return {"a0": 3.0}
        if mode == "sumo":
            return {}
        raise AssertionError(f"unexpected reward mode: {mode}")

    monkeypatch.setattr(
        "ece324_tango.asce.trainers.local_mappo_backend.rewards_from_metrics",
        _rewards_from_metrics,
    )

    LocalMappoBackend().evaluate(cfg)
    df = pd.read_csv(out_csv)

    for col in [
        "objective_mean_reward",
        "objective_delay_proxy",
        "objective_throughput_proxy",
        "objective_fairness_jain",
    ]:
        assert col in df.columns

    max_pressure_row = df.loc[df["controller"] == "max_pressure"].iloc[0]
    # Two-step episode: shaped reward on first live step (1.5), raw reward 0.0 on terminal step.
    assert max_pressure_row["mean_reward"] == 0.75
    assert max_pressure_row["objective_mean_reward"] == 3.0
