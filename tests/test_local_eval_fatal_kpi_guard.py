from pathlib import Path
from types import SimpleNamespace

import numpy as np
from traci.exceptions import FatalTraCIError

from ece324_tango.asce.trainers.base import EvalConfig
from ece324_tango.asce.trainers.local_mappo_backend import LocalMappoBackend


class _DummyActionSpace:
    n = 2


class _FatalOnFirstStepEnv:
    def __init__(self):
        self.fatal_seen = False

    def reset(self, seed=None):
        return {"a0": np.asarray([0.0], dtype=np.float32)}

    def action_spaces(self, agent):
        return _DummyActionSpace()

    def step(self, actions):
        self.fatal_seen = True
        raise FatalTraCIError("Connection closed by SUMO")

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


class _KPIAssertsNotCalledAfterFatal:
    def __init__(self):
        self.count = 0

    def update(self, env):
        self.count += 1
        if getattr(env, "fatal_seen", False):
            raise AssertionError(
                "kpi.update() should not run after FatalTraCIError in eval loop"
            )

    def summary(self):
        return SimpleNamespace(
            time_loss_s=0.0,
            person_time_loss_s=0.0,
            avg_trip_time_s=0.0,
            arrived_vehicles=0,
        )


def test_local_eval_skips_kpi_update_after_fatal_step(monkeypatch, tmp_path: Path):
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"placeholder")
    cfg = EvalConfig(
        model_path=model_path,
        out_csv=tmp_path / "eval.csv",
        net_file="dummy.net.xml",
        route_file="dummy.rou.xml",
        seconds=30,
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
        use_obs_norm=False,
    )

    monkeypatch.setattr(
        "ece324_tango.asce.trainers.local_mappo_backend.create_parallel_env",
        lambda **kwargs: _FatalOnFirstStepEnv(),
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
        _KPIAssertsNotCalledAfterFatal,
    )

    LocalMappoBackend().evaluate(cfg)
