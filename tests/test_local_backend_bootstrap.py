from pathlib import Path

import numpy as np

from ece324_tango.asce.trainers.base import TrainConfig
from ece324_tango.asce.trainers.local_mappo_backend import LocalMappoBackend


class _DummyActionSpace:
    n = 2


class _DummyEnv:
    def __init__(self, step_output):
        self._step_output = step_output
        self._stepped = False

    def reset(self, seed=None):
        self._stepped = False
        return {"a0": np.asarray([0.1], dtype=np.float32)}

    def action_spaces(self, agent):
        return _DummyActionSpace()

    def step(self, actions):
        if self._stepped:
            raise AssertionError("Expected exactly one step in this synthetic episode.")
        self._stepped = True
        return self._step_output

    def close(self):
        return None


class _FakeTrainer:
    instances = []

    def __init__(self, *args, **kwargs):
        self.obs_norm = None
        self.gobs_norm = None
        self.last_values_seen = None
        self.act_calls = 0
        _FakeTrainer.instances.append(self)

    def act_batch(self, obs_list, global_obs, n_valid_actions_list):
        return [{"action": 0, "logp": -0.1, "value": 0.0}]

    def act(self, obs, global_obs, n_valid_actions=None):
        self.act_calls += 1
        return {"action": 0, "logp": -0.1, "value": 42.0}

    def build_batch(self, trajectories, last_values=None):
        self.last_values_seen = dict(last_values or {})
        return {
            "obs": np.asarray([[0.0]], dtype=np.float32),
            "global_obs": np.asarray([[0.0]], dtype=np.float32),
            "actions": np.asarray([0], dtype=np.int64),
            "logp": np.asarray([0.0], dtype=np.float32),
            "returns": np.asarray([0.0], dtype=np.float32),
            "advantages": np.asarray([0.0], dtype=np.float32),
            "n_valid_actions": np.asarray([2], dtype=np.int64),
        }

    def update(self, batch, ppo_epochs=1, minibatch_size=1):
        return {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}

    def save(self, out_path):
        return None


def _train_cfg(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        model_path=tmp_path / "model.pt",
        rollout_csv=tmp_path / "rollout.csv",
        episode_metrics_csv=tmp_path / "episode_metrics.csv",
        net_file="dummy.net.xml",
        route_file="dummy.rou.xml",
        scenario_id="baseline",
        episodes=1,
        seconds=5,
        delta_time=5,
        ppo_epochs=1,
        minibatch_size=1,
        seed=7,
        use_gui=False,
        device="cpu",
        backend_verbose=False,
        reward_mode="objective",
        reward_delay_weight=1.0,
        reward_throughput_weight=1.0,
        reward_fairness_weight=0.25,
    )


def test_local_backend_bootstraps_last_value_on_truncation(monkeypatch, tmp_path: Path):
    trunc_step = (
        {"a0": np.asarray([0.2], dtype=np.float32)},
        {"a0": 0.0},
        {"a0": False},
        {"a0": True},
        {},
    )
    env = _DummyEnv(step_output=trunc_step)
    _FakeTrainer.instances.clear()

    monkeypatch.setattr(
        "ece324_tango.asce.trainers.local_mappo_backend.create_parallel_env",
        lambda **kwargs: env,
    )
    monkeypatch.setattr(
        "ece324_tango.asce.trainers.local_mappo_backend.MAPPOTrainer",
        _FakeTrainer,
    )

    LocalMappoBackend().train(_train_cfg(tmp_path))
    trainer = _FakeTrainer.instances[0]
    assert trainer.last_values_seen == {"a0": 42.0}
    assert trainer.act_calls == 1


def test_local_backend_does_not_bootstrap_on_true_termination(
    monkeypatch, tmp_path: Path
):
    term_step = (
        {"a0": np.asarray([0.2], dtype=np.float32)},
        {"a0": 0.0},
        {"a0": True},
        {"a0": False},
        {},
    )
    env = _DummyEnv(step_output=term_step)
    _FakeTrainer.instances.clear()

    monkeypatch.setattr(
        "ece324_tango.asce.trainers.local_mappo_backend.create_parallel_env",
        lambda **kwargs: env,
    )
    monkeypatch.setattr(
        "ece324_tango.asce.trainers.local_mappo_backend.MAPPOTrainer",
        _FakeTrainer,
    )

    LocalMappoBackend().train(_train_cfg(tmp_path))
    trainer = _FakeTrainer.instances[0]
    assert trainer.last_values_seen == {}
    assert trainer.act_calls == 0
