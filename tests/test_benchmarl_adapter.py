from __future__ import annotations

from gymnasium.spaces import Box, Discrete
import numpy as np

from ece324_tango.asce.trainers.benchmarl_task import SumoParallelAdapter


class _DummyBaseEnv:
    def __init__(self):
        self.ts_ids = ["tls_0", "tls_1"]

    def reset(self, seed=None):
        return {
            "tls_0": np.array([0.0, 1.0], dtype=np.float32),
            "tls_1": np.array([1.0, 0.0], dtype=np.float32),
        }

    def step(self, actions):
        obs = {
            "tls_0": np.array([0.5, 0.5], dtype=np.float32),
            "tls_1": np.array([0.3, 0.7], dtype=np.float32),
        }
        rewards = {"tls_0": 1.0, "tls_1": -1.0}
        dones = {"tls_0": False, "tls_1": True}
        infos = {"tls_0": {}, "tls_1": {}}
        return obs, rewards, dones, infos

    def observation_spaces(self, agent):
        return Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

    def action_spaces(self, agent):
        return Discrete(4)

    def close(self):
        return None


def test_sumo_parallel_adapter_reset_and_step_shapes():
    adapter = SumoParallelAdapter(_DummyBaseEnv())
    obs, infos = adapter.reset(seed=7)
    assert set(obs.keys()) == {"tls_0", "tls_1"}
    assert set(infos.keys()) == {"tls_0", "tls_1"}

    next_obs, rewards, terminations, truncations, next_infos = adapter.step(
        {"tls_0": 0, "tls_1": 1}
    )
    assert set(next_obs.keys()) == {"tls_0", "tls_1"}
    assert rewards["tls_0"] == 1.0
    assert terminations["tls_1"] is True
    assert truncations["tls_0"] is False
    assert set(next_infos.keys()) == {"tls_0", "tls_1"}


def test_sumo_parallel_adapter_spaces_proxy_to_base_env():
    adapter = SumoParallelAdapter(_DummyBaseEnv())
    assert isinstance(adapter.observation_space("tls_0"), Box)
    assert isinstance(adapter.action_space("tls_0"), Discrete)
