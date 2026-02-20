from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from gymnasium import spaces
import numpy as np

from ece324_tango.asce.env import create_parallel_env


@dataclass
class XuanceSumoEnvConfig:
    net_file: str
    route_file: str
    seed: int
    use_gui: bool
    seconds: int
    delta_time: int


def register_xuance_sumo_env(env_name: str = "sumo_custom") -> str:
    from xuance.environment import RawMultiAgentEnv
    from xuance.environment.multi_agent_env import REGISTRY_MULTI_AGENT_ENV

    class XuanceSumoEnv(RawMultiAgentEnv):
        def __init__(self, config):
            super().__init__()
            self._base_env = create_parallel_env(
                net_file=config.sumo_net_file,
                route_file=config.sumo_route_file,
                seed=config.env_seed,
                use_gui=getattr(config, "use_gui", False),
                seconds=int(config.sumo_seconds),
                delta_time=int(config.sumo_delta_time),
            )
            self.agents: List[str] = list(self._base_env.ts_ids)
            self.num_agents = len(self.agents)
            self.agent_groups = [self.agents]
            self.observation_space = {a: self._base_env.observation_spaces(a) for a in self.agents}
            self.action_space = {a: self._base_env.action_spaces(a) for a in self.agents}

            obs_dim = int(
                sum(int(np.prod(self.observation_space[a].shape)) for a in self.agents)
            )
            self.state_space = spaces.Box(
                low=-1e9,
                high=1e9,
                shape=(obs_dim,),
                dtype=np.float32,
            )
            self.max_episode_steps = max(1, int(config.sumo_seconds) // int(config.sumo_delta_time))
            self.render_mode = "human"
            self._episode_step = 0
            self._last_obs = {
                a: np.zeros(self.observation_space[a].shape, dtype=np.float32) for a in self.agents
            }

        def state(self):
            return np.concatenate(
                [np.asarray(self._last_obs[a], dtype=np.float32).ravel() for a in self.agents],
                axis=0,
            )

        def agent_mask(self):
            return {a: True for a in self.agents}

        def avail_actions(self):
            return {a: np.ones(self.action_space[a].n, dtype=np.bool_) for a in self.agents}

        def reset(self):
            obs = self._base_env.reset(seed=self._base_env.sumo_seed)
            self._episode_step = 0
            self._last_obs = {a: np.asarray(obs[a], dtype=np.float32) for a in self.agents}
            info = {
                "infos": {a: {} for a in self.agents},
                "individual_episode_rewards": {a: 0.0 for a in self.agents},
                "state": self.state(),
                "agent_mask": self.agent_mask(),
                "avail_actions": self.avail_actions(),
            }
            return obs, info

        def step(self, actions):
            obs, rewards, dones, infos = self._base_env.step(actions)
            self._episode_step += 1
            self._last_obs = {a: np.asarray(obs[a], dtype=np.float32) for a in self.agents}
            terminated = {a: bool(dones.get(a, False)) for a in self.agents}
            truncated = bool(self._episode_step >= self.max_episode_steps)
            rewards = {a: float(np.asarray(r).reshape(-1)[0]) for a, r in rewards.items()}
            info = {
                "infos": infos,
                "individual_episode_rewards": rewards,
                "state": self.state(),
                "agent_mask": self.agent_mask(),
                "avail_actions": self.avail_actions(),
            }
            return obs, rewards, terminated, truncated, info

        def close(self):
            self._base_env.close()

        def render(self, *args, **kwargs):
            return None

    REGISTRY_MULTI_AGENT_ENV[env_name] = XuanceSumoEnv
    return env_name
