from __future__ import annotations

from dataclasses import dataclass
from typing import List

from gymnasium import spaces
import numpy as np

from ece324_tango.asce.env import create_parallel_env
from ece324_tango.asce.runtime import extract_reset_obs
from ece324_tango.asce.traffic_metrics import (
    RewardWeights,
    compute_metrics_for_agents,
    rewards_from_metrics,
)


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
                quiet_sumo=bool(getattr(config, "sumo_quiet", False)),
            )
            self.env = self._base_env
            self.agents: List[str] = list(self._base_env.ts_ids)
            self.num_agents = len(self.agents)
            self.agent_groups = [self.agents]
            self.observation_space = {
                a: self._base_env.observation_spaces(a) for a in self.agents
            }
            self.action_space = {
                a: self._base_env.action_spaces(a) for a in self.agents
            }

            obs_dim = int(
                sum(int(np.prod(self.observation_space[a].shape)) for a in self.agents)
            )
            self.state_space = spaces.Box(
                low=-1e9,
                high=1e9,
                shape=(obs_dim,),
                dtype=np.float32,
            )
            self.max_episode_steps = max(
                1, int(config.sumo_seconds) // int(config.sumo_delta_time)
            )
            self.render_mode = "human"
            self._episode_step = 0
            self._last_obs = {
                a: np.zeros(self.observation_space[a].shape, dtype=np.float32)
                for a in self.agents
            }
            self._scenario_id = getattr(config, "scenario_id", "baseline")
            self._reward_mode = (
                str(getattr(config, "reward_mode", "objective")).strip().lower()
            )
            self._reward_weights = RewardWeights(
                delay=float(getattr(config, "reward_delay_weight", 1.0)),
                throughput=float(getattr(config, "reward_throughput_weight", 1.0)),
                fairness=float(getattr(config, "reward_fairness_weight", 0.25)),
            )

        def get_env_info(self):
            return {
                "state_space": self.state_space,
                "observation_space": self.observation_space,
                "action_space": self.action_space,
                "agents": self.agents,
                "num_agents": self.num_agents,
                "max_episode_steps": self.max_episode_steps,
            }

        def state(self):
            return np.concatenate(
                [
                    np.asarray(self._last_obs[a], dtype=np.float32).ravel()
                    for a in self.agents
                ],
                axis=0,
            )

        def agent_mask(self):
            return {a: True for a in self.agents}

        def avail_actions(self):
            return {
                a: np.ones(self.action_space[a].n, dtype=np.bool_) for a in self.agents
            }

        def reset(self):
            obs = extract_reset_obs(self._base_env.reset(seed=self._base_env.sumo_seed))
            self._episode_step = 0
            obs = {a: np.asarray(obs[a], dtype=np.float32) for a in self.agents}
            self._last_obs = obs
            info = {
                "infos": {a: {} for a in self.agents},
                "individual_episode_rewards": {a: 0.0 for a in self.agents},
                "state": self.state(),
                "agent_mask": self.agent_mask(),
                "avail_actions": self.avail_actions(),
            }
            return obs, info

        def step(self, actions):
            step_out = self._base_env.step(actions)
            self._episode_step += 1
            if len(step_out) == 5:
                obs, rewards, terminations, truncations, infos = step_out
                terminated = {a: bool(terminations.get(a, False)) for a in self.agents}
                truncated = bool(
                    all(bool(truncations.get(a, False)) for a in self.agents)
                )
            else:
                obs, rewards, dones, infos = step_out
                terminated = {a: bool(dones.get(a, False)) for a in self.agents}
                truncated = False
            obs = {a: np.asarray(obs[a], dtype=np.float32) for a in self.agents}
            self._last_obs = obs
            truncated = bool(truncated or self._episode_step >= self.max_episode_steps)
            rewards = {
                a: float(np.asarray(r).reshape(-1)[0]) for a, r in rewards.items()
            }
            sim_time = float(self._episode_step) * float(
                getattr(self._base_env, "delta_time", 1)
            )
            metrics_by_agent = compute_metrics_for_agents(
                env=self._base_env,
                agent_ids=self.agents,
                time_step=sim_time,
                actions={
                    a: int(np.asarray(actions[a]).reshape(-1)[0]) for a in self.agents
                },
                action_green_dur=float(getattr(self._base_env, "delta_time", 1)),
                scenario_id=self._scenario_id,
                observations=obs,
            )
            shaped_rewards = rewards_from_metrics(
                metrics_by_agent, mode=self._reward_mode, weights=self._reward_weights
            )
            if shaped_rewards:
                rewards = shaped_rewards
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
