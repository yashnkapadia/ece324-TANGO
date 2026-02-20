from __future__ import annotations

from typing import Dict, List, Optional

from pettingzoo.utils.env import ParallelEnv
from torchrl.data import Composite
from torchrl.envs import EnvBase
from torchrl.envs.libs.pettingzoo import PettingZooWrapper

from benchmarl.environments.common import TaskClass

from ece324_tango.asce.env import create_parallel_env


class SumoParallelAdapter(ParallelEnv):
    """PettingZoo ParallelEnv adapter around sumo-rl SumoEnvironment."""

    metadata = {"render_modes": ["human", "rgb_array"], "name": "sumo_parallel_adapter"}

    def __init__(self, base_env):
        self.base_env = base_env
        self.possible_agents = list(base_env.ts_ids)
        self.agents = list(self.possible_agents)

    def reset(self, seed=None, options=None):
        obs = self.base_env.reset(seed=seed)
        self.agents = list(obs.keys())
        infos = {agent: {} for agent in self.possible_agents}
        return obs, infos

    def step(self, actions):
        obs, rewards, dones, infos = self.base_env.step(actions)
        terminations = {a: bool(dones.get(a, False)) for a in self.possible_agents}
        truncations = {a: False for a in self.possible_agents}
        self.agents = [a for a in self.possible_agents if not (terminations[a] or truncations[a])]
        return obs, rewards, terminations, truncations, infos

    def observation_space(self, agent):
        return self.base_env.observation_spaces(agent)

    def action_space(self, agent):
        return self.base_env.action_spaces(agent)

    def close(self):
        self.base_env.close()


class SumoBenchmarlTask(TaskClass):
    @staticmethod
    def env_name() -> str:
        return "sumo_custom"

    def get_env_fun(self, num_envs: int, continuous_actions: bool, seed: Optional[int], device):
        def make_env():
            base_env = create_parallel_env(
                net_file=self.config["net_file"],
                route_file=self.config["route_file"],
                seed=self.config["seed"] if seed is None else seed,
                use_gui=self.config.get("use_gui", False),
                seconds=self.config["seconds"],
                delta_time=self.config["delta_time"],
            )
            pz_env = SumoParallelAdapter(base_env)
            return PettingZooWrapper(
                env=pz_env,
                categorical_actions=True,
                seed=seed,
                use_mask=False,
                done_on_any=True,
            )

        return make_env

    def supports_continuous_actions(self) -> bool:
        return False

    def supports_discrete_actions(self) -> bool:
        return True

    def max_steps(self, env: EnvBase) -> int:
        return int(max(1, self.config["seconds"] // self.config["delta_time"]))

    def has_render(self, env: EnvBase) -> bool:
        return False

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        return env.group_map

    def observation_spec(self, env: EnvBase) -> Composite:
        observation_spec = env.observation_spec.clone()
        for group in self.group_map(env):
            group_obs_spec = observation_spec[group]
            for key in list(group_obs_spec.keys()):
                if key != "observation":
                    del group_obs_spec[key]
        if "state" in observation_spec.keys():
            del observation_spec["state"]
        return observation_spec

    def info_spec(self, env: EnvBase):
        return None

    def state_spec(self, env: EnvBase):
        return None

    def action_spec(self, env: EnvBase) -> Composite:
        return env.full_action_spec

    def action_mask_spec(self, env: EnvBase):
        return None
