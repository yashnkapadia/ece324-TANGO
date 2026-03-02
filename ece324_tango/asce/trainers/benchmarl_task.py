from __future__ import annotations

from typing import Dict, List, Optional

from pettingzoo.utils.env import ParallelEnv
from torchrl.data import Composite
from torchrl.envs import EnvBase
from torchrl.envs.libs.pettingzoo import PettingZooWrapper

from benchmarl.environments.common import TaskClass

from ece324_tango.asce.env import create_parallel_env
from ece324_tango.asce.runtime import extract_reset_obs
from ece324_tango.asce.traffic_metrics import (
    RewardWeights,
    compute_metrics_for_agents,
    rewards_from_metrics,
)


class SumoParallelAdapter(ParallelEnv):
    """PettingZoo ParallelEnv adapter around sumo-rl SumoEnvironment."""

    metadata = {"render_modes": ["human", "rgb_array"], "name": "sumo_parallel_adapter"}

    def __init__(self, base_env):
        self.base_env = base_env
        self.possible_agents = list(base_env.ts_ids)
        self.agents = list(self.possible_agents)
        self.scenario_id = "baseline"
        self.reward_mode = "sumo"
        self.reward_weights = RewardWeights(delay=1.0, throughput=1.0, fairness=0.25)
        self._step_idx = 0

    def configure_reward(self, scenario_id: str, mode: str, weights: RewardWeights):
        self.scenario_id = scenario_id
        self.reward_mode = mode
        self.reward_weights = weights

    def reset(self, seed=None, options=None):
        obs = extract_reset_obs(self.base_env.reset(seed=seed))
        self.agents = list(obs.keys())
        self._step_idx = 0
        infos = {agent: {} for agent in self.possible_agents}
        return obs, infos

    def step(self, actions):
        step_out = self.base_env.step(actions)
        if len(step_out) == 5:
            obs, rewards, terminations, truncations, infos = step_out
            dones = {
                a: bool(terminations.get(a, False) or truncations.get(a, False))
                for a in self.possible_agents
            }
        else:
            obs, rewards, dones, infos = step_out
        self._step_idx += 1
        sim_time = float(self._step_idx) * float(
            getattr(self.base_env, "delta_time", 1)
        )
        metrics_by_agent = compute_metrics_for_agents(
            env=self.base_env,
            agent_ids=self.possible_agents,
            time_step=sim_time,
            actions={a: int(actions.get(a, 0)) for a in self.possible_agents},
            action_green_dur=float(getattr(self.base_env, "delta_time", 1)),
            scenario_id=self.scenario_id,
            observations=obs,
        )
        shaped = rewards_from_metrics(
            metrics_by_agent, mode=self.reward_mode, weights=self.reward_weights
        )
        if shaped:
            rewards = shaped
        terminations = {a: bool(dones.get(a, False)) for a in self.possible_agents}
        truncations = {a: False for a in self.possible_agents}
        self.agents = [
            a for a in self.possible_agents if not (terminations[a] or truncations[a])
        ]
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

    def get_env_fun(
        self, num_envs: int, continuous_actions: bool, seed: Optional[int], device
    ):
        def make_env():
            base_env = create_parallel_env(
                net_file=self.config["net_file"],
                route_file=self.config["route_file"],
                seed=self.config["seed"] if seed is None else seed,
                use_gui=self.config.get("use_gui", False),
                seconds=self.config["seconds"],
                delta_time=self.config["delta_time"],
                quiet_sumo=self.config.get("quiet_sumo", False),
            )
            pz_env = SumoParallelAdapter(base_env)
            pz_env.configure_reward(
                scenario_id=self.config.get("scenario_id", "baseline"),
                mode=self.config.get("reward_mode", "objective"),
                weights=RewardWeights(
                    delay=float(self.config.get("reward_delay_weight", 1.0)),
                    throughput=float(self.config.get("reward_throughput_weight", 1.0)),
                    fairness=float(self.config.get("reward_fairness_weight", 0.25)),
                ),
            )
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
