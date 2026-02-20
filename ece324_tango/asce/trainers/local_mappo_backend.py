from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from loguru import logger

from ece324_tango.asce.baselines import FixedTimeController, MaxPressureController
from ece324_tango.asce.env import create_parallel_env, flatten_obs_by_agent
from ece324_tango.asce.kpi import KPITracker
from ece324_tango.asce.mappo import MAPPOTrainer, Transition
from ece324_tango.asce.runtime import extract_reset_obs, extract_step, jain_index
from ece324_tango.asce.schema import ASCE_DATASET_COLUMNS
from ece324_tango.asce.traffic_metrics import (
    RewardWeights,
    compute_metrics_for_agents,
    rewards_from_metrics,
)
from ece324_tango.asce.trainers.base import AsceTrainerBackend, EvalConfig, TrainConfig


class LocalMappoBackend(AsceTrainerBackend):
    name = "local_mappo"

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def train(self, cfg: TrainConfig) -> None:
        cfg.model_path.parent.mkdir(parents=True, exist_ok=True)
        cfg.rollout_csv.parent.mkdir(parents=True, exist_ok=True)
        cfg.episode_metrics_csv.parent.mkdir(parents=True, exist_ok=True)

        env = create_parallel_env(
            net_file=cfg.net_file,
            route_file=cfg.route_file,
            seed=cfg.seed,
            use_gui=cfg.use_gui,
            seconds=cfg.seconds,
            delta_time=cfg.delta_time,
            quiet_sumo=not cfg.backend_verbose,
        )

        obs = extract_reset_obs(env.reset(seed=cfg.seed))
        if not obs:
            raise RuntimeError("No observations received from SUMO environment.")

        ordered_agents = sorted(obs.keys())
        first_agent = ordered_agents[0]
        obs_dim = int(np.asarray(obs[first_agent], dtype=np.float32).size)
        global_obs_dim = int(flatten_obs_by_agent(obs, ordered_agents).size)

        action_dims = {a: int(env.action_spaces(a).n) for a in ordered_agents}
        n_actions = min(action_dims.values())

        resolved_device = self._resolve_device(cfg.device)
        logger.info(f"Training device: {resolved_device}")

        trainer = MAPPOTrainer(
            obs_dim=obs_dim,
            global_obs_dim=global_obs_dim,
            n_actions=n_actions,
            device=resolved_device,
        )

        all_rows: List[dict] = []
        ep_metrics: List[dict] = []
        reward_weights = RewardWeights(
            delay=cfg.reward_delay_weight,
            throughput=cfg.reward_throughput_weight,
            fairness=cfg.reward_fairness_weight,
        )

        for ep in range(cfg.episodes):
            obs = extract_reset_obs(env.reset(seed=cfg.seed + ep))
            if not obs:
                continue

            trajectories: Dict[str, List[Transition]] = {a: [] for a in sorted(obs.keys())}
            done = False
            ep_reward = 0.0
            ep_steps = 0

            while not done:
                active_agents = sorted(obs.keys())
                gobs = flatten_obs_by_agent(obs, active_agents)

                actions = {}
                action_meta = {}
                for agent in active_agents:
                    out = trainer.act(np.asarray(obs[agent], dtype=np.float32), gobs)
                    actions[agent] = int(out["action"] % action_dims[agent])
                    action_meta[agent] = out

                next_obs, rewards, done, infos = extract_step(env.step(actions))
                sim_time = float(ep_steps + 1) * float(cfg.delta_time)
                metrics_by_agent = compute_metrics_for_agents(
                    env=env,
                    agent_ids=active_agents,
                    time_step=sim_time,
                    actions=actions,
                    action_green_dur=float(cfg.delta_time),
                    scenario_id=cfg.scenario_id,
                    observations=obs,
                )
                shaped_rewards = rewards_from_metrics(
                    metrics_by_agent=metrics_by_agent,
                    mode=cfg.reward_mode,
                    weights=reward_weights,
                )
                if shaped_rewards:
                    rewards = shaped_rewards

                global_reward = float(np.mean(list(rewards.values()))) if rewards else 0.0
                ep_reward += global_reward
                ep_steps += 1

                for agent in active_agents:
                    a_obs = np.asarray(obs[agent], dtype=np.float32)
                    all_rows.append(metrics_by_agent[agent].to_row())

                    trajectories[agent].append(
                        Transition(
                            obs=a_obs,
                            global_obs=gobs,
                            action=int(actions[agent]),
                            logp=float(action_meta[agent]["logp"]),
                            reward=global_reward,
                            done=done,
                            value=float(action_meta[agent]["value"]),
                        )
                    )

                obs = next_obs

            batch = trainer.build_batch(trajectories)
            losses = trainer.update(batch=batch, ppo_epochs=cfg.ppo_epochs, minibatch_size=cfg.minibatch_size)

            ep_metrics.append(
                {
                    "episode": ep,
                    "seed": cfg.seed + ep,
                    "scenario_id": cfg.scenario_id,
                    "mean_global_reward": ep_reward / max(1, ep_steps),
                    "steps": ep_steps,
                    "actor_loss": losses["actor_loss"],
                    "critic_loss": losses["critic_loss"],
                    "entropy": losses["entropy"],
                }
            )
            logger.info(
                f"Episode {ep}: reward={ep_reward / max(1, ep_steps):.4f}, "
                f"actor_loss={losses['actor_loss']:.4f}, critic_loss={losses['critic_loss']:.4f}"
            )

        trainer.save(str(cfg.model_path))
        logger.success(f"Saved MAPPO model: {cfg.model_path}")

        rollout_df = pd.DataFrame(all_rows)
        if not rollout_df.empty:
            rollout_df = rollout_df[ASCE_DATASET_COLUMNS]
        rollout_df.to_csv(cfg.rollout_csv, index=False)
        pd.DataFrame(ep_metrics).to_csv(cfg.episode_metrics_csv, index=False)

        logger.success(f"Saved rollout samples: {cfg.rollout_csv}")
        logger.success(f"Saved episode metrics: {cfg.episode_metrics_csv}")

    def evaluate(self, cfg: EvalConfig) -> None:
        cfg.out_csv.parent.mkdir(parents=True, exist_ok=True)
        resolved_device = self._resolve_device(cfg.device)
        logger.info(f"Inference device: {resolved_device}")

        records = []
        controllers = ["mappo", "fixed_time", "max_pressure"]

        for controller_name in controllers:
            env = create_parallel_env(
                net_file=cfg.net_file,
                route_file=cfg.route_file,
                seed=cfg.seed,
                use_gui=cfg.use_gui,
                seconds=cfg.seconds,
                delta_time=cfg.delta_time,
                quiet_sumo=not cfg.backend_verbose,
            )
            obs = extract_reset_obs(env.reset(seed=cfg.seed))
            if not obs:
                raise RuntimeError("No observations received from SUMO environment.")

            ordered_agents = sorted(obs.keys())
            action_dims = {a: int(env.action_spaces(a).n) for a in ordered_agents}

            fixed = FixedTimeController(action_size_by_agent=action_dims, green_duration_s=cfg.delta_time)
            max_pressure = MaxPressureController(action_size_by_agent=action_dims)
            reward_weights = RewardWeights(
                delay=cfg.reward_delay_weight,
                throughput=cfg.reward_throughput_weight,
                fairness=cfg.reward_fairness_weight,
            )

            trainer = None
            if controller_name == "mappo":
                obs_dim = int(np.asarray(obs[ordered_agents[0]], dtype=np.float32).size)
                global_obs_dim = int(flatten_obs_by_agent(obs, ordered_agents).size)
                n_actions = min(action_dims.values())
                trainer = MAPPOTrainer(
                    obs_dim=obs_dim,
                    global_obs_dim=global_obs_dim,
                    n_actions=n_actions,
                    device=resolved_device,
                )
                trainer.load(str(cfg.model_path))

            for ep in range(cfg.episodes):
                obs = extract_reset_obs(env.reset(seed=cfg.seed + ep))
                done = False
                ep_rewards = []
                per_agent_reward_totals: Dict[str, float] = {a: 0.0 for a in sorted(obs.keys())}
                steps = 0
                kpi = KPITracker()

                while not done:
                    active_agents = sorted(obs.keys())

                    if controller_name == "mappo":
                        gobs = flatten_obs_by_agent(obs, active_agents)
                        actions = {}
                        for a in active_agents:
                            out = trainer.act(np.asarray(obs[a], dtype=np.float32), gobs)
                            actions[a] = int(out["action"] % action_dims[a])
                    elif controller_name == "fixed_time":
                        actions = fixed.actions(obs)
                    else:
                        actions = max_pressure.actions(obs, env=env)

                    obs, rewards, done, infos = extract_step(env.step(actions))
                    sim_time = float(steps + 1) * float(cfg.delta_time)
                    metrics_by_agent = compute_metrics_for_agents(
                        env=env,
                        agent_ids=active_agents,
                        time_step=sim_time,
                        actions=actions,
                        action_green_dur=float(cfg.delta_time),
                        scenario_id="baseline",
                        observations=obs,
                    )
                    shaped_rewards = rewards_from_metrics(
                        metrics_by_agent=metrics_by_agent,
                        mode=cfg.reward_mode,
                        weights=reward_weights,
                    )
                    if shaped_rewards:
                        rewards = shaped_rewards
                    if rewards:
                        ep_rewards.append(float(np.mean(list(rewards.values()))))
                        for a, r in rewards.items():
                            per_agent_reward_totals[a] = per_agent_reward_totals.get(a, 0.0) + float(r)
                    steps += 1
                    kpi.update(env)

                avg_reward = float(np.mean(ep_rewards)) if ep_rewards else 0.0
                throughput_proxy = float(sum(max(0.0, v) for v in per_agent_reward_totals.values()))
                delay_proxy = float(-avg_reward)
                fairness = jain_index(list(per_agent_reward_totals.values()))
                k = kpi.summary()

                records.append(
                    {
                        "controller": controller_name,
                        "episode": ep,
                        "seed": cfg.seed + ep,
                        "steps": steps,
                        "mean_reward": avg_reward,
                        "delay_proxy": delay_proxy,
                        "throughput_proxy": throughput_proxy,
                        "fairness_jain": fairness,
                        "time_loss_s": k.time_loss_s,
                        "person_time_loss_s": k.person_time_loss_s,
                        "avg_trip_time_s": k.avg_trip_time_s,
                        "arrived_vehicles": k.arrived_vehicles,
                    }
                )

            env.close()

        pd.DataFrame(records).to_csv(cfg.out_csv, index=False)
        logger.success(f"Saved evaluation metrics: {cfg.out_csv}")
