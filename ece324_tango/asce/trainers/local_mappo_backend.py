from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from loguru import logger
from traci.exceptions import FatalTraCIError

from ece324_tango.asce.baselines import FixedTimeController, MaxPressureController
from ece324_tango.asce.env import (
    create_parallel_env,
    flatten_obs_by_agent,
    pad_observation,
)
from ece324_tango.asce.kpi import KPITracker
from ece324_tango.asce.mappo import MAPPOTrainer, Transition
from ece324_tango.asce.runtime import (
    extract_reset_obs,
    extract_step,
    extract_step_details,
    jain_index,
)
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
        try:
            obs = extract_reset_obs(env.reset(seed=cfg.seed))
            if not obs:
                raise RuntimeError("No observations received from SUMO environment.")

            ordered_agents = sorted(obs.keys())
            obs_dim = max(
                int(np.asarray(obs[a], dtype=np.float32).size) for a in ordered_agents
            )
            global_obs_dim = int(flatten_obs_by_agent(obs, ordered_agents).size)

            action_dims = {a: int(env.action_spaces(a).n) for a in ordered_agents}
            n_actions = max(action_dims.values())
            max_pressure = MaxPressureController(action_size_by_agent=action_dims)

            resolved_device = self._resolve_device(cfg.device)
            logger.info(f"Training device: {resolved_device}")

            trainer = MAPPOTrainer(
                obs_dim=obs_dim,
                global_obs_dim=global_obs_dim,
                n_actions=n_actions,
                device=resolved_device,
                use_obs_norm=cfg.use_obs_norm,
            )

            all_rows: List[dict] = []
            ep_metrics: List[dict] = []
            reward_weights = RewardWeights(
                delay=cfg.reward_delay_weight,
                throughput=cfg.reward_throughput_weight,
                fairness=cfg.reward_fairness_weight,
                residual=cfg.reward_residual_weight,
            )

            for ep in range(cfg.episodes):
                obs = extract_reset_obs(env.reset(seed=cfg.seed + ep))
                if not obs:
                    continue

                trajectories: Dict[str, List[Transition]] = {
                    a: [] for a in sorted(obs.keys())
                }
                done = False
                max_steps = max(1, int(cfg.seconds // cfg.delta_time))
                episode_terminated = False
                episode_truncated = False
                bootstrap_obs = None
                ep_reward = 0.0
                ep_steps = 0

                while not done:
                    active_agents = sorted(obs.keys())
                    gobs = flatten_obs_by_agent(obs, active_agents)

                    padded_obs_list = [
                        pad_observation(
                            np.asarray(obs[a], dtype=np.float32), target_dim=obs_dim
                        )
                        for a in active_agents
                    ]
                    n_valid_list = [action_dims[a] for a in active_agents]
                    # Update normalizer stats before acting:
                    # obs_norm gets one sample per agent; gobs_norm gets one per step
                    if trainer.obs_norm is not None:
                        for padded_obs in padded_obs_list:
                            trainer.obs_norm.update(padded_obs)
                    if trainer.gobs_norm is not None:
                        trainer.gobs_norm.update(gobs)
                    batch_out = trainer.act_batch(padded_obs_list, gobs, n_valid_list)
                    actions = {
                        a: int(batch_out[i]["action"])
                        for i, a in enumerate(active_agents)
                    }
                    mp_actions = max_pressure.actions(obs, env=env)
                    action_meta = {a: batch_out[i] for i, a in enumerate(active_agents)}

                    terminated = False
                    truncated = False
                    try:
                        next_obs, rewards, done, _infos, terminated, truncated = (
                            extract_step_details(env.step(actions))
                        )
                        if (
                            done
                            and not terminated
                            and not truncated
                            and (ep_steps + 1) >= max_steps
                        ):
                            # Legacy dones-only API: infer timeout from configured horizon.
                            truncated = True
                    except FatalTraCIError:
                        # SUMO process terminated early (demand exhausted before sim end).
                        # Treat as terminal step so the episode is cleanly finalised.
                        logger.warning(
                            f"Episode {ep}: SUMO connection closed at step {ep_steps} "
                            "(demand exhausted). Treating as episode end."
                        )
                        done = True
                        terminated = True
                        truncated = False
                        next_obs = {}
                        rewards = {a: 0.0 for a in active_agents}
                    if done and truncated and next_obs:
                        bootstrap_obs = {
                            a: np.asarray(next_obs[a], dtype=np.float32)
                            for a in sorted(next_obs.keys())
                        }
                    episode_terminated = episode_terminated or (done and terminated)
                    episode_truncated = episode_truncated or (done and truncated)
                    sim_time = float(ep_steps + 1) * float(cfg.delta_time)
                    if not done:
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
                            mp_deviation_by_agent={
                                a: float(int(actions.get(a, 0) != mp_actions.get(a, 0)))
                                for a in active_agents
                            }
                            if cfg.reward_mode == "residual_mp"
                            else None,
                        )
                        if shaped_rewards:
                            rewards = shaped_rewards
                    else:
                        metrics_by_agent = {}

                    global_reward = (
                        float(np.mean(list(rewards.values()))) if rewards else 0.0
                    )
                    ep_reward += global_reward
                    ep_steps += 1

                    for i, agent in enumerate(active_agents):
                        a_obs = padded_obs_list[i]
                        if metrics_by_agent:
                            all_rows.append(metrics_by_agent[agent].to_row())
                        agent_reward = float(rewards.get(agent, global_reward))
                        trajectories[agent].append(
                            Transition(
                                obs=a_obs,
                                global_obs=gobs,
                                action=int(actions[agent]),
                                logp=float(action_meta[agent]["logp"]),
                                reward=agent_reward,
                                done=bool(terminated),
                                value=float(action_meta[agent]["value"]),
                                n_valid_actions=int(action_dims[agent]),
                            )
                        )

                    obs = next_obs

                # Bootstrap value for truncated (non-terminal) episodes
                last_values: Dict[str, float] = {}
                if episode_truncated and not episode_terminated and bootstrap_obs:
                    final_agents = sorted(bootstrap_obs.keys())
                    final_gobs = flatten_obs_by_agent(bootstrap_obs, final_agents)
                    for agent in final_agents:
                        padded = pad_observation(
                            np.asarray(bootstrap_obs[agent], dtype=np.float32),
                            target_dim=obs_dim,
                        )
                        v = trainer.act(
                            padded, final_gobs, n_valid_actions=action_dims[agent]
                        )
                        last_values[agent] = v["value"]

                batch = trainer.build_batch(trajectories, last_values=last_values)
                losses = trainer.update(
                    batch=batch,
                    ppo_epochs=cfg.ppo_epochs,
                    minibatch_size=cfg.minibatch_size,
                )

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
        finally:
            env.close()

    def evaluate(self, cfg: EvalConfig) -> None:
        cfg.out_csv.parent.mkdir(parents=True, exist_ok=True)
        resolved_device = self._resolve_device(cfg.device)
        logger.info(f"Inference device: {resolved_device}")
        if not cfg.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {cfg.model_path}")
        ckpt_use_obs_norm = MAPPOTrainer.checkpoint_use_obs_norm(str(cfg.model_path))
        if ckpt_use_obs_norm != bool(cfg.use_obs_norm):
            raise RuntimeError(
                f"Checkpoint use_obs_norm={ckpt_use_obs_norm} but eval requested "
                f"use_obs_norm={cfg.use_obs_norm}. Re-run with matching --use-obs-norm/--no-use-obs-norm."
            )

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
            try:
                obs = extract_reset_obs(env.reset(seed=cfg.seed))
                if not obs:
                    raise RuntimeError(
                        "No observations received from SUMO environment."
                    )

                ordered_agents = sorted(obs.keys())
                action_dims = {a: int(env.action_spaces(a).n) for a in ordered_agents}

                fixed = FixedTimeController(
                    action_size_by_agent=action_dims, green_duration_s=cfg.delta_time
                )
                max_pressure = MaxPressureController(action_size_by_agent=action_dims)
                reward_weights = RewardWeights(
                    delay=cfg.reward_delay_weight,
                    throughput=cfg.reward_throughput_weight,
                    fairness=cfg.reward_fairness_weight,
                    residual=cfg.reward_residual_weight,
                )

                trainer = None
                if controller_name == "mappo":
                    obs_dim = max(
                        int(np.asarray(obs[a], dtype=np.float32).size)
                        for a in ordered_agents
                    )
                    global_obs_dim = int(flatten_obs_by_agent(obs, ordered_agents).size)
                    n_actions = max(action_dims.values())
                    trainer = MAPPOTrainer(
                        obs_dim=obs_dim,
                        global_obs_dim=global_obs_dim,
                        n_actions=n_actions,
                        device=resolved_device,
                        use_obs_norm=cfg.use_obs_norm,
                    )
                    trainer.load(str(cfg.model_path))

                for ep in range(cfg.episodes):
                    obs = extract_reset_obs(env.reset(seed=cfg.seed + ep))
                    done = False
                    if controller_name == "fixed_time" and hasattr(fixed, "reset"):
                        fixed.reset()
                    ep_rewards = []
                    per_agent_reward_totals: Dict[str, float] = {
                        a: 0.0 for a in sorted(obs.keys())
                    }
                    objective_ep_rewards = []
                    objective_per_agent_reward_totals: Dict[str, float] = {
                        a: 0.0 for a in sorted(obs.keys())
                    }
                    steps = 0
                    kpi = KPITracker()

                    while not done:
                        active_agents = sorted(obs.keys())
                        prev_obs = {
                            a: np.asarray(obs[a], dtype=np.float32)
                            for a in active_agents
                        }
                        skip_kpi_update = False

                        if controller_name == "mappo":
                            gobs = flatten_obs_by_agent(obs, active_agents)
                            padded_obs_list = [
                                pad_observation(
                                    np.asarray(obs[a], dtype=np.float32),
                                    target_dim=obs_dim,
                                )
                                for a in active_agents
                            ]
                            n_valid_list = [action_dims[a] for a in active_agents]
                            batch_out = trainer.act_batch(
                                padded_obs_list, gobs, n_valid_list
                            )
                            actions = {
                                a: int(batch_out[i]["action"])
                                for i, a in enumerate(active_agents)
                            }
                        elif controller_name == "fixed_time":
                            actions = fixed.actions(obs)
                        else:
                            actions = max_pressure.actions(obs, env=env)
                        mp_actions = max_pressure.actions(obs, env=env)

                        try:
                            obs, rewards, done, _infos = extract_step(env.step(actions))
                        except FatalTraCIError:
                            logger.warning(
                                f"Eval {controller_name} ep{ep}: SUMO connection closed at step {steps} "
                                "(demand exhausted). Treating as episode end."
                            )
                            done = True
                            skip_kpi_update = True
                            obs = {}
                            rewards = {a: 0.0 for a in active_agents}
                        sim_time = float(steps + 1) * float(cfg.delta_time)
                        objective_shaped_rewards: Dict[str, float] = {}
                        if not done:
                            metrics_by_agent = compute_metrics_for_agents(
                                env=env,
                                agent_ids=active_agents,
                                time_step=sim_time,
                                actions=actions,
                                action_green_dur=float(cfg.delta_time),
                                scenario_id="baseline",
                                observations=prev_obs,
                            )
                            shaped_rewards = rewards_from_metrics(
                                metrics_by_agent=metrics_by_agent,
                                mode=cfg.reward_mode,
                                weights=reward_weights,
                                mp_deviation_by_agent={
                                    a: float(int(actions.get(a, 0) != mp_actions.get(a, 0)))
                                    for a in active_agents
                                }
                                if cfg.reward_mode == "residual_mp"
                                else None,
                            )
                            objective_shaped_rewards = (
                                shaped_rewards
                                if cfg.reward_mode == "objective"
                                else rewards_from_metrics(
                                    metrics_by_agent=metrics_by_agent,
                                    mode="objective",
                                    weights=reward_weights,
                                )
                            )
                            if shaped_rewards:
                                rewards = shaped_rewards
                        if rewards:
                            ep_rewards.append(float(np.mean(list(rewards.values()))))
                            for a, r in rewards.items():
                                per_agent_reward_totals[a] = (
                                    per_agent_reward_totals.get(a, 0.0) + float(r)
                                )
                        if objective_shaped_rewards:
                            objective_ep_rewards.append(
                                float(np.mean(list(objective_shaped_rewards.values())))
                            )
                            for a, r in objective_shaped_rewards.items():
                                objective_per_agent_reward_totals[a] = (
                                    objective_per_agent_reward_totals.get(a, 0.0)
                                    + float(r)
                                )
                        steps += 1
                        if not skip_kpi_update:
                            kpi.update(env)

                    avg_reward = float(np.mean(ep_rewards)) if ep_rewards else 0.0
                    throughput_proxy = float(
                        sum(max(0.0, v) for v in per_agent_reward_totals.values())
                    )
                    delay_proxy = float(-avg_reward)
                    fairness = jain_index(list(per_agent_reward_totals.values()))
                    objective_avg_reward = (
                        float(np.mean(objective_ep_rewards))
                        if objective_ep_rewards
                        else 0.0
                    )
                    objective_throughput_proxy = float(
                        sum(
                            max(0.0, v)
                            for v in objective_per_agent_reward_totals.values()
                        )
                    )
                    objective_delay_proxy = float(-objective_avg_reward)
                    objective_fairness = jain_index(
                        list(objective_per_agent_reward_totals.values())
                    )
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
                            "objective_mean_reward": objective_avg_reward,
                            "objective_delay_proxy": objective_delay_proxy,
                            "objective_throughput_proxy": objective_throughput_proxy,
                            "objective_fairness_jain": objective_fairness,
                            "time_loss_s": k.time_loss_s,
                            "person_time_loss_s": k.person_time_loss_s,
                            "avg_trip_time_s": k.avg_trip_time_s,
                            "arrived_vehicles": k.arrived_vehicles,
                            "vehicle_delay_jain": k.vehicle_delay_jain,
                        }
                    )
            finally:
                env.close()

        pd.DataFrame(records).to_csv(cfg.out_csv, index=False)
        logger.success(f"Saved evaluation metrics: {cfg.out_csv}")
