from __future__ import annotations

import importlib.util
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from loguru import logger

from ece324_tango.asce.baselines import FixedTimeController, MaxPressureController
from ece324_tango.asce.env import create_parallel_env
from ece324_tango.asce.kpi import KPITracker
from ece324_tango.asce.runtime import extract_reset_obs, extract_step, jain_index
from ece324_tango.asce.schema import ASCE_DATASET_COLUMNS
from ece324_tango.asce.traffic_metrics import (
    RewardWeights,
    compute_metrics_for_agents,
    rewards_from_metrics,
)
from ece324_tango.asce.trainers.base import AsceTrainerBackend, EvalConfig, TrainConfig
from ece324_tango.asce.trainers.benchmarl_task import SumoBenchmarlTask
from ece324_tango.asce.trainers.local_mappo_backend import LocalMappoBackend
from ece324_tango.asce.trainers.noise_control import quiet_output


class BenchmarlBackend(AsceTrainerBackend):
    name = "benchmarl"

    @staticmethod
    def _ensure_available():
        if importlib.util.find_spec("benchmarl") is None:
            raise RuntimeError(
                "BenchMARL backend selected but package 'benchmarl' is not installed. "
                "Install BenchMARL first or use trainer_backend='local_mappo'."
            )

    @staticmethod
    def _resolve_device(device: str) -> str:
        return LocalMappoBackend._resolve_device(device)

    @staticmethod
    def _build_experiment(
        train_cfg: TrainConfig | EvalConfig, seed: int, device: str, quiet_sumo: bool
    ):
        from benchmarl.algorithms import MappoConfig
        from benchmarl.experiment import Experiment, ExperimentConfig
        from benchmarl.models import MlpConfig

        task = SumoBenchmarlTask(
            name="sumo_tls",
            config={
                "net_file": train_cfg.net_file,
                "route_file": train_cfg.route_file,
                "seconds": train_cfg.seconds,
                "delta_time": train_cfg.delta_time,
                "seed": seed,
                "use_gui": train_cfg.use_gui,
                "quiet_sumo": quiet_sumo,
                "scenario_id": getattr(train_cfg, "scenario_id", "baseline"),
                "reward_mode": train_cfg.reward_mode,
                "reward_delay_weight": float(train_cfg.reward_delay_weight),
                "reward_throughput_weight": float(train_cfg.reward_throughput_weight),
                "reward_fairness_weight": float(train_cfg.reward_fairness_weight),
            },
        )

        exp_cfg = ExperimentConfig.get_from_yaml()
        exp_cfg.max_n_iters = 1
        exp_cfg.max_n_frames = None
        exp_cfg.on_policy_collected_frames_per_batch = 64
        exp_cfg.on_policy_n_envs_per_worker = 1
        exp_cfg.on_policy_n_minibatch_iters = 1
        exp_cfg.on_policy_minibatch_size = 64
        exp_cfg.evaluation = False
        exp_cfg.loggers = []
        exp_cfg.create_json = False
        exp_cfg.prefer_continuous_actions = False
        exp_cfg.train_device = device
        exp_cfg.buffer_device = device
        exp_cfg.sampling_device = "cpu"
        exp_cfg.checkpoint_at_end = False
        exp_cfg.parallel_collection = False
        exp_cfg.collect_with_grad = True

        return Experiment(
            task=task,
            algorithm_config=MappoConfig.get_from_yaml(),
            model_config=MlpConfig.get_from_yaml(),
            critic_model_config=MlpConfig.get_from_yaml(),
            seed=seed,
            config=exp_cfg,
        )

    @staticmethod
    def _rollout_episode_stats(exp, deterministic: bool = True):
        from torchrl.envs.utils import ExplorationType, set_exploration_type

        exp.test_env.reset()
        exploration = (
            ExplorationType.DETERMINISTIC if deterministic else ExplorationType.RANDOM
        )
        with set_exploration_type(exploration):
            rollout = exp.test_env.rollout(
                max_steps=exp.max_steps,
                policy=exp.policy,
                auto_cast_to_device=True,
                break_when_any_done=True,
            )

        group_rewards: List[np.ndarray] = []
        per_agent_totals: Dict[str, float] = {}

        for group_name, agents in exp.group_map.items():
            reward_t = (
                rollout.get(("next", group_name, "reward"))
                .squeeze(-1)
                .detach()
                .cpu()
                .numpy()
            )
            if reward_t.ndim == 1:
                reward_t = reward_t[:, None]
            group_rewards.append(reward_t)
            for idx, agent in enumerate(agents):
                per_agent_totals[agent] = per_agent_totals.get(agent, 0.0) + float(
                    reward_t[:, idx].sum()
                )

        stacked = (
            np.concatenate(group_rewards, axis=1)
            if group_rewards
            else np.zeros((1, 1), dtype=np.float32)
        )
        mean_reward = float(stacked.mean())
        steps = int(stacked.shape[0])
        return rollout, mean_reward, per_agent_totals, steps

    @staticmethod
    def _rollout_to_schema_rows_from_replay(rollout, cfg: TrainConfig) -> List[dict]:
        rows: List[dict] = []
        top_keys = sorted(
            [
                k
                for k in rollout.keys()
                if k not in {"next", "done", "terminated", "truncated"}
            ]
        )
        n_steps = int(rollout.batch_size[0]) if len(rollout.batch_size) > 0 else 0

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
        try:
            for t in range(n_steps):
                actions = {}
                for agent in top_keys:
                    action_t = rollout.get((agent, "action"))[t].detach().cpu().numpy()
                    actions[agent] = int(np.asarray(action_t).reshape(-1)[0])
                next_obs, _, done, _ = extract_step(env.step(actions))
                sim_time = float((t + 1) * cfg.delta_time)
                metrics_by_agent = compute_metrics_for_agents(
                    env=env,
                    agent_ids=top_keys,
                    time_step=sim_time,
                    actions=actions,
                    action_green_dur=float(cfg.delta_time),
                    scenario_id=cfg.scenario_id,
                    observations=obs,
                )
                for agent in top_keys:
                    rows.append(metrics_by_agent[agent].to_row())
                obs = next_obs
                if done:
                    break
        finally:
            env.close()
        return rows

    @staticmethod
    def _kpi_from_rollout_replay(rollout, cfg: EvalConfig):
        top_keys = sorted(
            [
                k
                for k in rollout.keys()
                if k not in {"next", "done", "terminated", "truncated"}
            ]
        )
        n_steps = int(rollout.batch_size[0]) if len(rollout.batch_size) > 0 else 0

        env = create_parallel_env(
            net_file=cfg.net_file,
            route_file=cfg.route_file,
            seed=cfg.seed,
            use_gui=cfg.use_gui,
            seconds=cfg.seconds,
            delta_time=cfg.delta_time,
            quiet_sumo=not cfg.backend_verbose,
        )
        extract_reset_obs(env.reset(seed=cfg.seed))
        kpi = KPITracker()
        try:
            for t in range(n_steps):
                actions = {}
                for agent in top_keys:
                    action_t = rollout.get((agent, "action"))[t].detach().cpu().numpy()
                    actions[agent] = int(np.asarray(action_t).reshape(-1)[0])
                _, _, done, _ = extract_step(env.step(actions))
                kpi.update(env)
                if done:
                    break
        finally:
            env.close()
        return kpi.summary()

    def train(self, cfg: TrainConfig) -> None:
        self._ensure_available()
        cfg.model_path.parent.mkdir(parents=True, exist_ok=True)
        cfg.rollout_csv.parent.mkdir(parents=True, exist_ok=True)
        cfg.episode_metrics_csv.parent.mkdir(parents=True, exist_ok=True)

        resolved_device = self._resolve_device(cfg.device)
        logger.info(f"BenchMARL training device: {resolved_device}")

        with quiet_output(enabled=not cfg.backend_verbose):
            exp = self._build_experiment(
                cfg,
                seed=cfg.seed,
                device=resolved_device,
                quiet_sumo=not cfg.backend_verbose,
            )
        exp.config.max_n_iters = max(1, int(cfg.episodes))
        exp.config.on_policy_n_minibatch_iters = max(1, int(cfg.ppo_epochs))
        exp.config.on_policy_minibatch_size = max(32, int(cfg.minibatch_size))

        try:
            with quiet_output(enabled=not cfg.backend_verbose):
                exp.run()
                rollout, mean_reward, per_agent_totals, steps = (
                    self._rollout_episode_stats(exp, deterministic=True)
                )
            torch.save(
                {
                    "backend": self.name,
                    "state_dict": exp.state_dict(),
                    "seed": cfg.seed,
                    "seconds": cfg.seconds,
                    "delta_time": cfg.delta_time,
                },
                cfg.model_path,
            )
            logger.success(f"Saved BenchMARL model checkpoint: {cfg.model_path}")

            rollout_rows = self._rollout_to_schema_rows_from_replay(
                rollout=rollout, cfg=cfg
            )
            rollout_df = pd.DataFrame(rollout_rows)
            if rollout_df.empty:
                rollout_df = pd.DataFrame(columns=ASCE_DATASET_COLUMNS)
            else:
                rollout_df = rollout_df[ASCE_DATASET_COLUMNS]
            rollout_df.to_csv(cfg.rollout_csv, index=False)

            pd.DataFrame(
                [
                    {
                        "episode": 0,
                        "seed": cfg.seed,
                        "scenario_id": cfg.scenario_id,
                        "mean_global_reward": mean_reward,
                        "steps": steps,
                        "actor_loss": float("nan"),
                        "critic_loss": float("nan"),
                        "entropy": float("nan"),
                        "fairness_jain": jain_index(list(per_agent_totals.values())),
                    }
                ]
            ).to_csv(cfg.episode_metrics_csv, index=False)
        finally:
            try:
                exp.close()
            except RuntimeError:
                logger.warning(
                    "BenchMARL experiment env already closed; skipping duplicate close."
                )

    def evaluate(self, cfg: EvalConfig) -> None:
        self._ensure_available()
        cfg.out_csv.parent.mkdir(parents=True, exist_ok=True)
        resolved_device = self._resolve_device(cfg.device)
        logger.info(f"BenchMARL inference device: {resolved_device}")

        payload = torch.load(cfg.model_path, map_location="cpu")
        if payload.get("backend") != self.name:
            raise RuntimeError(
                f"Checkpoint {cfg.model_path} is not a BenchMARL artifact. "
                "Train with --trainer-backend benchmarl first."
            )

        records: List[dict] = []
        for ep in range(cfg.episodes):
            episode_seed = cfg.seed + ep
            with quiet_output(enabled=not cfg.backend_verbose):
                exp = self._build_experiment(
                    cfg,
                    seed=episode_seed,
                    device=resolved_device,
                    quiet_sumo=not cfg.backend_verbose,
                )
            try:
                exp.load_state_dict(payload["state_dict"])
                with quiet_output(enabled=not cfg.backend_verbose):
                    rollout, mean_reward, per_agent_totals, steps = (
                        self._rollout_episode_stats(exp, deterministic=True)
                    )
                k = self._kpi_from_rollout_replay(rollout=rollout, cfg=cfg)
                objective_mean_reward = (
                    mean_reward if cfg.reward_mode == "objective" else float("nan")
                )
                objective_delay_proxy = (
                    float(-mean_reward)
                    if cfg.reward_mode == "objective"
                    else float("nan")
                )
                objective_throughput_proxy = (
                    float(sum(max(0.0, value) for value in per_agent_totals.values()))
                    if cfg.reward_mode == "objective"
                    else float("nan")
                )
                objective_fairness = (
                    jain_index(list(per_agent_totals.values()))
                    if cfg.reward_mode == "objective"
                    else float("nan")
                )
                records.append(
                    {
                        "controller": "mappo",
                        "episode": ep,
                        "seed": episode_seed,
                        "steps": steps,
                        "mean_reward": mean_reward,
                        "delay_proxy": float(-mean_reward),
                        "throughput_proxy": float(
                            sum(max(0.0, value) for value in per_agent_totals.values())
                        ),
                        "fairness_jain": jain_index(list(per_agent_totals.values())),
                        "objective_mean_reward": objective_mean_reward,
                        "objective_delay_proxy": objective_delay_proxy,
                        "objective_throughput_proxy": objective_throughput_proxy,
                        "objective_fairness_jain": objective_fairness,
                        "time_loss_s": k.time_loss_s,
                        "person_time_loss_s": k.person_time_loss_s,
                        "avg_trip_time_s": k.avg_trip_time_s,
                        "arrived_vehicles": k.arrived_vehicles,
                    }
                )
            finally:
                try:
                    exp.close()
                except RuntimeError:
                    logger.warning(
                        "BenchMARL experiment env already closed; skipping duplicate close."
                    )

        with quiet_output(enabled=not cfg.backend_verbose):
            baseline_env = create_parallel_env(
                net_file=cfg.net_file,
                route_file=cfg.route_file,
                seed=cfg.seed,
                use_gui=cfg.use_gui,
                seconds=cfg.seconds,
                delta_time=cfg.delta_time,
                quiet_sumo=not cfg.backend_verbose,
            )
        try:
            obs = extract_reset_obs(baseline_env.reset(seed=cfg.seed))
            action_dims = {
                a: int(baseline_env.action_spaces(a).n) for a in sorted(obs.keys())
            }
            fixed = FixedTimeController(
                action_size_by_agent=action_dims, green_duration_s=cfg.delta_time
            )
            max_pressure = MaxPressureController(action_size_by_agent=action_dims)
            reward_weights = RewardWeights(
                delay=cfg.reward_delay_weight,
                throughput=cfg.reward_throughput_weight,
                fairness=cfg.reward_fairness_weight,
            )

            for controller_name, controller in [
                ("fixed_time", fixed),
                ("max_pressure", max_pressure),
            ]:
                for ep in range(cfg.episodes):
                    obs = extract_reset_obs(baseline_env.reset(seed=cfg.seed + ep))
                    done = False
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
                        with quiet_output(enabled=not cfg.backend_verbose):
                            if controller_name == "fixed_time":
                                actions = controller.actions(obs)
                            else:
                                actions = controller.actions(obs, env=baseline_env)
                            obs, rewards, done, _ = extract_step(
                                baseline_env.step(actions)
                            )
                        sim_time = float(steps + 1) * float(cfg.delta_time)
                        metrics_by_agent = compute_metrics_for_agents(
                            env=baseline_env,
                            agent_ids=sorted(obs.keys()),
                            time_step=sim_time,
                            actions=actions,
                            action_green_dur=float(cfg.delta_time),
                            scenario_id="baseline",
                            observations=obs,
                        )
                        shaped = rewards_from_metrics(
                            metrics_by_agent,
                            mode=cfg.reward_mode,
                            weights=reward_weights,
                        )
                        objective_shaped = (
                            shaped
                            if cfg.reward_mode == "objective"
                            else rewards_from_metrics(
                                metrics_by_agent,
                                mode="objective",
                                weights=reward_weights,
                            )
                        )
                        if shaped:
                            rewards = shaped
                        if rewards:
                            ep_rewards.append(float(np.mean(list(rewards.values()))))
                            for a, r in rewards.items():
                                per_agent_reward_totals[a] = (
                                    per_agent_reward_totals.get(a, 0.0) + float(r)
                                )
                        if objective_shaped:
                            objective_ep_rewards.append(
                                float(np.mean(list(objective_shaped.values())))
                            )
                            for a, r in objective_shaped.items():
                                objective_per_agent_reward_totals[a] = (
                                    objective_per_agent_reward_totals.get(a, 0.0)
                                    + float(r)
                                )
                        steps += 1
                        kpi.update(baseline_env)

                    avg_reward = float(np.mean(ep_rewards)) if ep_rewards else 0.0
                    objective_avg_reward = (
                        float(np.mean(objective_ep_rewards))
                        if objective_ep_rewards
                        else 0.0
                    )
                    k = kpi.summary()
                    records.append(
                        {
                            "controller": controller_name,
                            "episode": ep,
                            "seed": cfg.seed + ep,
                            "steps": steps,
                            "mean_reward": avg_reward,
                            "delay_proxy": float(-avg_reward),
                            "throughput_proxy": float(
                                sum(
                                    max(0.0, value)
                                    for value in per_agent_reward_totals.values()
                                )
                            ),
                            "fairness_jain": jain_index(
                                list(per_agent_reward_totals.values())
                            ),
                            "objective_mean_reward": objective_avg_reward,
                            "objective_delay_proxy": float(-objective_avg_reward),
                            "objective_throughput_proxy": float(
                                sum(
                                    max(0.0, value)
                                    for value in objective_per_agent_reward_totals.values()
                                )
                            ),
                            "objective_fairness_jain": jain_index(
                                list(objective_per_agent_reward_totals.values())
                            ),
                            "time_loss_s": k.time_loss_s,
                            "person_time_loss_s": k.person_time_loss_s,
                            "avg_trip_time_s": k.avg_trip_time_s,
                            "arrived_vehicles": k.arrived_vehicles,
                        }
                    )
        finally:
            baseline_env.close()

        pd.DataFrame(records).to_csv(cfg.out_csv, index=False)
        logger.success(f"Saved evaluation metrics: {cfg.out_csv}")
