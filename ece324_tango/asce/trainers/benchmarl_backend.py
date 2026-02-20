from __future__ import annotations

import importlib.util
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from loguru import logger

from ece324_tango.asce.baselines import FixedTimeController, MaxPressureController
from ece324_tango.asce.env import create_parallel_env, split_ns_ew_from_obs
from ece324_tango.asce.runtime import extract_reset_obs, extract_step, jain_index
from ece324_tango.asce.schema import ASCE_DATASET_COLUMNS
from ece324_tango.asce.trainers.base import AsceTrainerBackend, EvalConfig, TrainConfig
from ece324_tango.asce.trainers.benchmarl_task import SumoBenchmarlTask
from ece324_tango.asce.trainers.local_mappo_backend import LocalMappoBackend


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
    def _build_experiment(train_cfg: TrainConfig | EvalConfig, seed: int, device: str):
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
        exploration = ExplorationType.DETERMINISTIC if deterministic else ExplorationType.RANDOM
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
            reward_t = rollout.get(("next", group_name, "reward")).squeeze(-1).detach().cpu().numpy()
            if reward_t.ndim == 1:
                reward_t = reward_t[:, None]
            group_rewards.append(reward_t)
            for idx, agent in enumerate(agents):
                per_agent_totals[agent] = per_agent_totals.get(agent, 0.0) + float(reward_t[:, idx].sum())

        stacked = np.concatenate(group_rewards, axis=1) if group_rewards else np.zeros((1, 1), dtype=np.float32)
        mean_reward = float(stacked.mean())
        steps = int(stacked.shape[0])
        return rollout, mean_reward, per_agent_totals, steps

    @staticmethod
    def _rollout_to_schema_rows(rollout, delta_time: int, scenario_id: str) -> List[dict]:
        rows: List[dict] = []
        top_keys = [k for k in rollout.keys() if k not in {"next", "done", "terminated", "truncated"}]
        n_steps = int(rollout.batch_size[0]) if len(rollout.batch_size) > 0 else 0
        for t in range(n_steps):
            sim_time = float((t + 1) * delta_time)
            time_of_day = float((8.0 * 3600.0 + sim_time) / 86400.0)
            for agent in top_keys:
                obs_t = rollout.get((agent, "observation"))[t].detach().cpu().numpy().ravel()
                q_ns, q_ew, arr_ns, arr_ew = split_ns_ew_from_obs(obs_t)
                reward_t = float(rollout.get(("next", agent, "reward"))[t].mean().item())
                action_t = int(rollout.get((agent, "action"))[t].reshape(-1)[0].item())
                rows.append(
                    {
                        "intersection_id": str(agent),
                        "time_step": sim_time,
                        "queue_ns": int(round(q_ns)),
                        "queue_ew": int(round(q_ew)),
                        "arrivals_ns": int(round(arr_ns)),
                        "arrivals_ew": int(round(arr_ew)),
                        "avg_speed_ns": -1.0,
                        "avg_speed_ew": -1.0,
                        "current_phase": -1,
                        "time_of_day": time_of_day,
                        "action_phase": action_t,
                        "action_green_dur": float(delta_time),
                        "delay": float(max(0.0, -reward_t)),
                        "queue_total": int(round(q_ns + q_ew)),
                        "throughput": int(round(arr_ns + arr_ew)),
                        "scenario_id": scenario_id,
                    }
                )
        return rows

    def train(self, cfg: TrainConfig) -> None:
        self._ensure_available()
        cfg.model_path.parent.mkdir(parents=True, exist_ok=True)
        cfg.rollout_csv.parent.mkdir(parents=True, exist_ok=True)
        cfg.episode_metrics_csv.parent.mkdir(parents=True, exist_ok=True)

        resolved_device = self._resolve_device(cfg.device)
        logger.info(f"BenchMARL training device: {resolved_device}")

        exp = self._build_experiment(cfg, seed=cfg.seed, device=resolved_device)
        exp.config.max_n_iters = max(1, int(cfg.episodes))
        exp.config.on_policy_n_minibatch_iters = max(1, int(cfg.ppo_epochs))
        exp.config.on_policy_minibatch_size = max(32, int(cfg.minibatch_size))

        try:
            exp.run()
            rollout, mean_reward, per_agent_totals, steps = self._rollout_episode_stats(exp, deterministic=True)
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

            rollout_rows = self._rollout_to_schema_rows(
                rollout=rollout,
                delta_time=cfg.delta_time,
                scenario_id=cfg.scenario_id,
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
                logger.warning("BenchMARL experiment env already closed; skipping duplicate close.")

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

        exp = self._build_experiment(cfg, seed=cfg.seed, device=resolved_device)
        exp.load_state_dict(payload["state_dict"])

        records: List[dict] = []
        try:
            for ep in range(cfg.episodes):
                _, mean_reward, per_agent_totals, steps = self._rollout_episode_stats(exp, deterministic=True)
                records.append(
                    {
                        "controller": "mappo",
                        "episode": ep,
                        "seed": cfg.seed + ep,
                        "steps": steps,
                        "mean_reward": mean_reward,
                        "delay_proxy": float(-mean_reward),
                        "throughput_proxy": float(
                            sum(max(0.0, value) for value in per_agent_totals.values())
                        ),
                        "fairness_jain": jain_index(list(per_agent_totals.values())),
                    }
                )
        finally:
            try:
                exp.close()
            except RuntimeError:
                logger.warning("BenchMARL experiment env already closed; skipping duplicate close.")

        baseline_env = create_parallel_env(
            net_file=cfg.net_file,
            route_file=cfg.route_file,
            seed=cfg.seed,
            use_gui=cfg.use_gui,
            seconds=cfg.seconds,
            delta_time=cfg.delta_time,
        )
        try:
            obs = extract_reset_obs(baseline_env.reset(seed=cfg.seed))
            action_dims = {a: int(baseline_env.action_spaces(a).n) for a in sorted(obs.keys())}
            fixed = FixedTimeController(action_size_by_agent=action_dims, green_duration_s=cfg.delta_time)
            max_pressure = MaxPressureController(action_size_by_agent=action_dims)

            for controller_name, controller in [("fixed_time", fixed), ("max_pressure", max_pressure)]:
                for ep in range(cfg.episodes):
                    obs = extract_reset_obs(baseline_env.reset(seed=cfg.seed + ep))
                    done = False
                    ep_rewards = []
                    per_agent_reward_totals: Dict[str, float] = {a: 0.0 for a in sorted(obs.keys())}
                    steps = 0

                    while not done:
                        if controller_name == "fixed_time":
                            actions = controller.actions(obs)
                        else:
                            actions = controller.actions(obs, env=baseline_env)
                        obs, rewards, done, _ = extract_step(baseline_env.step(actions))
                        if rewards:
                            ep_rewards.append(float(np.mean(list(rewards.values()))))
                            for a, r in rewards.items():
                                per_agent_reward_totals[a] = per_agent_reward_totals.get(a, 0.0) + float(r)
                        steps += 1

                    avg_reward = float(np.mean(ep_rewards)) if ep_rewards else 0.0
                    records.append(
                        {
                            "controller": controller_name,
                            "episode": ep,
                            "seed": cfg.seed + ep,
                            "steps": steps,
                            "mean_reward": avg_reward,
                            "delay_proxy": float(-avg_reward),
                            "throughput_proxy": float(
                                sum(max(0.0, value) for value in per_agent_reward_totals.values())
                            ),
                            "fairness_jain": jain_index(list(per_agent_reward_totals.values())),
                        }
                    )
        finally:
            baseline_env.close()

        pd.DataFrame(records).to_csv(cfg.out_csv, index=False)
        logger.success(f"Saved evaluation metrics: {cfg.out_csv}")
