from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import shutil
from typing import Dict, List

import numpy as np
import pandas as pd
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
from ece324_tango.asce.trainers.local_mappo_backend import LocalMappoBackend
from ece324_tango.asce.trainers.noise_control import quiet_output
from ece324_tango.asce.trainers.xuance_compat import apply_xuance_value_norm_patch
from ece324_tango.asce.trainers.xuance_env import register_xuance_sumo_env


class XuanceBackend(AsceTrainerBackend):
    name = "xuance"
    _CONFIG_PATH = Path(__file__).with_name("configs") / "xuance_mappo_sumo.yaml"

    @staticmethod
    def _ensure_available():
        if importlib.util.find_spec("xuance") is None:
            raise RuntimeError(
                "Xuance backend selected but package 'xuance' is not installed. "
                "Install Xuance first or use trainer_backend='local_mappo'."
            )

    @staticmethod
    def _resolve_device(device: str) -> str:
        resolved = LocalMappoBackend._resolve_device(device)
        return "cuda:0" if resolved == "cuda" else resolved

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    def _build_runner(self, cfg: TrainConfig | EvalConfig, seed: int):
        from xuance.common import get_arguments
        from xuance.torch.runners.runner_marl import RunnerMARL

        if self._env_bool("TANGO_XUANCE_PATCH_VALUE_NORM", True):
            patched = apply_xuance_value_norm_patch()
            if patched:
                logger.info("Applied local Xuance value_norm compatibility patch.")

        if not self._CONFIG_PATH.exists():
            raise RuntimeError(f"Missing Xuance config file: {self._CONFIG_PATH}")

        register_xuance_sumo_env(env_name="sumo_custom")
        args = get_arguments(
            method="mappo",
            env="sumo_custom",
            env_id="grid4x4",
            config_path=str(self._CONFIG_PATH.resolve()),
        )

        args.parallels = 1
        args.seed = seed
        args.env_seed = seed
        args.render = False
        args.test_mode = False

        args.device = self._resolve_device(cfg.device)
        args.sumo_net_file = cfg.net_file
        args.sumo_route_file = cfg.route_file
        args.sumo_seconds = int(cfg.seconds)
        args.sumo_delta_time = int(cfg.delta_time)
        args.use_gui = bool(cfg.use_gui)
        args.sumo_quiet = not cfg.backend_verbose
        args.scenario_id = getattr(cfg, "scenario_id", "baseline")
        args.reward_mode = cfg.reward_mode
        args.reward_delay_weight = float(cfg.reward_delay_weight)
        args.reward_throughput_weight = float(cfg.reward_throughput_weight)
        args.reward_fairness_weight = float(cfg.reward_fairness_weight)

        # Stable settings for custom SUMO adapter under Xuance MAPPO.
        args.use_value_norm = self._env_bool("TANGO_XUANCE_USE_VALUE_NORM", True)
        args.use_gae = self._env_bool("TANGO_XUANCE_USE_GAE", True)
        args.use_advnorm = self._env_bool("TANGO_XUANCE_USE_ADVNORM", False)
        args.n_epochs = 1
        args.n_minibatch = 1
        args.buffer_size = 16
        args.running_steps = 16

        args.log_dir = "./logs/xuance_mappo/"
        args.model_dir = "./models/xuance_mappo/"
        return RunnerMARL(args)

    @staticmethod
    def _safe_close_runner(runner):
        try:
            runner.agents.finish()
        except Exception:
            pass
        try:
            runner.envs.close()
        except Exception:
            pass

    @staticmethod
    def _run_episode_with_agent(
        cfg: TrainConfig | EvalConfig,
        agent,
        deterministic: bool = False,
    ):
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
        done = False
        steps = 0
        mean_rewards = []
        per_agent_totals: Dict[str, float] = {a: 0.0 for a in sorted(obs.keys())}
        rows: List[dict] = []
        reward_weights = RewardWeights(
            delay=cfg.reward_delay_weight,
            throughput=cfg.reward_throughput_weight,
            fairness=cfg.reward_fairness_weight,
        )
        kpi = KPITracker()

        try:
            while not done:
                obs_batch = [obs]
                policy_out = agent.action(
                    obs_dict=obs_batch,
                    state=None,
                    avail_actions_dict=None,
                    rnn_hidden_actor=None,
                    rnn_hidden_critic=None,
                    test_mode=deterministic,
                )
                actions_for_env = policy_out["actions"][0]
                actions = {
                    a: int(np.asarray(actions_for_env[a]).reshape(-1)[0]) for a in sorted(obs.keys())
                }

                next_obs, rewards, done, _ = extract_step(env.step(actions))
                steps += 1
                sim_time = float(steps * cfg.delta_time)
                metrics_by_agent = compute_metrics_for_agents(
                    env=env,
                    agent_ids=sorted(obs.keys()),
                    time_step=sim_time,
                    actions=actions,
                    action_green_dur=float(cfg.delta_time),
                    scenario_id=getattr(cfg, "scenario_id", "baseline"),
                    observations=obs,
                )
                shaped = rewards_from_metrics(metrics_by_agent, mode=cfg.reward_mode, weights=reward_weights)
                if shaped:
                    rewards = shaped
                if rewards:
                    mean_rewards.append(float(np.mean(list(rewards.values()))))
                    for a, r in rewards.items():
                        per_agent_totals[a] = per_agent_totals.get(a, 0.0) + float(r)
                kpi.update(env)

                for agent_id in sorted(obs.keys()):
                    rows.append(metrics_by_agent[agent_id].to_row())
                obs = next_obs
        finally:
            env.close()

        mean_reward = float(np.mean(mean_rewards)) if mean_rewards else 0.0
        return rows, mean_reward, per_agent_totals, steps, kpi.summary()

    def train(self, cfg: TrainConfig) -> None:
        self._ensure_available()
        cfg.model_path.parent.mkdir(parents=True, exist_ok=True)
        cfg.rollout_csv.parent.mkdir(parents=True, exist_ok=True)
        cfg.episode_metrics_csv.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Xuance training device: {self._resolve_device(cfg.device)}")
        with quiet_output(enabled=not cfg.backend_verbose):
            runner = self._build_runner(cfg, seed=cfg.seed)
        runner.config.running_steps = max(16, int(cfg.episodes * max(1, cfg.seconds // cfg.delta_time)))
        runner.config.buffer_size = max(16, int(cfg.minibatch_size))
        runner.config.n_epochs = max(1, int(cfg.ppo_epochs))
        runner.config.n_minibatch = 1

        logger.info(
            "Xuance stability flags: "
            f"use_gae={runner.config.use_gae}, "
            f"use_value_norm={runner.config.use_value_norm}, "
            f"use_advnorm={runner.config.use_advnorm}"
        )

        try:
            with quiet_output(enabled=not cfg.backend_verbose):
                runner.run()
            saved_path = f"{runner.agents.model_dir_save}/final_train_model.pth"
            shutil.copyfile(saved_path, cfg.model_path)
            xuance_ckpt_dir = cfg.model_path.parent / f"{cfg.model_path.stem}_xuance"
            xuance_ckpt_dir.mkdir(parents=True, exist_ok=True)
            seed_export_dir = xuance_ckpt_dir / "seed_export"
            seed_export_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(saved_path, seed_export_dir / "final_train_model.pth")
            logger.success(f"Saved Xuance MAPPO model: {cfg.model_path}")

            with quiet_output(enabled=not cfg.backend_verbose):
                rows, mean_reward, per_agent_totals, steps, _ = self._run_episode_with_agent(
                    cfg=cfg,
                    agent=runner.agents,
                    deterministic=False,
                )
            rollout_df = pd.DataFrame(rows)
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
        except ValueError as exc:
            if "setting an array element with a sequence" in str(exc):
                raise RuntimeError(
                    "Xuance MAPPO failed with current normalization settings. "
                    "If needed, disable value normalization with TANGO_XUANCE_USE_VALUE_NORM=0. "
                    "Compatibility patch toggle: TANGO_XUANCE_PATCH_VALUE_NORM."
                ) from exc
            raise
        finally:
            self._safe_close_runner(runner)

    def evaluate(self, cfg: EvalConfig) -> None:
        self._ensure_available()
        cfg.out_csv.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Xuance inference device: {self._resolve_device(cfg.device)}")

        with quiet_output(enabled=not cfg.backend_verbose):
            runner = self._build_runner(cfg, seed=cfg.seed)
        records: List[dict] = []
        try:
            xuance_ckpt_dir = cfg.model_path.parent / f"{cfg.model_path.stem}_xuance"
            if xuance_ckpt_dir.exists():
                with quiet_output(enabled=not cfg.backend_verbose):
                    runner.agents.load_model(str(xuance_ckpt_dir))
            else:
                raise RuntimeError(
                    f"Missing Xuance checkpoint directory: {xuance_ckpt_dir}. "
                    "Re-run training with --trainer-backend xuance first."
                )
            for ep in range(cfg.episodes):
                with quiet_output(enabled=not cfg.backend_verbose):
                    rows, mean_reward, per_agent_totals, steps, k = self._run_episode_with_agent(
                        cfg=cfg,
                        agent=runner.agents,
                        deterministic=True,
                    )
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
                        "time_loss_s": k.time_loss_s,
                        "person_time_loss_s": k.person_time_loss_s,
                        "avg_trip_time_s": k.avg_trip_time_s,
                        "arrived_vehicles": k.arrived_vehicles,
                    }
                )
        finally:
            self._safe_close_runner(runner)

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
            action_dims = {a: int(baseline_env.action_spaces(a).n) for a in sorted(obs.keys())}
            fixed = FixedTimeController(action_size_by_agent=action_dims, green_duration_s=cfg.delta_time)
            max_pressure = MaxPressureController(action_size_by_agent=action_dims)
            reward_weights = RewardWeights(
                delay=cfg.reward_delay_weight,
                throughput=cfg.reward_throughput_weight,
                fairness=cfg.reward_fairness_weight,
            )

            for controller_name, controller in [("fixed_time", fixed), ("max_pressure", max_pressure)]:
                for ep in range(cfg.episodes):
                    obs = extract_reset_obs(baseline_env.reset(seed=cfg.seed + ep))
                    done = False
                    ep_rewards = []
                    per_agent_reward_totals: Dict[str, float] = {a: 0.0 for a in sorted(obs.keys())}
                    steps = 0
                    kpi = KPITracker()

                    while not done:
                        with quiet_output(enabled=not cfg.backend_verbose):
                            if controller_name == "fixed_time":
                                actions = controller.actions(obs)
                            else:
                                actions = controller.actions(obs, env=baseline_env)
                            obs, rewards, done, _ = extract_step(baseline_env.step(actions))
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
                        shaped = rewards_from_metrics(metrics_by_agent, mode=cfg.reward_mode, weights=reward_weights)
                        if shaped:
                            rewards = shaped
                        if rewards:
                            ep_rewards.append(float(np.mean(list(rewards.values()))))
                            for a, r in rewards.items():
                                per_agent_reward_totals[a] = per_agent_reward_totals.get(a, 0.0) + float(r)
                        steps += 1
                        kpi.update(baseline_env)

                    avg_reward = float(np.mean(ep_rewards)) if ep_rewards else 0.0
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
                                sum(max(0.0, value) for value in per_agent_reward_totals.values())
                            ),
                            "fairness_jain": jain_index(list(per_agent_reward_totals.values())),
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
