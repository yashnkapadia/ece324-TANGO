from __future__ import annotations

import json
import signal
from pathlib import Path
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
from ece324_tango.asce.obs_norm import ObsRunningNorm
from ece324_tango.asce.mappo import (
    MAPPOTrainer,
    ResidualMAPPOTrainer,
    Transition,
    augment_obs_with_mp,
)
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


def _worker_init():
    """Pool initializer: workers ignore SIGINT so the parent handles it."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Workers inherit the parent's stdout/stderr fds (same terminal the parent's
    # rich.Live TUI is drawing on). Silence all three layers:
    #   1. OS-level dup2 → catches C-extension output (SUMO binary step logs)
    #   2. Python sys.stdout/stderr replacement → catches TraCI print() calls
    #   3. loguru.remove() → removes tqdm/default handlers added by config.py
    #   4. warnings.filterwarnings → suppresses gymnasium UserWarning to stderr
    import os
    import sys
    import warnings

    _devnull = open(os.devnull, "w")
    os.dup2(_devnull.fileno(), 1)
    os.dup2(_devnull.fileno(), 2)
    sys.stdout = _devnull
    sys.stderr = _devnull
    warnings.filterwarnings("ignore")

    from loguru import logger as _logger

    _logger.remove()


def _eval_controller_sequence(eval_baselines: list[str]) -> list[str]:
    return ["mappo", *eval_baselines]


def _format_eval_summary(
    scenario_name: str,
    train_ep: int,
    results: dict[str, float],
) -> tuple[float, str]:
    mappo_ptl = results.get("mappo", 0.0)
    mp_ptl = results.get("max_pressure", 1.0)
    ratio = mappo_ptl / max(mp_ptl, 1.0)

    metric_parts = [f"MAPPO={mappo_ptl:.0f}s", f"MP={mp_ptl:.0f}s"]
    if "fixed_time" in results:
        metric_parts.append(f"FT={results['fixed_time']:.0f}s")
    if "nema" in results:
        metric_parts.append(f"NEMA={results['nema']:.0f}s")

    ratio_parts = [f"MAPPO/MP={ratio:.3f}"]
    if "nema" in results:
        ratio_parts.append(
            f"MAPPO/NEMA={mappo_ptl / max(results['nema'], 1.0):.3f}"
        )

    log_line = (
        f"  EVAL ep {train_ep} [{scenario_name}]: person-time-loss -> "
        f"{', '.join(metric_parts)} | {', '.join(ratio_parts)}"
    )
    return ratio, log_line


def _sanitize_scenario_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)
    return safe.strip("_") or "scenario"


def _build_scenario_best_paths(model_path: Path, scenario_names: list[str]) -> dict[str, Path]:
    return {
        scenario_name: model_path.with_name(
            f"{model_path.stem}_best_{_sanitize_scenario_name(scenario_name)}{model_path.suffix}"
        )
        for scenario_name in scenario_names
    }


def _build_train_state_path(model_path: Path) -> Path:
    return model_path.with_name(f"{model_path.stem}_train_state.json")


def _run_episode_worker(args: dict) -> dict:
    """Run a single SUMO episode in a subprocess for parallel collection.

    All imports are local to avoid CUDA initialization in the subprocess.
    The worker creates its own SUMO env + model, runs one episode, and returns
    trajectories + metrics as pickle-safe dicts.
    """
    import os
    import sys
    import time as _time

    os.environ["LIBSUMO_AS_TRACI"] = "1"
    if "/usr/share/sumo/tools" not in sys.path:
        sys.path.insert(0, "/usr/share/sumo/tools")

    import numpy as np
    from traci.exceptions import FatalTraCIError

    from ece324_tango.asce.baselines import MaxPressureController
    from ece324_tango.asce.env import (
        create_parallel_env,
        flatten_obs_by_agent,
        pad_observation,
    )
    from ece324_tango.asce.mappo import (
        MAPPOTrainer,
        ResidualMAPPOTrainer,
        Transition,
        augment_obs_with_mp,
    )
    from ece324_tango.asce.runtime import extract_reset_obs, extract_step_details
    from ece324_tango.asce.traffic_metrics import (
        RewardWeights,
        compute_metrics_for_agents,
        rewards_from_metrics,
    )

    # Unpack args
    net_file = args["net_file"]
    route_file = args["route_file"]
    seed = args["seed"]
    seconds = args["seconds"]
    delta_time = args["delta_time"]
    scenario_id = args["scenario_id"]
    model_state_dict = args["model_state_dict"]
    obs_dim = args["obs_dim"]
    global_obs_dim = args["global_obs_dim"]
    n_actions = args["n_actions"]
    action_dims_dict = args["action_dims_dict"]
    ordered_agents = args["ordered_agents"]
    reward_mode = args["reward_mode"]
    reward_weights_dict = args["reward_weights_dict"]
    residual_mode = args["residual_mode"]
    use_obs_norm = args["use_obs_norm"]
    episode_num = args["episode_num"]

    ep_t0 = _time.time()

    # Create environment
    env = create_parallel_env(
        net_file=net_file,
        route_file=route_file,
        seed=seed,
        use_gui=False,
        seconds=seconds,
        delta_time=delta_time,
        quiet_sumo=True,
    )

    try:
        obs = extract_reset_obs(env.reset(seed=seed))
        if not obs:
            env.close()
            return {
                "trajectories": {},
                "last_values": {},
                "all_rows": [],
                "ep_reward": 0.0,
                "ep_steps": 0,
                "gate_fraction": 0.0,
                "episode_num": episode_num,
                "scenario_id": scenario_id,
                "raw_obs_for_norm": [],
                "raw_gobs_for_norm": [],
                "elapsed": _time.time() - ep_t0,
            }

        # Create trainer on CPU (no CUDA in subprocess)
        if residual_mode == "action_gate":
            trainer = ResidualMAPPOTrainer(
                obs_dim=obs_dim,
                global_obs_dim=global_obs_dim,
                n_actions=n_actions,
                residual_mode="action_gate",
                device="cpu",
                use_obs_norm=use_obs_norm,
            )
        else:
            trainer = MAPPOTrainer(
                obs_dim=obs_dim,
                global_obs_dim=global_obs_dim,
                n_actions=n_actions,
                device="cpu",
                use_obs_norm=use_obs_norm,
            )

        # Load model state dict
        import torch

        trainer.actor.load_state_dict(model_state_dict["actor"])
        trainer.critic.load_state_dict(model_state_dict["critic"])
        if trainer.obs_norm is not None and model_state_dict.get("obs_norm") is not None:
            trainer.obs_norm.load_state_dict(model_state_dict["obs_norm"])
        if trainer.gobs_norm is not None and model_state_dict.get("gobs_norm") is not None:
            trainer.gobs_norm.load_state_dict(model_state_dict["gobs_norm"])
        trainer.actor.eval()
        trainer.critic.eval()

        max_pressure = MaxPressureController(action_size_by_agent=action_dims_dict)
        reward_weights = RewardWeights(
            delay=reward_weights_dict["delay"],
            throughput=reward_weights_dict["throughput"],
            fairness=reward_weights_dict["fairness"],
            residual=reward_weights_dict["residual"],
        )

        trajectories: dict[str, list] = {a: [] for a in sorted(obs.keys())}
        all_rows: list[dict] = []
        raw_obs_for_norm: list[np.ndarray] = []
        raw_gobs_for_norm: list[np.ndarray] = []
        done = False
        max_steps = max(1, int(seconds // delta_time))
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
            n_valid_list = [action_dims_dict[a] for a in active_agents]

            # Compute MP actions BEFORE env.step
            mp_actions = max_pressure.actions(obs, env=env)
            mp_actions_list = [mp_actions.get(a, 0) for a in active_agents]

            aug_obs_list = None
            if residual_mode == "action_gate":
                aug_arr = augment_obs_with_mp(
                    np.stack(padded_obs_list), mp_actions_list, n_actions
                )
                aug_obs_list = [aug_arr[i] for i in range(len(active_agents))]

            # Update obs_norm locally (same as sequential path) so act_batch
            # sees properly normalized observations during this episode.
            if residual_mode == "action_gate" and aug_obs_list is not None:
                if trainer.obs_norm is not None:
                    for aug_obs in aug_obs_list:
                        trainer.obs_norm.update(aug_obs)
                raw_obs_for_norm.extend(aug_obs.copy() for aug_obs in aug_obs_list)
            else:
                if trainer.obs_norm is not None:
                    for padded_obs in padded_obs_list:
                        trainer.obs_norm.update(padded_obs)
                raw_obs_for_norm.extend(padded_obs.copy() for padded_obs in padded_obs_list)
            if trainer.gobs_norm is not None:
                trainer.gobs_norm.update(gobs)
            raw_gobs_for_norm.append(gobs.copy())

            if residual_mode == "action_gate":
                batch_out = trainer.act_batch_residual(
                    padded_obs_list, gobs, n_valid_list, mp_actions_list
                )
            else:
                batch_out = trainer.act_batch(padded_obs_list, gobs, n_valid_list)

            actions = {
                a: int(batch_out[i]["action"])
                for i, a in enumerate(active_agents)
            }
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
                    truncated = True
            except FatalTraCIError:
                done = True
                terminated = False
                truncated = True
                next_obs = {}
                rewards = {a: 0.0 for a in active_agents}
                bootstrap_obs = {
                    a: np.asarray(obs[a], dtype=np.float32) for a in active_agents
                }
            if done and truncated and next_obs:
                bootstrap_obs = {
                    a: np.asarray(next_obs[a], dtype=np.float32)
                    for a in sorted(next_obs.keys())
                }
            episode_terminated = episode_terminated or (done and terminated)
            episode_truncated = episode_truncated or (done and truncated)
            sim_time = float(ep_steps + 1) * float(delta_time)
            if not done:
                metrics_by_agent = compute_metrics_for_agents(
                    env=env,
                    agent_ids=active_agents,
                    time_step=sim_time,
                    actions=actions,
                    action_green_dur=float(delta_time),
                    scenario_id=scenario_id,
                    observations=obs,
                )
                shaped_rewards = rewards_from_metrics(
                    metrics_by_agent=metrics_by_agent,
                    mode=reward_mode,
                    weights=reward_weights,
                    mp_deviation_by_agent={
                        a: float(int(actions.get(a, 0) != mp_actions.get(a, 0)))
                        for a in active_agents
                    }
                    if reward_mode == "residual_mp"
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
                a_obs = (
                    aug_obs_list[i] if aug_obs_list is not None else padded_obs_list[i]
                )
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
                        n_valid_actions=int(action_dims_dict[agent]),
                        mp_action=int(action_meta[agent].get("mp_action", 0)),
                        gate=int(action_meta[agent].get("gate", 0)),
                    )
                )

            obs = next_obs

        # Bootstrap value for truncated episodes
        last_values: dict[str, float] = {}
        if episode_truncated and not episode_terminated and bootstrap_obs:
            final_agents = sorted(bootstrap_obs.keys())
            final_gobs = pad_observation(
                flatten_obs_by_agent(bootstrap_obs, final_agents),
                target_dim=global_obs_dim,
            )
            gobs_n = (
                trainer.gobs_norm.normalize(final_gobs)
                if trainer.gobs_norm is not None
                else np.asarray(final_gobs, dtype=np.float32)
            )
            with torch.no_grad():
                gobs_t = torch.tensor(
                    gobs_n, dtype=torch.float32, device=trainer.device
                ).unsqueeze(0)
                bootstrap_value = float(trainer.critic(gobs_t).item())
            for agent in final_agents:
                last_values[agent] = bootstrap_value

        gate_fraction = 0.0
        if residual_mode == "action_gate":
            all_gates = [t.gate for traj in trajectories.values() for t in traj]
            gate_fraction = float(np.mean(all_gates)) if all_gates else 0.0

        elapsed = _time.time() - ep_t0
    finally:
        env.close()

    return {
        "trajectories": trajectories,
        "last_values": last_values,
        "all_rows": all_rows,
        "ep_reward": ep_reward,
        "ep_steps": ep_steps,
        "gate_fraction": gate_fraction,
        "episode_num": episode_num,
        "scenario_id": scenario_id,
        "raw_obs_for_norm": raw_obs_for_norm,
        "raw_gobs_for_norm": raw_gobs_for_norm,
        "elapsed": elapsed,
    }


def _run_eval_worker(args: dict) -> dict:
    """Run one scenario eval in a subprocess so training isn't blocked."""
    import os
    import sys
    import time as _time

    os.environ["LIBSUMO_AS_TRACI"] = "1"
    libsumo_path = "/usr/share/sumo/tools/libsumo/"
    if libsumo_path not in sys.path:
        sys.path.insert(0, libsumo_path)

    import numpy as np_w
    import torch

    from ece324_tango.asce.baselines import (
        FixedTimeController as FTC,
        MaxPressureController as MPC,
    )
    from ece324_tango.asce.env import (
        create_parallel_env as _cpe,
        flatten_obs_by_agent as _foba,
        pad_observation as _po,
    )
    from ece324_tango.asce.kpi import KPITracker as _KPI
    from ece324_tango.asce.mappo import ResidualMAPPOTrainer as _RMT
    from ece324_tango.asce.runtime import extract_reset_obs as _ero
    from ece324_tango.sumo_rl.environment.env import SumoEnvironment as _SE

    t0 = _time.time()
    net_file = args["net_file"]
    eval_route = args["eval_route"]
    model_state = args["model_state_dict"]
    obs_dim = args["obs_dim"]
    global_obs_dim = args["global_obs_dim"]
    n_actions = args["n_actions"]
    action_dims = args["action_dims"]
    ordered_agents = args["ordered_agents"]
    train_ep = args["train_ep"]
    seconds = args["seconds"]
    delta_time = args["delta_time"]
    seed = args["seed"]
    residual_mode = args["residual_mode"]
    use_obs_norm = args["use_obs_norm"]
    eval_baselines = args["eval_baselines"]

    # Build a throwaway trainer with the snapshotted weights
    trainer = _RMT(
        obs_dim=obs_dim,
        global_obs_dim=global_obs_dim,
        n_actions=n_actions,
        residual_mode=residual_mode,
        device="cpu",
        use_obs_norm=use_obs_norm,
    )
    trainer.actor.load_state_dict(model_state["actor"])
    trainer.critic.load_state_dict(model_state["critic"])
    if trainer.obs_norm is not None and model_state.get("obs_norm") is not None:
        trainer.obs_norm.load_state_dict(model_state["obs_norm"])
    if trainer.gobs_norm is not None and model_state.get("gobs_norm") is not None:
        trainer.gobs_norm.load_state_dict(model_state["gobs_norm"])
    trainer.actor.eval()
    trainer.critic.eval()

    from pathlib import Path as _P

    scenario_name = _P(eval_route).stem.removesuffix(".rou")
    results = {}

    for ctrl in _eval_controller_sequence(eval_baselines):
        try:
            if ctrl == "nema":
                ev = _SE(
                    net_file=net_file,
                    route_file=eval_route,
                    use_gui=False,
                    num_seconds=seconds,
                    delta_time=delta_time,
                    sumo_seed=seed,
                    single_agent=False,
                    sumo_warnings=False,
                    fixed_ts=True,
                    additional_sumo_cmd="--no-step-log true",
                )
            else:
                ev = _cpe(
                    net_file=net_file,
                    route_file=eval_route,
                    seed=seed,
                    use_gui=False,
                    seconds=seconds,
                    delta_time=delta_time,
                    quiet_sumo=True,
                )
            obs_raw = ev.reset(seed=seed)
            obs = _ero(obs_raw)
            if not obs:
                ev.close()
                continue
            agents = sorted(obs.keys())
            a_dims = {a: int(ev.action_spaces(a).n) for a in agents}
            mp = MPC(action_size_by_agent=a_dims)
            ft = FTC(action_size_by_agent=a_dims, green_duration_s=delta_time)
            kpi = _KPI()
            done = False

            while not done:
                active = sorted(obs.keys())
                if ctrl == "mappo":
                    gobs = _foba(obs, active)
                    padded = [
                        _po(np_w.asarray(obs[a], dtype=np_w.float32), target_dim=obs_dim)
                        for a in active
                    ]
                    n_valid = [a_dims[a] for a in active]
                    if residual_mode == "action_gate":
                        mp_acts = mp.actions(obs, env=ev)
                        mp_list = [mp_acts.get(a, 0) for a in active]
                        batch_out = trainer.act_batch_residual(padded, gobs, n_valid, mp_list)
                    else:
                        batch_out = trainer.act_batch(padded, gobs, n_valid)
                    actions = {a: int(batch_out[i]["action"]) for i, a in enumerate(active)}
                elif ctrl == "max_pressure":
                    actions = mp.actions(obs, env=ev)
                elif ctrl == "fixed_time":
                    actions = ft.actions(obs)
                else:
                    actions = {}

                try:
                    result = ev.step(actions)
                    if ctrl == "nema":
                        obs, _, dones, _ = result
                        done = dones.get("__all__", False) if isinstance(dones, dict) else dones
                    else:
                        obs, _, done_flag, _ = result
                        done = (
                            done_flag.get("__all__", False)
                            if isinstance(done_flag, dict)
                            else done_flag
                        )
                except Exception:
                    done = True
                kpi.update(ev)

            k = kpi.summary()
            results[ctrl] = k.person_time_loss_s
        except Exception:
            pass
        finally:
            try:
                ev.close()
            except Exception:
                pass

    ratio, log_line = _format_eval_summary(scenario_name, train_ep, results)

    elapsed = _time.time() - t0
    return {
        "train_ep": train_ep,
        "scenario_result": {
            "scenario_name": scenario_name,
            "ratio": ratio,
            "results": results,
            "log_line": log_line,
        },
        "elapsed": elapsed,
    }


class LocalMappoBackend(AsceTrainerBackend):
    name = "local_mappo"

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _update_best_eval_checkpoints(
        self,
        trainer,
        per_scenario_results: list[dict],
        best_eval_ratio: float,
        best_model_path: str,
        scenario_best_ratios: dict[str, float],
        scenario_best_paths: dict[str, Path],
    ) -> float:
        worst = max(res["ratio"] for res in per_scenario_results)
        if worst < best_eval_ratio:
            best_eval_ratio = worst
            trainer.save(str(best_model_path))
            logger.info(
                f"  New overall best model: MAPPO/MP={worst:.3f} → {best_model_path}"
            )

        for res in per_scenario_results:
            scenario_name = res["scenario_name"]
            ratio = res["ratio"]
            if ratio < scenario_best_ratios.get(scenario_name, float("inf")):
                scenario_best_ratios[scenario_name] = ratio
                scenario_best_path = scenario_best_paths[scenario_name]
                trainer.save(str(scenario_best_path))
                logger.info(
                    f"  New best for [{scenario_name}]: MAPPO/MP={ratio:.3f} → "
                    f"{scenario_best_path}"
                )

        return best_eval_ratio

    def _save_train_state(
        self,
        state_path: Path,
        start_episode: int,
        best_eval_ratio: float,
        scenario_best_ratios: dict[str, float],
        last_eval_ep: int | str | None,
        last_eval_ratios: dict[str, float],
    ) -> None:
        worst_scenario = (
            max(last_eval_ratios, key=last_eval_ratios.get)
            if last_eval_ratios
            else ""
        )
        payload = {
            "start_episode": int(start_episode),
            "best_eval_ratio": float(best_eval_ratio)
            if best_eval_ratio != float("inf")
            else None,
            "scenario_best_ratios": {
                k: (float(v) if v != float("inf") else None)
                for k, v in scenario_best_ratios.items()
            },
            "last_eval_ep": last_eval_ep,
            "last_eval_ratios": {k: float(v) for k, v in last_eval_ratios.items()},
            "best_eval_scenario": worst_scenario,
        }
        state_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    def _load_train_state(self, state_path: Path) -> dict:
        if not state_path.exists():
            return {}
        try:
            return json.loads(state_path.read_text())
        except Exception as exc:
            logger.warning(f"Failed to load train state from {state_path}: {exc}")
            return {}

    def train(self, cfg: TrainConfig) -> None:
        cfg.model_path.parent.mkdir(parents=True, exist_ok=True)
        cfg.rollout_csv.parent.mkdir(parents=True, exist_ok=True)
        cfg.episode_metrics_csv.parent.mkdir(parents=True, exist_ok=True)

        import time as _time

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
            logger.info("Initial env.reset() ...")
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

            # Scale LR for batched training: 1/sqrt(num_workers) reduces
            # effective step size to compensate for larger gradient batches.
            lr_kwargs = {}
            if cfg.num_workers > 1 and cfg.scale_lr_by_workers:
                import math as _math

                lr_scale = 1.0 / _math.sqrt(cfg.num_workers)
                lr_kwargs = {
                    "actor_lr": 3e-4 * lr_scale,
                    "critic_lr": 1e-3 * lr_scale,
                }
                logger.info(
                    f"LR scaled by 1/sqrt({cfg.num_workers})={lr_scale:.3f}: "
                    f"actor_lr={lr_kwargs['actor_lr']:.2e}, "
                    f"critic_lr={lr_kwargs['critic_lr']:.2e}"
                )

            if cfg.residual_mode == "action_gate":
                trainer = ResidualMAPPOTrainer(
                    obs_dim=obs_dim,
                    global_obs_dim=global_obs_dim,
                    n_actions=n_actions,
                    residual_mode="action_gate",
                    device=resolved_device,
                    use_obs_norm=cfg.use_obs_norm,
                    **lr_kwargs,
                )
            else:
                trainer = MAPPOTrainer(
                    obs_dim=obs_dim,
                    global_obs_dim=global_obs_dim,
                    n_actions=n_actions,
                    device=resolved_device,
                    use_obs_norm=cfg.use_obs_norm,
                    **lr_kwargs,
                )

            train_state_path = _build_train_state_path(cfg.model_path)
            train_state = self._load_train_state(train_state_path)

            # Resume from checkpoint if requested
            start_episode = 0
            if cfg.resume and cfg.model_path.exists():
                trainer.load(str(cfg.model_path))
                # Infer start_episode from existing metrics CSV
                if cfg.episode_metrics_csv.exists():
                    existing_metrics = pd.read_csv(cfg.episode_metrics_csv)
                    start_episode = int(existing_metrics["episode"].max()) + 1
                    logger.info(
                        f"Resumed from {cfg.model_path} at episode {start_episode}"
                    )
                else:
                    logger.info(f"Resumed model from {cfg.model_path}, starting at episode 0")

            # Warm-start: load weights from a prior model without resuming episode count
            elif cfg.warm_start_model:
                # Capture correct normalizer dims before load() overwrites them
                # (ResidualMAPPOTrainer uses augmented_dim = obs_dim + n_actions)
                fresh_obs_norm_dim = trainer.obs_norm.dim if trainer.obs_norm else obs_dim
                fresh_gobs_norm_dim = (
                    trainer.gobs_norm.dim if trainer.gobs_norm else global_obs_dim
                )
                trainer.load(cfg.warm_start_model)
                logger.info(f"Warm-started weights from {cfg.warm_start_model}")
                if cfg.reset_obs_norm:
                    if trainer.obs_norm is not None:
                        trainer.obs_norm = ObsRunningNorm(fresh_obs_norm_dim)
                    if trainer.gobs_norm is not None:
                        trainer.gobs_norm = ObsRunningNorm(fresh_gobs_norm_dim)
                    logger.info("Obs normalizer reset — will re-fit to curriculum distribution")

            all_rows: List[dict] = []
            ep_metrics: List[dict] = []
            best_eval_ratio = float("inf")  # best MAPPO/MP ratio (lower is better)
            best_model_path = cfg.model_path.with_name(
                cfg.model_path.stem + "_best" + cfg.model_path.suffix
            )
            eval_scenario_names = (
                [Path(rf).stem.removesuffix(".rou") for rf in cfg.route_files]
                if cfg.route_files
                else [Path(cfg.route_file).stem.removesuffix(".rou")]
            )
            scenario_best_paths = _build_scenario_best_paths(
                cfg.model_path, eval_scenario_names
            )
            scenario_best_ratios = {
                scenario_name: float("inf") for scenario_name in eval_scenario_names
            }
            last_eval_ep = None
            last_eval_ratios: dict[str, float] = {}

            if cfg.resume:
                if cfg.episode_metrics_csv.exists():
                    try:
                        ep_metrics_df = pd.read_csv(cfg.episode_metrics_csv)
                        ep_metrics = ep_metrics_df.to_dict("records")
                    except Exception as exc:
                        logger.warning(
                            f"Failed to restore episode metrics from {cfg.episode_metrics_csv}: {exc}"
                        )
                if cfg.rollout_csv.exists():
                    try:
                        rollout_df = pd.read_csv(cfg.rollout_csv)
                        all_rows = rollout_df.to_dict("records")
                    except Exception as exc:
                        logger.warning(
                            f"Failed to restore rollout rows from {cfg.rollout_csv}: {exc}"
                        )
                if train_state:
                    if train_state.get("start_episode") is not None:
                        start_episode = max(start_episode, int(train_state["start_episode"]))
                    if train_state.get("best_eval_ratio") is not None:
                        best_eval_ratio = float(train_state["best_eval_ratio"])
                    for scenario_name, ratio in train_state.get(
                        "scenario_best_ratios", {}
                    ).items():
                        if (
                            scenario_name in scenario_best_ratios
                            and ratio is not None
                        ):
                            scenario_best_ratios[scenario_name] = float(ratio)
                    last_eval_ep = train_state.get("last_eval_ep")
                    last_eval_ratios = {
                        k: float(v)
                        for k, v in train_state.get("last_eval_ratios", {}).items()
                    }

            # Resolve effective reward mode: action_gate supersedes residual_mp
            if cfg.residual_mode == "action_gate" and cfg.reward_mode == "residual_mp":
                logger.warning(
                    "residual_mode=action_gate supersedes residual_mp reward mode. "
                    "Switching to 'objective' reward for this run."
                )
                effective_reward_mode = "objective"
            else:
                effective_reward_mode = cfg.reward_mode

            reward_weights = RewardWeights(
                delay=cfg.reward_delay_weight,
                throughput=cfg.reward_throughput_weight,
                fairness=cfg.reward_fairness_weight,
                residual=cfg.reward_residual_weight,
            )

            # Graceful interrupt: checkpoint on SIGINT/SIGTERM
            _interrupt_requested = False

            def _handle_interrupt(signum, frame):
                nonlocal _interrupt_requested
                _interrupt_requested = True
                sig_name = signal.Signals(signum).name
                logger.warning(f"Received {sig_name} — will checkpoint after current episode")

            prev_sigint = signal.signal(signal.SIGINT, _handle_interrupt)
            prev_sigterm = signal.signal(signal.SIGTERM, _handle_interrupt)

            if cfg.num_workers > 1:
                best_eval_ratio = self._train_parallel(
                    cfg=cfg,
                    trainer=trainer,
                    obs_dim=obs_dim,
                    global_obs_dim=global_obs_dim,
                    n_actions=n_actions,
                    action_dims=action_dims,
                    ordered_agents=ordered_agents,
                    effective_reward_mode=effective_reward_mode,
                    reward_weights=reward_weights,
                    start_episode=start_episode,
                    all_rows=all_rows,
                    ep_metrics=ep_metrics,
                    env=env,
                    _interrupt_requested_ref=lambda: _interrupt_requested,
                    best_eval_ratio=best_eval_ratio,
                    best_model_path=str(best_model_path),
                    scenario_best_ratios=scenario_best_ratios,
                    scenario_best_paths=scenario_best_paths,
                    train_state_path=train_state_path,
                    last_eval_ep=last_eval_ep,
                    last_eval_ratios=last_eval_ratios,
                )
                train_state = self._load_train_state(train_state_path)
                if train_state.get("best_eval_ratio") is not None:
                    best_eval_ratio = float(train_state["best_eval_ratio"])
                last_eval_ep = train_state.get("last_eval_ep")
                last_eval_ratios = {
                    k: float(v)
                    for k, v in train_state.get("last_eval_ratios", {}).items()
                }
                for scenario_name, ratio in train_state.get(
                    "scenario_best_ratios", {}
                ).items():
                    if scenario_name in scenario_best_ratios and ratio is not None:
                        scenario_best_ratios[scenario_name] = float(ratio)

            for ep in range(start_episode, cfg.episodes):
                if cfg.num_workers > 1:
                    break  # parallel path already handled all episodes

                # Curriculum: select scenario for this episode
                if cfg.route_files:
                    scenario_pool_seq = cfg.route_files
                    scenario_ids_seq = [
                        Path(rf).stem.removesuffix(".rou") for rf in scenario_pool_seq
                    ]
                    current_route = scenario_pool_seq[ep % len(scenario_pool_seq)]
                    current_scenario_id = scenario_ids_seq[ep % len(scenario_pool_seq)]
                    env.close()
                    env = create_parallel_env(
                        net_file=cfg.net_file,
                        route_file=current_route,
                        seed=cfg.seed + ep,
                        use_gui=cfg.use_gui,
                        seconds=cfg.seconds,
                        delta_time=cfg.delta_time,
                        quiet_sumo=not cfg.backend_verbose,
                    )
                else:
                    current_scenario_id = cfg.scenario_id

                ep_t0 = _time.time()
                logger.info(f"Episode {ep}: resetting env ...")
                obs = extract_reset_obs(env.reset(seed=cfg.seed + ep))
                logger.info(
                    f"Episode {ep}: reset done in {_time.time() - ep_t0:.1f}s, "
                    f"agents={len(obs) if obs else 0}"
                )
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
                step_t0 = _time.time()

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

                    # Compute MP actions BEFORE env.step (Pitfall C5)
                    mp_actions = max_pressure.actions(obs, env=env)
                    mp_actions_list = [mp_actions.get(a, 0) for a in active_agents]

                    # For action_gate, augment obs with MP one-hot (obs_dim + n_actions).
                    # This augmented version is what obs_norm, the actor, and transitions use.
                    aug_obs_list = None
                    if cfg.residual_mode == "action_gate":
                        aug_arr = augment_obs_with_mp(
                            np.stack(padded_obs_list), mp_actions_list, n_actions
                        )
                        aug_obs_list = [aug_arr[i] for i in range(len(active_agents))]

                    # Update normalizer stats before acting
                    if cfg.residual_mode == "action_gate" and trainer.obs_norm is not None:
                        for aug_obs in aug_obs_list:
                            trainer.obs_norm.update(aug_obs)
                    elif trainer.obs_norm is not None:
                        for padded_obs in padded_obs_list:
                            trainer.obs_norm.update(padded_obs)
                    if trainer.gobs_norm is not None:
                        trainer.gobs_norm.update(gobs)

                    if cfg.residual_mode == "action_gate":
                        batch_out = trainer.act_batch_residual(
                            padded_obs_list, gobs, n_valid_list, mp_actions_list
                        )
                    else:
                        batch_out = trainer.act_batch(
                            padded_obs_list, gobs, n_valid_list
                        )
                    actions = {
                        a: int(batch_out[i]["action"])
                        for i, a in enumerate(active_agents)
                    }
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
                        # This is a truncation, NOT a terminal state — the episode was
                        # cut short by an external event, so we should bootstrap from the
                        # critic's last value estimate rather than assuming value=0 (FIX-02).
                        logger.warning(
                            f"Episode {ep}: SUMO connection closed at step {ep_steps} "
                            "(demand exhausted). Treating as truncation for bootstrapping."
                        )
                        done = True
                        terminated = False
                        truncated = True
                        next_obs = {}
                        rewards = {a: 0.0 for a in active_agents}
                        # Use raw obs (not padded) so flatten_obs_by_agent produces
                        # the same global_obs_dim that gobs_norm was initialized with.
                        bootstrap_obs = {
                            a: np.asarray(obs[a], dtype=np.float32)
                            for a in active_agents
                        }
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
                            mode=effective_reward_mode,
                            weights=reward_weights,
                            mp_deviation_by_agent={
                                a: float(int(actions.get(a, 0) != mp_actions.get(a, 0)))
                                for a in active_agents
                            }
                            if effective_reward_mode == "residual_mp"
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

                    if ep_steps % 20 == 0:
                        elapsed = _time.time() - step_t0
                        logger.debug(
                            f"  ep {ep} step {ep_steps}/{max_steps} "
                            f"({elapsed:.1f}s, {elapsed/ep_steps:.2f}s/step)"
                        )

                    for i, agent in enumerate(active_agents):
                        # Store augmented obs for action_gate so PPO update
                        # sees the same shape that obs_norm and GatedActor expect.
                        a_obs = aug_obs_list[i] if aug_obs_list is not None else padded_obs_list[i]
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
                                mp_action=int(action_meta[agent].get("mp_action", 0)),
                                gate=int(action_meta[agent].get("gate", 0)),
                            )
                        )

                    obs = next_obs

                # Bootstrap value for truncated (non-terminal) episodes
                last_values: Dict[str, float] = {}
                if episode_truncated and not episode_terminated and bootstrap_obs:
                    import torch

                    final_agents = sorted(bootstrap_obs.keys())
                    final_gobs = pad_observation(
                        flatten_obs_by_agent(bootstrap_obs, final_agents),
                        target_dim=global_obs_dim,
                    )
                    # Bootstrap only needs the critic value (V(s)), not actor outputs.
                    # Calling trainer.act() would fail for GatedActor (returns tuple).
                    gobs_n = (
                        trainer.gobs_norm.normalize(final_gobs)
                        if trainer.gobs_norm is not None
                        else np.asarray(final_gobs, dtype=np.float32)
                    )
                    with torch.no_grad():
                        gobs_t = torch.tensor(
                            gobs_n, dtype=torch.float32, device=trainer.device
                        ).unsqueeze(0)
                        bootstrap_value = float(trainer.critic(gobs_t).item())
                    for agent in final_agents:
                        last_values[agent] = bootstrap_value

                sim_elapsed = _time.time() - ep_t0
                logger.info(
                    f"Episode {ep}: sim done ({ep_steps} steps in {sim_elapsed:.1f}s), "
                    f"updating PPO ..."
                )
                batch = trainer.build_batch(trajectories, last_values=last_values)
                losses = trainer.update(
                    batch=batch,
                    ppo_epochs=cfg.ppo_epochs,
                    minibatch_size=cfg.minibatch_size,
                )

                gate_fraction = 0.0
                if cfg.residual_mode == "action_gate":
                    all_gates = [
                        t.gate
                        for traj in trajectories.values()
                        for t in traj
                    ]
                    gate_fraction = float(np.mean(all_gates)) if all_gates else 0.0

                ep_metrics.append(
                    {
                        "episode": ep,
                        "seed": cfg.seed + ep,
                        "scenario_id": current_scenario_id,
                        "mean_global_reward": ep_reward / max(1, ep_steps),
                        "steps": ep_steps,
                        "actor_loss": losses["actor_loss"],
                        "critic_loss": losses["critic_loss"],
                        "entropy": losses["entropy"],
                        "gate_fraction": gate_fraction,
                    }
                )
                logger.info(
                    f"Episode {ep}: reward={ep_reward / max(1, ep_steps):.4f}, "
                    f"actor_loss={losses['actor_loss']:.4f}, critic_loss={losses['critic_loss']:.4f}, "
                    f"gate_fraction={gate_fraction:.3f}"
                )

                # Periodic checkpoint
                if cfg.checkpoint_every > 0 and (ep + 1) % cfg.checkpoint_every == 0:
                    trainer.save(str(cfg.model_path))
                    pd.DataFrame(ep_metrics).to_csv(cfg.episode_metrics_csv, index=False)
                    self._save_train_state(
                        state_path=train_state_path,
                        start_episode=ep + 1,
                        best_eval_ratio=best_eval_ratio,
                        scenario_best_ratios=scenario_best_ratios,
                        last_eval_ep=last_eval_ep,
                        last_eval_ratios=last_eval_ratios,
                    )
                    logger.info(f"  Checkpoint saved at episode {ep}")

                # Periodic baseline evaluation
                if cfg.eval_every > 0 and (ep + 1) % cfg.eval_every == 0:
                    eval_res = self._run_inline_eval(
                        cfg, env, trainer, obs_dim, global_obs_dim, action_dims,
                        n_actions, ordered_agents, ep,
                        route_files=cfg.route_files if cfg.route_files else None,
                    )
                    best_eval_ratio = self._update_best_eval_checkpoints(
                        trainer=trainer,
                        per_scenario_results=eval_res["per_scenario"],
                        best_eval_ratio=best_eval_ratio,
                        best_model_path=str(best_model_path),
                        scenario_best_ratios=scenario_best_ratios,
                        scenario_best_paths=scenario_best_paths,
                    )
                    last_eval_ep = ep
                    last_eval_ratios = {
                        res["scenario_name"]: res["ratio"]
                        for res in eval_res["per_scenario"]
                    }
                    self._save_train_state(
                        state_path=train_state_path,
                        start_episode=ep + 1,
                        best_eval_ratio=best_eval_ratio,
                        scenario_best_ratios=scenario_best_ratios,
                        last_eval_ep=last_eval_ep,
                        last_eval_ratios=last_eval_ratios,
                    )

                # Graceful exit on interrupt
                if _interrupt_requested:
                    logger.warning(
                        f"Interrupted after episode {ep} — saving checkpoint ..."
                    )
                    trainer.save(str(cfg.model_path))
                    pd.DataFrame(ep_metrics).to_csv(cfg.episode_metrics_csv, index=False)
                    rollout_df = pd.DataFrame(all_rows)
                    if not rollout_df.empty:
                        rollout_df = rollout_df[ASCE_DATASET_COLUMNS]
                    rollout_df.to_csv(cfg.rollout_csv, index=False)
                    self._save_train_state(
                        state_path=train_state_path,
                        start_episode=ep + 1,
                        best_eval_ratio=best_eval_ratio,
                        scenario_best_ratios=scenario_best_ratios,
                        last_eval_ep=last_eval_ep,
                        last_eval_ratios=last_eval_ratios,
                    )
                    logger.success(
                        f"Checkpoint saved at episode {ep}. "
                        f"Resume with --resume --episodes {cfg.episodes}"
                    )
                    break

            signal.signal(signal.SIGINT, prev_sigint)
            signal.signal(signal.SIGTERM, prev_sigterm)

            trainer.save(str(cfg.model_path))
            logger.success(f"Saved MAPPO model: {cfg.model_path}")

            rollout_df = pd.DataFrame(all_rows)
            if not rollout_df.empty:
                rollout_df = rollout_df[ASCE_DATASET_COLUMNS]
            rollout_df.to_csv(cfg.rollout_csv, index=False)
            pd.DataFrame(ep_metrics).to_csv(cfg.episode_metrics_csv, index=False)
            self._save_train_state(
                state_path=train_state_path,
                start_episode=cfg.episodes,
                best_eval_ratio=best_eval_ratio,
                scenario_best_ratios=scenario_best_ratios,
                last_eval_ep=last_eval_ep,
                last_eval_ratios=last_eval_ratios,
            )

            logger.success(f"Saved rollout samples: {cfg.rollout_csv}")
            logger.success(f"Saved episode metrics: {cfg.episode_metrics_csv}")

            # Post-training multi-seed evaluation
            if cfg.final_eval_seeds > 0 and not _interrupt_requested:
                logger.info(
                    f"Running {cfg.final_eval_seeds}-seed final evaluation ..."
                )
                final_ratios = []
                for s in range(cfg.final_eval_seeds):
                    eval_seed = cfg.seed + 1000 + s
                    eval_res = self._run_inline_eval(
                        cfg, env, trainer, obs_dim, global_obs_dim,
                        action_dims, n_actions, ordered_agents,
                        train_ep=f"final-s{s}",
                        eval_seed=eval_seed,
                        route_files=cfg.route_files if cfg.route_files else None,
                    )
                    final_ratios.append(eval_res["worst_ratio"])
                mean_ratio = float(np.mean(final_ratios))
                std_ratio = float(np.std(final_ratios))
                logger.success(
                    f"Final eval ({cfg.final_eval_seeds} seeds): "
                    f"MAPPO/MP = {mean_ratio:.3f} +/- {std_ratio:.3f}  "
                    f"(seeds: {[f'{r:.3f}' for r in final_ratios]})"
                )
        finally:
            env.close()

    def _train_parallel(
        self,
        cfg: TrainConfig,
        trainer,
        obs_dim: int,
        global_obs_dim: int,
        n_actions: int,
        action_dims: dict,
        ordered_agents: list,
        effective_reward_mode: str,
        reward_weights: RewardWeights,
        start_episode: int,
        all_rows: list,
        ep_metrics: list,
        env,
        _interrupt_requested_ref,
        best_eval_ratio: float = float("inf"),
        best_model_path: str = "",
        scenario_best_ratios: dict[str, float] | None = None,
        scenario_best_paths: dict[str, Path] | None = None,
        train_state_path: Path | None = None,
        last_eval_ep: int | str | None = None,
        last_eval_ratios: dict[str, float] | None = None,
    ) -> float:
        """Run training with parallel episode collection via multiprocessing.Pool."""
        import multiprocessing
        import time as _time

        import torch

        scenario_best_ratios = scenario_best_ratios or {}
        scenario_best_paths = scenario_best_paths or {}
        last_eval_ratios = last_eval_ratios or {}
        reward_weights_dict = {
            "delay": reward_weights.delay,
            "throughput": reward_weights.throughput,
            "fairness": reward_weights.fairness,
            "residual": reward_weights.residual,
        }

        eval_workers = max(1, cfg.eval_workers)
        mp_ctx = multiprocessing.get_context("spawn")
        pool = mp_ctx.Pool(
            processes=cfg.num_workers, maxtasksperchild=1,
            initializer=_worker_init,
        )
        eval_pool = mp_ctx.Pool(
            processes=eval_workers, maxtasksperchild=1,
            initializer=_worker_init,
        )
        bg_eval_result = None

        # Live status panel
        from ece324_tango.asce.trainers.training_tui import TrainingStatus

        scenario_pool_names = [
            Path(rf).stem.removesuffix(".rou")
            for rf in (cfg.route_files if cfg.route_files else [cfg.route_file])
        ]
        tui = TrainingStatus(
            total_episodes=cfg.episodes,
            num_workers=cfg.num_workers,
            scenario_names=scenario_pool_names,
            device=str(trainer.device) if hasattr(trainer, "device") else "cuda",
            scenario_id=cfg.scenario_id,
            start_episode=start_episode,
            initial_best_ratio=best_eval_ratio,
            initial_best_scenario=(
                max(last_eval_ratios, key=last_eval_ratios.get) if last_eval_ratios else ""
            ),
            initial_eval_ep=last_eval_ep,
            initial_eval_ratios=last_eval_ratios,
        )
        tui.start()

        try:
            for ep_batch_start in range(
                start_episode, cfg.episodes, cfg.num_workers
            ):
                batch_t0 = _time.time()
                batch_size = min(cfg.num_workers, cfg.episodes - ep_batch_start)

                # Serialize current model state (CPU tensors for pickling)
                model_state_dict = {
                    "actor": {
                        k: v.cpu() for k, v in trainer.actor.state_dict().items()
                    },
                    "critic": {
                        k: v.cpu() for k, v in trainer.critic.state_dict().items()
                    },
                    "obs_norm": trainer.obs_norm.state_dict()
                    if trainer.obs_norm is not None
                    else None,
                    "gobs_norm": trainer.gobs_norm.state_dict()
                    if trainer.gobs_norm is not None
                    else None,
                }

                # Determine scenario pool for curriculum round-robin
                scenario_pool = cfg.route_files if cfg.route_files else [cfg.route_file]
                scenario_ids = [
                    Path(rf).stem.removesuffix(".rou") for rf in scenario_pool
                ]

                # Log scenario assignments for this batch
                if len(scenario_pool) > 1:
                    assignments = [
                        f"w{i}={scenario_ids[(ep_batch_start + i) % len(scenario_pool)]}"
                        for i in range(batch_size)
                    ]
                    logger.info(
                        f"Batch ep {ep_batch_start}-{ep_batch_start + batch_size - 1} "
                        f"scenarios: {', '.join(assignments)}"
                    )

                worker_args = [
                    {
                        "net_file": cfg.net_file,
                        "route_file": scenario_pool[
                            (ep_batch_start + i) % len(scenario_pool)
                        ],
                        "seed": cfg.seed + ep_batch_start + i,
                        "seconds": cfg.seconds,
                        "delta_time": cfg.delta_time,
                        "scenario_id": scenario_ids[
                            (ep_batch_start + i) % len(scenario_pool)
                        ],
                        "model_state_dict": model_state_dict,
                        "obs_dim": obs_dim,
                        "global_obs_dim": global_obs_dim,
                        "n_actions": n_actions,
                        "action_dims_dict": action_dims,
                        "ordered_agents": ordered_agents,
                        "reward_mode": effective_reward_mode,
                        "reward_weights_dict": reward_weights_dict,
                        "residual_mode": cfg.residual_mode,
                        "use_obs_norm": cfg.use_obs_norm,
                        "episode_num": ep_batch_start + i,
                    }
                    for i in range(batch_size)
                ]

                async_result = pool.map_async(_run_episode_worker, worker_args)
                # Poll with timeout so Ctrl+C can break through
                while not async_result.ready():
                    if _interrupt_requested_ref():
                        break
                    async_result.wait(timeout=2.0)
                if _interrupt_requested_ref() and not async_result.ready():
                    # Workers ignore SIGINT, so terminate them
                    pool.terminate()
                    pool.join()
                    # Save what we have
                    logger.warning(
                        f"Interrupted during batch ep {ep_batch_start} "
                        f"— saving checkpoint ..."
                    )
                    trainer.save(str(cfg.model_path))
                    pd.DataFrame(ep_metrics).to_csv(
                        cfg.episode_metrics_csv, index=False
                    )
                    if train_state_path is not None:
                        self._save_train_state(
                            state_path=train_state_path,
                            start_episode=ep_batch_start,
                            best_eval_ratio=best_eval_ratio,
                            scenario_best_ratios=scenario_best_ratios,
                            last_eval_ep=last_eval_ep,
                            last_eval_ratios=last_eval_ratios,
                        )
                    logger.success(
                        f"Checkpoint saved. "
                        f"Resume with --resume --episodes {cfg.episodes}"
                    )
                    tui.stop()
                    eval_pool.terminate()
                    eval_pool.join()
                    return best_eval_ratio
                results = async_result.get()

                # Update obs_norm in main process from worker-collected raw observations
                for res in results:
                    if trainer.obs_norm is not None:
                        for raw_obs in res["raw_obs_for_norm"]:
                            trainer.obs_norm.update(raw_obs)
                    if trainer.gobs_norm is not None:
                        for raw_gobs in res["raw_gobs_for_norm"]:
                            trainer.gobs_norm.update(raw_gobs)

                # Build per-episode batches with correct GAE, then merge
                batches = []
                for res in results:
                    if res["trajectories"]:
                        batch = trainer.build_batch(
                            res["trajectories"],
                            last_values=res["last_values"],
                        )
                        batches.append(batch)

                if batches:
                    merged_batch = {
                        k: np.concatenate([b[k] for b in batches])
                        for k in batches[0]
                    }
                    losses = trainer.update(
                        batch=merged_batch,
                        ppo_epochs=cfg.ppo_epochs,
                        minibatch_size=cfg.minibatch_size,
                    )
                else:
                    losses = {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}

                # Collect metrics and rows from all workers
                batch_elapsed = _time.time() - batch_t0
                worker_times = [r["elapsed"] for r in results]
                progress_pct = min(100, (ep_batch_start + batch_size) / cfg.episodes * 100)
                logger.info(
                    f"Workers completed: ep {ep_batch_start}-"
                    f"{ep_batch_start + batch_size - 1} "
                    f"({progress_pct:.0f}%), "
                    f"avg {np.mean(worker_times):.1f}s/episode, "
                    f"batch wall {batch_elapsed:.1f}s"
                )

                for res in results:
                    ep_num = res["episode_num"]
                    all_rows.extend(res["all_rows"])
                    ep_metrics.append(
                        {
                            "episode": ep_num,
                            "seed": cfg.seed + ep_num,
                            "scenario_id": res["scenario_id"],
                            "mean_global_reward": res["ep_reward"]
                            / max(1, res["ep_steps"]),
                            "steps": res["ep_steps"],
                            "actor_loss": losses["actor_loss"],
                            "critic_loss": losses["critic_loss"],
                            "entropy": losses["entropy"],
                            "gate_fraction": res["gate_fraction"],
                        }
                    )
                    logger.info(
                        f"Episode {ep_num} [{res['scenario_id']}]: "
                        f"reward={res['ep_reward'] / max(1, res['ep_steps']):.4f}, "
                        f"steps={res['ep_steps']}, "
                        f"gate_fraction={res['gate_fraction']:.3f}"
                    )

                # Update TUI panel
                batch_gate_fracs = [r["gate_fraction"] for r in results]
                batch_rewards = [
                    r["ep_reward"] / max(1, r["ep_steps"]) for r in results
                ]
                tui.update_batch(
                    last_ep=ep_batch_start + batch_size - 1,
                    batch_wall_s=batch_elapsed,
                    avg_gate_frac=float(np.mean(batch_gate_fracs)),
                    avg_reward=float(np.mean(batch_rewards)),
                )

                # Periodic checkpoint — trigger if any episode in batch crosses boundary
                last_ep = ep_batch_start + batch_size - 1
                first_ep = ep_batch_start
                if cfg.checkpoint_every > 0 and (
                    last_ep // cfg.checkpoint_every > (first_ep - 1) // cfg.checkpoint_every
                    if first_ep > 0
                    else (last_ep + 1) >= cfg.checkpoint_every
                ):
                    trainer.save(str(cfg.model_path))
                    pd.DataFrame(ep_metrics).to_csv(
                        cfg.episode_metrics_csv, index=False
                    )
                    if train_state_path is not None:
                        self._save_train_state(
                            state_path=train_state_path,
                            start_episode=last_ep + 1,
                            best_eval_ratio=best_eval_ratio,
                            scenario_best_ratios=scenario_best_ratios,
                            last_eval_ep=last_eval_ep,
                            last_eval_ratios=last_eval_ratios,
                        )
                    logger.info(f"  Checkpoint saved at episode {last_ep}")

                # Collect background eval results (if any finished)
                if bg_eval_result is not None and bg_eval_result.ready():
                    try:
                        worker_eval_results = bg_eval_result.get(timeout=1)
                        ev_res = {
                            "train_ep": worker_eval_results[0]["train_ep"],
                            "per_scenario": [
                                res["scenario_result"] for res in worker_eval_results
                            ],
                            "elapsed": max(res["elapsed"] for res in worker_eval_results),
                        }
                        parts = ", ".join(
                            f"{res['scenario_name']} MAPPO/MP={res['ratio']:.3f}"
                            for res in ev_res["per_scenario"]
                        )
                        for res in ev_res["per_scenario"]:
                            logger.info(res["log_line"])
                        worst = max(res["ratio"] for res in ev_res["per_scenario"])
                        if len(ev_res["per_scenario"]) > 1:
                            logger.info(
                                f"  EVAL ep {ev_res['train_ep']}: {parts} | "
                                f"worst={worst:.3f} ({ev_res['elapsed']:.0f}s)"
                            )
                        best_eval_ratio = self._update_best_eval_checkpoints(
                            trainer=trainer,
                            per_scenario_results=ev_res["per_scenario"],
                            best_eval_ratio=best_eval_ratio,
                            best_model_path=str(best_model_path),
                            scenario_best_ratios=scenario_best_ratios,
                            scenario_best_paths=scenario_best_paths,
                        )
                        last_eval_ep = ev_res["train_ep"]
                        last_eval_ratios = {
                            res["scenario_name"]: res["ratio"]
                            for res in ev_res["per_scenario"]
                        }
                        if train_state_path is not None:
                            self._save_train_state(
                                state_path=train_state_path,
                                start_episode=last_ep + 1,
                                best_eval_ratio=best_eval_ratio,
                                scenario_best_ratios=scenario_best_ratios,
                                last_eval_ep=last_eval_ep,
                                last_eval_ratios=last_eval_ratios,
                            )
                        tui.update_eval_results(
                            eval_ep=ev_res["train_ep"],
                            ratios={
                                res["scenario_name"]: res["ratio"]
                                for res in ev_res["per_scenario"]
                            },
                        )
                    except Exception as e:
                        logger.warning(f"  Background eval failed: {e}")
                    bg_eval_result = None

                # Periodic baseline evaluation — launch in background
                if cfg.eval_every > 0 and (
                    last_ep // cfg.eval_every > (first_ep - 1) // cfg.eval_every
                    if first_ep > 0
                    else (last_ep + 1) >= cfg.eval_every
                ):
                    if bg_eval_result is not None and not bg_eval_result.ready():
                        logger.info(
                            f"  Eval still running from prior batch — skipping eval at ep {last_ep}"
                        )
                    else:
                        eval_routes = cfg.route_files if cfg.route_files else [cfg.route_file]
                        eval_model_state_dict = {
                            "actor": {
                                k: v.cpu() for k, v in trainer.actor.state_dict().items()
                            },
                            "critic": {
                                k: v.cpu() for k, v in trainer.critic.state_dict().items()
                            },
                            "obs_norm": trainer.obs_norm.state_dict()
                            if trainer.obs_norm is not None
                            else None,
                            "gobs_norm": trainer.gobs_norm.state_dict()
                            if trainer.gobs_norm is not None
                            else None,
                        }
                        eval_args = [
                            {
                                "net_file": cfg.net_file,
                                "eval_route": eval_route,
                                "model_state_dict": eval_model_state_dict,
                                "obs_dim": obs_dim,
                                "global_obs_dim": global_obs_dim,
                                "n_actions": n_actions,
                                "action_dims": action_dims,
                                "ordered_agents": ordered_agents,
                                "train_ep": last_ep,
                                "seconds": cfg.seconds,
                                "delta_time": cfg.delta_time,
                                "seed": cfg.seed,
                                "residual_mode": cfg.residual_mode,
                                "use_obs_norm": cfg.use_obs_norm,
                                "eval_baselines": cfg.eval_baselines,
                            }
                            for eval_route in eval_routes
                        ]
                        bg_eval_result = eval_pool.map_async(
                            _run_eval_worker, eval_args
                        )
                        logger.info(
                            f"  Background eval launched for ep {last_ep} "
                            f"({len(eval_routes)} scenarios, {eval_workers} eval workers)"
                        )
                        tui.update_eval_started()

                # Graceful exit on interrupt
                if _interrupt_requested_ref():
                    logger.warning(
                        f"Interrupted after episode {last_ep} — saving checkpoint ..."
                    )
                    # Kill background eval immediately
                    if bg_eval_result is not None and not bg_eval_result.ready():
                        logger.info("  Terminating background eval ...")
                        eval_pool.terminate()
                        eval_pool.join()
                        # Recreate pool so finally block doesn't error
                        eval_pool = mp_ctx.Pool(
                            processes=eval_workers, maxtasksperchild=1,
                            initializer=_worker_init,
                        )
                        bg_eval_result = None
                    trainer.save(str(cfg.model_path))
                    pd.DataFrame(ep_metrics).to_csv(
                        cfg.episode_metrics_csv, index=False
                    )
                    rollout_df = pd.DataFrame(all_rows)
                    if not rollout_df.empty:
                        rollout_df = rollout_df[ASCE_DATASET_COLUMNS]
                    rollout_df.to_csv(cfg.rollout_csv, index=False)
                    if train_state_path is not None:
                        self._save_train_state(
                            state_path=train_state_path,
                            start_episode=last_ep + 1,
                            best_eval_ratio=best_eval_ratio,
                            scenario_best_ratios=scenario_best_ratios,
                            last_eval_ep=last_eval_ep,
                            last_eval_ratios=last_eval_ratios,
                        )
                    logger.success(
                        f"Checkpoint saved at episode {last_ep}. "
                        f"Resume with --resume --episodes {cfg.episodes}"
                    )
                    break

            # Drain any pending background eval before returning
            if bg_eval_result is not None:
                try:
                    if bg_eval_result.ready():
                        worker_eval_results = bg_eval_result.get(timeout=1)
                        ev_res = {
                            "train_ep": worker_eval_results[0]["train_ep"],
                            "per_scenario": [
                                res["scenario_result"] for res in worker_eval_results
                            ],
                            "elapsed": max(res["elapsed"] for res in worker_eval_results),
                        }
                        for res in ev_res["per_scenario"]:
                            logger.info(res["log_line"])
                        best_eval_ratio = self._update_best_eval_checkpoints(
                            trainer=trainer,
                            per_scenario_results=ev_res["per_scenario"],
                            best_eval_ratio=best_eval_ratio,
                            best_model_path=str(best_model_path),
                            scenario_best_ratios=scenario_best_ratios,
                            scenario_best_paths=scenario_best_paths,
                        )
                        last_eval_ep = ev_res["train_ep"]
                        last_eval_ratios = {
                            res["scenario_name"]: res["ratio"]
                            for res in ev_res["per_scenario"]
                        }
                        if train_state_path is not None:
                            self._save_train_state(
                                state_path=train_state_path,
                                start_episode=cfg.episodes,
                                best_eval_ratio=best_eval_ratio,
                                scenario_best_ratios=scenario_best_ratios,
                                last_eval_ep=last_eval_ep,
                                last_eval_ratios=last_eval_ratios,
                            )
                    else:
                        logger.info("  Background eval still running — skipping final drain")
                except Exception:
                    pass
        finally:
            tui.stop()
            pool.terminate()
            pool.join()
            eval_pool.terminate()
            eval_pool.join()
        return best_eval_ratio

    def _run_inline_eval(self, cfg, train_env, trainer, obs_dim, global_obs_dim,
                         action_dims, n_actions, ordered_agents, train_ep,
                         eval_seed: int | None = None,
                         route_files: list[str] | None = None):
        """Run one-episode eval for MAPPO vs baselines on the training scenario(s).

        When route_files is provided, evaluates on ALL scenarios and returns the
        worst-case MAPPO/MP ratio (highest ratio = worst performance).

        Configured baselines come from cfg.eval_baselines and must include
        max_pressure because model selection uses the MAPPO/MP ratio.
        """
        eval_routes = route_files if route_files else [cfg.route_file]
        per_scenario_results: list[dict] = []

        for eval_route in eval_routes:
            scenario_name = Path(eval_route).stem.removesuffix(".rou")
            ratio = self._run_single_eval(
                cfg, trainer, obs_dim, global_obs_dim, action_dims, n_actions,
                ordered_agents, train_ep, eval_route, scenario_name,
                eval_seed=eval_seed,
            )
            per_scenario_results.append(
                {"scenario_name": scenario_name, "ratio": ratio}
            )

        if len(per_scenario_results) > 1:
            parts = ", ".join(
                f"{res['scenario_name']} MAPPO/MP={res['ratio']:.3f}"
                for res in per_scenario_results
            )
            worst = max(res["ratio"] for res in per_scenario_results)
            logger.info(f"  EVAL ep {train_ep}: {parts} | worst={worst:.3f}")
        else:
            worst = per_scenario_results[0]["ratio"]

        return {
            "worst_ratio": worst,
            "per_scenario": per_scenario_results,
        }

    def _run_single_eval(self, cfg, trainer, obs_dim, global_obs_dim,
                         action_dims, n_actions, ordered_agents, train_ep,
                         route_file: str, scenario_name: str,
                         eval_seed: int | None = None):
        """Run one-episode eval on a single scenario for MAPPO vs baselines."""
        import torch
        from ece324_tango.sumo_rl.environment.env import SumoEnvironment

        seed = eval_seed if eval_seed is not None else cfg.seed
        results = {}
        for controller_name in _eval_controller_sequence(cfg.eval_baselines):
            if controller_name == "nema":
                eval_env = SumoEnvironment(
                    net_file=cfg.net_file,
                    route_file=route_file,
                    use_gui=False,
                    num_seconds=cfg.seconds,
                    delta_time=cfg.delta_time,
                    sumo_seed=seed,
                    single_agent=False,
                    sumo_warnings=False,
                    fixed_ts=True,
                    additional_sumo_cmd="--no-step-log true",
                )
            else:
                eval_env = create_parallel_env(
                    net_file=cfg.net_file,
                    route_file=route_file,
                    seed=seed,
                    use_gui=False,
                    seconds=cfg.seconds,
                    delta_time=cfg.delta_time,
                    quiet_sumo=True,
                )
            try:
                obs_raw = eval_env.reset(seed=seed)
                obs = extract_reset_obs(obs_raw)
                if not obs:
                    continue
                agents = sorted(obs.keys())
                a_dims = {a: int(eval_env.action_spaces(a).n) for a in agents}
                mp = MaxPressureController(action_size_by_agent=a_dims)
                ft = FixedTimeController(
                    action_size_by_agent=a_dims, green_duration_s=cfg.delta_time
                )
                kpi = KPITracker()
                done = False

                while not done:
                    active = sorted(obs.keys())
                    if controller_name == "mappo":
                        gobs = flatten_obs_by_agent(obs, active)
                        padded = [
                            pad_observation(
                                np.asarray(obs[a], dtype=np.float32), target_dim=obs_dim
                            )
                            for a in active
                        ]
                        n_valid = [a_dims[a] for a in active]
                        if cfg.residual_mode == "action_gate":
                            mp_acts = mp.actions(obs, env=eval_env)
                            mp_list = [mp_acts.get(a, 0) for a in active]
                            batch_out = trainer.act_batch_residual(
                                padded, gobs, n_valid, mp_list
                            )
                        else:
                            batch_out = trainer.act_batch(padded, gobs, n_valid)
                        actions = {
                            a: int(batch_out[i]["action"])
                            for i, a in enumerate(active)
                        }
                    elif controller_name == "max_pressure":
                        actions = mp.actions(obs, env=eval_env)
                    elif controller_name == "fixed_time":
                        actions = ft.actions(obs)
                    else:
                        actions = {}

                    try:
                        result = eval_env.step(actions)
                        if controller_name == "nema":
                            obs, _, dones, _ = result
                            done = dones.get("__all__", False) if isinstance(dones, dict) else dones
                        else:
                            obs, _, done, _ = result
                            done = done.get("__all__", False) if isinstance(done, dict) else done
                    except FatalTraCIError:
                        done = True
                    kpi.update(eval_env)

                k = kpi.summary()
                results[controller_name] = k.person_time_loss_s
            finally:
                eval_env.close()

        ratio, log_line = _format_eval_summary(scenario_name, train_ep, results)
        logger.info(log_line)
        return ratio

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
                    if cfg.residual_mode == "action_gate":
                        trainer = ResidualMAPPOTrainer(
                            obs_dim=obs_dim,
                            global_obs_dim=global_obs_dim,
                            n_actions=n_actions,
                            residual_mode="action_gate",
                            device=resolved_device,
                            use_obs_norm=cfg.use_obs_norm,
                        )
                    else:
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
                    ep_gate_total = 0
                    ep_gate_steps = 0

                    while not done:
                        active_agents = sorted(obs.keys())
                        prev_obs = {
                            a: np.asarray(obs[a], dtype=np.float32)
                            for a in active_agents
                        }
                        skip_kpi_update = False

                        # Compute MP actions BEFORE env.step (Pitfall C5)
                        mp_actions = max_pressure.actions(obs, env=env)

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
                            if cfg.residual_mode == "action_gate":
                                mp_actions_list = [
                                    mp_actions.get(a, 0) for a in active_agents
                                ]
                                batch_out = trainer.act_batch_residual(
                                    padded_obs_list, gobs, n_valid_list, mp_actions_list
                                )
                                gate_vals = [
                                    int(batch_out[i].get("gate", 0))
                                    for i in range(len(active_agents))
                                ]
                                ep_gate_total += sum(gate_vals)
                                ep_gate_steps += len(gate_vals)
                            else:
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
                            "gate_fraction": float(ep_gate_total / max(1, ep_gate_steps))
                            if cfg.residual_mode == "action_gate"
                            else 0.0,
                        }
                    )
            finally:
                env.close()

        pd.DataFrame(records).to_csv(cfg.out_csv, index=False)
        logger.success(f"Saved evaluation metrics: {cfg.out_csv}")
