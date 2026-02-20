from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from loguru import logger
import typer

from ece324_tango.asce.env import (
    create_parallel_env,
    flatten_obs_by_agent,
    get_default_sumo_files,
    split_ns_ew_from_obs,
)
from ece324_tango.asce.mappo import MAPPOTrainer, Transition
from ece324_tango.asce.schema import ASCE_DATASET_COLUMNS
from ece324_tango.config import MODELS_DIR, PROCESSED_DATA_DIR, RESULTS_DIR

app = typer.Typer(add_completion=False)


def _current_phase(env, agent_id: str) -> int:
    try:
        ts = env.unwrapped.traffic_signals[agent_id]
        return int(getattr(ts, "green_phase", 0))
    except Exception:
        return 0


def _extract_reset_obs(reset_output):
    if isinstance(reset_output, tuple):
        return reset_output[0]
    return reset_output


def _extract_step(step_output):
    if len(step_output) == 5:
        obs, rewards, terminations, truncations, infos = step_output
        done = _safe_done(terminations, truncations)
        return obs, rewards, done, infos
    obs, rewards, dones, infos = step_output
    if "__all__" in dones:
        done = bool(dones["__all__"])
    else:
        done = all(bool(v) for v in dones.values()) if dones else False
    return obs, rewards, done, infos


def _safe_done(terminations: Dict[str, bool], truncations: Dict[str, bool]) -> bool:
    keys = set(terminations.keys()) | set(truncations.keys())
    return all(bool(terminations.get(k, False) or truncations.get(k, False)) for k in keys) if keys else False


@app.command()
def main(
    model_path: Path = MODELS_DIR / "asce_mappo.pt",
    rollout_csv: Path = PROCESSED_DATA_DIR / "asce_rollout_samples.csv",
    episode_metrics_csv: Path = RESULTS_DIR / "asce_train_episode_metrics.csv",
    net_file: str = "",
    route_file: str = "",
    scenario_id: str = "baseline",
    episodes: int = 5,
    seconds: int = 3600,
    delta_time: int = 5,
    ppo_epochs: int = 4,
    minibatch_size: int = 512,
    seed: int = 7,
    use_gui: bool = False,
    device: str = "auto",
):
    """Train minimal MAPPO controller on a sample SUMO-RL network."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    rollout_csv.parent.mkdir(parents=True, exist_ok=True)
    episode_metrics_csv.parent.mkdir(parents=True, exist_ok=True)

    if not net_file or not route_file:
        net_file, route_file = get_default_sumo_files()

    logger.info(f"Using SUMO net: {net_file}")
    logger.info(f"Using SUMO route: {route_file}")
    env = create_parallel_env(
        net_file=net_file,
        route_file=route_file,
        seed=seed,
        use_gui=use_gui,
        seconds=seconds,
        delta_time=delta_time,
    )

    obs = _extract_reset_obs(env.reset(seed=seed))
    if not obs:
        raise RuntimeError("No observations received from SUMO environment.")

    ordered_agents = sorted(obs.keys())
    first_agent = ordered_agents[0]
    obs_dim = int(np.asarray(obs[first_agent], dtype=np.float32).size)
    global_obs_dim = int(flatten_obs_by_agent(obs, ordered_agents).size)

    action_dims = {a: int(env.action_spaces(a).n) for a in ordered_agents}
    n_actions = min(action_dims.values())

    if device == "auto":
        import torch

        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved_device = device

    logger.info(f"Training device: {resolved_device}")

    trainer = MAPPOTrainer(
        obs_dim=obs_dim,
        global_obs_dim=global_obs_dim,
        n_actions=n_actions,
        device=resolved_device,
    )

    all_rows: List[dict] = []
    ep_metrics: List[dict] = []

    for ep in range(episodes):
        obs = _extract_reset_obs(env.reset(seed=seed + ep))
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

            next_obs, rewards, done, infos = _extract_step(env.step(actions))

            global_reward = float(np.mean(list(rewards.values()))) if rewards else 0.0
            ep_reward += global_reward
            ep_steps += 1

            sim_time = float(ep_steps * delta_time)
            time_of_day = float((8.0 * 3600.0 + sim_time) / 86400.0)

            for agent in active_agents:
                a_obs = np.asarray(obs[agent], dtype=np.float32)
                q_ns, q_ew, arr_ns, arr_ew = split_ns_ew_from_obs(a_obs)
                queue_total = int(round(q_ns + q_ew))
                throughput = int(round(arr_ns + arr_ew))
                delay = float(max(0.0, -float(rewards.get(agent, global_reward))))

                row = {
                    "intersection_id": agent,
                    "time_step": sim_time,
                    "queue_ns": int(round(q_ns)),
                    "queue_ew": int(round(q_ew)),
                    "arrivals_ns": int(round(arr_ns)),
                    "arrivals_ew": int(round(arr_ew)),
                    "avg_speed_ns": -1.0,
                    "avg_speed_ew": -1.0,
                    "current_phase": _current_phase(env, agent),
                    "time_of_day": time_of_day,
                    "action_phase": int(actions[agent]),
                    "action_green_dur": float(delta_time),
                    "delay": delay,
                    "queue_total": queue_total,
                    "throughput": throughput,
                    "scenario_id": scenario_id,
                }
                all_rows.append(row)

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
        losses = trainer.update(batch=batch, ppo_epochs=ppo_epochs, minibatch_size=minibatch_size)

        ep_metrics.append(
            {
                "episode": ep,
                "seed": seed + ep,
                "scenario_id": scenario_id,
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

    trainer.save(str(model_path))
    logger.success(f"Saved MAPPO model: {model_path}")

    rollout_df = pd.DataFrame(all_rows)
    if not rollout_df.empty:
        rollout_df = rollout_df[ASCE_DATASET_COLUMNS]
    rollout_df.to_csv(rollout_csv, index=False)
    pd.DataFrame(ep_metrics).to_csv(episode_metrics_csv, index=False)

    logger.success(f"Saved rollout samples: {rollout_csv}")
    logger.success(f"Saved episode metrics: {episode_metrics_csv}")


if __name__ == "__main__":
    app()
