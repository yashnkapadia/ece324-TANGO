from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from loguru import logger
import typer

from ece324_tango.asce.baselines import FixedTimeController, MaxPressureController
from ece324_tango.asce.env import create_parallel_env, flatten_obs_by_agent, get_default_sumo_files
from ece324_tango.asce.mappo import MAPPOTrainer
from ece324_tango.config import MODELS_DIR, RESULTS_DIR

app = typer.Typer(add_completion=False)


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


def _jain_index(values: List[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    den = arr.size * np.square(arr).sum()
    if den <= 0:
        return 0.0
    return float(np.square(arr.sum()) / den)


@app.command()
def main(
    model_path: Path = MODELS_DIR / "asce_mappo.pt",
    out_csv: Path = RESULTS_DIR / "asce_eval_metrics.csv",
    net_file: str = "",
    route_file: str = "",
    seconds: int = 3600,
    delta_time: int = 5,
    episodes: int = 3,
    seed: int = 17,
    use_gui: bool = False,
    device: str = "auto",
):
    """Evaluate MAPPO vs fixed-time and queue-greedy baselines."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if not net_file or not route_file:
        net_file, route_file = get_default_sumo_files()

    if device == "auto":
        import torch

        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved_device = device
    logger.info(f"Inference device: {resolved_device}")

    records = []
    controllers = ["mappo", "fixed_time", "max_pressure"]

    for controller_name in controllers:
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
        action_dims = {a: int(env.action_spaces(a).n) for a in ordered_agents}

        fixed = FixedTimeController(action_size_by_agent=action_dims, green_duration_s=delta_time)
        max_pressure = MaxPressureController(action_size_by_agent=action_dims)

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
            trainer.load(str(model_path))

        for ep in range(episodes):
            obs = _extract_reset_obs(env.reset(seed=seed + ep))
            done = False
            ep_rewards = []
            per_agent_reward_totals: Dict[str, float] = {a: 0.0 for a in sorted(obs.keys())}
            steps = 0

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

                obs, rewards, done, infos = _extract_step(env.step(actions))
                if rewards:
                    ep_rewards.append(float(np.mean(list(rewards.values()))))
                    for a, r in rewards.items():
                        per_agent_reward_totals[a] = per_agent_reward_totals.get(a, 0.0) + float(r)
                steps += 1

            avg_reward = float(np.mean(ep_rewards)) if ep_rewards else 0.0
            throughput_proxy = float(sum(max(0.0, v) for v in per_agent_reward_totals.values()))
            delay_proxy = float(-avg_reward)
            fairness = _jain_index(list(per_agent_reward_totals.values()))

            records.append(
                {
                    "controller": controller_name,
                    "episode": ep,
                    "seed": seed + ep,
                    "steps": steps,
                    "mean_reward": avg_reward,
                    "delay_proxy": delay_proxy,
                    "throughput_proxy": throughput_proxy,
                    "fairness_jain": fairness,
                }
            )

        env.close()

    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    logger.success(f"Saved evaluation metrics: {out_csv}")


if __name__ == "__main__":
    app()
