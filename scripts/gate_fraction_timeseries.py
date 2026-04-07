"""Roll out the trained ASCE policy and record per-step gate-override fraction.

Produces:
  - reports/results/gate_timeseries_demand_surge.csv  (per-step fraction)
  - reports/final/asce_gate_timeseries.pdf            (figure for the report)

Used to support the "override rate is highest during demand transitions" claim
in the report's gate-behavior paragraph.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("LIBSUMO_AS_TRACI", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from loguru import logger

from ece324_tango.asce.baselines import MaxPressureController
from ece324_tango.asce.env import (
    create_parallel_env,
    flatten_obs_by_agent,
    pad_observation,
)
from ece324_tango.asce.mappo import ResidualMAPPOTrainer
from ece324_tango.asce.runtime import extract_reset_obs, extract_step_details
from ece324_tango.config import PROJ_ROOT

NET_FILE = str(PROJ_ROOT / "sumo" / "network" / "osm.net.xml.gz")
SCEN_DIR = PROJ_ROOT / "sumo" / "demand" / "curriculum"
DEFAULT_MODEL = PROJ_ROOT / "models" / "asce_mappo_curriculum_best.pt"
RESULTS_DIR = PROJ_ROOT / "reports" / "results"
FIG_DIR = PROJ_ROOT / "reports" / "final"

# Demand Surge spec from generate_curriculum.py: 2x eastbound surge from
# second 300 to second 600 of simulation, on top of PM Peak base demand.
SURGE_START_S = 300
SURGE_END_S = 600


def load_trainer(model_path: Path) -> tuple[ResidualMAPPOTrainer, int]:
    state = torch.load(model_path, map_location="cpu", weights_only=False)
    n_actions = int(state["actor"]["phase_head.weight"].shape[0])
    global_obs_dim = int(state["critic"]["net.0.weight"].shape[1])
    actor_in = int(state["actor"]["body.0.weight"].shape[1])
    obs_dim = actor_in - n_actions  # action_gate concatenates MP one-hot
    use_obs_norm = bool(state.get("use_obs_norm", True))

    trainer = ResidualMAPPOTrainer(
        obs_dim=obs_dim,
        global_obs_dim=global_obs_dim,
        n_actions=n_actions,
        residual_mode="action_gate",
        device="cpu",
        use_obs_norm=use_obs_norm,
    )
    trainer.actor.load_state_dict(state["actor"])
    trainer.critic.load_state_dict(state["critic"])
    if trainer.obs_norm is not None and state.get("obs_norm") is not None:
        trainer.obs_norm.load_state_dict(state["obs_norm"])
    if trainer.gobs_norm is not None and state.get("gobs_norm") is not None:
        trainer.gobs_norm.load_state_dict(state["gobs_norm"])
    trainer.actor.eval()
    trainer.critic.eval()
    return trainer, obs_dim


def rollout_gate_series(
    trainer: ResidualMAPPOTrainer,
    obs_dim: int,
    scenario: str,
    seed: int,
    seconds: int,
    delta_time: int,
) -> pd.DataFrame:
    route = SCEN_DIR / f"{scenario}.rou.xml"
    env = create_parallel_env(
        net_file=NET_FILE,
        route_file=str(route),
        seed=seed,
        use_gui=False,
        seconds=seconds,
        delta_time=delta_time,
        quiet_sumo=True,
    )
    try:
        obs = extract_reset_obs(env.reset(seed=seed))
        if not obs:
            return pd.DataFrame()
        action_dims = {a: int(env.action_spaces(a).n) for a in sorted(obs.keys())}
        mp = MaxPressureController(action_size_by_agent=action_dims)

        rows: list[dict] = []
        done = False
        step = 0
        while not done:
            active = sorted(obs.keys())
            padded = [
                pad_observation(np.asarray(obs[a], dtype=np.float32), target_dim=obs_dim)
                for a in active
            ]
            gobs = flatten_obs_by_agent(obs, active)
            n_valid = [action_dims[a] for a in active]
            mp_acts = mp.actions(obs, env=env)
            mp_list = [mp_acts.get(a, 0) for a in active]
            batch_out = trainer.act_batch_residual(padded, gobs, n_valid, mp_list)

            n_override = sum(int(b["gate"] == 1) for b in batch_out)
            n_total = len(batch_out)
            actions = {a: int(batch_out[i]["action"]) for i, a in enumerate(active)}

            try:
                next_obs, _, done, _, _, _ = extract_step_details(env.step(actions))
            except Exception:
                done = True
                next_obs = {}

            step += 1
            sim_time = float(step) * float(delta_time)
            rows.append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "sim_time_s": sim_time,
                    "n_agents": n_total,
                    "n_override": n_override,
                    "gate_fraction": n_override / n_total if n_total else float("nan"),
                }
            )
            obs = next_obs
        return pd.DataFrame(rows)
    finally:
        env.close()


def smooth(series: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(series) <= window:
        return series
    pad = window // 2
    padded = np.pad(series, (pad, pad), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")[: len(series)]


def main(
    scenarios: tuple[str, ...] = ("am_peak", "demand_surge"),
    seed: int = 1000,
    seconds: int = 1200,
) -> None:
    trainer, obs_dim = load_trainer(DEFAULT_MODEL)
    logger.info(f"Loaded {DEFAULT_MODEL.name}: obs_dim={obs_dim}")

    all_rows: list[pd.DataFrame] = []
    for scenario in scenarios:
        logger.info(f"Rolling out {scenario} (seed={seed}, seconds={seconds})")
        df = rollout_gate_series(trainer, obs_dim, scenario, seed, seconds, delta_time=5)
        if df.empty:
            logger.warning(f"No data for {scenario}")
            continue
        all_rows.append(df)
        logger.info(
            f"  {scenario}: {len(df)} steps, mean gate frac = {df['gate_fraction'].mean():.3f}"
        )

    if not all_rows:
        return

    combined = pd.concat(all_rows, ignore_index=True)
    out_csv = RESULTS_DIR / "gate_timeseries.csv"
    combined.to_csv(out_csv, index=False)
    logger.success(f"Saved -> {out_csv.relative_to(PROJ_ROOT)}")

    # Plot.
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    colors = {"demand_surge": "#d62728", "am_peak": "#1f77b4"}
    labels = {"demand_surge": "Demand Surge", "am_peak": "AM Peak (control)"}
    for scenario in scenarios:
        sub = combined[combined["scenario"] == scenario]
        if sub.empty:
            continue
        t = sub["sim_time_s"].to_numpy()
        g = sub["gate_fraction"].to_numpy()
        ax.plot(t, smooth(g, 5), color=colors.get(scenario, "k"), lw=1.4,
                label=labels.get(scenario, scenario))

    if "demand_surge" in scenarios:
        ax.axvspan(SURGE_START_S, SURGE_END_S, color="#d62728", alpha=0.10,
                   label="Eastbound surge (300–600 s)")

    ax.set_xlabel("Simulation time (s)")
    ax.set_ylabel("Override fraction")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    fig.tight_layout()
    fig_path = FIG_DIR / "asce_gate_timeseries.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    logger.success(f"Saved -> {fig_path.relative_to(PROJ_ROOT)}")


if __name__ == "__main__":
    main()
