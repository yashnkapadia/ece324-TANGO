"""Generate per-step rollout datasets for MP and FT baselines across scenarios.

Produces Parquet files at ``data/pira/{controller}_{scenario}.parquet``
matching the ASCE_DATASET_COLUMNS schema, ready for PIRA training.

Usage::

    pixi run python scripts/generate_baseline_dataset.py
    pixi run python scripts/generate_baseline_dataset.py --seconds 300 --seeds 3
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import typer
from loguru import logger

from ece324_tango.asce.baselines import FixedTimeController, MaxPressureController
from ece324_tango.asce.env import create_parallel_env
from ece324_tango.asce.runtime import extract_reset_obs, extract_step_details
from ece324_tango.asce.schema import ASCE_DATASET_COLUMNS
from ece324_tango.asce.traffic_metrics import compute_metrics_for_agents
from ece324_tango.config import PROJ_ROOT

app = typer.Typer(add_completion=False)

NET_FILE = str(PROJ_ROOT / "sumo" / "network" / "osm.net.xml")
SCENARIO_DIR = PROJ_ROOT / "sumo" / "demand" / "curriculum"
OUT_DIR = PROJ_ROOT / "data" / "pira"

SCENARIOS = {
    "am_peak": SCENARIO_DIR / "am_peak.rou.xml",
    "pm_peak": SCENARIO_DIR / "pm_peak.rou.xml",
    "demand_surge": SCENARIO_DIR / "demand_surge.rou.xml",
    "midday_multimodal": SCENARIO_DIR / "midday_multimodal.rou.xml",
}

CONTROLLERS = {
    "max_pressure": lambda dims, dt: MaxPressureController(action_size_by_agent=dims),
    "fixed_time": lambda dims, dt: FixedTimeController(
        action_size_by_agent=dims, green_duration_s=dt
    ),
}


def _run_baseline(
    controller_name: str,
    scenario_id: str,
    route_file: str,
    seconds: int,
    delta_time: int,
    seed: int,
) -> list[dict]:
    """Run one baseline controller on one scenario for one seed, return rows."""
    env = create_parallel_env(
        net_file=NET_FILE,
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
            return []

        agents = sorted(obs.keys())
        action_dims = {a: int(env.action_spaces(a).n) for a in agents}
        controller = CONTROLLERS[controller_name](action_dims, delta_time)

        rows: list[dict] = []
        done = False
        step = 0

        while not done:
            active = sorted(obs.keys())
            if controller_name == "max_pressure":
                actions = controller.actions(obs, env=env)
            else:
                actions = controller.actions(obs)

            try:
                next_obs, _, done, _, _, _ = extract_step_details(env.step(actions))
            except Exception:
                done = True
                next_obs = {}

            step += 1
            sim_time = float(step) * float(delta_time)

            if not done:
                metrics = compute_metrics_for_agents(
                    env=env,
                    agent_ids=active,
                    time_step=sim_time,
                    actions=actions,
                    action_green_dur=float(delta_time),
                    scenario_id=scenario_id,
                    observations=obs,
                )
                for agent_id in active:
                    rows.append(metrics[agent_id].to_row())

            obs = next_obs

        return rows
    finally:
        env.close()


@app.command()
def main(
    seconds: int = typer.Option(900, help="Simulation duration per episode"),
    delta_time: int = typer.Option(5, help="Decision interval (seconds)"),
    seeds: int = typer.Option(5, help="Number of random seeds per scenario"),
    base_seed: int = typer.Option(42, help="Starting seed"),
):
    """Generate MP/FT baseline rollout datasets for PIRA training."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for scenario_id, route_path in SCENARIOS.items():
        if not route_path.exists():
            logger.warning(f"Skipping {scenario_id}: {route_path} not found")
            continue

        for ctrl_name in CONTROLLERS:
            all_rows: list[dict] = []

            for s in range(seeds):
                seed = base_seed + s
                logger.info(f"{ctrl_name} / {scenario_id} / seed={seed} " f"({s + 1}/{seeds})")
                rows = _run_baseline(
                    controller_name=ctrl_name,
                    scenario_id=scenario_id,
                    route_file=str(route_path),
                    seconds=seconds,
                    delta_time=delta_time,
                    seed=seed,
                )
                all_rows.extend(rows)
                logger.info(f"  → {len(rows)} rows")

            if not all_rows:
                logger.warning(f"No data for {ctrl_name}/{scenario_id}")
                continue

            df = pd.DataFrame(all_rows)
            df = df[[c for c in ASCE_DATASET_COLUMNS if c in df.columns]]

            out_path = OUT_DIR / f"{ctrl_name}_{scenario_id}.parquet"
            df.to_parquet(out_path, index=False)
            logger.success(f"Saved {len(df)} rows → {out_path.relative_to(PROJ_ROOT)}")

    # Also produce a single merged file for convenience
    parts = [p for p in OUT_DIR.glob("*.parquet") if p.name != "baseline_dataset.parquet"]
    if parts:
        merged = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        merged_path = OUT_DIR / "baseline_dataset.parquet"
        merged.to_parquet(merged_path, index=False)
        logger.success(f"Merged {len(merged)} total rows → {merged_path.relative_to(PROJ_ROOT)}")


if __name__ == "__main__":
    app()
