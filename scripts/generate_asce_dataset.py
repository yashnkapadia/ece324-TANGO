"""Generate per-step rollout dataset from a trained ASCE checkpoint.

Companion to ``generate_baseline_dataset.py``: rolls the trained MAPPO
policy out across the four curriculum scenarios and writes one Parquet
file per scenario in ``data/pira/`` matching ``ASCE_DATASET_COLUMNS``.

This script exists so PIRA training has an "agent" rollout dataset to
sit alongside the Max-Pressure / Fixed-Time baseline datasets. PIRA
itself is deferred (see ``PIRA.md`` and final-report Section 4.6), so
this generator is intentionally a thin smoke-test rather than a full
production pipeline: by default it runs the headline checkpoint with a
single seed and a short horizon, just enough to confirm the file lands
on disk and matches the expected schema.

Usage::

    pixi run generate-asce-dataset
    pixi run generate-asce-dataset --seeds 5 --seconds 900
    pixi run python scripts/generate_asce_dataset.py --model-path models/asce_mappo_curriculum.pt
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root is importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# libsumo (faster than TraCI) is what the trainer uses; mirror that here.
os.environ.setdefault("LIBSUMO_AS_TRACI", "1")

import numpy as np
import pandas as pd
import torch
import typer
from loguru import logger

from ece324_tango.asce.baselines import MaxPressureController
from ece324_tango.asce.env import (
    create_parallel_env,
    flatten_obs_by_agent,
    pad_observation,
)
from ece324_tango.asce.mappo import ResidualMAPPOTrainer
from ece324_tango.asce.runtime import extract_reset_obs, extract_step_details
from ece324_tango.asce.schema import ASCE_DATASET_COLUMNS
from ece324_tango.asce.traffic_metrics import compute_metrics_for_agents
from ece324_tango.config import PROJ_ROOT

app = typer.Typer(add_completion=False)

NET_FILE = str(PROJ_ROOT / "sumo" / "network" / "osm.net.xml.gz")
SCENARIO_DIR = PROJ_ROOT / "sumo" / "demand" / "curriculum"
DEFAULT_MODEL = PROJ_ROOT / "models" / "asce_mappo_curriculum_best.pt"
OUT_DIR = PROJ_ROOT / "data" / "pira"

SCENARIOS = {
    "am_peak": SCENARIO_DIR / "am_peak.rou.xml",
    "pm_peak": SCENARIO_DIR / "pm_peak.rou.xml",
    "demand_surge": SCENARIO_DIR / "demand_surge.rou.xml",
    "midday_multimodal": SCENARIO_DIR / "midday_multimodal.rou.xml",
}


def _load_trainer(model_path: Path) -> tuple[ResidualMAPPOTrainer, int, bool]:
    """Load a checkpoint into a fresh trainer in eval mode.

    Returns ``(trainer, obs_dim, use_action_gate)``. The curriculum
    checkpoints stored on disk are bare ``model_state_dict``-style
    payloads with keys ``actor``, ``critic``, ``obs_norm``,
    ``gobs_norm``, ``use_obs_norm``, ``residual_mode`` — no separate
    metadata wrapper — so the network dims are recovered from the
    actor / critic weight shapes.
    """
    state = torch.load(model_path, map_location="cpu", weights_only=False)

    # Recover dims from weight shapes. The actor MLP's first Linear is
    # ``net.0`` (input -> hidden); the phase head's output dim is the
    # padded action space.
    n_actions = int(state["actor"]["phase_head.weight"].shape[0])
    global_obs_dim = int(state["critic"]["net.0.weight"].shape[1])
    residual_mode = state.get("residual_mode", "action_gate")
    use_obs_norm = bool(state.get("use_obs_norm", True))

    # In action_gate mode the GatedActor input is (obs_dim + n_actions),
    # because the trainer concatenates the Max-Pressure one-hot to each
    # observation internally. Recover the un-augmented obs_dim so the
    # observation padding below matches what the trainer expects.
    actor_in = int(state["actor"]["body.0.weight"].shape[1])
    obs_dim = actor_in - n_actions if residual_mode == "action_gate" else actor_in

    trainer = ResidualMAPPOTrainer(
        obs_dim=obs_dim,
        global_obs_dim=global_obs_dim,
        n_actions=n_actions,
        residual_mode=residual_mode,
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
    return trainer, obs_dim, residual_mode == "action_gate"


def _run_asce(
    trainer: ResidualMAPPOTrainer,
    obs_dim: int,
    use_action_gate: bool,
    scenario_id: str,
    route_file: str,
    seconds: int,
    delta_time: int,
    seed: int,
) -> list[dict]:
    """Run the trained policy on one scenario for one seed."""
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

            if use_action_gate:
                mp_acts = mp.actions(obs, env=env)
                mp_list = [mp_acts.get(a, 0) for a in active]
                batch_out = trainer.act_batch_residual(padded, gobs, n_valid, mp_list)
            else:
                batch_out = trainer.act_batch(padded, gobs, n_valid)

            actions = {a: int(batch_out[i]["action"]) for i, a in enumerate(active)}

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
    model_path: Path = typer.Option(
        DEFAULT_MODEL, help="Path to the trained ASCE checkpoint to roll out."
    ),
    seconds: int = typer.Option(900, help="Simulation duration per episode (s)"),
    delta_time: int = typer.Option(5, help="Decision interval (s)"),
    seeds: int = typer.Option(5, help="Number of random seeds per scenario"),
    base_seed: int = typer.Option(42, help="Starting seed"),
    scenario: str = typer.Option(
        "all", help="Single scenario id, or 'all' for the full curriculum"
    ),
):
    """Generate ASCE rollout dataset for PIRA training.

    Defaults match ``generate_baseline_dataset.py`` (5 seeds × 900 s ×
    4 scenarios) so the ASCE rollouts can be paired one-to-one with the
    Max-Pressure / Fixed-Time baseline rollouts. PIRA training is
    deferred (final report Section 4.6); this script is the dataset
    side of that pipeline. Validate end-to-end with a thin smoke run::

        pixi run generate-asce-dataset --seeds 1 --seconds 60 --scenario am_peak
    """
    if not model_path.exists():
        raise typer.BadParameter(f"Checkpoint not found: {model_path}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer, obs_dim, use_action_gate = _load_trainer(model_path)
    logger.info(
        f"Loaded {model_path.name}: obs_dim={obs_dim}, "
        f"residual_mode={'action_gate' if use_action_gate else 'none'}"
    )

    scenarios = SCENARIOS if scenario == "all" else {scenario: SCENARIOS[scenario]}

    for scenario_id, route_path in scenarios.items():
        if not route_path.exists():
            logger.warning(f"Skipping {scenario_id}: {route_path} not found")
            continue

        all_rows: list[dict] = []
        for s in range(seeds):
            seed = base_seed + s
            logger.info(f"asce / {scenario_id} / seed={seed} ({s + 1}/{seeds})")
            rows = _run_asce(
                trainer=trainer,
                obs_dim=obs_dim,
                use_action_gate=use_action_gate,
                scenario_id=scenario_id,
                route_file=str(route_path),
                seconds=seconds,
                delta_time=delta_time,
                seed=seed,
            )
            all_rows.extend(rows)
            logger.info(f"  -> {len(rows)} rows")

        if not all_rows:
            logger.warning(f"No data generated for {scenario_id}")
            continue

        df = pd.DataFrame(all_rows)
        df = df[[c for c in ASCE_DATASET_COLUMNS if c in df.columns]]

        # Smoke-check: every required column must be present after the
        # projection above. Fail loudly if the schema drifts.
        missing = [c for c in ASCE_DATASET_COLUMNS if c not in df.columns]
        if missing:
            raise RuntimeError(
                f"ASCE rollout for {scenario_id} is missing schema columns: {missing}"
            )

        out_path = OUT_DIR / f"asce_{scenario_id}.parquet"
        df.to_parquet(out_path, index=False)
        logger.success(f"Saved {len(df)} rows -> {out_path.relative_to(PROJ_ROOT)}")


if __name__ == "__main__":
    app()
