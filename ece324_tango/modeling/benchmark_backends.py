from __future__ import annotations

from datetime import datetime
from pathlib import Path
import time
from typing import List

import pandas as pd
from loguru import logger
import typer

from ece324_tango.asce.env import get_default_sumo_files
from ece324_tango.asce.trainers import EvalConfig, TrainConfig, get_backend
from ece324_tango.config import RESULTS_DIR

app = typer.Typer(add_completion=False)


def _parse_csv_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


@app.command()
def main(
    backends: str = "benchmarl,xuance",
    seeds: str = "7,17,27",
    episodes: int = 1,
    seconds: int = 30,
    delta_time: int = 5,
    ppo_epochs: int = 1,
    minibatch_size: int = 64,
    device: str = "cpu",
    reward_mode: str = "objective",
    reward_delay_weight: float = 1.0,
    reward_throughput_weight: float = 1.0,
    reward_fairness_weight: float = 0.25,
    scenario_id: str = "baseline",
    net_file: str = "",
    route_file: str = "",
    use_gui: bool = False,
    backend_verbose: bool = False,
    out_dir: Path = RESULTS_DIR / "backend_compare",
    run_id: str = "",
):
    """Benchmark backend train/eval speed and MAPPO reward over multiple seeds."""
    selected_backends = _parse_csv_list(backends)
    selected_seeds = [int(s) for s in _parse_csv_list(seeds)]
    if not selected_backends:
        raise typer.BadParameter("No backends provided.")
    if not selected_seeds:
        raise typer.BadParameter("No seeds provided.")

    if not net_file or not route_file:
        net_file, route_file = get_default_sumo_files()

    if not run_id:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Benchmark run dir: {run_dir}")

    rows = []
    for backend_name in selected_backends:
        backend = get_backend(backend_name)
        for seed in selected_seeds:
            model_path = run_dir / f"{backend_name}_s{seed}.pt"
            rollout_csv = run_dir / f"{backend_name}_s{seed}_rollout.csv"
            train_metrics_csv = run_dir / f"{backend_name}_s{seed}_train.csv"
            eval_csv = run_dir / f"{backend_name}_s{seed}_eval.csv"

            train_cfg = TrainConfig(
                model_path=model_path,
                rollout_csv=rollout_csv,
                episode_metrics_csv=train_metrics_csv,
                net_file=net_file,
                route_file=route_file,
                scenario_id=scenario_id,
                episodes=episodes,
                seconds=seconds,
                delta_time=delta_time,
                ppo_epochs=ppo_epochs,
                minibatch_size=minibatch_size,
                seed=seed,
                use_gui=use_gui,
                device=device,
                backend_verbose=backend_verbose,
                reward_mode=reward_mode,
                reward_delay_weight=reward_delay_weight,
                reward_throughput_weight=reward_throughput_weight,
                reward_fairness_weight=reward_fairness_weight,
            )

            t0 = time.perf_counter()
            backend.train(train_cfg)
            train_seconds = time.perf_counter() - t0

            eval_cfg = EvalConfig(
                model_path=model_path,
                out_csv=eval_csv,
                net_file=net_file,
                route_file=route_file,
                seconds=seconds,
                delta_time=delta_time,
                episodes=1,
                seed=seed,
                use_gui=use_gui,
                device=device,
                backend_verbose=backend_verbose,
                reward_mode=reward_mode,
                reward_delay_weight=reward_delay_weight,
                reward_throughput_weight=reward_throughput_weight,
                reward_fairness_weight=reward_fairness_weight,
            )

            t1 = time.perf_counter()
            backend.evaluate(eval_cfg)
            eval_seconds = time.perf_counter() - t1

            eval_df = pd.read_csv(eval_csv)
            mappo = eval_df[eval_df["controller"] == "mappo"].iloc[0]
            row = {
                "backend": backend_name,
                "seed": seed,
                "train_seconds": round(train_seconds, 3),
                "eval_seconds": round(eval_seconds, 3),
                "mappo_mean_reward": float(mappo["mean_reward"]),
                "mappo_delay_proxy": float(mappo["delay_proxy"]),
                "mappo_throughput_proxy": float(mappo["throughput_proxy"]),
                "mappo_fairness_jain": float(mappo["fairness_jain"]),
                "model_path": str(model_path),
                "eval_csv": str(eval_csv),
            }
            rows.append(row)
            logger.info(
                f"Done backend={backend_name} seed={seed} "
                f"train={row['train_seconds']:.3f}s eval={row['eval_seconds']:.3f}s "
                f"reward={row['mappo_mean_reward']:.6f}"
            )

    summary_df = pd.DataFrame(rows)
    summary_csv = run_dir / "summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    agg = (
        summary_df.groupby("backend", as_index=False)[
            [
                "train_seconds",
                "eval_seconds",
                "mappo_mean_reward",
                "mappo_delay_proxy",
                "mappo_throughput_proxy",
                "mappo_fairness_jain",
            ]
        ]
        .mean()
        .sort_values(by="mappo_mean_reward", ascending=False)
    )
    agg_csv = run_dir / "aggregate.csv"
    agg.to_csv(agg_csv, index=False)

    logger.success(f"Wrote summary: {summary_csv}")
    logger.success(f"Wrote aggregate: {agg_csv}")
    print("\nPer-seed summary")
    print(summary_df.to_string(index=False))
    print("\nAggregate means")
    print(agg.to_string(index=False))


if __name__ == "__main__":
    app()
