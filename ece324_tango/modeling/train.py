from __future__ import annotations

from pathlib import Path

from loguru import logger
import typer

from ece324_tango.asce.env import get_default_sumo_files
from ece324_tango.config import MODELS_DIR, PROCESSED_DATA_DIR, RESULTS_DIR
from ece324_tango.asce.trainers import TrainConfig, get_backend

app = typer.Typer(add_completion=False)


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
    trainer_backend: str = "local_mappo",
    backend_verbose: bool = False,
    reward_mode: str = "objective",
    reward_delay_weight: float = 1.0,
    reward_throughput_weight: float = 1.0,
    reward_fairness_weight: float = 0.25,
):
    """Train ASCE controller using selected trainer backend."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    rollout_csv.parent.mkdir(parents=True, exist_ok=True)
    episode_metrics_csv.parent.mkdir(parents=True, exist_ok=True)

    if not net_file or not route_file:
        net_file, route_file = get_default_sumo_files()

    logger.info(f"Using SUMO net: {net_file}")
    logger.info(f"Using SUMO route: {route_file}")
    logger.info(f"Trainer backend: {trainer_backend}")
    backend = get_backend(trainer_backend)
    cfg = TrainConfig(
        model_path=model_path,
        rollout_csv=rollout_csv,
        episode_metrics_csv=episode_metrics_csv,
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
    backend.train(cfg)


if __name__ == "__main__":
    app()
