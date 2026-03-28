from __future__ import annotations

from pathlib import Path

from loguru import logger
import typer

from ece324_tango.asce.env import get_default_sumo_files
from ece324_tango.asce.trainers import TrainConfig
from ece324_tango.asce.trainers.local_mappo_backend import LocalMappoBackend
from ece324_tango.config import MODELS_DIR, PROCESSED_DATA_DIR, RESULTS_DIR

app = typer.Typer(add_completion=False)
_VALID_REWARD_MODES = {"objective", "sumo", "time_loss", "residual_mp"}


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
    backend_verbose: bool = False,
    reward_mode: str = typer.Option(
        "objective",
        help="Reward mode: objective | sumo | time_loss | residual_mp",
    ),
    reward_delay_weight: float = 1.0,
    reward_throughput_weight: float = 1.0,
    reward_fairness_weight: float = 0.25,
    reward_residual_weight: float = 0.25,
    use_obs_norm: bool = typer.Option(
        True,
        "--use-obs-norm/--no-use-obs-norm",
        help="Enable running observation normalization (Welford per-feature)",
    ),
):
    """Train the local ASCE MAPPO controller."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    rollout_csv.parent.mkdir(parents=True, exist_ok=True)
    episode_metrics_csv.parent.mkdir(parents=True, exist_ok=True)

    if not net_file or not route_file:
        net_file, route_file = get_default_sumo_files()
    reward_mode = str(reward_mode).strip().lower()
    if reward_mode not in _VALID_REWARD_MODES:
        raise typer.BadParameter(
            f"Unsupported reward mode: {reward_mode!r}. "
            f"Expected one of: {', '.join(sorted(_VALID_REWARD_MODES))}."
        )

    logger.info(f"Using SUMO net: {net_file}")
    logger.info(f"Using SUMO route: {route_file}")
    backend = LocalMappoBackend()
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
        reward_residual_weight=reward_residual_weight,
        use_obs_norm=use_obs_norm,
    )
    backend.train(cfg)


if __name__ == "__main__":
    app()
