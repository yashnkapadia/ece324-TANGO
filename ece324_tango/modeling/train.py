from __future__ import annotations

from pathlib import Path

from loguru import logger
import typer

from ece324_tango.asce.env import get_default_sumo_files
from ece324_tango.asce.trainers import TrainConfig
from ece324_tango.asce.trainers.local_mappo_backend import LocalMappoBackend
from ece324_tango.config import MODELS_DIR, PROCESSED_DATA_DIR, RESULTS_DIR

app = typer.Typer(add_completion=False)
_VALID_REWARD_MODES = {"objective", "person_objective", "sumo", "time_loss", "residual_mp"}
_VALID_RESIDUAL_MODES = {"none", "action_gate"}


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
        help="Reward mode: objective | person_objective | sumo | time_loss | residual_mp",
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
    residual_mode: str = typer.Option(
        "none",
        help="Residual mode: none | action_gate",
    ),
    checkpoint_every: int = typer.Option(
        0,
        help="Save checkpoint every N episodes (0 = end only)",
    ),
    eval_every: int = typer.Option(
        0,
        help="Run baseline eval every N episodes during training (0 = disabled)",
    ),
    resume: bool = typer.Option(
        False,
        "--resume/--no-resume",
        help="Resume training from existing model_path checkpoint",
    ),
    num_workers: int = typer.Option(
        1,
        "--num-workers",
        help="Number of parallel SUMO workers for episode collection (1=sequential)",
    ),
    scale_lr_by_workers: bool = typer.Option(
        True,
        "--scale-lr/--no-scale-lr",
        help="Scale learning rate by 1/sqrt(num_workers) for batched training stability",
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
    residual_mode = str(residual_mode).strip().lower()
    if residual_mode not in _VALID_RESIDUAL_MODES:
        raise typer.BadParameter(
            f"Unsupported residual mode: {residual_mode!r}. "
            f"Expected one of: {', '.join(sorted(_VALID_RESIDUAL_MODES))}."
        )

    if num_workers < 1:
        raise typer.BadParameter(
            f"--num-workers must be >= 1, got {num_workers}."
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
        residual_mode=residual_mode,
        checkpoint_every=checkpoint_every,
        eval_every=eval_every,
        resume=resume,
        num_workers=num_workers,
        scale_lr_by_workers=scale_lr_by_workers,
    )
    backend.train(cfg)


if __name__ == "__main__":
    app()
