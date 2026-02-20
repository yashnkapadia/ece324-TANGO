from __future__ import annotations

from pathlib import Path

from loguru import logger
import typer

from ece324_tango.asce.env import get_default_sumo_files
from ece324_tango.asce.trainers import EvalConfig, get_backend
from ece324_tango.config import MODELS_DIR, RESULTS_DIR

app = typer.Typer(add_completion=False)


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
    trainer_backend: str = "local_mappo",
    backend_verbose: bool = False,
    reward_mode: str = "objective",
    reward_delay_weight: float = 1.0,
    reward_throughput_weight: float = 1.0,
    reward_fairness_weight: float = 0.25,
):
    """Evaluate ASCE controller vs baselines using selected backend."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if not net_file or not route_file:
        net_file, route_file = get_default_sumo_files()

    logger.info(f"Trainer backend: {trainer_backend}")
    backend = get_backend(trainer_backend)
    cfg = EvalConfig(
        model_path=model_path,
        out_csv=out_csv,
        net_file=net_file,
        route_file=route_file,
        seconds=seconds,
        delta_time=delta_time,
        episodes=episodes,
        seed=seed,
        use_gui=use_gui,
        device=device,
        backend_verbose=backend_verbose,
        reward_mode=reward_mode,
        reward_delay_weight=reward_delay_weight,
        reward_throughput_weight=reward_throughput_weight,
        reward_fairness_weight=reward_fairness_weight,
    )
    backend.evaluate(cfg)


if __name__ == "__main__":
    app()
