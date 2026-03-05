from __future__ import annotations

from pathlib import Path

import typer

from ece324_tango.asce.env import get_default_sumo_files
from ece324_tango.asce.trainers import EvalConfig
from ece324_tango.asce.trainers.local_mappo_backend import LocalMappoBackend
from ece324_tango.config import MODELS_DIR, RESULTS_DIR

app = typer.Typer(add_completion=False)
_VALID_REWARD_MODES = {"objective", "sumo", "time_loss"}


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
    backend_verbose: bool = False,
    reward_mode: str = typer.Option(
        "objective",
        help="Reward mode: objective | sumo | time_loss",
    ),
    reward_delay_weight: float = 1.0,
    reward_throughput_weight: float = 1.0,
    reward_fairness_weight: float = 0.25,
    use_obs_norm: bool = typer.Option(
        True,
        "--use-obs-norm/--no-use-obs-norm",
        help="Enable running observation normalization (Welford per-feature)",
    ),
):
    """Evaluate the local ASCE MAPPO controller vs baselines."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if not net_file or not route_file:
        net_file, route_file = get_default_sumo_files()
    reward_mode = str(reward_mode).strip().lower()
    if reward_mode not in _VALID_REWARD_MODES:
        raise typer.BadParameter(
            f"Unsupported reward mode: {reward_mode!r}. "
            f"Expected one of: {', '.join(sorted(_VALID_REWARD_MODES))}."
        )

    backend = LocalMappoBackend()
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
        use_obs_norm=use_obs_norm,
    )
    backend.evaluate(cfg)


if __name__ == "__main__":
    app()
