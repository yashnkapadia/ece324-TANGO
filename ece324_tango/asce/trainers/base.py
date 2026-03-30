from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    model_path: Path
    rollout_csv: Path
    episode_metrics_csv: Path
    net_file: str
    route_file: str
    scenario_id: str
    episodes: int
    seconds: int
    delta_time: int
    ppo_epochs: int
    minibatch_size: int
    seed: int
    use_gui: bool
    device: str
    backend_verbose: bool
    reward_mode: str
    reward_delay_weight: float
    reward_throughput_weight: float
    reward_fairness_weight: float
    reward_residual_weight: float
    use_obs_norm: bool = False
    residual_mode: str = "none"
    checkpoint_every: int = 0  # save checkpoint every N episodes (0 = end only)
    eval_every: int = 0  # run baseline eval every N episodes (0 = disabled)
    resume: bool = False  # resume from model_path if it exists


@dataclass
class EvalConfig:
    model_path: Path
    out_csv: Path
    net_file: str
    route_file: str
    seconds: int
    delta_time: int
    episodes: int
    seed: int
    use_gui: bool
    device: str
    backend_verbose: bool
    reward_mode: str
    reward_delay_weight: float
    reward_throughput_weight: float
    reward_fairness_weight: float
    reward_residual_weight: float
    use_obs_norm: bool = False
    residual_mode: str = "none"


class AsceTrainerBackend:
    name: str = "base"

    def train(self, cfg: TrainConfig) -> None:
        raise NotImplementedError

    def evaluate(self, cfg: EvalConfig) -> None:
        raise NotImplementedError
