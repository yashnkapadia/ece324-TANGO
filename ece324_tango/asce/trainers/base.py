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


class AsceTrainerBackend:
    name: str = "base"

    def train(self, cfg: TrainConfig) -> None:
        raise NotImplementedError

    def evaluate(self, cfg: EvalConfig) -> None:
        raise NotImplementedError
