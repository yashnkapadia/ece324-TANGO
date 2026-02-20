from __future__ import annotations

import importlib.util

from loguru import logger

from ece324_tango.asce.trainers.base import AsceTrainerBackend, EvalConfig, TrainConfig
from ece324_tango.asce.trainers.local_mappo_backend import LocalMappoBackend


class BenchmarlBackend(AsceTrainerBackend):
    """BenchMARL spike backend.

    Current spike behavior:
    - verifies whether benchmarl is installed,
    - falls back to LocalMappoBackend while preserving CLI/backend wiring.

    This keeps pipeline parity while we build a native BenchMARL task adapter.
    """

    name = "benchmarl"

    def __init__(self):
        self._fallback = LocalMappoBackend()

    @staticmethod
    def _ensure_available():
        if importlib.util.find_spec("benchmarl") is None:
            raise RuntimeError(
                "BenchMARL backend selected but package 'benchmarl' is not installed. "
                "Install BenchMARL first or use trainer_backend='local_mappo'."
            )

    def train(self, cfg: TrainConfig) -> None:
        self._ensure_available()
        logger.warning("BenchMARL spike backend currently reuses local MAPPO pipeline for parity.")
        self._fallback.train(cfg)

    def evaluate(self, cfg: EvalConfig) -> None:
        self._ensure_available()
        logger.warning("BenchMARL spike backend currently reuses local MAPPO evaluation for parity.")
        self._fallback.evaluate(cfg)
