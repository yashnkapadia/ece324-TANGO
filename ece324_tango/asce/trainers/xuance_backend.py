from __future__ import annotations

import importlib.util

from loguru import logger

from ece324_tango.asce.trainers.base import AsceTrainerBackend, EvalConfig, TrainConfig
from ece324_tango.asce.trainers.local_mappo_backend import LocalMappoBackend


class XuanceBackend(AsceTrainerBackend):
    """Xuance spike backend.

    Current spike behavior:
    - verifies whether xuance is installed,
    - falls back to LocalMappoBackend while preserving CLI/backend wiring.

    This keeps pipeline parity while we build a native Xuance env adapter.
    """

    name = "xuance"

    def __init__(self):
        self._fallback = LocalMappoBackend()

    @staticmethod
    def _ensure_available():
        if importlib.util.find_spec("xuance") is None:
            raise RuntimeError(
                "Xuance backend selected but package 'xuance' is not installed. "
                "Install Xuance first or use trainer_backend='local_mappo'."
            )

    def train(self, cfg: TrainConfig) -> None:
        self._ensure_available()
        logger.warning("Xuance spike backend currently reuses local MAPPO pipeline for parity.")
        self._fallback.train(cfg)

    def evaluate(self, cfg: EvalConfig) -> None:
        self._ensure_available()
        logger.warning("Xuance spike backend currently reuses local MAPPO evaluation for parity.")
        self._fallback.evaluate(cfg)
