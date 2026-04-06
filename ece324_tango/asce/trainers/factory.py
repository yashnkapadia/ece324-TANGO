from __future__ import annotations

from ece324_tango.asce.trainers.base import AsceTrainerBackend
from ece324_tango.asce.trainers.local_mappo_backend import LocalMappoBackend


def get_backend(name: str) -> AsceTrainerBackend:
    key = name.strip().lower()
    if key == "local_mappo":
        return LocalMappoBackend()
    raise ValueError(f"Unknown trainer backend '{name}'. Expected one of: local_mappo")
