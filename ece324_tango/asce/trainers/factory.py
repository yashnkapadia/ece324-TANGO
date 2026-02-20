from __future__ import annotations

from ece324_tango.asce.trainers.base import AsceTrainerBackend
from ece324_tango.asce.trainers.benchmarl_backend import BenchmarlBackend
from ece324_tango.asce.trainers.local_mappo_backend import LocalMappoBackend
from ece324_tango.asce.trainers.xuance_backend import XuanceBackend


def get_backend(name: str) -> AsceTrainerBackend:
    key = name.strip().lower()
    if key == "local_mappo":
        return LocalMappoBackend()
    if key == "benchmarl":
        return BenchmarlBackend()
    if key == "xuance":
        return XuanceBackend()
    raise ValueError(
        f"Unknown trainer backend '{name}'. Expected one of: local_mappo, benchmarl, xuance"
    )
