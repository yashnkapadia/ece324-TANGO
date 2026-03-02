from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np

DEFAULT_SAMPLE_ROOT = "RESCO/grid4x4"


def get_default_sumo_files() -> Tuple[str, str]:
    """Return sample SUMO net/route files from sumo-rl package data."""
    import sumo_rl

    root = Path(sumo_rl.__file__).resolve().parent / "nets" / DEFAULT_SAMPLE_ROOT
    net_file = root / "grid4x4.net.xml"
    route_file = root / "grid4x4_1.rou.xml"
    if not net_file.exists() or not route_file.exists():
        raise FileNotFoundError(
            f"Could not find sample SUMO files under {root}. "
            "Install sumo-rl package data or pass --net-file/--route-file explicitly."
        )
    return str(net_file), str(route_file)


def create_parallel_env(
    net_file: str,
    route_file: str,
    seed: int,
    use_gui: bool,
    seconds: int,
    delta_time: int,
    quiet_sumo: bool = False,
):
    """Create a sumo-rl multi-agent environment."""
    from sumo_rl.environment.env import SumoEnvironment

    env = SumoEnvironment(
        net_file=net_file,
        route_file=route_file,
        use_gui=use_gui,
        num_seconds=seconds,
        delta_time=delta_time,
        sumo_seed=seed,
        single_agent=False,
        sumo_warnings=not quiet_sumo,
        additional_sumo_cmd="--no-step-log true" if quiet_sumo else None,
    )
    return env


def flatten_obs_by_agent(obs: Dict[str, np.ndarray], ordered_agents: Iterable[str]) -> np.ndarray:
    parts = [np.asarray(obs[a], dtype=np.float32).ravel() for a in ordered_agents]
    return np.concatenate(parts, axis=0)


def pad_observation(obs: np.ndarray, target_dim: int) -> np.ndarray:
    vec = np.asarray(obs, dtype=np.float32).ravel()
    if vec.size > target_dim:
        raise ValueError(
            f"Observation length {vec.size} exceeds target_dim={target_dim}. "
            "Increase target_dim or inspect observation schema drift."
        )
    if vec.size == target_dim:
        return vec
    out = np.zeros((target_dim,), dtype=np.float32)
    out[: vec.size] = vec
    return out


def split_ns_ew_from_obs(obs: np.ndarray) -> Tuple[float, float, float, float]:
    """Best-effort splitter for queue/arrival proxies.

    sumo-rl default observations are often:
    [phase_one_hot, min_green, lane_densities..., lane_queues...]
    We keep this robust and fallback-safe for prototype logging.
    """
    vec = np.asarray(obs, dtype=np.float32).ravel()
    if vec.size < 8:
        total = float(np.clip(vec.sum(), 0.0, None))
        return total / 2.0, total / 2.0, total / 2.0, total / 2.0

    half = vec.size // 2
    first = vec[:half]
    second = vec[half:]
    queue_ns = float(np.clip(first.sum(), 0.0, None))
    queue_ew = float(np.clip(second.sum(), 0.0, None))
    arrivals_ns = float(np.clip(first.mean() * max(1, first.size), 0.0, None))
    arrivals_ew = float(np.clip(second.mean() * max(1, second.size), 0.0, None))
    return queue_ns, queue_ew, arrivals_ns, arrivals_ew
