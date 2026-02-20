from __future__ import annotations

from typing import Dict, List

import numpy as np
from ece324_tango.error_reporting import report_exception


def extract_reset_obs(reset_output):
    if isinstance(reset_output, tuple):
        return reset_output[0]
    return reset_output


def safe_done(terminations: Dict[str, bool], truncations: Dict[str, bool]) -> bool:
    keys = set(terminations.keys()) | set(truncations.keys())
    if not keys:
        return False
    return all(bool(terminations.get(k, False) or truncations.get(k, False)) for k in keys)


def extract_step(step_output):
    if len(step_output) == 5:
        obs, rewards, terminations, truncations, infos = step_output
        done = safe_done(terminations, truncations)
        return obs, rewards, done, infos
    obs, rewards, dones, infos = step_output
    if "__all__" in dones:
        done = bool(dones["__all__"])
    else:
        done = all(bool(v) for v in dones.values()) if dones else False
    return obs, rewards, done, infos


def current_phase(env, agent_id: str) -> int:
    try:
        ts = env.unwrapped.traffic_signals[agent_id]
        return int(getattr(ts, "green_phase", 0))
    except Exception as exc:
        report_exception(
            context="runtime.current_phase_fallback",
            exc=exc,
            details={"agent_id": agent_id},
            once_key=f"current_phase:{agent_id}",
        )
        return 0


def jain_index(values: List[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    den = arr.size * np.square(arr).sum()
    if den <= 0:
        return 0.0
    return float(np.square(arr.sum()) / den)
