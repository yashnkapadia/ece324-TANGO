from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class FixedTimeController:
    """Cycle phases with equal green splits per phase, advancing on a timer.

    Each phase gets ``phase_duration_s`` seconds of green before the controller
    advances to the next phase.  With ``delta_time`` (the RL step interval),
    the controller holds each phase for ``phase_duration_s // delta_time`` steps
    (minimum 1).  This represents a reasonable default fixed-time plan without
    tuning to specific demand data — the kind of plan an engineer would deploy
    before counts are available.

    For the truest real-world FT baseline on the Toronto corridor, use
    ``fixed_ts=True`` in sumo-rl instead (runs the native NEMA program).
    """

    action_size_by_agent: Dict[str, int]
    green_duration_s: int  # RL step interval (delta_time), used for timing
    phase_duration_s: int = 30  # green time per phase before cycling
    cursor: Dict[str, int] = field(default_factory=dict)
    _step_counter: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        if not self.cursor:
            self.cursor = {k: 0 for k in self.action_size_by_agent}
        if not self._step_counter:
            self._step_counter = {k: 0 for k in self.action_size_by_agent}
        self._steps_per_phase = max(1, self.phase_duration_s // max(1, self.green_duration_s))

    def reset(self) -> None:
        self.cursor = {k: 0 for k in self.action_size_by_agent}
        self._step_counter = {k: 0 for k in self.action_size_by_agent}

    def actions(self, observations: Dict[str, np.ndarray]) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for agent, n_actions in self.action_size_by_agent.items():
            out[agent] = self.cursor[agent] % n_actions
            self._step_counter[agent] += 1
            if self._step_counter[agent] >= self._steps_per_phase:
                self._step_counter[agent] = 0
                self.cursor[agent] = (self.cursor[agent] + 1) % n_actions
        return out


@dataclass
class QueueGreedyController:
    """Proxy max-pressure baseline using local queue-like observation features.

    This is a practical MVP placeholder when exact upstream/downstream edge
    pressure computation is unavailable from the environment wrapper.
    """

    action_size_by_agent: Dict[str, int]

    def actions(self, observations: Dict[str, np.ndarray]) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for agent, obs in observations.items():
            n_actions = self.action_size_by_agent[agent]
            vec = np.asarray(obs, dtype=np.float32).ravel()
            if vec.size == 0:
                out[agent] = 0
                continue
            groups: List[np.ndarray] = np.array_split(vec, n_actions)
            scores = [float(np.clip(g.sum(), 0.0, None)) for g in groups]
            out[agent] = int(np.argmax(scores))
        return out


@dataclass
class MaxPressureController:
    """Edge-level max-pressure controller using TLS links and phase states.

    Pressure(action) = sum_{green movements} (queue_in - queue_out)
    where queues come from lane halting numbers.
    """

    action_size_by_agent: Dict[str, int]

    def _phase_pressure(self, env, ts_id: str, phase_idx: int) -> float:
        ts = env.traffic_signals[ts_id]
        phase_state = ts.green_phases[phase_idx].state
        controlled_links = env.sumo.trafficlight.getControlledLinks(ts_id)

        pressure = 0.0
        for i, signal_state in enumerate(phase_state):
            if signal_state not in ("g", "G"):
                continue
            if i >= len(controlled_links):
                continue
            group = controlled_links[i]
            if not group:
                continue
            # Each group can hold alternative links for the same signal index.
            for link in group:
                incoming_lane = link[0]
                outgoing_lane = link[1]
                q_in = float(env.sumo.lane.getLastStepHaltingNumber(incoming_lane))
                q_out = float(env.sumo.lane.getLastStepHaltingNumber(outgoing_lane))
                pressure += q_in - q_out
        return pressure

    def actions(self, observations: Dict[str, np.ndarray], env) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for agent in observations.keys():
            n_actions = self.action_size_by_agent[agent]
            best_action: Optional[int] = None
            best_score = -float("inf")
            for a in range(n_actions):
                score = self._phase_pressure(env=env, ts_id=agent, phase_idx=a)
                if score > best_score:
                    best_score = score
                    best_action = a
            out[agent] = int(best_action if best_action is not None else 0)
        return out
