from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, List, Sequence

import numpy as np

from ece324_tango.asce.env import split_ns_ew_from_obs
from ece324_tango.asce.runtime import jain_index
from ece324_tango.error_reporting import report_exception


@dataclass
class IntersectionMetrics:
    intersection_id: str
    time_step: float
    queue_ns: int
    queue_ew: int
    arrivals_ns: int
    arrivals_ew: int
    avg_speed_ns: float
    avg_speed_ew: float
    current_phase: int
    time_of_day: float
    action_phase: int
    action_green_dur: float
    delay: float
    queue_total: int
    throughput: int
    scenario_id: str

    def to_row(self) -> dict:
        return {
            "intersection_id": self.intersection_id,
            "time_step": self.time_step,
            "queue_ns": self.queue_ns,
            "queue_ew": self.queue_ew,
            "arrivals_ns": self.arrivals_ns,
            "arrivals_ew": self.arrivals_ew,
            "avg_speed_ns": self.avg_speed_ns,
            "avg_speed_ew": self.avg_speed_ew,
            "current_phase": self.current_phase,
            "time_of_day": self.time_of_day,
            "action_phase": self.action_phase,
            "action_green_dur": self.action_green_dur,
            "delay": self.delay,
            "queue_total": self.queue_total,
            "throughput": self.throughput,
            "scenario_id": self.scenario_id,
        }


@dataclass
class RewardWeights:
    delay: float
    throughput: float
    fairness: float


def _edge_axis(env, edge_id: str) -> str:
    shape = env.sumo.edge.getShape(edge_id)
    if not shape or len(shape) < 2:
        return "ns"
    (x0, y0), (x1, y1) = shape[0], shape[-1]
    dx = float(x1) - float(x0)
    dy = float(y1) - float(y0)
    return "ew" if abs(dx) > abs(dy) else "ns"


def _incoming_edges_for_ts(env, ts_id: str) -> Dict[str, List[str]]:
    links = env.sumo.trafficlight.getControlledLinks(ts_id)
    incoming_lanes = set()
    for group in links:
        for link in group:
            if not link:
                continue
            lane_in = link[0]
            if lane_in:
                incoming_lanes.add(lane_in)

    edge_ids = set()
    for lane_id in incoming_lanes:
        try:
            edge_ids.add(env.sumo.lane.getEdgeID(lane_id))
        except Exception as exc:
            report_exception(
                context="traffic_metrics.incoming_edge_lookup_failed",
                exc=exc,
                details={"ts_id": ts_id, "lane_id": lane_id},
                once_key=f"incoming_edge_lookup:{ts_id}:{lane_id}",
            )
            continue

    ns_edges: List[str] = []
    ew_edges: List[str] = []
    for edge_id in sorted(edge_ids):
        try:
            axis = _edge_axis(env, edge_id)
        except Exception as exc:
            report_exception(
                context="traffic_metrics.edge_axis_failed",
                exc=exc,
                details={"ts_id": ts_id, "edge_id": edge_id},
                once_key=f"edge_axis:{ts_id}:{edge_id}",
            )
            axis = "ns"
        if axis == "ew":
            ew_edges.append(edge_id)
        else:
            ns_edges.append(edge_id)
    return {"ns": ns_edges, "ew": ew_edges}


def _sum_edge_metric(env, edge_ids: Sequence[str], fn_name: str) -> float:
    total = 0.0
    fn = getattr(env.sumo.edge, fn_name)
    for edge_id in edge_ids:
        total += float(fn(edge_id))
    return total


def _mean_edge_speed(env, edge_ids: Sequence[str]) -> float:
    if not edge_ids:
        return 0.0
    vals = [float(env.sumo.edge.getLastStepMeanSpeed(edge_id)) for edge_id in edge_ids]
    return float(np.mean(vals)) if vals else 0.0


def compute_metrics_for_agent(
    env,
    agent_id: str,
    time_step: float,
    action_phase: int,
    action_green_dur: float,
    scenario_id: str,
    start_hour: float = 8.0,
    obs_fallback: np.ndarray | None = None,
) -> IntersectionMetrics:
    try:
        grouped = _incoming_edges_for_ts(env, agent_id)
        ns_edges = grouped["ns"]
        ew_edges = grouped["ew"]
        all_edges = ns_edges + ew_edges
        if not all_edges:
            raise RuntimeError("No approach edges found")

        queue_ns = int(round(_sum_edge_metric(env, ns_edges, "getLastStepHaltingNumber")))
        queue_ew = int(round(_sum_edge_metric(env, ew_edges, "getLastStepHaltingNumber")))
        arrivals_ns = int(round(_sum_edge_metric(env, ns_edges, "getLastStepVehicleNumber")))
        arrivals_ew = int(round(_sum_edge_metric(env, ew_edges, "getLastStepVehicleNumber")))
        avg_speed_ns = _mean_edge_speed(env, ns_edges)
        avg_speed_ew = _mean_edge_speed(env, ew_edges)
        delay = float(_sum_edge_metric(env, all_edges, "getWaitingTime"))
        queue_total = int(round(_sum_edge_metric(env, all_edges, "getLastStepHaltingNumber")))
        throughput = int(round(_sum_edge_metric(env, all_edges, "getLastStepVehicleNumber")))
        current_phase = int(env.sumo.trafficlight.getPhase(agent_id))
    except Exception as exc:
        report_exception(
            context="traffic_metrics.compute_metrics_fallback",
            exc=exc,
            details={"agent_id": agent_id, "time_step": time_step, "scenario_id": scenario_id},
            once_key=f"metrics_fallback:{scenario_id}:{agent_id}",
        )
        obs = np.zeros((1,), dtype=np.float32) if obs_fallback is None else np.asarray(obs_fallback)
        q_ns, q_ew, arr_ns, arr_ew = split_ns_ew_from_obs(obs)
        queue_ns = int(round(q_ns))
        queue_ew = int(round(q_ew))
        arrivals_ns = int(round(arr_ns))
        arrivals_ew = int(round(arr_ew))
        avg_speed_ns = -1.0
        avg_speed_ew = -1.0
        delay = float(max(0.0, queue_ns + queue_ew))
        queue_total = int(round(queue_ns + queue_ew))
        throughput = int(round(arrivals_ns + arrivals_ew))
        current_phase = -1

    time_of_day = float((start_hour * 3600.0 + float(time_step)) / 86400.0)
    return IntersectionMetrics(
        intersection_id=agent_id,
        time_step=float(time_step),
        queue_ns=queue_ns,
        queue_ew=queue_ew,
        arrivals_ns=arrivals_ns,
        arrivals_ew=arrivals_ew,
        avg_speed_ns=float(avg_speed_ns),
        avg_speed_ew=float(avg_speed_ew),
        current_phase=current_phase,
        time_of_day=time_of_day,
        action_phase=int(action_phase),
        action_green_dur=float(action_green_dur),
        delay=float(delay),
        queue_total=queue_total,
        throughput=throughput,
        scenario_id=scenario_id,
    )


def compute_metrics_for_agents(
    env,
    agent_ids: Iterable[str],
    time_step: float,
    actions: Dict[str, int],
    action_green_dur: float,
    scenario_id: str,
    observations: Dict[str, np.ndarray] | None = None,
) -> Dict[str, IntersectionMetrics]:
    metrics: Dict[str, IntersectionMetrics] = {}
    for agent_id in agent_ids:
        fallback = None
        if observations is not None and agent_id in observations:
            fallback = np.asarray(observations[agent_id], dtype=np.float32)
        metrics[agent_id] = compute_metrics_for_agent(
            env=env,
            agent_id=agent_id,
            time_step=time_step,
            action_phase=int(actions.get(agent_id, 0)),
            action_green_dur=action_green_dur,
            scenario_id=scenario_id,
            obs_fallback=fallback,
        )
    return metrics


def rewards_from_metrics(
    metrics_by_agent: Dict[str, IntersectionMetrics],
    mode: str,
    weights: RewardWeights,
) -> Dict[str, float]:
    if mode == "sumo":
        return {}

    throughputs = [float(m.throughput) for m in metrics_by_agent.values()]
    fairness = jain_index(throughputs)
    rewards: Dict[str, float] = {}
    for agent_id, m in metrics_by_agent.items():
        delay_term = math.log1p(max(0.0, float(m.delay)))
        throughput_term = math.log1p(max(0.0, float(m.throughput)))
        reward = (
            -weights.delay * delay_term
            + weights.throughput * throughput_term
            + weights.fairness * float(fairness)
        )
        rewards[agent_id] = float(reward)
    return rewards
