import pytest

from ece324_tango.asce.traffic_metrics import (
    IntersectionMetrics,
    RewardWeights,
    _incoming_edges_for_ts,
    compute_metrics_for_agent,
    rewards_from_metrics,
)


def _metric(agent_id: str, delay: float, throughput: int) -> IntersectionMetrics:
    return IntersectionMetrics(
        intersection_id=agent_id,
        time_step=5.0,
        queue_ns=1,
        queue_ew=1,
        arrivals_ns=throughput,
        arrivals_ew=0,
        avg_speed_ns=5.0,
        avg_speed_ew=5.0,
        current_phase=0,
        time_of_day=0.0,
        action_phase=0,
        action_green_dur=5.0,
        delay=delay,
        queue_total=2,
        throughput=throughput,
        scenario_id="baseline",
    )


def test_rewards_from_metrics_objective_prefers_higher_throughput_lower_delay():
    metrics = {
        "a": _metric("a", delay=1.0, throughput=8),
        "b": _metric("b", delay=10.0, throughput=2),
    }
    rewards = rewards_from_metrics(
        metrics_by_agent=metrics,
        mode="objective",
        weights=RewardWeights(delay=1.0, throughput=1.0, fairness=0.0),
    )
    assert rewards["a"] > rewards["b"]


def test_rewards_from_metrics_sumo_mode_returns_empty():
    metrics = {"a": _metric("a", delay=1.0, throughput=8)}
    rewards = rewards_from_metrics(
        metrics_by_agent=metrics,
        mode="sumo",
        weights=RewardWeights(delay=1.0, throughput=1.0, fairness=0.25),
    )
    assert rewards == {}


def test_rewards_from_metrics_time_loss_mode_tracks_delay_only():
    metrics = {
        "a": _metric("a", delay=1.0, throughput=1),
        "b": _metric("b", delay=10.0, throughput=100),
    }
    rewards = rewards_from_metrics(
        metrics_by_agent=metrics,
        mode="time_loss",
        weights=RewardWeights(delay=1.0, throughput=999.0, fairness=999.0),
    )
    assert rewards["a"] == pytest.approx(-0.69314718056)
    assert rewards["b"] == pytest.approx(-2.39789527280)
    assert rewards["a"] > rewards["b"]


def test_rewards_from_metrics_unknown_mode_raises():
    metrics = {"a": _metric("a", delay=1.0, throughput=8)}
    with pytest.raises(ValueError, match="Unsupported reward mode"):
        rewards_from_metrics(
            metrics_by_agent=metrics,
            mode="not_a_mode",
            weights=RewardWeights(delay=1.0, throughput=1.0, fairness=0.25),
        )


class _DummyLaneDomain:
    def __init__(self):
        self.edge_by_lane = {
            "lane_ns": "edge_ns",
            "lane_ew": "edge_ew",
        }
        self.shape_by_lane = {
            "lane_ns": [(0.0, 0.0), (0.0, 20.0)],
            "lane_ew": [(0.0, 0.0), (20.0, 0.0)],
        }

    def getEdgeID(self, lane_id: str) -> str:
        return self.edge_by_lane[lane_id]

    def getShape(self, lane_id: str):
        return self.shape_by_lane[lane_id]


class _DummyEdgeDomain:
    def __init__(self, fail_waiting_time: bool = False):
        self.fail_waiting_time = fail_waiting_time

    def getLastStepHaltingNumber(self, edge_id: str) -> float:
        return 1.0

    def getLastStepVehicleNumber(self, edge_id: str) -> float:
        return 2.0

    def getLastStepMeanSpeed(self, edge_id: str) -> float:
        return 5.0

    def getWaitingTime(self, edge_id: str) -> float:
        if self.fail_waiting_time:
            raise TypeError("unexpected metric type failure")
        return 3.0


class _DummyTrafficLightDomain:
    def getControlledLinks(self, ts_id: str):
        return [
            [("lane_ns", "out_lane_0", "via_0")],
            [("lane_ew", "out_lane_1", "via_1")],
        ]

    def getPhase(self, ts_id: str) -> int:
        return 0


class _DummySumo:
    def __init__(self, fail_waiting_time: bool = False):
        self.lane = _DummyLaneDomain()
        self.edge = _DummyEdgeDomain(fail_waiting_time=fail_waiting_time)
        self.trafficlight = _DummyTrafficLightDomain()


class _DummyEnv:
    def __init__(self, fail_waiting_time: bool = False):
        self.sumo = _DummySumo(fail_waiting_time=fail_waiting_time)


def test_incoming_edges_are_split_by_lane_geometry_axis():
    env = _DummyEnv()
    grouped = _incoming_edges_for_ts(env, "tls_0")
    assert grouped["ns"] == ["edge_ns"]
    assert grouped["ew"] == ["edge_ew"]


def test_compute_metrics_does_not_swallow_unexpected_type_errors():
    env = _DummyEnv(fail_waiting_time=True)
    with pytest.raises(TypeError, match="unexpected metric type failure"):
        compute_metrics_for_agent(
            env=env,
            agent_id="tls_0",
            time_step=5.0,
            action_phase=0,
            action_green_dur=5.0,
            scenario_id="baseline",
            obs_fallback=None,
        )
