from ece324_tango.asce.traffic_metrics import IntersectionMetrics, RewardWeights, rewards_from_metrics


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
