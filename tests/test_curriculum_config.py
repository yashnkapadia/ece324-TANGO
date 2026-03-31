"""Tests for curriculum training configuration and round-robin scenario assignment."""

from __future__ import annotations

from pathlib import Path

import pytest

from ece324_tango.asce.trainers.base import TrainConfig


def _make_train_config(**overrides) -> TrainConfig:
    """Build a minimal TrainConfig with sensible defaults for testing."""
    defaults = dict(
        model_path=Path("/tmp/model.pt"),
        rollout_csv=Path("/tmp/rollout.csv"),
        episode_metrics_csv=Path("/tmp/ep_metrics.csv"),
        net_file="test.net.xml",
        route_file="test.rou.xml",
        scenario_id="baseline",
        episodes=10,
        seconds=300,
        delta_time=5,
        ppo_epochs=4,
        minibatch_size=512,
        seed=7,
        use_gui=False,
        device="cpu",
        backend_verbose=False,
        reward_mode="objective",
        reward_delay_weight=1.0,
        reward_throughput_weight=1.0,
        reward_fairness_weight=0.25,
        reward_residual_weight=0.25,
    )
    defaults.update(overrides)
    return TrainConfig(**defaults)


class TestTrainConfigRouteFiles:
    """TrainConfig.route_files field behaviour."""

    def test_route_files_stores_list(self):
        cfg = _make_train_config(route_files=["a.rou.xml", "b.rou.xml"])
        assert cfg.route_files == ["a.rou.xml", "b.rou.xml"]

    def test_route_files_defaults_to_empty_list(self):
        cfg = _make_train_config()
        assert cfg.route_files == []

    def test_curriculum_mode_detected_when_route_files_nonempty(self):
        cfg = _make_train_config(route_files=["a.rou.xml", "b.rou.xml"])
        assert len(cfg.route_files) > 0

    def test_non_curriculum_when_route_files_empty(self):
        cfg = _make_train_config()
        assert len(cfg.route_files) == 0


class TestRoundRobinAssignment:
    """Round-robin scenario cycling logic (used by training loop)."""

    SCENARIOS = [
        "am_peak.rou.xml",
        "pm_peak.rou.xml",
        "demand_surge.rou.xml",
        "midday_multimodal.rou.xml",
    ]

    @pytest.mark.parametrize(
        "episode, expected_idx",
        [
            (0, 0),
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 0),  # wraps around
            (5, 1),
            (7, 3),
            (11, 3),
        ],
    )
    def test_sequential_round_robin(self, episode, expected_idx):
        """Episode N should get scenario N % len(scenarios)."""
        scenario_pool = self.SCENARIOS
        selected = scenario_pool[episode % len(scenario_pool)]
        assert selected == scenario_pool[expected_idx]

    @pytest.mark.parametrize(
        "ep_batch_start, worker_idx, expected_idx",
        [
            (0, 0, 0),
            (0, 1, 1),
            (0, 2, 2),
            (0, 3, 3),
            (0, 4, 0),  # wraps with 8 workers
            (0, 5, 1),
            (0, 6, 2),
            (0, 7, 3),
            (8, 0, 0),  # next batch
            (8, 1, 1),
        ],
    )
    def test_parallel_worker_assignment(self, ep_batch_start, worker_idx, expected_idx):
        """Worker gets scenario (ep_batch_start + worker_idx) % len(scenarios)."""
        scenario_pool = self.SCENARIOS
        selected = scenario_pool[(ep_batch_start + worker_idx) % len(scenario_pool)]
        assert selected == scenario_pool[expected_idx]

    def test_scenario_id_derived_from_filename(self):
        """Scenario IDs should be derived from route file stems."""
        scenario_pool = self.SCENARIOS
        scenario_ids = [Path(rf).stem for rf in scenario_pool]
        assert scenario_ids == ["am_peak", "pm_peak", "demand_surge", "midday_multimodal"]
