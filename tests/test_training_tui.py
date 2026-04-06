from ece324_tango.asce.trainers.local_mappo_backend import (
    LocalMappoBackend,
    _infer_start_episode_from_log,
)
from ece324_tango.asce.trainers.training_tui import TrainingStatus


def test_training_tui_uses_worst_ratio_for_best_summary():
    tui = TrainingStatus(
        total_episodes=100,
        num_workers=8,
        scenario_names=["am_peak", "pm_peak"],
    )

    tui.update_eval_results(
        eval_ep=63,
        ratios={"am_peak": 0.81, "pm_peak": 0.94},
    )

    assert tui.best_ratio == 0.94
    assert tui.best_scenario == "pm_peak"


def test_train_state_round_trip(tmp_path):
    backend = LocalMappoBackend()
    state_path = tmp_path / "resume_state.json"

    backend._save_train_state(
        state_path=state_path,
        start_episode=64,
        best_eval_ratio=0.91,
        scenario_best_ratios={"am_peak": 0.88, "pm_peak": 0.95},
        last_eval_ep=63,
        last_eval_ratios={"am_peak": 0.88, "pm_peak": 0.95},
    )

    restored = backend._load_train_state(state_path)
    assert restored["start_episode"] == 64
    assert restored["best_eval_ratio"] == 0.91
    assert restored["scenario_best_ratios"]["am_peak"] == 0.88
    assert restored["last_eval_ep"] == 63
    assert restored["last_eval_ratios"]["pm_peak"] == 0.95


def test_infer_start_episode_from_log(tmp_path):
    log_file = tmp_path / "run.log"
    log_file.write_text(
        "\n".join(
            [
                "2026-03-31 INFO Checkpoint saved at episode 31.",
                "2026-03-31 INFO Resumed from models/asce_mappo_curriculum.pt at episode 32",
                "2026-03-31 INFO Checkpoint saved at episode 104.",
            ]
        )
    )

    assert _infer_start_episode_from_log(str(log_file)) == 105
