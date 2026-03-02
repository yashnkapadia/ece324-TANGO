from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest


RUN_SLOW = os.getenv("RUN_SLOW_INTEGRATION") == "1"


def _run_cmd(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise AssertionError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize("backend", ["benchmarl", "xuance"])
def test_backend_cli_train_and_eval_smoke(tmp_path: Path, backend: str):
    if not RUN_SLOW:
        pytest.skip("Set RUN_SLOW_INTEGRATION=1 to run slow integration tests.")
    if importlib.util.find_spec(backend) is None:
        pytest.skip(f"Missing optional dependency: {backend}")

    model_path = tmp_path / f"asce_{backend}.pt"
    rollout_csv = tmp_path / f"asce_{backend}_rollout.csv"
    train_metrics_csv = tmp_path / f"asce_{backend}_train_metrics.csv"
    eval_csv = tmp_path / f"asce_{backend}_eval_metrics.csv"

    train_cmd = [
        sys.executable,
        "-m",
        "ece324_tango.modeling.train",
        "--trainer-backend",
        backend,
        "--episodes",
        "1",
        "--seconds",
        "30",
        "--delta-time",
        "5",
        "--device",
        "cpu",
        "--model-path",
        str(model_path),
        "--rollout-csv",
        str(rollout_csv),
        "--episode-metrics-csv",
        str(train_metrics_csv),
    ]
    _run_cmd(train_cmd)

    assert model_path.exists()
    assert rollout_csv.exists()
    assert train_metrics_csv.exists()

    eval_cmd = [
        sys.executable,
        "-m",
        "ece324_tango.modeling.predict",
        "--trainer-backend",
        backend,
        "--episodes",
        "1",
        "--seconds",
        "30",
        "--delta-time",
        "5",
        "--device",
        "cpu",
        "--model-path",
        str(model_path),
        "--out-csv",
        str(eval_csv),
    ]
    _run_cmd(eval_cmd)

    assert eval_csv.exists()
    df = pd.read_csv(eval_csv)
    assert {"mappo", "fixed_time", "max_pressure"}.issubset(
        set(df["controller"].astype(str).tolist())
    )
