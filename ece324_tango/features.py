from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger
import typer

from ece324_tango.config import PROCESSED_DATA_DIR

app = typer.Typer(add_completion=False)


@app.command()
def build_asce_features(
    input_path: Path = PROCESSED_DATA_DIR / "asce_rollout_samples.csv",
    output_path: Path = PROCESSED_DATA_DIR / "asce_features.csv",
):
    """Build a lightweight ASCE feature table for analysis/debugging."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    df = pd.read_csv(input_path)
    feat = df.copy()
    feat["queue_imbalance"] = feat["queue_ns"] - feat["queue_ew"]
    feat["arrival_imbalance"] = feat["arrivals_ns"] - feat["arrivals_ew"]
    feat["is_peak"] = ((feat["time_of_day"] >= 7 / 24) & (feat["time_of_day"] <= 10 / 24)).astype(int)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feat.to_csv(output_path, index=False)
    logger.success(f"Wrote features: {output_path}")


if __name__ == "__main__":
    app()
