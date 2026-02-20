from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger
import typer

from ece324_tango.asce.schema import ASCE_DATASET_COLUMNS
from ece324_tango.config import PROCESSED_DATA_DIR

app = typer.Typer(add_completion=False)


@app.command()
def validate_schema(input_path: Path = PROCESSED_DATA_DIR / "asce_rollout_samples.csv"):
    """Validate ASCE rollout CSV matches agreed dataset contract."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    df = pd.read_csv(input_path, nrows=50)
    missing = [c for c in ASCE_DATASET_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in ASCE_DATASET_COLUMNS]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    logger.info(f"Schema valid for: {input_path}")
    if extra:
        logger.warning(f"Extra columns present: {extra}")


if __name__ == "__main__":
    app()
