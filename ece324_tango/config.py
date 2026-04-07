import multiprocessing as _mp
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from ece324_tango.error_reporting import report_exception  # noqa: F401

load_dotenv()

PROJ_ROOT = Path(__file__).resolve().parents[1]

if _mp.current_process().name == "MainProcess":
    logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RESULTS_DIR = REPORTS_DIR / "results"

_tui_active: bool = False

try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: None if _tui_active else tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError as exc:
    report_exception(
        context="config.tqdm_not_available",
        exc=exc,
        once_key="config_tqdm_not_available",
    )
