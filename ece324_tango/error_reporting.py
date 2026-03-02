from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict

from loguru import logger


_REPORTED_ONCE_KEYS: set[str] = set()
_ERROR_LOG_PATH = (
    Path(__file__).resolve().parents[1] / "reports" / "results" / "error_events.jsonl"
)


def report_exception(
    *,
    context: str,
    exc: BaseException,
    details: Dict[str, Any] | None = None,
    once_key: str | None = None,
) -> None:
    """Log and persist non-fatal exceptions/fallback events."""
    if once_key and once_key in _REPORTED_ONCE_KEYS:
        return
    if once_key:
        _REPORTED_ONCE_KEYS.add(once_key)

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "context": context,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "details": details or {},
    }
    _ERROR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _ERROR_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")

    logger.warning(f"{context}: {type(exc).__name__}: {exc}")
