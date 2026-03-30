# Coding Conventions

**Analysis Date:** 2026-03-29

## Naming Patterns

**Files:**
- Snake case for all module files: `error_reporting.py`, `traffic_metrics.py`, `local_mappo_backend.py`
- Underscores separate logical concepts: `obs_norm.py`, `test_mappo_core.py`

**Functions:**
- Snake case for all function names: `occupancy_for_vehicle_type()`, `extract_step_details()`, `compute_metrics_for_agent()`
- Private functions prefixed with single underscore: `_lane_axis()`, `_incoming_edges_for_ts()`, `_zero_module_parameters()`
- Test functions follow pattern `test_<behavior_being_tested>`: `test_normalize_converges_to_zero_mean_unit_variance()`, `test_pad_observation_right_pads_with_zeros()`

**Variables:**
- Lowercase with underscores: `sim_time`, `active_ids`, `n_actions`, `action_size_by_agent`
- Aggregated data structures use descriptive suffixes: `_by_agent`, `_by_vehicle`, `_by_lane`
- Dictionary keys use snake case: `"intersection_id"`, `"time_step"`, `"queue_ns"`

**Types and Classes:**
- PascalCase for classes: `Actor`, `Critic`, `KPITracker`, `MAPPOTrainer`, `FixedTimeController`, `MaxPressureController`
- PascalCase for dataclasses: `IntersectionMetrics`, `RewardWeights`, `EpisodeKPI`, `TrainConfig`, `EvalConfig`, `Transition`
- Use `@dataclass` decorator for data-holding classes (see `ece324_tango/asce/trainers/base.py`)

## Code Style

**Formatting:**
- Tool: Black (with flake8 secondary)
- Line length: 99 characters
- Configured in `pyproject.toml` with `line-length = 99`

**Linting:**
- Primary: Ruff (>= 0.15.4, < 0.16) - Modern, fast linter used in pixi.toml
- Secondary: Flake8 - Configured in `setup.cfg` with ignored rules: E731 (lambda assignment), E266 (multiple leading # for comment), E501 (long lines - handled by Black), C901 (complex function), W503 (line break before operator)
- Tool order: isort → Black → Ruff (imports ordered, formatted, checked)

**Type Hints:**
- Use future annotations: `from __future__ import annotations` at top of all files for forward references
- Type hints present throughout: function parameters, return types, variable assignments
- Union types use pipe notation where possible: `Dict[str, Any] | None` instead of `Optional[Dict]`
- Generics use modern syntax: `Dict[str, List[float]]`, `Sequence[str]`

## Import Organization

**Order:**
1. `from __future__ import annotations` (always first when used)
2. Standard library imports (sorted): `datetime`, `json`, `math`, `types`, etc.
3. Third-party imports (sorted): `numpy`, `pandas`, `torch`, `loguru`, `typer`, `pytest`, etc.
4. Local imports (sorted): `from ece324_tango.*`

**Path Aliases:**
- No path aliases used; all imports are explicit relative to project root
- First-party package declared in `pyproject.toml`: `known_first_party = ["ece324_tango"]`
- isort configured with `profile = "black"` and `force_sort_within_sections = true`

**Examples:**
```python
# From ece324_tango/config.py
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from ece324_tango.error_reporting import report_exception

# From ece324_tango/asce/mappo.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from ece324_tango.asce.obs_norm import ObsRunningNorm
```

## Error Handling

**Patterns:**
- Use `report_exception()` from `ece324_tango/error_reporting.py` for non-fatal errors that should be logged and persisted
- Function signature: `report_exception(context: str, exc: BaseException, details: Dict[str, Any] | None = None, once_key: str | None = None) -> None`
- Errors are logged to `reports/results/error_events.jsonl` with timestamp, context, error type, and optional details
- Use `once_key` parameter to report the same error only once (deduplicate repeated errors)
- For fatal/assertion errors: raise exceptions (see `test_local_backend_bootstrap.py` line 27: `raise AssertionError`)
- For expected exceptions in tests: use `pytest.raises()` context manager

**Examples:**
```python
# From ece324_tango/config.py - graceful fallback
try:
    from tqdm import tqdm
    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError as exc:
    report_exception(
        context="config.tqdm_not_available",
        exc=exc,
        once_key="config_tqdm_not_available",
    )

# From ece324_tango/asce/runtime.py - with context details
try:
    ts = env.unwrapped.traffic_signals[agent_id]
    return int(getattr(ts, "green_phase", 0))
except Exception as exc:
    report_exception(
        context="runtime.current_phase_fallback",
        exc=exc,
        details={"agent_id": agent_id},
        once_key=f"current_phase:{agent_id}",
    )
    return 0  # Safe fallback value
```

## Logging

**Framework:** loguru

**Patterns:**
- Import: `from loguru import logger`
- Log at appropriate level based on severity:
  - `logger.info()`: General progress and milestones
  - `logger.warning()`: Non-fatal errors and fallback events (used in `report_exception()`)
  - `logger.debug()`: Detailed diagnostic info (not commonly used in codebase)
- Logger configured in `ece324_tango/config.py` to integrate with tqdm progress bars
- Use f-strings for formatting: `logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")`

**Examples:**
```python
# From ece324_tango/config.py
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

# From ece324_tango/error_reporting.py
logger.warning(f"{context}: {type(exc).__name__}: {exc}")
```

## Comments

**When to Comment:**
- Docstrings on functions that perform non-obvious operations or have complex logic
- Inline comments explaining "why" for non-obvious design decisions, not "what" the code does
- Module-level docstrings sparse (focus on code clarity rather than comments)

**JSDoc/Docstrings:**
- Use triple-quoted string docstrings on complex functions
- Documented return types in function signatures (prefer type hints over docstring type declarations)
- Example from `ece324_tango/asce/runtime.py`:
```python
def extract_step_details(step_output):
    """Normalize step output and preserve end-type when available.

    Returns:
        obs, rewards, done, infos, terminated, truncated
    """
```

## Function Design

**Size:**
- Most functions are 5-40 lines; larger functions decomposed into helpers
- Factory functions (e.g., `get_backend()` in `ece324_tango/asce/trainers/factory.py`) may be longer

**Parameters:**
- Use keyword-only arguments for clarity in complex functions: `report_exception(*, context, exc, ...)`
- Default parameters used for optional configurations (e.g., `hidden_dim: int = 128` in `Actor.__init__`)
- Avoid positional-only long parameter lists; prefer configuration objects (dataclasses)

**Return Values:**
- Return concrete values, not None, when possible; use type hints to signal intent
- Single return value per function (not multiple; use tuples or dataclasses for multiple outputs)
- Functions with side effects (void return) explicitly return `None` or omit return
- Early returns used for guard clauses and fallbacks

**Examples:**
```python
# From ece324_tango/asce/baselines.py - keyword-only
def actions(self, observations: Dict[str, np.ndarray]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for agent, obs in observations.items():
        # ... logic ...
    return out

# From ece324_tango/asce/kpi.py - early return pattern
def update(self, env) -> None:
    sim_time = float(env.sumo.simulation.getTime())
    active_ids = list(env.sumo.vehicle.getIDList())

    for vid in active_ids:
        try:
            current_tl = float(env.sumo.vehicle.getTimeLoss(vid))
        except Exception as exc:
            report_exception(...)
            continue  # Guard clause
```

## Module Design

**Exports:**
- Most modules export all public classes/functions; no `__all__` except in `ece324_tango/__init__.py`
- Public API defined implicitly (underscore-prefixed items are private)

**Barrel Files:**
- Used in `__init__.py` files to expose public APIs:
  - `ece324_tango/asce/trainers/__init__.py` exports `TrainConfig`, `EvalConfig`, `get_backend`
  - `ece324_tango/__init__.py` minimal: `__all__ = ["config"]`

**Module Organization:**
- Each module focuses on one domain/responsibility: `traffic_metrics.py`, `kpi.py`, `mappo.py`, `baselines.py`
- Dataclass-heavy design: domain objects are immutable dataclasses passed between functions
- Factory pattern used for backend selection: `ece324_tango/asce/trainers/factory.py`

---

*Convention analysis: 2026-03-29*
