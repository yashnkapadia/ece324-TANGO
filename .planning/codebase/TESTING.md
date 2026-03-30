# Testing Patterns

**Analysis Date:** 2026-03-29

## Test Framework

**Runner:**
- pytest (latest version)
- Config file: `pyproject.toml` (under `[tool.pytest.ini_options]`)
- Markers defined:
  - `slow`: Long-running tests
  - `integration`: Cross-module or CLI integration tests

**Assertion Library:**
- pytest built-in assertions + `numpy.testing` for numerical arrays

**Run Commands:**
```bash
pytest                    # Run all tests
pytest -v               # Verbose mode (show test names)
pytest tests/test_kpi.py # Run specific test file
pytest -k "test_name"   # Run tests matching pattern
pytest --tb=short       # Show short tracebacks
pytest -m slow          # Run only slow-marked tests
pytest --co             # Collect tests (dry run)
```

## Test File Organization

**Location:**
- All tests in `tests/` directory at project root
- 19 test files total (41 test functions across files)
- Co-located pattern: test file `test_X.py` corresponds to module `ece324_tango/.../X.py`

**Naming:**
- `test_<functionality>()` for simple unit tests
- `test_<unit>_<behavior>()` for parameterized or multiple related tests
- Private test helpers prefixed with `_`: `_DummyEnv`, `_FakeTrainer`, `_metric()`

**Structure:**
```
tests/
├── conftest.py                                    # Pytest fixtures and shared setup
├── test_kpi.py                                   # Tests for ece324_tango/asce/kpi.py
├── test_traffic_metrics.py                       # Tests for ece324_tango/asce/traffic_metrics.py
├── test_obs_norm.py                              # Tests for ece324_tango/asce/obs_norm.py
├── test_observation_padding.py                   # Tests for ece324_tango/asce/env.py
├── test_mappo_core.py                            # Tests for ece324_tango/asce/mappo.py
├── test_baseline_max_pressure.py                 # Tests for ece324_tango/asce/baselines.py
├── test_runtime_step_flags.py                    # Tests for ece324_tango/asce/runtime.py
├── test_data.py                                  # Tests for schema/config validation
├── test_local_backend_bootstrap.py               # Tests for trainer backend
├── test_local_backend_eval_guard.py              # Tests for trainer eval guards
├── test_local_eval_objective_scoring.py          # Tests for evaluation metrics
├── test_local_eval_fallback_observation_alignment.py
├── test_local_eval_fatal_kpi_guard.py
├── test_local_eval_fixed_time_reset.py
├── test_cli_obs_norm_defaults.py                 # Tests for CLI argument handling
├── test_backend_selection.py                     # Tests for backend factory
├── test_device_resolution.py                     # Tests for device setup
└── test_obs_norm_parity.py                       # Tests for normalization consistency
```

## Test Structure

**Suite Organization:**
- No explicit test classes (flat structure)
- Shared fixtures defined in `conftest.py` (currently minimal: path setup)
- Related tests grouped in same file by functionality domain

**Common Pattern:**
```python
from [module] import [function_or_class]

def _helper_function():
    """Private test helper."""
    pass

def test_happy_path():
    # Arrange
    input_data = _helper_function()

    # Act
    result = function_under_test(input_data)

    # Assert
    assert result == expected_value

def test_error_case():
    # Arrange
    invalid_input = ...

    # Act & Assert
    with pytest.raises(ValueError, match="pattern"):
        function_under_test(invalid_input)
```

**Patterns Observed:**

1. **Simple unit test** (from `tests/test_data.py`):
```python
def test_asce_schema_columns_stable():
    expected = {...}
    assert set(ASCE_DATASET_COLUMNS) == expected
```

2. **Assertion on numerical arrays** (from `tests/test_obs_norm.py`):
```python
def test_normalize_converges_to_zero_mean_unit_variance():
    rng = np.random.default_rng(42)
    norm = ObsRunningNorm(dim=4)
    for _ in range(1000):
        x = rng.normal(loc=[2.0, -1.0, 0.0, 5.0], scale=[1.0, 2.0, 0.5, 3.0])
        norm.update(x)
    out = norm.normalize(np.array([2.0, -1.0, 0.0, 5.0], dtype=np.float32))
    np.testing.assert_allclose(out, 0.0, atol=0.1)  # Numpy testing API
```

3. **State roundtrip test** (from `tests/test_obs_norm.py`):
```python
def test_state_dict_roundtrip():
    norm = ObsRunningNorm(dim=3)
    norm.update(np.array([1.0, 2.0, 3.0]))
    state = norm.state_dict()

    norm2 = ObsRunningNorm(dim=3)
    norm2.load_state_dict(state)
    np.testing.assert_array_equal(norm.normalize(x), norm2.normalize(x))
```

## Mocking

**Framework:** pytest's `monkeypatch` fixture

**Patterns:**
```python
def test_with_mocks(monkeypatch, tmp_path):
    """Use monkeypatch for module-level and class substitution."""
    # Mock a module-level factory function
    monkeypatch.setattr(
        "ece324_tango.asce.trainers.local_mappo_backend.create_parallel_env",
        lambda **kwargs: DummyEnv(),
    )

    # Mock a class
    monkeypatch.setattr(
        "ece324_tango.asce.trainers.local_mappo_backend.MAPPOTrainer",
        FakeTrainer,
    )

    # Call code under test
    LocalMappoBackend().train(cfg)

    # Assert on side effects
    trainer = FakeTrainer.instances[0]
    assert trainer.act_calls == 1
```

**What to Mock:**
- External environment interactions: `create_parallel_env()`, `KPITracker`, SUMO integration
- Heavy computations: `MAPPOTrainer` (use fake with minimal logic)
- I/O operations: file writes, CSV output (use `tmp_path` fixture)
- Time-dependent code: rarely needed here

**What NOT to Mock:**
- Core domain logic: `rewards_from_metrics()`, `extract_step_details()` (test real implementation)
- Dataclass construction: dataclasses are transparent
- Utility functions: math, array operations (test directly)
- Built-in types: lists, dicts, numpy arrays

**Examples from `tests/test_local_backend_bootstrap.py`:**
```python
class _DummyActionSpace:
    n = 2

class _DummyEnv:
    """Minimal fake environment for testing bootstrap logic."""
    def __init__(self, step_output):
        self._step_output = step_output
        self._stepped = False

    def reset(self, seed=None):
        self._stepped = False
        return {"a0": np.asarray([0.1], dtype=np.float32)}

    def action_spaces(self, agent):
        return _DummyActionSpace()

    def step(self, actions):
        if self._stepped:
            raise AssertionError("Expected exactly one step in this synthetic episode.")
        self._stepped = True
        return self._step_output

class _FakeTrainer:
    instances = []

    def __init__(self, *args, **kwargs):
        self.obs_norm = None
        self.act_calls = 0
        _FakeTrainer.instances.append(self)

    def act(self, obs, global_obs, n_valid_actions=None):
        self.act_calls += 1
        return {"action": 0, "logp": -0.1, "value": 42.0}
```

## Fixtures and Factories

**Test Data:**
- Factory helper functions create domain objects: `_metric()` in `tests/test_traffic_metrics.py`, `_train_cfg()` in `tests/test_local_backend_bootstrap.py`
- Numpy random seeding for reproducibility: `np.random.default_rng(42)` in `tests/test_obs_norm.py`
- Temporary file paths via pytest's `tmp_path` fixture

**Location:**
- Conftest: `tests/conftest.py` - minimal, only path setup
- Inline helpers: private functions `_metric()`, `_DummyEnv`, `_FakeTrainer` defined in test files where needed
- No shared fixtures directory; test data created fresh per test

**Examples:**
```python
# From tests/test_traffic_metrics.py - Factory for test objects
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

# From tests/test_local_backend_bootstrap.py - Config factory
def _train_cfg(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        model_path=tmp_path / "model.pt",
        rollout_csv=tmp_path / "rollout.csv",
        episode_metrics_csv=tmp_path / "episode_metrics.csv",
        net_file="dummy.net.xml",
        route_file="dummy.rou.xml",
        scenario_id="baseline",
        episodes=1,
        seconds=5,
        delta_time=5,
        # ... more fields ...
    )
```

## Coverage

**Requirements:** Not enforced (no coverage threshold in pytest config)

**View Coverage:**
```bash
pytest --cov=ece324_tango --cov-report=html  # Generate HTML report
pytest --cov=ece324_tango --cov-report=term  # Terminal report
```

**Status:** 41 tests across 19 files provide reasonable coverage for:
- Core traffic metric calculations
- Training backend logic and trainer state management
- Evaluation and scoring mechanisms
- Data validation and schema consistency
- Normalization and numerical stability
- Device and backend selection

**Gaps:** No explicit coverage measurement; gaps likely in:
- CLI argument parsing (partial coverage in `test_cli_obs_norm_defaults.py`)
- Visualization code (`ece324_tango/plots.py`)
- Baseline controller edge cases

## Test Types

**Unit Tests:**
- Scope: Individual functions and methods in isolation
- Approach: Direct function calls with controlled inputs
- Examples:
  - `test_occupancy_for_vehicle_type()` - Tests KPI calculation
  - `test_normalize_converges_to_zero_mean_unit_variance()` - Tests observation normalization
  - `test_update_respects_action_mask_from_batch()` - Tests MAPPO trainer action masking
- Assertion style: Direct equality and numpy assertions

**Integration Tests:**
- Scope: Component interactions (trainer + environment + metrics)
- Approach: Mock only external I/O, test real domain logic flow
- Examples:
  - `test_local_backend_bootstraps_last_value_on_truncation()` - Tests trainer, backend, env interaction
  - `test_local_eval_records_objective_score_for_all_controllers()` - Tests evaluation pipeline with mocked I/O
  - `test_compute_metrics_does_not_swallow_unexpected_type_errors()` - Tests error propagation
- Marked with `@pytest.mark.integration` (none currently, but infrastructure exists)

**E2E Tests:**
- Status: Not used
- CLI could benefit from E2E tests (e.g., `pixi run validate-asce-schema`)

## Common Patterns

**Async Testing:**
- Not used in this codebase (synchronous Python)

**Error Testing:**
```python
# From tests/test_traffic_metrics.py
def test_rewards_from_metrics_unknown_mode_raises():
    metrics = {"a0": _metric("a", delay=1.0, throughput=8)}
    with pytest.raises(ValueError, match="Unsupported reward mode"):
        rewards_from_metrics(
            metrics_by_agent=metrics,
            mode="not_a_mode",
            weights=RewardWeights(delay=1.0, throughput=1.0, fairness=0.25),
        )

# From tests/test_observation_padding.py
def test_pad_observation_rejects_smaller_target():
    src = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    with pytest.raises(ValueError, match="target_dim"):
        pad_observation(src, target_dim=2)
```

**Behavior-Driven Assertions:**
```python
# From tests/test_traffic_metrics.py
def test_rewards_from_metrics_objective_prefers_higher_throughput_lower_delay():
    metrics = {
        "a": _metric("a", delay=1.0, throughput=8),
        "b": _metric("b", delay=10.0, throughput=2),
    }
    rewards = rewards_from_metrics(...)
    assert rewards["a"] > rewards["b"]  # Behavioral assertion

def test_rewards_from_metrics_residual_mp_penalizes_deviation_from_max_pressure():
    # ... setup ...
    assert rewards["a"] > rewards["b"]
    assert rewards["a"] - rewards["b"] == pytest.approx(0.5)  # Numerical tolerance
```

**Approximate Equality:**
```python
# From tests/test_traffic_metrics.py - Used for floating-point comparisons
assert rewards["a"] == pytest.approx(-0.69314718056)
assert rewards["a"] - rewards["b"] == pytest.approx(0.5)
```

---

*Testing analysis: 2026-03-29*
