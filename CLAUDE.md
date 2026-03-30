<!-- GSD:project-start source:PROJECT.md -->
## Project

**TANGO — Milestone 2**

TANGO (Traffic Adaptive Network Guidance & Optimization) is a two-model traffic optimization system for a Toronto corridor. ASCE (Adaptive Signal Control Engine) uses multi-agent reinforcement learning (MAPPO) to control 8 signalized intersections in real time. PIRA (Planning Infrastructure Response Analyzer) will be a GNN surrogate for scenario planning. This milestone focuses on making ASCE competitive with Max-Pressure and building the data pipeline for PIRA.

**Core Value:** MAPPO must match or beat Max-Pressure on person-time-loss under nominal conditions and generalize better under irregular demand — this is the proposal's success criterion and the foundation for PIRA and the final report.

### Constraints

- **Compute**: RTX 4070 Laptop GPU — training must complete in reasonable time (<1 day per experiment)
- **Simulation**: SUMO FatalTraCIError when demand exhausts before sim end (~285s in 300s episodes) — must handle gracefully or extend demand
- **Timeline**: Final report deadline approaching — prioritize closing MP gap over stretch goals
- **Dependencies**: demand studio requires `dash`, `plotly`, `pyproj`, `lxml`, `sumolib` — some may need installation
<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->
## Technology Stack

## Languages
- Python 3.12.12 - All source code, ML training, simulation, data processing
## Runtime
- Linux 64-bit (conda-forge packages)
- CUDA 12.8 support built-in
- Pixi (conda-based) - Defined in `pixi.toml`
- Lockfile: `pixi.lock` (present, fully locked)
- Python package manager: pip (included via pixi)
## Frameworks
- PyTorch 2.7.1 (CUDA 12.6) - Neural network training and inference
- SUMO (Simulation of Urban Mobility) v1.26.0 - Traffic simulator via TraCI interface
- Typer - Command-line interface building
- NumPy - Array computations, observation processing
- Pandas - CSV/data handling, metrics aggregation
- Matplotlib - Figure generation
- scikit-learn - Metrics and utilities
- pytest - Test runner
- black 26.1.0 - Code formatter (line-length: 99)
- isort 7.0.0 - Import sorting (profile: black)
- flake8 7.3.0 - Linter (config in `setup.cfg`)
- ruff >=0.15.4,<0.16 - Fast linter/formatter
- Jupyter/IPython - Interactive development
- flit_core >=3.2,<4 - Build backend
- python-dotenv - Environment variable loading
- OpenMPI >=5.0.8,<6 - Multi-agent distributed training (optional)
- loguru - Structured logging with tqdm integration
- tqdm - Progress bars
- IPython/Jupyter - Interactive notebooks
## Key Dependencies
- `sumo-rl 1.4.5` - Multi-agent RL interface to SUMO
- `pytorch-gpu 2.7.1` - GPU-accelerated tensor computation
- `numpy` - Core numerical operations for observation padding, queue calculations
- `pandas` - Training/evaluation metrics CSV I/O
- `loguru` - Exception-safe logging via `ece324_tango/error_reporting.py`
- `typer` - Type-safe CLI argument parsing with help text
- `python-dotenv` - Environment configuration loading
- `sumolib 1.26.0` - SUMO utilities (imported transitively via sumo-rl)
## Configuration
- Loaded via `python-dotenv` in `ece324_tango/config.py`
- No `.env` file present in repo (secrets not committed)
- Paths configured in `ece324_tango/config.py`:
- `pyproject.toml` - Flit-based package config, Black/isort/pytest settings
- `setup.cfg` - Flake8 configuration (E731, E266, E501, C901, W503 ignored)
- `pixi.toml` - Workspace definition, dependencies, and Pixi tasks
- `train-asce` - Basic training on sample grid
- `train-asce-toronto-demand` - Full training pipeline (30 episodes, 300 s, objective reward)
- `eval-asce-toronto-demand` - Evaluation on Toronto network
- `validate-asce-schema` - Data validation via `ece324_tango.dataset`
- `plot-interim-figures` - Report figure generation via `ece324_tango.plots`
## Platform Requirements
- Linux 64-bit OS (WSL2 compatible, tested on Linux 6.6.87.2-microsoft-standard-WSL2)
- NVIDIA GPU with CUDA 12.x support (recommended for training)
- ~6GB disk for Pixi environment (.pixi/)
- ~3.5MB for SUMO network files (osm.net.xml.gz)
- Same as development (GPU optional for inference, slower on CPU)
- Docker containerization not currently in use
- CUDA 12.8 declaratively specified
- CuDNN 9.10.2.21 included in lock file
- PyTorch auto-detects CUDA availability at runtime
## Dependencies by Module
- sumo-rl, numpy
- torch, numpy
- typer, loguru, torch (via LocalMappoBackend)
- numpy, pandas, torch, loguru, traci.exceptions (from sumo-rl)
- matplotlib, pandas, numpy
- python-dotenv, loguru, pathlib, tqdm
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

## Naming Patterns
- Snake case for all module files: `error_reporting.py`, `traffic_metrics.py`, `local_mappo_backend.py`
- Underscores separate logical concepts: `obs_norm.py`, `test_mappo_core.py`
- Snake case for all function names: `occupancy_for_vehicle_type()`, `extract_step_details()`, `compute_metrics_for_agent()`
- Private functions prefixed with single underscore: `_lane_axis()`, `_incoming_edges_for_ts()`, `_zero_module_parameters()`
- Test functions follow pattern `test_<behavior_being_tested>`: `test_normalize_converges_to_zero_mean_unit_variance()`, `test_pad_observation_right_pads_with_zeros()`
- Lowercase with underscores: `sim_time`, `active_ids`, `n_actions`, `action_size_by_agent`
- Aggregated data structures use descriptive suffixes: `_by_agent`, `_by_vehicle`, `_by_lane`
- Dictionary keys use snake case: `"intersection_id"`, `"time_step"`, `"queue_ns"`
- PascalCase for classes: `Actor`, `Critic`, `KPITracker`, `MAPPOTrainer`, `FixedTimeController`, `MaxPressureController`
- PascalCase for dataclasses: `IntersectionMetrics`, `RewardWeights`, `EpisodeKPI`, `TrainConfig`, `EvalConfig`, `Transition`
- Use `@dataclass` decorator for data-holding classes (see `ece324_tango/asce/trainers/base.py`)
## Code Style
- Tool: Black (with flake8 secondary)
- Line length: 99 characters
- Configured in `pyproject.toml` with `line-length = 99`
- Primary: Ruff (>= 0.15.4, < 0.16) - Modern, fast linter used in pixi.toml
- Secondary: Flake8 - Configured in `setup.cfg` with ignored rules: E731 (lambda assignment), E266 (multiple leading # for comment), E501 (long lines - handled by Black), C901 (complex function), W503 (line break before operator)
- Tool order: isort → Black → Ruff (imports ordered, formatted, checked)
- Use future annotations: `from __future__ import annotations` at top of all files for forward references
- Type hints present throughout: function parameters, return types, variable assignments
- Union types use pipe notation where possible: `Dict[str, Any] | None` instead of `Optional[Dict]`
- Generics use modern syntax: `Dict[str, List[float]]`, `Sequence[str]`
## Import Organization
- No path aliases used; all imports are explicit relative to project root
- First-party package declared in `pyproject.toml`: `known_first_party = ["ece324_tango"]`
- isort configured with `profile = "black"` and `force_sort_within_sections = true`
## Error Handling
- Use `report_exception()` from `ece324_tango/error_reporting.py` for non-fatal errors that should be logged and persisted
- Function signature: `report_exception(context: str, exc: BaseException, details: Dict[str, Any] | None = None, once_key: str | None = None) -> None`
- Errors are logged to `reports/results/error_events.jsonl` with timestamp, context, error type, and optional details
- Use `once_key` parameter to report the same error only once (deduplicate repeated errors)
- For fatal/assertion errors: raise exceptions (see `test_local_backend_bootstrap.py` line 27: `raise AssertionError`)
- For expected exceptions in tests: use `pytest.raises()` context manager
## Logging
- Import: `from loguru import logger`
- Log at appropriate level based on severity:
- Logger configured in `ece324_tango/config.py` to integrate with tqdm progress bars
- Use f-strings for formatting: `logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")`
## Comments
- Docstrings on functions that perform non-obvious operations or have complex logic
- Inline comments explaining "why" for non-obvious design decisions, not "what" the code does
- Module-level docstrings sparse (focus on code clarity rather than comments)
- Use triple-quoted string docstrings on complex functions
- Documented return types in function signatures (prefer type hints over docstring type declarations)
- Example from `ece324_tango/asce/runtime.py`:
## Function Design
- Most functions are 5-40 lines; larger functions decomposed into helpers
- Factory functions (e.g., `get_backend()` in `ece324_tango/asce/trainers/factory.py`) may be longer
- Use keyword-only arguments for clarity in complex functions: `report_exception(*, context, exc, ...)`
- Default parameters used for optional configurations (e.g., `hidden_dim: int = 128` in `Actor.__init__`)
- Avoid positional-only long parameter lists; prefer configuration objects (dataclasses)
- Return concrete values, not None, when possible; use type hints to signal intent
- Single return value per function (not multiple; use tuples or dataclasses for multiple outputs)
- Functions with side effects (void return) explicitly return `None` or omit return
- Early returns used for guard clauses and fallbacks
## Module Design
- Most modules export all public classes/functions; no `__all__` except in `ece324_tango/__init__.py`
- Public API defined implicitly (underscore-prefixed items are private)
- Used in `__init__.py` files to expose public APIs:
- Each module focuses on one domain/responsibility: `traffic_metrics.py`, `kpi.py`, `mappo.py`, `baselines.py`
- Dataclass-heavy design: domain objects are immutable dataclasses passed between functions
- Factory pattern used for backend selection: `ece324_tango/asce/trainers/factory.py`
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

## Pattern Overview
- Decentralized actor execution: each signalized intersection makes independent phase decisions using shared policy
- Centralized critic: global observation (concatenated agent observations) for value estimation
- Environment abstraction: sumo-rl wrapper around SUMO traffic microsimulation
- Training/evaluation pipelines with pluggable reward modes and configurable baselines
- Observation normalization using Welford's online algorithm for stable training
## Layers
- Purpose: CLI interfaces for training and evaluation workflows
- Location: `ece324_tango/modeling/train.py`, `ece324_tango/modeling/predict.py`
- Contains: Typer CLI apps, configuration parsing, backend selection
- Depends on: `LocalMappoBackend`, `TrainConfig`, `EvalConfig`
- Used by: Pixi tasks, direct Python module execution
- Purpose: Orchestrate multi-agent training loop and baseline evaluation
- Location: `ece324_tango/asce/trainers/local_mappo_backend.py`
- Contains: Episode loop, environment interaction, trajectory collection, metric computation
- Depends on: `MAPPOTrainer`, baselines, SUMO environment, KPI tracking
- Used by: Modeling entry points
- Purpose: Actor-critic networks with batch processing for multiple agents
- Location: `ece324_tango/asce/mappo.py`
- Contains: Actor (local policy), Critic (global value), PPO update logic, batch processing
- Depends on: PyTorch, observation normalization
- Used by: Backend for action selection and policy updates
- Purpose: SUMO environment creation and observation manipulation
- Location: `ece324_tango/asce/env.py`
- Contains: Environment factory, observation flattening/padding, default SUMO file resolution
- Depends on: sumo-rl package
- Used by: Backend for environment instantiation
- Purpose: Deterministic signal control policies for comparison
- Location: `ece324_tango/asce/baselines.py`
- Contains: FixedTimeController (uniform cycling), MaxPressureController (queue-differential optimization)
- Depends on: SUMO environment state
- Used by: Backend evaluation loop
- Purpose: Compute KPIs and rewards from SUMO/agent state
- Location: `ece324_tango/asce/kpi.py`, `ece324_tango/asce/traffic_metrics.py`
- Contains: Vehicle-level time-loss tracking, person-weighted delays, fairness (Jain index), throughput
- Depends on: SUMO TraCI API, numpy
- Used by: Backend for episode metrics and reward signals
- Purpose: Handle environment API variations and helper calculations
- Location: `ece324_tango/asce/runtime.py`
- Contains: Step output normalization (legacy vs. Gymnasium), phase extraction, Jain index
- Depends on: None (numpy for calculations)
- Used by: Backend, baselines, trainers
- Purpose: Online feature-wise standardization using Welford's algorithm
- Location: `ece324_tango/asce/obs_norm.py`
- Contains: `ObsRunningNorm` class with update/normalize/state persistence
- Depends on: numpy
- Used by: MAPPOTrainer during training and inference
- Purpose: Schema validation and feature engineering on rollout data
- Location: `ece324_tango/asce/schema.py`, `ece324_tango/dataset.py`, `ece324_tango/features.py`
- Contains: ASCE dataset column definitions, rollout CSV schema validation, feature extraction (queue imbalance, peak detection)
- Depends on: pandas, pathlib
- Used by: Post-training analysis and reproducibility
- Purpose: Generate interim report figures from training/evaluation artifacts
- Location: `ece324_tango/plots.py`
- Contains: Matplotlib figure rendering (training curves, KPI comparisons)
- Depends on: pandas, matplotlib
- Used by: `plot-interim-figures` pixi task
## Data Flow
- **Episode state:** observations, actions, rewards accumulated in trajectory dictionaries keyed by agent_id
- **Normalizer state:** running mean/variance per feature updated during training, saved/restored with model
- **Baseline state:** Fixed-Time has internal cursor (phase counter), Max-Pressure is stateless
- **Error state:** Non-fatal exceptions logged to `reports/results/error_events.jsonl` with once-keys for deduplication
## Key Abstractions
- Purpose: Immutable configuration for training and evaluation
- Examples: `ece324_tango/asce/trainers/base.py`
- Pattern: Dataclass with required and optional fields; passed to backend.train/evaluate
- Purpose: Actor-critic policy with batch action selection and PPO updates
- Examples: `ece324_tango/asce/mappo.py`
- Pattern: Stateful class with `act_batch()` for inference, `update()` for learning, normalizers optional
- Purpose: Abstract interface for training/evaluation implementations
- Examples: `ece324_tango/asce/trainers/base.py` (base), `ece324_tango/asce/trainers/local_mappo_backend.py` (implementation)
- Pattern: Base class with `train(cfg)` and `evaluate(cfg)` methods; factory in `ece324_tango/asce/trainers/factory.py`
- Purpose: Record of a single agent's step outcome
- Examples: `ece324_tango/asce/mappo.py`
- Pattern: Dataclass with obs, action, reward, done, value, logprob, n_valid_actions
- Purpose: Structured reward/metric definitions
- Examples: `ece324_tango/asce/traffic_metrics.py`
- Pattern: Dataclasses for type safety; enables multi-mode reward functions
## Entry Points
- Location: `ece324_tango/modeling/train.py::main()`
- Triggers: `pixi run train-asce-toronto-demand` or direct module invocation
- Responsibilities: Parse CLI args, create LocalMappoBackend, instantiate TrainConfig, call backend.train()
- Location: `ece324_tango/modeling/predict.py::main()`
- Triggers: `pixi run eval-asce-toronto-demand` or direct module invocation
- Responsibilities: Parse CLI args, load trained model, create EvalConfig, call backend.evaluate()
- Location: `ece324_tango/plots.py` (module-level, no app)
- Triggers: `pixi run plot-interim-figures`
- Responsibilities: Load train/eval CSVs, render 2x2 subplot figure, save to reports/figures/
- Location: `ece324_tango/dataset.py::validate_schema()`
- Triggers: `pixi run validate-asce-schema`
- Responsibilities: Check rollout CSV has required ASCE_DATASET_COLUMNS
- Location: `ece324_tango/features.py::build_asce_features()`
- Triggers: Direct Python call or future pixi task
- Responsibilities: Compute derived features (queue imbalance, arrival imbalance, peak detection) and write feature CSV
## Error Handling
- **SUMO TraCI failures:** FatalTraCIError caught; episode marked done if happens mid-simulation
- **Vehicle/lane lookups:** Missing vehicle type/departure/lane defaults applied; context + once_key logged to error_events.jsonl
- **Observation/action mismatches:** RuntimeError raised if reset returns empty obs or observation size exceeds target_dim
- **Invalid normalizer states:** ValueError raised if non-finite values or dimension mismatches detected
- **Missing SUMO files:** FileNotFoundError if net/route files not found and not provided explicitly
## Cross-Cutting Concerns
- Tool: loguru (configured in `ece324_tango/config.py`)
- Integration: writes to stderr, optionally integrates with tqdm progress bars
- Handlers: module-level logger in each source file
- Input validation: CLI args via typer (auto type-coercion, bounds checking)
- Data contracts: ASCE_DATASET_COLUMNS enforced by `dataset.validate_schema()`
- Schema drift detection: observation size padding with warnings
- Abstraction: `LocalMappoBackend._resolve_device(device: str) → str`
- Behavior: "auto" → CUDA if available else CPU; else returns string as-is
- Used: trainer initialization and checkpoint device placement
- Applied at sample collection: per-agent logits have invalid actions set to `-inf` before sampling
- Applied during PPO update: per-transition n_valid_actions mask enforced in clip ratio calculation
- Guarantees: invalid actions never sampled, loss computed only over valid action space
- Context manager in `ece324_tango/asce/trainers/noise_control.py::quiet_output()`
- Suppresses third-party stdout/stderr and tqdm via environment variable and file descriptor manipulation
- Used: backend with `backend_verbose=False` parameter
<!-- GSD:architecture-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd:quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd:debug` for investigation and bug fixing
- `/gsd:execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->



<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd:profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
