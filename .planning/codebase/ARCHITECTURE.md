# Architecture

**Analysis Date:** 2026-03-29

## Pattern Overview

**Overall:** Multi-agent reinforcement learning (MAPPO) for decentralized traffic signal control with shared parameter actor and centralized critic.

**Key Characteristics:**
- Decentralized actor execution: each signalized intersection makes independent phase decisions using shared policy
- Centralized critic: global observation (concatenated agent observations) for value estimation
- Environment abstraction: sumo-rl wrapper around SUMO traffic microsimulation
- Training/evaluation pipelines with pluggable reward modes and configurable baselines
- Observation normalization using Welford's online algorithm for stable training

## Layers

**Entry Points (Modeling Layer):**
- Purpose: CLI interfaces for training and evaluation workflows
- Location: `ece324_tango/modeling/train.py`, `ece324_tango/modeling/predict.py`
- Contains: Typer CLI apps, configuration parsing, backend selection
- Depends on: `LocalMappoBackend`, `TrainConfig`, `EvalConfig`
- Used by: Pixi tasks, direct Python module execution

**Backend Layer (Training/Evaluation):**
- Purpose: Orchestrate multi-agent training loop and baseline evaluation
- Location: `ece324_tango/asce/trainers/local_mappo_backend.py`
- Contains: Episode loop, environment interaction, trajectory collection, metric computation
- Depends on: `MAPPOTrainer`, baselines, SUMO environment, KPI tracking
- Used by: Modeling entry points

**Policy Layer (Multi-Agent MAPPO):**
- Purpose: Actor-critic networks with batch processing for multiple agents
- Location: `ece324_tango/asce/mappo.py`
- Contains: Actor (local policy), Critic (global value), PPO update logic, batch processing
- Depends on: PyTorch, observation normalization
- Used by: Backend for action selection and policy updates

**Environment Layer:**
- Purpose: SUMO environment creation and observation manipulation
- Location: `ece324_tango/asce/env.py`
- Contains: Environment factory, observation flattening/padding, default SUMO file resolution
- Depends on: sumo-rl package
- Used by: Backend for environment instantiation

**Baseline Controllers:**
- Purpose: Deterministic signal control policies for comparison
- Location: `ece324_tango/asce/baselines.py`
- Contains: FixedTimeController (uniform cycling), MaxPressureController (queue-differential optimization)
- Depends on: SUMO environment state
- Used by: Backend evaluation loop

**Metrics & Reward Layer:**
- Purpose: Compute KPIs and rewards from SUMO/agent state
- Location: `ece324_tango/asce/kpi.py`, `ece324_tango/asce/traffic_metrics.py`
- Contains: Vehicle-level time-loss tracking, person-weighted delays, fairness (Jain index), throughput
- Depends on: SUMO TraCI API, numpy
- Used by: Backend for episode metrics and reward signals

**Runtime Utilities:**
- Purpose: Handle environment API variations and helper calculations
- Location: `ece324_tango/asce/runtime.py`
- Contains: Step output normalization (legacy vs. Gymnasium), phase extraction, Jain index
- Depends on: None (numpy for calculations)
- Used by: Backend, baselines, trainers

**Observation Normalization:**
- Purpose: Online feature-wise standardization using Welford's algorithm
- Location: `ece324_tango/asce/obs_norm.py`
- Contains: `ObsRunningNorm` class with update/normalize/state persistence
- Depends on: numpy
- Used by: MAPPOTrainer during training and inference

**Data Layer:**
- Purpose: Schema validation and feature engineering on rollout data
- Location: `ece324_tango/asce/schema.py`, `ece324_tango/dataset.py`, `ece324_tango/features.py`
- Contains: ASCE dataset column definitions, rollout CSV schema validation, feature extraction (queue imbalance, peak detection)
- Depends on: pandas, pathlib
- Used by: Post-training analysis and reproducibility

**Visualization Layer:**
- Purpose: Generate interim report figures from training/evaluation artifacts
- Location: `ece324_tango/plots.py`
- Contains: Matplotlib figure rendering (training curves, KPI comparisons)
- Depends on: pandas, matplotlib
- Used by: `plot-interim-figures` pixi task

## Data Flow

**Training Episode Flow:**

1. Environment reset: `env.reset(seed=seed)` → per-agent observations
2. Observation padding: pad individual agent obs to target dimension
3. Normalization: apply `ObsRunningNorm` if enabled, update running stats
4. Batch action selection: single GPU call to actor for all agents → actions per agent + log-probabilities + values
5. Max-Pressure computation: compute pressure for all actions (for reward mode)
6. Environment step: `env.step(actions)` → next observations, rewards, dones
7. Metric computation: extract queue/speed/phase from per-agent obs, compute time-loss from vehicle IDs
8. Reward calculation: weighted sum of delay, throughput, fairness, residual (Max-Pressure residual)
9. GAE advantage computation: generalized advantage estimation with gamma=0.99, gae_lambda=0.95
10. Trajectory accumulation: append transition to per-agent trajectory lists
11. PPO update: minibatch replay with action masking per sample, clip_eps=0.2
12. Logging: episode metrics (mean reward, KPIs) → CSV

**Evaluation Episode Flow:**

1. Load trained MAPPO checkpoint with normalizer state
2. For each baseline (MAPPO, FixedTime, MaxPressure):
   - Reset environment
   - Per step: get action (MAPPO uses normalizer, baselines use local heuristics)
   - Track KPIs (time-loss, person-time-loss, vehicle delays, Jain index)
3. Aggregate over seeds → summary statistics (mean/std per controller)
4. Write results CSV

**State Management:**

- **Episode state:** observations, actions, rewards accumulated in trajectory dictionaries keyed by agent_id
- **Normalizer state:** running mean/variance per feature updated during training, saved/restored with model
- **Baseline state:** Fixed-Time has internal cursor (phase counter), Max-Pressure is stateless
- **Error state:** Non-fatal exceptions logged to `reports/results/error_events.jsonl` with once-keys for deduplication

## Key Abstractions

**TrainConfig / EvalConfig:**
- Purpose: Immutable configuration for training and evaluation
- Examples: `ece324_tango/asce/trainers/base.py`
- Pattern: Dataclass with required and optional fields; passed to backend.train/evaluate

**MAPPOTrainer:**
- Purpose: Actor-critic policy with batch action selection and PPO updates
- Examples: `ece324_tango/asce/mappo.py`
- Pattern: Stateful class with `act_batch()` for inference, `update()` for learning, normalizers optional

**AsceTrainerBackend:**
- Purpose: Abstract interface for training/evaluation implementations
- Examples: `ece324_tango/asce/trainers/base.py` (base), `ece324_tango/asce/trainers/local_mappo_backend.py` (implementation)
- Pattern: Base class with `train(cfg)` and `evaluate(cfg)` methods; factory in `ece324_tango/asce/trainers/factory.py`

**Transition:**
- Purpose: Record of a single agent's step outcome
- Examples: `ece324_tango/asce/mappo.py`
- Pattern: Dataclass with obs, action, reward, done, value, logprob, n_valid_actions

**RewardWeights / IntersectionMetrics:**
- Purpose: Structured reward/metric definitions
- Examples: `ece324_tango/asce/traffic_metrics.py`
- Pattern: Dataclasses for type safety; enables multi-mode reward functions

## Entry Points

**Training:**
- Location: `ece324_tango/modeling/train.py::main()`
- Triggers: `pixi run train-asce-toronto-demand` or direct module invocation
- Responsibilities: Parse CLI args, create LocalMappoBackend, instantiate TrainConfig, call backend.train()

**Evaluation:**
- Location: `ece324_tango/modeling/predict.py::main()`
- Triggers: `pixi run eval-asce-toronto-demand` or direct module invocation
- Responsibilities: Parse CLI args, load trained model, create EvalConfig, call backend.evaluate()

**Figure Generation:**
- Location: `ece324_tango/plots.py` (module-level, no app)
- Triggers: `pixi run plot-interim-figures`
- Responsibilities: Load train/eval CSVs, render 2x2 subplot figure, save to reports/figures/

**Data Validation:**
- Location: `ece324_tango/dataset.py::validate_schema()`
- Triggers: `pixi run validate-asce-schema`
- Responsibilities: Check rollout CSV has required ASCE_DATASET_COLUMNS

**Feature Engineering:**
- Location: `ece324_tango/features.py::build_asce_features()`
- Triggers: Direct Python call or future pixi task
- Responsibilities: Compute derived features (queue imbalance, arrival imbalance, peak detection) and write feature CSV

## Error Handling

**Strategy:** Graceful degradation with centralized once-per-error logging

**Patterns:**
- **SUMO TraCI failures:** FatalTraCIError caught; episode marked done if happens mid-simulation
- **Vehicle/lane lookups:** Missing vehicle type/departure/lane defaults applied; context + once_key logged to error_events.jsonl
- **Observation/action mismatches:** RuntimeError raised if reset returns empty obs or observation size exceeds target_dim
- **Invalid normalizer states:** ValueError raised if non-finite values or dimension mismatches detected
- **Missing SUMO files:** FileNotFoundError if net/route files not found and not provided explicitly

**Reporting:** `ece324_tango/error_reporting.py::report_exception()` writes JSON lines with timestamp, context, error_type, error message, and user-provided details.

## Cross-Cutting Concerns

**Logging:**
- Tool: loguru (configured in `ece324_tango/config.py`)
- Integration: writes to stderr, optionally integrates with tqdm progress bars
- Handlers: module-level logger in each source file

**Validation:**
- Input validation: CLI args via typer (auto type-coercion, bounds checking)
- Data contracts: ASCE_DATASET_COLUMNS enforced by `dataset.validate_schema()`
- Schema drift detection: observation size padding with warnings

**Device Resolution:**
- Abstraction: `LocalMappoBackend._resolve_device(device: str) → str`
- Behavior: "auto" → CUDA if available else CPU; else returns string as-is
- Used: trainer initialization and checkpoint device placement

**Action Masking:**
- Applied at sample collection: per-agent logits have invalid actions set to `-inf` before sampling
- Applied during PPO update: per-transition n_valid_actions mask enforced in clip ratio calculation
- Guarantees: invalid actions never sampled, loss computed only over valid action space

**Quiet Mode:**
- Context manager in `ece324_tango/asce/trainers/noise_control.py::quiet_output()`
- Suppresses third-party stdout/stderr and tqdm via environment variable and file descriptor manipulation
- Used: backend with `backend_verbose=False` parameter

---

*Architecture analysis: 2026-03-29*
