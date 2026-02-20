# Prototype Log

## 2026-02-19
- Implemented Phase 0 scaffolding under `ece324_tango` package.
- Added pixi tasks for ASCE train/eval/schema validation.
- Implemented Phase 1 minimal ASCE MAPPO pipeline:
  - shared actor MLP and centralized critic,
  - sample SUMO network bootstrap via sumo-rl package data,
  - fixed-time and queue-greedy baseline eval script,
  - rollout logging to the agreed dataset schema.
- Added schema tests in `tests/test_data.py`.
- Added GPU device flag support in training/eval (`--device auto|cuda|cpu`).
- Migrated pixi GPU setup to conda-forge style with `pytorch-gpu`, `cuda-version = 12.8.*`, and `[system-requirements] cuda = \"12\"`.
- Verified in pixi env: `torch.cuda.is_available() == True` on RTX 4070 Laptop GPU.
- Replaced max-pressure proxy baseline with edge-level max-pressure controller using controlled TLS links and per-lane halting counts from TraCI.

## Follow-up
- Replace proxy max-pressure with true edge-level pressure from traci edge graph.
- Replace proxy NS/EW queue/arrival/speed values with direct per-approach aggregations.
- Add person-weighted delay and timeLoss extraction for proposal-aligned KPI reporting.

## 2026-02-19 (backend abstraction pass)
- Added trainer backend architecture under `ece324_tango/asce/trainers/`.
- Rewired `ece324_tango/modeling/train.py` and `ece324_tango/modeling/predict.py` to use backend selection:
  - `trainer_backend=local_mappo|benchmarl|xuance`
- Added runtime utility helpers in `ece324_tango/asce/runtime.py`.
- Added spike backend modules for BenchMARL/Xuance with package gating and parity fallback to local pipeline.
- Added test coverage for backend factory selection, max-pressure action validity, and device resolution.
- Added strict handoff memory docs:
  - `docs/notes/runbook.md`
  - `docs/notes/adr/ADR-0001-backend-strategy.md`

## Next 3 tasks
1. Implement native Xuance custom MARL env adapter (replace fallback).
2. Add integration smoke tests for backend-specific runs under optional test marks.
3. Reduce BenchMARL runtime overhead and noisy logging in sample-network mode.

## 2026-02-20 (pixi workflow correction)
- Corrected dependency-management policy:
  - Use `pixi add` for conda dependencies.
  - Use `pixi add --pypi` for PyPI dependencies.
  - Use `pixi install` to sync/update the environment from manifest + lock.
- Explicitly banned `pixi run pip install ...` for project dependency management in runbook.
- Added `benchmarl` and `xuance` via `pixi add --pypi ...`, updating both `pixi.toml` and `pixi.lock`.

## 2026-02-20 (native BenchMARL adapter)
- Replaced BenchMARL fallback with a native path:
  - Added `SumoParallelAdapter` + `SumoBenchmarlTask` under `ece324_tango/asce/trainers/benchmarl_task.py`.
  - BenchMARL backend now runs `benchmarl` MAPPO on SUMO through TorchRL PettingZoo wrapper.
  - BenchMARL train now saves a backend-tagged checkpoint payload and writes rollout/schema CSV + episode metrics CSV.
  - BenchMARL eval now loads checkpoint payload, evaluates MAPPO policy, and reports alongside fixed-time and max-pressure baselines.
- Added adapter unit tests in `tests/test_benchmarl_adapter.py`.
- Smoke validated:
  - `train --trainer-backend benchmarl` succeeds on sample network.
  - `predict --trainer-backend benchmarl` succeeds and writes metrics CSV.

## Remaining from plan
1. Add backend-marked integration tests (optional/slow marker) for benchmarl and xuance train/eval CLI runs.
2. Reduce BenchMARL training noise/runtime overhead and tighten env close lifecycle handling.
3. Improve Xuance adapter robustness so value normalization/GAE can be safely re-enabled for stronger baseline parity.

## 2026-02-20 (native Xuance adapter)
- Added OpenMPI to pixi environment to satisfy Xuance runtime dependency (`libmpi.so`).
- Replaced Xuance fallback with native path:
  - Added custom SUMO environment registration for Xuance in `ece324_tango/asce/trainers/xuance_env.py`.
  - Implemented native Xuance MAPPO train/eval backend in `ece324_tango/asce/trainers/xuance_backend.py`.
  - Training now exports both a direct model artifact (`models/*.pt`) and Xuance-compatible checkpoint folder (`*_xuance/seed_export`).
  - Evaluation loads the Xuance checkpoint folder and reports MAPPO against fixed-time and max-pressure baselines.
- Validated via short smoke runs on sample SUMO network for:
  - `--trainer-backend xuance` training CLI.
  - `--trainer-backend xuance` evaluation CLI.

## 2026-02-20 (integration tests)
- Added backend integration test module: `tests/test_backend_integration_slow.py`.
- Coverage includes both `benchmarl` and `xuance` CLI training/evaluation smoke paths.
- Tests are marked `slow` + `integration` and gated by env var:
  - run with `RUN_SLOW_INTEGRATION=1 pixi run pytest tests/test_backend_integration_slow.py`
  - default `pixi run pytest tests` remains fast (integration tests skip unless enabled).

## 2026-02-20 (noise controls + Xuance re-check)
- Added config-gated backend noise control:
  - CLI flag: `--backend-verbose` (default quiet mode).
  - Backends now pass quiet SUMO settings by default (`sumo_warnings=False`, `--no-step-log true`).
  - Added helper: `ece324_tango/asce/trainers/noise_control.py`.
- Re-checked Xuance settings matrix on SUMO adapter:
  - `use_gae=True, use_value_norm=False`: stable.
  - `use_value_norm=True` (with or without GAE): unstable shape/type failures in Xuance on-policy buffer path.
- Updated Xuance backend defaults accordingly and exposed env-var gates:
  - `TANGO_XUANCE_USE_GAE` (default `1`)
  - `TANGO_XUANCE_USE_VALUE_NORM` (now default `1` with local compat patch)
  - `TANGO_XUANCE_USE_ADVNORM` (default `0`)

## 2026-02-20 (local Xuance compat patch + fair compare)
- Added local runtime patch for Xuance on-policy buffer value normalization shape handling:
  - `ece324_tango/asce/trainers/xuance_compat.py`
  - toggle: `TANGO_XUANCE_PATCH_VALUE_NORM` (default `1`)
- Verified Xuance train works with `TANGO_XUANCE_USE_VALUE_NORM=1` and `TANGO_XUANCE_USE_GAE=1`.
- Fair micro-benchmark (cpu, 1 episode, 30s sim, 1 eval episode):
  - BenchMARL: train 66.267s, eval 37.809s, MAPPO mean_reward -0.000208
  - Xuance (patched): train 41.256s, eval 41.003s, MAPPO mean_reward -0.000208

## 2026-02-20 (Xuance custom-config path, no MPE bootstrap)
- Switched Xuance backend config load from framework MPE default bootstrap to project-owned config file:
  - `ece324_tango/asce/trainers/configs/xuance_mappo_sumo.yaml`
  - backend now calls `get_arguments(..., config_path=<project yaml>)`.
- Kept runtime overrides for scenario-specific values (seed, device, SUMO files/timing, quiet mode).
- Revalidated with smoke runs:
  - `train --trainer-backend xuance --episodes 1 --seconds 30 --delta-time 5`
  - `predict --trainer-backend xuance --episodes 1 --seconds 30 --delta-time 5`

## 2026-02-20 (objective reward + TraCI metrics)
- Added shared traffic metrics/reward module: `ece324_tango/asce/traffic_metrics.py`.
  - Computes per-intersection metrics from TraCI controlled links and edge-level aggregations.
  - Falls back to observation-based proxy splitting if TraCI edge mapping fails.
- Added objective reward mode with config gates:
  - CLI: `--reward-mode objective|sumo`
  - Weights: `--reward-delay-weight`, `--reward-throughput-weight`, `--reward-fairness-weight`
- Wired objective reward into:
  - local MAPPO train/eval loops,
  - Xuance custom env wrapper,
  - BenchMARL SUMO adapter,
  - baseline comparisons in backend eval paths.
- Local and Xuance rollout logging now uses TraCI-derived metrics path (with fallback).

## 2026-02-20 (objective-mode backend micro-compare)
- Objective-mode micro run (1 episode, 30s, delta=5, cpu):
  - BenchMARL train: 41.86s, eval: 26.13s
  - Xuance train: 25.98s, eval: 35.17s
- Eval outputs were numerically aligned between BenchMARL and Xuance in this micro run:
  - MAPPO mean_reward: `0.116181`
  - fixed_time mean_reward: `0.116181`
  - max_pressure mean_reward: `0.132357`
- Note: in this sandbox, both BenchMARL and Xuance required unsandboxed execution for some runs due `/dev/shm` and network permission constraints.

## 2026-02-20 (benchmarl rollout realism + multi-seed compare)
- BenchMARL rollout export path upgraded:
  - replaced observation-proxy CSV conversion with action-replay in raw SUMO env + TraCI metric extraction.
  - file: `ece324_tango/asce/trainers/benchmarl_backend.py` (`_rollout_to_schema_rows_from_replay`).
- Verified BenchMARL rollout CSV now reports non-placeholder fields (`avg_speed_*`, `current_phase`) consistent with TraCI path.
- Multi-seed objective-mode compare (backend in {benchmarl, xuance}, seeds {7,17,27}, episodes=1, seconds=30, delta=5, cpu):
  - BenchMARL mean: train 49.377s, eval 27.468s, mean_reward 0.118996, throughput_proxy 11.423643, fairness 0.188181
  - Xuance mean: train 26.083s, eval 39.349s, mean_reward 0.121229, throughput_proxy 11.638025, fairness 0.187422
- Raw summary artifact: `/tmp/tango_backend_sweep/summary.csv`

## 2026-02-20 (benchmark automation)
- Added reproducible benchmark CLI:
  - module: `ece324_tango/modeling/benchmark_backends.py`
  - pixi task: `pixi run benchmark-backends`
- CLI writes per-seed and aggregate outputs under:
  - `reports/results/backend_compare/<run_id>/summary.csv`
  - `reports/results/backend_compare/<run_id>/aggregate.csv`

## Next 3 tasks (proposal-aligned)
1. Run 10-seed ASCE significance evaluation using newly added proposal KPI primitives (`timeLoss`, occupancy-weighted person delay, per-trip travel time) against max-pressure/fixed-time.
2. Integrate Toronto corridor SUMO assets + TMC calibration into canonical dataset generation command (scenario IDs: baseline/construction/transit) with schema-complete exports.
3. Start PIRA data pipeline: generate scenario-level labels from trained ASCE rollouts and implement first GNN surrogate baseline with MAPE/R2/inference tracking.

## 2026-02-20 (proposal KPI extraction)
- Added KPI tracker module: `ece324_tango/asce/kpi.py`
  - network `timeLoss` accumulation from TraCI (`vehicle.getTimeLoss` deltas),
  - occupancy-weighted person delay (`person_time_loss_s`),
  - average trip time for arrived vehicles (`avg_trip_time_s`).
- Wired KPI fields into eval records for:
  - `local_mappo`,
  - `xuance`,
  - `benchmarl` (including rollout-action replay path).

## 2026-02-20 (LibSignal backend assessment)
- Reviewed LibSignal codebase (`run.py`, `trainer/tsc_trainer.py`, registry/config pipeline) for backend fit.
- Added `libsignal` backend name to factory as an explicit placeholder:
  - file: `ece324_tango/asce/trainers/libsignal_backend.py`
  - behavior: fail-fast runtime error with integration blocker context.
- Added assessment note:
  - `docs/notes/libsignal_backend_assessment.md`
  - decision: treat LibSignal as deferred backend until adapter can map our `net_file`/`route_file` and schema/KPI outputs.
