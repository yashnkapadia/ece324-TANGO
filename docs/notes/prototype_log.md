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
