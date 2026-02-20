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
1. Implement native BenchMARL task adapter (replace fallback).
2. Implement native Xuance custom MARL env adapter (replace fallback).
3. Add integration smoke tests for backend-specific runs under optional test marks.

## 2026-02-20 (pixi workflow correction)
- Corrected dependency-management policy:
  - Use `pixi add` for conda dependencies.
  - Use `pixi add --pypi` for PyPI dependencies.
  - Use `pixi install` to sync/update the environment from manifest + lock.
- Explicitly banned `pixi run pip install ...` for project dependency management in runbook.
- Added `benchmarl` and `xuance` via `pixi add --pypi ...`, updating both `pixi.toml` and `pixi.lock`.
