---
phase: quick
plan: 260330-ods
subsystem: training
tags: [multiprocessing, sumo, mappo, parallel, libsumo]

provides:
  - "--num-workers N CLI flag for parallel SUMO episode collection"
  - "_run_episode_worker subprocess function with libsumo"
  - "_train_parallel method with spawn-context Pool"
affects: [train-asce-toronto-demand, local_mappo_backend]

tech-stack:
  added: [multiprocessing spawn context]
  patterns: [parallel data collection with centralized PPO update]

key-files:
  created: []
  modified:
    - ece324_tango/asce/trainers/base.py
    - ece324_tango/modeling/train.py
    - ece324_tango/asce/trainers/local_mappo_backend.py

key-decisions:
  - "spawn context (not fork) to avoid CUDA deadlock in subprocesses"
  - "maxtasksperchild=1 to prevent libsumo global state leaks"
  - "Workers run on CPU only; GPU used only in main process for PPO update"
  - "Per-episode GAE then batch concatenation (not cross-episode GAE)"
  - "obs_norm updated in main process from worker-collected raw observations"

requirements-completed: []

duration: 4min
completed: 2026-03-30
---

# Quick 260330-ods: Parallel SUMO Training with Multiprocess Summary

**Multiprocessing spawn Pool for N parallel SUMO episodes per PPO update, targeting ~Nx wall-clock speedup on simulation bottleneck**

## What Was Done

### Task 1: Add num_workers to TrainConfig and CLI
- Added `num_workers: int = 1` to `TrainConfig` dataclass
- Added `--num-workers` typer.Option to train.py CLI with validation (>= 1)
- Commit: `3f96c43`

### Task 2: Parallel episode collection in LocalMappoBackend
- Added `_run_episode_worker()` module-level function:
  - Sets LIBSUMO_AS_TRACI=1 and /usr/share/sumo/tools in sys.path
  - Creates own SumoEnvironment, MAPPOTrainer/ResidualMAPPOTrainer on CPU
  - Loads model state dict (actor, critic, obs_norm, gobs_norm)
  - Runs full episode loop identical to sequential path
  - Collects raw observations for main-process norm update
  - Returns pickle-safe dict with trajectories, last_values, metrics, raw obs
- Added `_train_parallel()` method:
  - Creates spawn-context Pool(num_workers, maxtasksperchild=1)
  - Serializes model state to CPU tensors each batch
  - Maps worker args via pool.map()
  - Updates obs_norm/gobs_norm from worker raw observations
  - Builds per-episode GAE batches then concatenates for merged PPO update
  - Handles checkpoint, eval_every, graceful interrupt per batch
- Sequential path (num_workers=1) completely untouched
- Commit: `f4a4045`

### Task 3: Human verification (checkpoint)
- Awaiting manual smoke testing

## Deviations from Plan

None - plan executed exactly as written.

## Known Stubs

None.

## Verification

- `TrainConfig` has `num_workers` field: PASSED
- `--num-workers` shows in CLI help: PASSED
- `_run_episode_worker` importable: PASSED
- All 49 tests pass (1 deselected): PASSED
- Manual smoke test (--num-workers 1 and --num-workers 2+): PENDING (checkpoint)

## Self-Check: PASSED

All files exist, all commits verified.
