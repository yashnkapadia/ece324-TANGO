---
phase: 02-action-gate-residual-mappo
plan: 04
subsystem: asce-backend-cli
tags: [residual-mappo, action-gate, backend-wiring, cli]
dependency_graph:
  requires: [02-03]
  provides: [backend-action-gate-train, backend-action-gate-eval, cli-residual-mode]
  affects: [04-01]
tech_stack:
  added: []
  patterns: [residual-mode-dispatch, effective-reward-mode, augmented-obs-normalizer]
key_files:
  created: []
  modified:
    - ece324_tango/asce/trainers/local_mappo_backend.py
    - ece324_tango/modeling/train.py
    - ece324_tango/modeling/predict.py
decisions:
  - "effective_reward_mode: action_gate supersedes residual_mp, falls back to objective"
  - "Augmented obs normalizer update in train loop when action_gate active"
  - "gate_fraction=0.0 emitted for residual_mode=none (backward-compatible column)"
  - "MP actions computed before env.step in both train and eval loops (Pitfall C5)"
metrics:
  duration: ~5 min
  completed: 2026-03-30
  tasks_completed: 2
  tasks_total: 2
  files_modified: 3
---

# Phase 2 Plan 4: Wire Action-Gate into Backend, CLI, and Eval Loop Summary

Backend integration layer making action-gate residual MAPPO runnable via --residual-mode action_gate on both train and eval CLIs, with gate_fraction logged per episode.

## What Was Built

### Task 1: Backend train loop wiring (f326416)

- **Trainer instantiation**: branches on `cfg.residual_mode` -- ResidualMAPPOTrainer for `action_gate`, plain MAPPOTrainer for `none`
- **Action step ordering**: mp_actions computed BEFORE env.step (Pitfall C5); `act_batch_residual` called with mp_actions_list for action_gate mode
- **Augmented obs normalizer**: when action_gate is active, obs_norm.update receives augmented observations (obs + MP one-hot) matching the ResidualMAPPOTrainer's augmented_dim
- **Transition fields**: gate and mp_action populated via `action_meta.get()` with default 0, making construction identical for both modes
- **gate_fraction metric**: computed from all transitions after each episode, emitted as 0.0 for non-residual mode
- **effective_reward_mode**: action_gate supersedes residual_mp reward mode (logs warning, switches to objective)
- Import of `ResidualMAPPOTrainer` and `augment_obs_with_mp` added to backend

### Task 2: CLI flags and eval gate_fraction (9e2c28e)

- **train.py**: `--residual-mode` CLI option with `_VALID_RESIDUAL_MODES` validation, passed through to TrainConfig
- **predict.py**: identical `--residual-mode` CLI option with validation, passed through to EvalConfig
- **Eval loop**: ResidualMAPPOTrainer instantiated for action_gate mode; `act_batch_residual` called in mappo controller branch
- **Eval gate tracking**: `ep_gate_total`/`ep_gate_steps` accumulated per episode; `gate_fraction` emitted in records dict
- **MP actions in eval**: computed before env.step for all controllers (Pitfall C5 preserved)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing functionality] Augmented obs normalizer update for action_gate**
- **Found during:** Task 1
- **Issue:** The obs normalizer in the train loop was updating with raw padded obs, but ResidualMAPPOTrainer's obs_norm expects augmented observations (obs_dim + n_actions) when residual_mode=action_gate
- **Fix:** Added conditional branch: for action_gate, augment obs with MP one-hot before normalizer update; for none mode, update with raw padded obs as before
- **Files modified:** ece324_tango/asce/trainers/local_mappo_backend.py
- **Commit:** f326416

## Verification

- All 50 tests pass (7 action-gate + 4 mappo-core + 39 others)
- `train --help` shows `--residual-mode` option
- `predict --help` shows `--residual-mode` option
- `gate_fraction` present in both train() ep_metrics and evaluate() records
- `act_batch_residual` present in both train() and evaluate() code paths
- residual_mode="none" path unchanged: no new computations except gate_fraction=0.0 in metrics

## Known Stubs

None. All code paths are fully wired and functional.

## Self-Check: PASSED

- FOUND: ece324_tango/asce/trainers/local_mappo_backend.py
- FOUND: ece324_tango/modeling/train.py
- FOUND: ece324_tango/modeling/predict.py
- FOUND: 02-04-SUMMARY.md
- FOUND: commit f326416
- FOUND: commit 9e2c28e
