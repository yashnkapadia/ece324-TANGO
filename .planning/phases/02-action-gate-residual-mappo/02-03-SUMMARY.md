---
phase: 02-action-gate-residual-mappo
plan: 03
subsystem: asce-mappo
tags: [residual-mappo, action-gate, ppo, joint-logp]
dependency_graph:
  requires: [02-02]
  provides: [ResidualMAPPOTrainer, TrainConfig.residual_mode, EvalConfig.residual_mode]
  affects: [02-04]
tech_stack:
  added: []
  patterns: [joint-logp-gate-mask, residual-mode-dispatch]
key_files:
  created: []
  modified:
    - ece324_tango/asce/mappo.py
    - ece324_tango/asce/trainers/base.py
    - tests/test_action_gate_mappo.py
decisions:
  - "gate_decisions batch key (not gate) for PPO update — matches test expectations"
  - "Entropy bonus is gate_entropy + gate_mask * phase_entropy (weighted by gate decision)"
  - "Categorical index 0 = override (gate=1), index 1 = follow MP (gate=0) — preserved from GatedActor"
metrics:
  duration: ~8 min
  completed: 2026-03-30
  tasks_completed: 2
  tasks_total: 2
  files_modified: 3
---

# Phase 2 Plan 3: ResidualMAPPOTrainer with Joint Logp PPO Update Summary

ResidualMAPPOTrainer subclasses MAPPOTrainer, adding act_batch_residual (gate+phase sampling with MP augmentation) and a PPO update that uses gate_mask to zero-gradient the phase_head on gate=0 transitions.

## What Was Built

### Task 1: ResidualMAPPOTrainer (c8b04d4)

- **ResidualMAPPOTrainer** class inherits MAPPOTrainer, accepts `residual_mode="none"|"action_gate"` and `gate_init_bias`
- `residual_mode="action_gate"`: replaces Actor with GatedActor (augmented obs_dim + n_actions input)
- `residual_mode="none"`: identical behavior to MAPPOTrainer (plain Actor, unchanged act_batch/update)
- **act_batch_residual**: augments obs with MP one-hot, forward through GatedActor, samples gate and phase independently, computes joint logp as `gate_logp + (gate==1).float() * phase_logp`, dispatches mp_action for gate=0 or phase_action for gate=1
- **update override**: loads `gate_decisions` from batch, converts semantic gate to Categorical indices, recomputes joint logp with gate_mask ensuring zero gradient on phase_head for gate=0 transitions
- **build_batch override**: appends `gate_decisions` and `mp_action` arrays from Transition fields
- **save**: includes `residual_mode` in checkpoint metadata
- **load**: validates checkpoint `residual_mode` matches trainer config; raises RuntimeError on mismatch

### Task 2: Config Fields (4c9916a)

- Added `residual_mode: str = "none"` to both TrainConfig and EvalConfig
- Full backward compatibility: all existing call sites work without change

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed test dimension mismatches in test_action_gate_mappo.py**
- **Found during:** Task 1
- **Issue:** Tests written in 02-01 assumed gate_head/phase_head accept raw obs_dim input, but GatedActor (from 02-02) uses a shared body so gate_head accepts hidden_dim input. Tests also assumed forward() returns a tensor (not tuple) and that phase_head is indexable with [-1].
- **Fix:** Updated 4 tests: (1) test_joint_logp_gate0 uses augmented obs through full forward path, (2) test_joint_logp_gate1 unpacks forward tuple and uses augmented obs, (3) test_gate0_phase_head_zero_gradient uses augmented obs dim (7 not 4), (4) test_gate1_dispatches uses phase_head.bias not phase_head[-1].bias, (5) test_gate_warm_start uses forward_gate() not gate_head() directly
- **Files modified:** tests/test_action_gate_mappo.py
- **Commit:** c8b04d4

## Verification

- All 7 tests in test_action_gate_mappo.py pass
- All 4 tests in test_mappo_core.py pass (no regression)
- 11/11 total tests green

## Known Stubs

None. All code paths are fully wired.

## Self-Check: PASSED

- FOUND: ece324_tango/asce/mappo.py
- FOUND: ece324_tango/asce/trainers/base.py
- FOUND: tests/test_action_gate_mappo.py
- FOUND: 02-03-SUMMARY.md
- FOUND: commit c8b04d4
- FOUND: commit 4c9916a
