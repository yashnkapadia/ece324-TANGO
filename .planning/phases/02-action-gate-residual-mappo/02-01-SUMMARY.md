---
phase: 02-action-gate-residual-mappo
plan: 01
subsystem: asce
tags: [tdd, red, action-gate, testing]
dependency_graph:
  requires: []
  provides: [action-gate-test-suite]
  affects: [ece324_tango/asce/mappo.py]
tech_stack:
  added: []
  patterns: [tdd-red-green-refactor, gradient-isolation-testing]
key_files:
  created:
    - tests/test_action_gate_mappo.py
  modified: []
decisions:
  - "Gate index convention: index 1 = gate=0 (follow MP), index 0 = gate=1 (use phase head) -- matches bias sign convention"
  - "gate_decisions key in batch dict to track per-transition gate choices for gradient isolation"
metrics:
  duration: ~1 min
  completed: 2026-03-30T03:30:59Z
---

# Phase 2 Plan 01: Action-Gate Joint Log-Probability Tests (TDD RED) Summary

Seven failing unit tests pinning joint log-probability semantics, gradient isolation, and action dispatch for the action-gate residual MAPPO architecture.

## What Was Done

### Task 1: Write 7 failing tests for GatedActor and ResidualMAPPOTrainer

Created `tests/test_action_gate_mappo.py` with 7 test functions:

1. **test_joint_logp_gate0_equals_gate_logp** -- When gate=0, joint logp equals only log P(gate=0); phase logp is NOT included.
2. **test_joint_logp_gate1_equals_sum** -- When gate=1, joint logp equals log P(gate=1) + log P(phase action).
3. **test_gate0_phase_head_zero_gradient** -- PPO update on gate=0-only batch produces zero gradient on phase head parameters.
4. **test_gate0_dispatches_mp_action** -- Gate=0 returns the Max-Pressure action regardless of phase head output.
5. **test_gate1_dispatches_phase_action** -- Gate=1 returns the phase head's chosen action, ignoring MP.
6. **test_gate_warm_start_biases_toward_zero** -- Negative gate_init_bias makes gate=0 dominant (>85% of samples).
7. **test_residual_mode_none_unchanged** -- residual_mode="none" behaves like vanilla MAPPOTrainer with no gate_head.

### Verification

All 7 tests fail with `ImportError: cannot import name 'GatedActor' from 'ece324_tango.asce.mappo'` -- confirming the RED phase is correct. No syntax errors, no unexpected failures.

## Commits

| Task | Commit | Message |
|------|--------|---------|
| 1 | bb9a190 | test(02-01): add failing action-gate joint logp tests |

## Deviations from Plan

None -- plan executed exactly as written.

## Known Stubs

None -- test file contains no stubs; all assertions are concrete.

## Design Decisions

- **Gate index convention**: Index 1 maps to gate=0 (follow MP), index 0 maps to gate=1 (use phase head). This allows negative `gate_init_bias` to push the softmax toward the "follow MP" index naturally.
- **Batch key `gate_decisions`**: Tests expect this numpy array in the batch dict so the update method can distinguish gate=0 vs gate=1 transitions for gradient masking.
- **`phase_head` attribute name**: Tests reference `trainer.actor.phase_head` as the sub-network whose gradients must be zero on gate=0 transitions.
- **`act_batch_residual` method**: Tests call this new method (not `act_batch`) with the additional `mp_actions_list` parameter.

## Self-Check: PASSED
