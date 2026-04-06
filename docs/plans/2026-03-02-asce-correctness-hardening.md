# ASCE Correctness Hardening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix correctness bugs that can invalidate MAPPO training/evaluation conclusions, then lock behavior with targeted tests.

**Architecture:** Keep the existing backend split (`local_mappo`, `benchmarl`, `xuance`) and patch correctness at the lowest common layer first (MAPPO math + step semantics), then align backend eval behavior and CLI ergonomics. Use TDD for every fix and keep commits small.

**Tech Stack:** Python 3.12, PyTorch, SUMO-RL, pytest, pixi.

---

### Task 1: Add failing tests for MAPPO masked PPO consistency

**Files:**
- Modify: `tests/` (new test module, MAPPO-focused)
- Modify: `/home/as04/ece324-TANGO/ece324_tango/asce/mappo.py`

**Steps:**
1. Add a unit test that constructs logits with invalid actions and verifies:
   - sampling log-prob uses masked distribution;
   - update log-prob must be computed under the same mask.
2. Add a unit test checking entropy excludes invalid actions.
3. Run only these tests and verify they fail on current code.
4. Implement minimal mask plumbing in `MAPPOTrainer.update`.
5. Re-run tests and ensure pass.
6. Commit: `test+fix: enforce action-mask consistency in PPO update`.

### Task 2: Add failing tests for truncation-aware bootstrap and GAE

**Files:**
- Modify: `tests/` (new test module for runtime + local backend bootstrap flow)
- Modify: `/home/as04/ece324-TANGO/ece324_tango/asce/runtime.py`
- Modify: `/home/as04/ece324-TANGO/ece324_tango/asce/trainers/local_mappo_backend.py`
- Modify: `/home/as04/ece324-TANGO/ece324_tango/asce/mappo.py`

**Steps:**
1. Add a unit test proving timeout/truncation should bootstrap from `V(s_T)` instead of zero.
2. Add a unit test that true terminals still use zero bootstrap.
3. Add a runtime extraction test that preserves termination vs truncation signals.
4. Run tests and verify they fail on current code.
5. Implement minimal changes:
   - preserve step end-type (`terminated` vs `truncated`);
   - compute `last_values` on truncation only;
   - pass terminal-only mask into GAE recursion.
6. Re-run tests and ensure pass.
7. Commit: `test+fix: make GAE truncation-aware with critic bootstrap`.

### Task 3: Fix MAPPO eval seeding parity in Xuance and BenchMARL

**Files:**
- Modify: `/home/as04/ece324-TANGO/ece324_tango/asce/trainers/xuance_backend.py`
- Modify: `/home/as04/ece324-TANGO/ece324_tango/asce/trainers/benchmarl_backend.py`
- Modify: `tests/` (backend eval seed behavior tests with mocks)

**Steps:**
1. Add tests that confirm each eval episode uses `seed + ep` for MAPPO records.
2. Update Xuance eval episode runner to accept per-episode seed and use it.
3. Update BenchMARL eval to reinitialize/reset with per-episode seeds (or document deterministic path and fix reported seed semantics).
4. Re-run targeted tests.
5. Commit: `fix: align MAPPO eval seeding with reported episode seeds`.

### Task 4: Prevent obs-normalization mismatch at eval time

**Files:**
- Modify: `/home/as04/ece324-TANGO/ece324_tango/asce/mappo.py`
- Modify: `/home/as04/ece324-TANGO/ece324_tango/modeling/predict.py`
- Modify: `/home/as04/ece324-TANGO/pixi.toml`
- Modify: `tests/` (checkpoint/CLI mismatch tests)

**Steps:**
1. Add test for checkpoint metadata containing `use_obs_norm`.
2. Add test that eval errors/warns on mismatch between checkpoint and CLI flag.
3. Implement checkpoint metadata validation in load/eval path.
4. Add `--use-obs-norm` to Toronto eval task to match training defaults.
5. Run tests.
6. Commit: `fix: enforce obs-norm config parity between train and eval`.

### Task 5: Correct fallback observation alignment in local eval metrics path

**Files:**
- Modify: `/home/as04/ece324-TANGO/ece324_tango/asce/trainers/local_mappo_backend.py`
- Modify: `tests/test_traffic_metrics.py` or new local-eval fallback test

**Steps:**
1. Add test for fallback metric extraction using pre-step observation.
2. Patch eval loop to retain `prev_obs` for fallback before `env.step`.
3. Re-run tests.
4. Commit: `fix: use pre-step obs for metric fallback in eval`.

### Task 6: Add regression coverage for critical MAPPO math paths

**Files:**
- Add: `tests/test_mappo_core.py` (or similarly named module)

**Steps:**
1. Add tests for:
   - action masking in `act` and `act_batch`;
   - `build_batch` per-agent aggregation;
   - GAE with terminal vs truncated endpoints;
   - load/save with obs norm state.
2. Run fast suite.
3. Commit: `test: add MAPPO core regression coverage`.

### Task 7: Verify end-to-end and update documentation

**Files:**
- Modify: `/home/as04/ece324-TANGO/docs/notes/runbook.md`
- Modify: `/home/as04/ece324-TANGO/docs/notes/prototype_log.md`

**Steps:**
1. Run: `pixi run pytest tests -q`.
2. If available, run one short smoke train/eval for `local_mappo` to confirm no behavioral break.
3. Update runbook/prototype notes with fixed semantics (masking, truncation bootstrap, seed handling, obs-norm parity).
4. Commit: `docs: record correctness fixes and verification evidence`.

### Recommended commit order
1. Task 1
2. Task 2
3. Task 3
4. Task 4
5. Task 5
6. Task 6
7. Task 7

### Verification checklist
1. `pixi run pytest tests -q`
2. Optional focused run: `pixi run python -m ece324_tango.modeling.train --trainer-backend local_mappo --episodes 1 --seconds 30 --delta-time 5 --device cpu --model-path /tmp/asce_fix.pt --rollout-csv /tmp/asce_fix_rollout.csv --episode-metrics-csv /tmp/asce_fix_train.csv`
3. Optional focused eval: `pixi run python -m ece324_tango.modeling.predict --trainer-backend local_mappo --episodes 1 --seconds 30 --delta-time 5 --device cpu --model-path /tmp/asce_fix.pt --out-csv /tmp/asce_fix_eval.csv`
