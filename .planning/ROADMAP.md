# Roadmap: TANGO Milestone 2

## Overview

ASCE must close a 25% performance gap to Max-Pressure on person-time-loss. The path runs through five phases: fix simulation correctness bugs and build the scenario pool (Phase 1), implement action-gate residual MAPPO architecture (Phase 2), validate policy convergence at baseline before scaling up (Phase 3), train on curriculum scenarios to exploit Max-Pressure's failure modes (Phase 4), and finalize evaluation infrastructure with Parquet dataset logging for PIRA (Phase 5). Phases 1 and 2 can be worked in parallel; Phase 3 is a hard gate before curriculum begins.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Foundation Fixes + Scenario Pool** - Fix demand window bug, add pixi deps, generate 4+ curriculum scenario files
- [ ] **Phase 2: Action-Gate Residual MAPPO** - Two-head actor (gate + phase) with MP one-hot observation input and joint log-probability PPO update
- [ ] **Phase 3: Baseline Convergence Validation** - 200+ episode training run confirming policy converges with action-gate enabled before curriculum
- [ ] **Phase 4: Curriculum Training Integration** - CurriculumManager with round-robin scenario sampling, per-episode env rebuild, proposal KPI target
- [ ] **Phase 5: Eval Loop + Dataset Logging** - Streaming Parquet writer for all 3 controllers across all scenarios, PIRA-compatible schema

## Phase Details

### Phase 1: Foundation Fixes + Scenario Pool
**Goal**: Simulation correctness is restored and a scenario pool exists for curriculum training
**Depends on**: Nothing (first phase)
**Requirements**: FIX-01, FIX-02, DEM-01, DEM-02, DEM-03, DEM-04
**Success Criteria** (what must be TRUE):
  1. A training run on the fixed route file completes all 60 steps per episode without FatalTraCIError
  2. FatalTraCIError mid-episode (if it occurs) is treated as truncation with bootstrapped value, not as terminal with value=0
  3. Running `generate_demand.py` headlessly produces a valid `.rou.xml` file that SUMO accepts
  4. At least 4 `.rou.xml` scenario files exist covering AM peak, PM peak, off-peak, and incident/reduced-capacity demand profiles
  5. Multimodal demand (cars, trucks, buses) is present in the generated scenario files
**Plans**: TBD

### Phase 2: Action-Gate Residual MAPPO
**Goal**: The actor can learn when to follow or override Max-Pressure, with MP's recommendation visible as an explicit input
**Depends on**: Nothing (parallelizable with Phase 1)
**Requirements**: RES-01, RES-02, RES-03, RES-04, RES-05, RES-06
**Success Criteria** (what must be TRUE):
  1. Training with `residual_mode="action_gate"` completes without error and the gate_fraction metric is logged each episode
  2. When gate=0, the Max-Pressure action is executed; when gate=1, the actor's phase head action is executed
  3. The PPO update applies zero gradient to the phase head for gate=0 transitions (verifiable via unit test)
  4. The gate can be warm-started to bias toward follow-MP behavior in early training
  5. Setting `residual_mode="none"` produces identical behavior to the pre-Phase-2 baseline (backward compatibility)
**Plans**: TBD

### Phase 3: Baseline Convergence Validation
**Goal**: The action-gate policy demonstrates stable learning at 200+ episodes on a single fixed scenario before curriculum begins
**Depends on**: Phase 1 (fixed demand file), Phase 2 (action-gate architecture)
**Requirements**: FIX-03
**Success Criteria** (what must be TRUE):
  1. A training run of at least 200 episodes completes without crashing on the fixed demand file
  2. The rolling mean person-time-loss ratio (MAPPO / Max-Pressure) falls below 1.15 for at least 3 consecutive 10-episode windows
  3. Gate_fraction shows an increasing trend over training, confirming the policy learns when to override
**Plans**: TBD

### Phase 4: Curriculum Training Integration
**Goal**: MAPPO trains across varied demand scenarios and achieves the proposal's person-time-loss target on at least one evaluation scenario
**Depends on**: Phase 1 (scenario pool), Phase 3 (converged base policy)
**Requirements**: CUR-01, CUR-02, CUR-03
**Success Criteria** (what must be TRUE):
  1. TrainConfig accepts a list of route files and the training loop cycles through them across episodes
  2. The environment is rebuilt per episode with the selected scenario's route file
  3. Per-scenario person-time-loss ratios are tracked and logged across the curriculum pool
  4. Residual MAPPO achieves person-time-loss(MAPPO) <= 0.90 * person-time-loss(Max-Pressure) on at least one evaluation scenario
**Plans**: TBD

### Phase 5: Eval Loop + Dataset Logging
**Goal**: All three controllers are evaluated across all scenarios and their outputs are logged to a PIRA-compatible Parquet dataset
**Depends on**: Phase 4 (trained policy and scenario pool)
**Requirements**: DAT-01, DAT-02, DAT-03, DAT-04
**Success Criteria** (what must be TRUE):
  1. Running the evaluation loop produces a `dataset.parquet` file that can be read with `pd.read_parquet()` without error
  2. The Parquet file contains rows from all three controllers (mappo, fixed_time, max_pressure) with a `controller` column distinguishing them
  3. Each row includes `scenario_id` and all fields in `ASCE_DATASET_COLUMNS`
  4. All curriculum scenarios appear in the dataset for each controller
**Plans**: TBD

## Progress

**Execution Order:**
Phases 1 and 2 can be worked in parallel. Phase 3 requires both. Phase 4 requires Phase 3. Phase 5 requires Phase 4.

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Foundation Fixes + Scenario Pool | 0/? | Not started | - |
| 2. Action-Gate Residual MAPPO | 0/? | Not started | - |
| 3. Baseline Convergence Validation | 0/? | Not started | - |
| 4. Curriculum Training Integration | 0/? | Not started | - |
| 5. Eval Loop + Dataset Logging | 0/? | Not started | - |
