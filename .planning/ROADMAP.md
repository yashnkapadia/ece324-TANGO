# Roadmap: TANGO Milestone 2

## Overview

ASCE must close a 25% performance gap to Max-Pressure on person-time-loss. The intellectual core — what scenarios to train on and why — must come first, because it drives every downstream decision: how long flows run, what modes to include, what the CLI needs to produce, and how curriculum training is structured. In parallel with scenario design, the action-gate residual MAPPO architecture can be built since it has no dependency on demand specifics. Once both are ready, we validate convergence, run curriculum training, and finalize dataset logging for PIRA.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Curriculum Scenario Design** - Research TMC data, understand MP failure modes, design scenario portfolio with domain rationale, define what the demand CLI must produce
- [ ] **Phase 2: Action-Gate Residual MAPPO** - Two-head actor (gate + phase) with MP one-hot observation input and joint log-probability PPO update
- [ ] **Phase 3: Simulation Alignment + Headless Demand CLI** - Extend flow durations per scenario specs, fix truncation handling, build headless CLI guided by Phase 1 design
- [ ] **Phase 4: Baseline Convergence Validation** - 200+ episode training run confirming action-gate policy converges on fixed demand before curriculum
- [ ] **Phase 5: Curriculum Training Integration** - CurriculumManager with scenario sampling, per-episode env rebuild, proposal KPI target
- [ ] **Phase 6: Eval Loop + Dataset Logging** - Streaming Parquet writer for all 3 controllers across all scenarios, PIRA-compatible schema

## Phase Details

### Phase 1: Curriculum Scenario Design
**Goal**: A well-reasoned portfolio of demand scenarios exists with clear rationale for each, and the spec for what the demand CLI must produce is locked
**Depends on**: Nothing (frontloaded research phase)
**Requirements**: DEM-03, DEM-04
**Success Criteria** (what must be TRUE):
  1. TMC data has been explored to identify meaningful demand variation (time-of-day patterns, volume ranges, directional imbalances, mode splits)
  2. Max-Pressure's theoretical failure modes are documented (non-stationary demand, multimodal conflicts, demand surges, asymmetric loads)
  3. Each scenario has documented rationale explaining what MP assumption it violates or what generalization capability it tests
  4. Scenario specs define: time window, duration, demand scale, mode mix, and any capacity modifications — these specs drive Phase 3's CLI and flow durations
  5. At least 4 distinct demand regimes identified, with multimodal demand (cars, trucks, buses) where TMC data supports it
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
**Plans**: 4 plans

Plans:
- [ ] 02-01-PLAN.md — TDD: joint logp and gradient isolation tests (failing red tests first)
- [ ] 02-02-PLAN.md — GatedActor class, Transition extension, augment_obs_with_mp helper
- [ ] 02-03-PLAN.md — ResidualMAPPOTrainer with act_batch_residual, joint logp PPO update, TrainConfig residual_mode
- [ ] 02-04-PLAN.md — Backend wiring, gate_fraction logging, CLI --residual-mode flag

### Phase 3: Simulation Alignment + Headless Demand CLI
**Goal**: Simulation runs cleanly for all scenario durations and the demand CLI produces the exact .rou.xml files specified in Phase 1's scenario design
**Depends on**: Phase 1 (scenario specs drive CLI requirements and flow durations)
**Requirements**: FIX-01, FIX-02, DEM-01, DEM-02
**Success Criteria** (what must be TRUE):
  1. Flow durations in route files extend past simulation end time for each scenario; no FatalTraCIError from demand exhaustion
  2. If FatalTraCIError occurs mid-episode for other reasons, it is treated as truncation (bootstrap from critic) not terminal (value=0)
  3. Demand studio dependencies (dash, plotly, lxml, pyproj) installable via pixi
  4. Headless CLI produces valid .rou.xml files matching each scenario spec from Phase 1 — SUMO accepts them without errors
  5. Generated scenarios produce meaningfully different traffic patterns when simulated
**Plans**: TBD

### Phase 4: Baseline Convergence Validation
**Goal**: The action-gate policy demonstrates stable learning at 200+ episodes on a single fixed scenario before curriculum begins
**Depends on**: Phase 2 (action-gate architecture), Phase 3 (aligned demand files)
**Requirements**: FIX-03
**Success Criteria** (what must be TRUE):
  1. A training run of at least 200 episodes completes without crashing on the fixed demand file
  2. The rolling mean person-time-loss ratio (MAPPO / Max-Pressure) falls below 1.15 for at least 3 consecutive 10-episode windows
  3. Gate_fraction shows an increasing trend over training, confirming the policy learns when to override
**Plans**: TBD

### Phase 5: Curriculum Training Integration
**Goal**: MAPPO trains across varied demand scenarios and achieves the proposal's person-time-loss target on at least one evaluation scenario
**Depends on**: Phase 1 (scenario portfolio), Phase 4 (converged base policy)
**Requirements**: CUR-01, CUR-02, CUR-03
**Success Criteria** (what must be TRUE):
  1. TrainConfig accepts a list of route files and the training loop cycles through them across episodes
  2. The environment is rebuilt per episode with the selected scenario's route file
  3. Per-scenario person-time-loss ratios are tracked and logged across the curriculum pool
  4. Residual MAPPO achieves person-time-loss(MAPPO) <= 0.90 * person-time-loss(Max-Pressure) on at least one evaluation scenario
**Plans**: TBD

### Phase 6: Eval Loop + Dataset Logging
**Goal**: All three controllers are evaluated across all scenarios and their outputs are logged to a PIRA-compatible Parquet dataset
**Depends on**: Phase 5 (trained policy and scenario pool)
**Requirements**: DAT-01, DAT-02, DAT-03, DAT-04
**Success Criteria** (what must be TRUE):
  1. Running the evaluation loop produces a `dataset.parquet` file that can be read with `pd.read_parquet()` without error
  2. The Parquet file contains rows from all three controllers (mappo, fixed_time, max_pressure) with a `controller` column distinguishing them
  3. Each row includes `scenario_id` and all fields in `ASCE_DATASET_COLUMNS`
  4. All curriculum scenarios appear in the dataset for each controller
**Plans**: TBD

## Progress

**Execution Order:**
Phases 1 and 2 can be worked in parallel. Phase 3 depends on Phase 1. Phase 4 depends on Phases 2+3. Phase 5 depends on Phases 1+4. Phase 6 depends on Phase 5.

```
Phase 1 (Scenario Design) ──┬──→ Phase 3 (CLI + Alignment) ──┐
                              │                                  ├──→ Phase 4 (Convergence) ──→ Phase 5 (Curriculum) ──→ Phase 6 (Parquet)
Phase 2 (Action-Gate MAPPO) ─┘                                  │
                              └─────────────────────────────────┘
```

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Curriculum Scenario Design | 0/? | Not started | - |
| 2. Action-Gate Residual MAPPO | 0/4 | Planned | - |
| 3. Simulation Alignment + Headless Demand CLI | 0/? | Not started | - |
| 4. Baseline Convergence Validation | 0/? | Not started | - |
| 5. Curriculum Training Integration | 0/? | Not started | - |
| 6. Eval Loop + Dataset Logging | 0/? | Not started | - |
