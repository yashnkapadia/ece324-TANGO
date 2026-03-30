# Requirements: TANGO Milestone 2

**Defined:** 2026-03-29
**Core Value:** MAPPO must match or beat Max-Pressure on person-time-loss under nominal conditions and generalize better under irregular demand.

## v1 Requirements

### Foundation Fixes

- [ ] **FIX-01**: Demand route files extend beyond simulation end time to prevent FatalTraCIError from demand exhaustion
- [ ] **FIX-02**: FatalTraCIError mid-episode is treated as truncation (bootstrap from critic) not terminal (value=0)
- [ ] **FIX-03**: Training runs extended to 200+ episodes to allow policy convergence before curriculum

### Residual MAPPO Architecture

- [ ] **RES-01**: Max-Pressure action encoded as one-hot and appended to actor observation at each step
- [ ] **RES-02**: Actor has binary override gate head (gate logit) alongside existing phase head
- [ ] **RES-03**: When gate=0 (follow MP), the MP action is executed; when gate=1 (override), the phase head action is used
- [ ] **RES-04**: PPO update uses joint log-probability: log P(gate) + (gate==1) * log P(phase), with phase head contributing zero gradient when gate=0
- [ ] **RES-05**: Gate can be warm-started (bias toward gate=0 / follow MP) to stabilize early training
- [ ] **RES-06**: Evaluation logs gate_fraction metric (fraction of steps where MAPPO overrides MP)

### Demand Generation

- [ ] **DEM-01**: Headless CLI wrapper around demand studio's generate_scenario() callable without Dash UI
- [ ] **DEM-02**: CLI generates demand .rou.xml files with configurable time-of-day, demand scale, and mode mix
- [ ] **DEM-03**: At least 4 curriculum scenarios generated: AM peak, PM peak, off-peak, incident/reduced capacity
- [ ] **DEM-04**: Multimodal demand included (cars, trucks, buses) using TMC data

### Curriculum Training

- [ ] **CUR-01**: TrainConfig accepts list of route files for curriculum training
- [ ] **CUR-02**: Training loop cycles through route files across episodes (round-robin or scheduled)
- [ ] **CUR-03**: Residual MAPPO trained on curriculum achieves person-time-loss(MAPPO) <= 0.90 * person-time-loss(Max-Pressure) on at least one evaluation scenario

### Dataset Logging

- [ ] **DAT-01**: Evaluation loop logs step-level data to dataset.parquet using streaming PyArrow writer
- [ ] **DAT-02**: Parquet schema includes controller column (mappo/fixed_time/max_pressure)
- [ ] **DAT-03**: Parquet schema matches ASCE_DATASET_COLUMNS plus controller and scenario_id fields
- [ ] **DAT-04**: All 3 controllers (MAPPO, Fixed-Time, Max-Pressure) produce labeled rows in the same dataset

## v2 Requirements

### PIRA Integration

- **PIRA-01**: GNN surrogate trained on dataset.parquet to predict scenario-level metrics
- **PIRA-02**: PIRA inference under 5 seconds per scenario
- **PIRA-03**: PIRA MAPE <= 10% on held-out scenarios

### Advanced Features

- **ADV-01**: Pressure features appended to observation vector (PressLight-style)
- **ADV-02**: Neighbor observation concatenation for corridor coordination
- **ADV-03**: Zero-shot transfer evaluation on a second Toronto corridor

## Out of Scope

| Feature | Reason |
|---------|--------|
| GNN/attention architecture rewrite (CoLight, CityLight) | On 8-intersection linear corridor, adds complexity without proportional benefit |
| Pedestrian/cyclist signal phases | Vehicle-focused for milestone 2; TMC ped data available for future |
| Real-time deployment | Academic project; SUMO simulation only |
| Imitation learning pretraining | Residual approach is strictly better — learns when to deviate, not just to copy |
| Alternative RL algorithms (QMIX, SAC) | MAPPO validated in literature for this domain; switching adds risk |

## Traceability

| Requirement | Phase | Phase Name | Status |
|-------------|-------|------------|--------|
| FIX-01 | Phase 1 | Simulation Alignment + Headless Demand CLI | Pending |
| FIX-02 | Phase 1 | Simulation Alignment + Headless Demand CLI | Pending |
| FIX-03 | Phase 4 | Baseline Convergence Validation | Pending |
| RES-01 | Phase 2 | Action-Gate Residual MAPPO | Pending |
| RES-02 | Phase 2 | Action-Gate Residual MAPPO | Pending |
| RES-03 | Phase 2 | Action-Gate Residual MAPPO | Pending |
| RES-04 | Phase 2 | Action-Gate Residual MAPPO | Pending |
| RES-05 | Phase 2 | Action-Gate Residual MAPPO | Pending |
| RES-06 | Phase 2 | Action-Gate Residual MAPPO | Pending |
| DEM-01 | Phase 1 | Simulation Alignment + Headless Demand CLI | Pending |
| DEM-02 | Phase 1 | Simulation Alignment + Headless Demand CLI | Pending |
| DEM-03 | Phase 3 | Curriculum Scenario Design | Pending |
| DEM-04 | Phase 3 | Curriculum Scenario Design | Pending |
| CUR-01 | Phase 5 | Curriculum Training Integration | Pending |
| CUR-02 | Phase 5 | Curriculum Training Integration | Pending |
| CUR-03 | Phase 5 | Curriculum Training Integration | Pending |
| DAT-01 | Phase 6 | Eval Loop + Dataset Logging | Pending |
| DAT-02 | Phase 6 | Eval Loop + Dataset Logging | Pending |
| DAT-03 | Phase 6 | Eval Loop + Dataset Logging | Pending |
| DAT-04 | Phase 6 | Eval Loop + Dataset Logging | Pending |

**Coverage:**
- v1 requirements: 20 total
- Mapped to phases: 20
- Unmapped: 0

---
*Requirements defined: 2026-03-29*
*Last updated: 2026-03-29 after roadmap creation*
