# TANGO — Milestone 2

## What This Is

TANGO (Traffic Adaptive Network Guidance & Optimization) is a two-model traffic optimization system for a Toronto corridor. ASCE (Adaptive Signal Control Engine) uses multi-agent reinforcement learning (MAPPO) to control 8 signalized intersections in real time. PIRA (Planning Infrastructure Response Analyzer) will be a GNN surrogate for scenario planning. This milestone focuses on making ASCE competitive with Max-Pressure and building the data pipeline for PIRA.

## Core Value

MAPPO must match or beat Max-Pressure on person-time-loss under nominal conditions and generalize better under irregular demand — this is the proposal's success criterion and the foundation for PIRA and the final report.

## Requirements

### Validated

- ✓ ASCE end-to-end pipeline (train + eval) with MAPPO, Fixed-Time, Max-Pressure — existing
- ✓ TMC-calibrated demand on Dundas St W corridor (8 intersections, 70 flows) — existing
- ✓ Multi-objective reward (delay + throughput + fairness) — existing
- ✓ Observation normalization (Welford online per-feature) — existing
- ✓ Reward-level residual MAPPO mode (`residual_mp`) — existing
- ✓ KPI tracking (person-time-loss, trip time, throughput, Jain fairness) — existing
- ✓ Demand Studio app for TMC-based scenario generation — existing (data-setup branch)

### Active

- [ ] Action-level residual MAPPO: run Max-Pressure first, train MAPPO to output corrections
- [ ] Headless demand generation CLI using demand studio abstractions
- [ ] Curriculum training demand files (varied time-of-day, demand scale, multimodal)
- [ ] Train ASCE on curriculum scenarios to close the Max-Pressure gap
- [ ] Dataset.parquet logging from all 3 controllers for PIRA training data
- [ ] Achieve proposal target: person-time-loss(MAPPO) <= 0.90 * person-time-loss(Max-Pressure)

### Out of Scope

- PIRA GNN surrogate implementation — deferred to milestone 3
- Real-time deployment or hardware integration — academic project
- Pedestrian/cyclist signal optimization — vehicle-focused for now
- Alternative RL architectures (CityLight, IG-RL, CoLight) — stretch goals only

## Context

- ECE324 course project at University of Toronto, targeting IEEE ITSC 2026
- Team: Aryan Shrivastava, Kotaro Murakami, Yash Nishit Kapadia
- Current branch: `code-setup` (ASCE pipeline), `data-setup` (demand generation)
- Interim report submitted: MAPPO beats Fixed-Time but trails Max-Pressure by ~25%
- Training curve at 30 episodes suggests policy has not converged
- Max-Pressure is provably throughput-optimal under stationary single-commodity demand, which is exactly the current benchmark — curriculum training with varied/irregular demand is where MAPPO should gain advantage
- Demand Studio app (3031 lines, Dash/Plotly) pulled into code-setup; core function `generate_scenario()` can be called headlessly
- PIRA branch exists (`origin/PIRA`) with initial implementation requiring dataset.parquet

## Constraints

- **Compute**: RTX 4070 Laptop GPU — training must complete in reasonable time (<1 day per experiment)
- **Simulation**: SUMO FatalTraCIError when demand exhausts before sim end (~285s in 300s episodes) — must handle gracefully or extend demand
- **Timeline**: Final report deadline approaching — prioritize closing MP gap over stretch goals
- **Dependencies**: demand studio requires `dash`, `plotly`, `pyproj`, `lxml`, `sumolib` — some may need installation

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Action-level residual over imitation learning | Residual lets MAPPO selectively override MP only when beneficial, rather than blindly copying MP first | — Pending |
| Use demand studio abstractions headlessly | Avoids reinventing demand generation; reuses teammate's tested code | — Pending |
| Log dataset.parquet during eval for all controllers | PIRA needs labeled training data from multiple control strategies | — Pending |
| Curriculum training with varied demand | MP excels under stationary demand; irregular scenarios are where MAPPO should differentiate | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd:transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd:complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-03-29 after initialization*
