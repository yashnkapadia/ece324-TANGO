# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-29)

**Core value:** MAPPO must match or beat Max-Pressure on person-time-loss under nominal conditions and generalize better under irregular demand
**Current focus:** Reward tuning + parallel training; v1 MAPPO/MP=0.961 at ep 74, v3 person_objective at 0.981 at ep 23

## Current Position

Phase: 4 of 6 (Baseline Convergence Validation) -- IN PROGRESS
Plan: Reward tuning experiments + parallel training infrastructure
Status: Two reward variants being compared; parallel training (8 workers) deployed
Last activity: 2026-03-30 — Reward tuning (v3 person_objective), parallel SUMO training, graceful checkpoint

Progress: [███████░░░] ~65%

## Performance Metrics

**Velocity:**
- Total plans completed: 5 (across phases 2-3)
- Average duration: ~7 min/plan
- Total execution time: ~35 min plans + ~4 hours debugging/fixing Phase 4 infra

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 02 | 4 | ~20 min | ~5 min |
| 03 | 1 | ~10 min | ~10 min |
| 04 | 0 (direct) | ~4 hrs | N/A (infra fixes) |

## Accumulated Context

### Decisions

- [Init]: Action-level residual over imitation learning — residual lets MAPPO learn when to deviate, not just copy MP
- [Init]: Phase 4 is a hard convergence gate — do not start curriculum (Phase 5) until rolling mean ratio < 1.15x
- [Phase 2]: effective_reward_mode: action_gate supersedes residual_mp, falls back to objective
- [Phase 3]: 300s buffer for flow end times prevents demand exhaustion near episode boundary
- [Phase 4]: Vendored sumo-rl — native TLS programs MUST be preserved; sumo-rl's phase replacement crashes SUMO on complex junctions
- [Phase 4]: Toronto vType defaults: 40 km/h maxSpeed (Dundas speed limit), not SUMO's 50 km/h — affects baseline comparison fairness
- [Phase 4]: NEMA baseline is the truest FT comparison — it's what Toronto actually runs on these intersections
- [Phase 4]: Streetcars skipped until network gets tram lanes via netconvert
- [Phase 4]: --ignore-route-errors removed — scenarios are clean after demand studio fixes + strict_route_check=True

### Pending Todos

- Regenerate scenarios with Toronto vTypes (40 km/h) after Phase 4 run finishes
- Add tram lanes to osm.net.xml for streetcar support (Phase 4.5)
- Demand studio has medium/low bugs remaining from audit (heading range gaps, turn boundary cases)

### Blockers/Concerns

- [Phase 4] v1 (objective) 200ep run still going in background — at ep 85, MAPPO/MP=0.961 at ep 74
- [Phase 4] v3 (person_objective) 200ep run with 8 workers just started — ETA ~2 hours
- [Phase 4] Batched training (N episodes per PPO update) changes training dynamics vs sequential — not directly comparable episode-for-episode

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 260330-k3l | Reward tuning: person-occupancy weighting, per-vehicle delay normalization, local approach fairness | 2026-03-30 | pending | [260330-k3l](./quick/260330-k3l-reward-tuning-person-occupancy-weighting/) |
| 260330-ods | Parallel SUMO training with multiprocessing and libsumo (3.4x speedup) | 2026-03-30 | 3f96c43, f4a4045 | [260330-ods](./quick/260330-ods-parallel-sumo-training-with-multiprocess/) |
| 260330-ras | Generate MP/FT baseline datasets in Parquet for PIRA, update ASCE schema | 2026-03-30 | pending | [260330-ras](./quick/260330-ras-generate-mp-ft-baseline-datasets-in-parq/) |

## Session Continuity

Last session: 2026-03-30
Stopped at: v3 person_objective 200ep restarted with parallel fixes; baseline datasets generating
Resume file: None
