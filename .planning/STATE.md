# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-29)

**Core value:** MAPPO must match or beat Max-Pressure on person-time-loss under nominal conditions and generalize better under irregular demand
**Current focus:** Phase 4 training actively running; MAPPO/MP=0.968 at episode 24

## Current Position

Phase: 4 of 6 (Baseline Convergence Validation) -- IN PROGRESS
Plan: Phase 4 running (no formal plan created; direct execution)
Status: Training running, early convergence signal positive
Last activity: 2026-03-30 — Phase 4 infrastructure fixes + training launch

Progress: [██████░░░░] ~50%

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

- [Phase 4] Wall clock for 200 episodes ~5.5 hours on RTX 4070. Currently running.
- [Phase 4] MAPPO/MP already at 0.968 at ep 24 — may meet success criteria well before ep 200.
- [Phase 4] NEMA baseline not yet measured in current run (added after run started). Will appear in next run with --eval-every.

## Session Continuity

Last session: 2026-03-30
Stopped at: Phase 4 training running (~ep 25 of 200), all infra fixes committed
Resume file: None
