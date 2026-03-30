# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-29)

**Core value:** MAPPO must match or beat Max-Pressure on person-time-loss under nominal conditions and generalize better under irregular demand
**Current focus:** Phase 1 — Foundation Fixes + Scenario Pool

## Current Position

Phase: 1 of 5 (Foundation Fixes + Scenario Pool)
Plan: 0 of ? in current phase
Status: Ready to plan
Last activity: 2026-03-29 — Roadmap created; project initialized

Progress: [░░░░░░░░░░] 0%

## Performance Metrics

**Velocity:**
- Total plans completed: 0
- Average duration: -
- Total execution time: 0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| - | - | - | - |

**Recent Trend:**
- Last 5 plans: -
- Trend: -

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Init]: Action-level residual over imitation learning — residual lets MAPPO learn when to deviate, not just copy MP
- [Init]: Use demand studio abstractions headlessly — reuses teammate's tested code via `generate_scenario()`
- [Init]: Phase 1 + 2 parallelizable — no code dependency between foundation fixes and architecture work
- [Init]: Phase 3 is a hard convergence gate — do not start curriculum (Phase 4) until rolling mean ratio < 1.15x

### Pending Todos

None yet.

### Blockers/Concerns

- [Phase 1] MEDIUM confidence on headless demand path — `generate_scenario()` confirmed callable but not tested end-to-end with Pixi env after adding deps. First task: add dash/plotly/lxml/pyproj to pixi.toml and verify `.rou.xml` output. Fall back to manually authored files if this fails.
- [Phase 2] Joint log-probability correctness is highest-risk silent error — write unit test confirming zero gradient on phase head for gate=0 transitions before integration.
- [Phase 3] Wall-clock time for 200 episodes unknown — if >8 hours on RTX 4070, reduce to 100 episodes with tighter convergence criterion.

## Session Continuity

Last session: 2026-03-29
Stopped at: Roadmap and state initialized; no plans created yet
Resume file: None
