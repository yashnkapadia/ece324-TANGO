# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-29)

**Core value:** MAPPO must match or beat Max-Pressure on person-time-loss under nominal conditions and generalize better under irregular demand
**Current focus:** Phase 1 complete — verifying; Phase 2 planned (4 plans)

## Current Position

Phase: 2 of 6 (Action-Gate Residual MAPPO)
Plan: 1 of 4 complete
Status: Executing
Last activity: 2026-03-30 — TDD RED: 7 failing tests for action-gate joint logp semantics

Progress: [██░░░░░░░░] ~15%

## Performance Metrics

**Velocity:**
- Total plans completed: 0
- Average duration: -
- Total execution time: 0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 02 | 1 | ~1 min | ~1 min |

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
- [Init]: Phase 1 (design) + Phase 2 (architecture) parallelizable — no dependency between scenario research and action-gate code
- [Revision]: Frontload the intellectual work — scenario design drives everything: CLI shape, flow durations, training structure
- [Revision]: Demand exhaustion is not a bug — it's a config mismatch; flow durations depend on scenario specs from Phase 1
- [Init]: Phase 4 is a hard convergence gate — do not start curriculum (Phase 5) until rolling mean ratio < 1.15x

### Pending Todos

None yet.

### Blockers/Concerns

- [Phase 1] Curriculum design is the intellectual core — which scenarios, why, what MP assumptions they violate. Requires TMC data exploration and understanding demand studio capabilities/parameters.
- [Phase 2] Joint log-probability correctness is highest-risk silent error — write unit test confirming zero gradient on phase head for gate=0 transitions before integration.
- [Phase 3] MEDIUM confidence on headless demand path — `generate_scenario()` confirmed callable but not tested end-to-end. Blockers: pixi deps (dash/plotly/lxml/pyproj), NETWORK_CACHE population, 25-parameter defaults. Phase 1 design locks the CLI requirements.
- [Phase 4] Wall-clock time for 200 episodes unknown — if >8 hours on RTX 4070, reduce to 100 episodes with tighter convergence criterion.

## Session Continuity

Last session: 2026-03-30
Stopped at: Completed 02-01-PLAN.md (TDD RED — 7 failing action-gate tests)
Resume file: None
