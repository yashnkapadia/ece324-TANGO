# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-29)

**Core value:** MAPPO must match or beat Max-Pressure on person-time-loss under nominal conditions and generalize better under irregular demand
**Current focus:** Phase 1 — Simulation Alignment + Headless Demand CLI

## Current Position

Phase: 1 of 6 (Simulation Alignment + Headless Demand CLI)
Plan: 0 of ? in current phase
Status: Ready to plan
Last activity: 2026-03-29 — Roadmap revised to 6 phases; curriculum design separated as Phase 3

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
- [Init]: Phase 1 + 2 parallelizable — no code dependency between sim alignment and architecture work
- [Init]: Phase 3 is a dedicated research/design phase for curriculum scenarios — demand scenario design requires domain understanding, not just CLI invocation
- [Init]: Phase 4 is a hard convergence gate — do not start curriculum (Phase 5) until rolling mean ratio < 1.15x
- [Revision]: Demand exhaustion is not a bug — it's a config mismatch (flow end time < sim end time); reframed as "simulation alignment"
- [Revision]: Curriculum scenario design needs explicit research — what demand patterns break MP's stationarity assumption requires TMC data exploration and domain reasoning

### Pending Todos

None yet.

### Blockers/Concerns

- [Phase 1] MEDIUM confidence on headless demand path — `generate_scenario()` confirmed callable but not tested end-to-end with Pixi env after adding deps. First task: add dash/plotly/lxml/pyproj to pixi.toml and verify `.rou.xml` output. Fall back to manually authored files if this fails.
- [Phase 2] Joint log-probability correctness is highest-risk silent error — write unit test confirming zero gradient on phase head for gate=0 transitions before integration.
- [Phase 3] Curriculum design is the intellectual core of the project — which scenarios, why, what MP assumptions they violate. Requires TMC data exploration and understanding demand studio parameters.
- [Phase 4] Wall-clock time for 200 episodes unknown — if >8 hours on RTX 4070, reduce to 100 episodes with tighter convergence criterion.

## Session Continuity

Last session: 2026-03-29
Stopped at: Roadmap revised to 6 phases; ready to plan Phase 1
Resume file: None
