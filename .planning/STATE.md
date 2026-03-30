# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-29)

**Core value:** MAPPO must match or beat Max-Pressure on person-time-loss under nominal conditions and generalize better under irregular demand
**Current focus:** Phase 3 complete; Phase 2 in progress (1/4 plans); Phase 4 next after Phase 2

## Current Position

Phase: 2 of 6 (Action-Gate Residual MAPPO) + Phase 3 complete
Plan: 3/4 Phase 2 in progress; Phase 3 done (1/1)
Status: Executing Phase 2; Phase 3 complete
Last activity: 2026-03-30 — Plan 02-03: ResidualMAPPOTrainer with joint logp PPO update

Progress: [████░░░░░░] ~35%

## Performance Metrics

**Velocity:**
- Total plans completed: 0
- Average duration: -
- Total execution time: 0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 02 | 1 | ~1 min | ~1 min |
| 03 | 1 | ~10 min | ~10 min |

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
- [Phase 3]: 300s buffer for flow end times prevents demand exhaustion near episode boundary
- [Phase 3]: FatalTraCIError is truncation (bootstrap from critic), not terminal (value=0)

### Pending Todos

None yet.

### Blockers/Concerns

- [Phase 1] Curriculum design is the intellectual core — which scenarios, why, what MP assumptions they violate. Requires TMC data exploration and understanding demand studio capabilities/parameters.
- [Phase 2] RESOLVED — Joint logp implemented and verified: gate_mask zeros phase_logp for gate=0 transitions, all 7 tests pass including zero-gradient check.
- [Phase 3] RESOLVED — headless demand path works end-to-end; pixi deps installed; all 4 scenarios regenerated with buffer
- [Phase 4] Wall-clock time for 200 episodes unknown — if >8 hours on RTX 4070, reduce to 100 episodes with tighter convergence criterion.

## Session Continuity

Last session: 2026-03-30
Stopped at: Completed 02-03-PLAN.md (ResidualMAPPOTrainer with joint logp PPO update)
Resume file: None
