---
phase: 03-simulation-alignment
plan: 1-of-1
subsystem: simulation, demand-cli
tags: [fix, demand, truncation, pixi]
dependency_graph:
  requires: [01-curriculum-scenario-design]
  provides: [aligned-demand-files, truncation-bootstrap, generate-curriculum-task]
  affects: [04-baseline-convergence, 05-curriculum-training]
tech_stack:
  added: []
  patterns: [flow-end-buffer, truncation-bootstrap-from-critic]
key_files:
  created: []
  modified:
    - scripts/generate_curriculum.py
    - ece324_tango/asce/trainers/local_mappo_backend.py
    - pixi.toml
    - sumo/demand/curriculum/am_peak.rou.xml
    - sumo/demand/curriculum/pm_peak.rou.xml
    - sumo/demand/curriculum/midday_multimodal.rou.xml
    - sumo/demand/curriculum/demand_surge.rou.xml
decisions:
  - "300s buffer chosen for flow end times (5 min beyond sim end ensures no demand exhaustion)"
  - "FatalTraCIError bootstrap uses last padded obs when SUMO cannot provide next_obs"
metrics:
  duration: ~10 min
  completed: 2026-03-30
requirements: [FIX-01, FIX-02, DEM-01, DEM-02]
---

# Phase 3: Simulation Alignment + Headless Demand CLI Summary

Flow end times extended 300s past simulation_seconds with audit enforcement; FatalTraCIError treated as truncation (critic bootstrap) not terminal (value=0); pixi generate-curriculum task added.

## Requirements Completed

| Req | Description | Implementation |
|-----|-------------|----------------|
| FIX-01 | Flow duration buffer | `FLOW_END_BUFFER_S = 300` added to generate_curriculum.py; `simulation_end = spec.simulation_seconds + 300` passed to `generate_scenario()`; streetcar injection also uses buffered end; `_audit_scenario` now flags `end <= simulation_seconds` |
| FIX-02 | Truncation handling | FatalTraCIError catch in `train()` now sets `terminated=False, truncated=True` and builds `bootstrap_obs` from last padded observations; `build_batch` uses critic's value estimate for GAE bootstrapping |
| DEM-01 | Headless CLI pixi task | `generate-curriculum` task added to `pixi.toml` pointing to `scripts/generate_curriculum.py` |
| DEM-02 | Configurable demand parameters | Verified: `--list` shows all 4 scenarios; `--scenarios` accepts subset selection |

## Verification Results

### Flow End Time Verification

| Scenario | sim_seconds | Old flow end | New flow end | Status |
|----------|------------|-------------|-------------|--------|
| am_peak | 900 | 900 | 1200 | Correct |
| pm_peak | 900 | 900 | 1200 | Correct |
| midday_multimodal | 1200 | 1200 | 1500 | Correct |
| demand_surge | 1200 | 1200 | 1500 | Correct |

### Regeneration Output

All 4 scenarios regenerated successfully with audit PASS:
- am_peak: 176 vehicle flows, 31 ped flows, 6 streetcars
- pm_peak: 158 vehicle flows, 31 ped flows, 6 streetcars
- midday_multimodal: 179 vehicle flows, 31 ped flows, 8 streetcars
- demand_surge: 195 vehicle flows (incl 37 surge), 31 ped flows, 9 streetcars

### Test Suite

39 tests passed, 4 pre-existing failures (documented in deferred-items.md):
- test_action_gate_mappo.py: Phase 2 imports not yet implemented
- test_local_backend_bootstrap.py: _DummyEnv missing traffic_signals
- test_local_eval_fallback_observation_alignment.py: mock signature mismatch
- test_local_eval_objective_scoring.py: mock signature mismatch

No regressions caused by Phase 3 changes.

## Deviations from Plan

None -- plan executed exactly as written.

## Decisions Made

1. **300s buffer value**: 5 minutes ensures vehicles are still being injected well past any episode boundary; long enough to prevent demand exhaustion even with late-arriving vehicles, short enough to not massively inflate route file sizes.
2. **Bootstrap from last padded obs on FatalTraCIError**: When SUMO crashes, `next_obs` is unavailable. Using the last padded observations (current step's obs) as bootstrap input is the best available approximation; the critic's value estimate from these obs provides a meaningful non-zero bootstrap value.
3. **Surge flows excluded from audit timing check**: Surge flows have intentionally short time windows (e.g., 300-600s) that are below simulation_seconds by design.

## Commits

| Hash | Message |
|------|---------|
| c128587 | feat(03): simulation alignment -- flow buffer, truncation fix, pixi task |

## Known Stubs

None -- all implementations are complete and wired.

## Self-Check: PASSED

- All 7 modified files exist on disk
- All 4 regenerated .rou.xml files present
- SUMMARY.md created
- Commit c128587 verified in git log
