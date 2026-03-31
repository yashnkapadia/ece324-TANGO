---
phase: 05-curriculum-training-integration
plan: 01
subsystem: training
tags: [mappo, curriculum, round-robin, multi-scenario, sumo]

requires:
  - phase: 04-convergence-gate
    provides: "Stable MAPPO training pipeline with parallel workers"
provides:
  - "TrainConfig.route_files for curriculum scenario lists"
  - "CLI --route-files flag for comma-separated route file input"
  - "Round-robin scenario cycling in both sequential and parallel training paths"
  - "Multi-scenario inline eval returning worst-case MAPPO/MP ratio"
  - "Per-episode scenario_id in metrics CSV"
affects: [05-02, training-runs, evaluation]

tech-stack:
  added: []
  patterns:
    - "Round-robin scenario assignment: scenario_pool[(ep_batch_start + i) % len(scenario_pool)]"
    - "Scenario ID derived from filename: Path(rf).stem.removesuffix('.rou')"
    - "Worst-case ratio for multi-scenario best-model checkpointing"

key-files:
  created:
    - tests/test_curriculum_config.py
  modified:
    - ece324_tango/asce/trainers/base.py
    - ece324_tango/modeling/train.py
    - ece324_tango/asce/trainers/local_mappo_backend.py

key-decisions:
  - "Scenario ID derived from route filename stem minus .rou suffix (e.g., am_peak.rou.xml -> am_peak)"
  - "Inline eval returns worst-case MAPPO/MP ratio across all curriculum scenarios for conservative checkpointing"
  - "Refactored _run_inline_eval into wrapper + _run_single_eval for multi-scenario support"

patterns-established:
  - "Curriculum config: route_files as list[str] with default_factory=list"
  - "Round-robin: index = (ep_batch_start + worker_idx) % len(scenario_pool)"

requirements-completed: [CUR-01, CUR-02]

duration: 4min
completed: 2026-03-31
---

# Phase 05 Plan 01: Curriculum Training Config & Round-Robin Summary

**TrainConfig accepts multiple route files via --route-files, training loop cycles scenarios round-robin, inline eval runs on all curriculum scenarios returning worst-case ratio**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-31T01:41:02Z
- **Completed:** 2026-03-31T01:45:10Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- TrainConfig.route_files field with default_factory=list and CLI --route-files comma-separated parsing
- Round-robin scenario cycling in both parallel (_train_parallel) and sequential training paths
- Multi-scenario inline eval with per-scenario MAPPO/MP ratio logging and worst-case return for best-model checkpointing
- 23 unit tests covering config construction, round-robin assignment (sequential + parallel), and scenario ID derivation
- Per-episode scenario_id in metrics CSV from worker results

## Task Commits

Each task was committed atomically:

1. **Task 1 (RED): Add failing tests** - `e4c4caa` (test)
2. **Task 1 (GREEN): Add route_files to TrainConfig and --route-files CLI** - `c7d31a9` (feat)
3. **Task 2: Training loop curriculum + multi-scenario eval** - `7351fe3` (feat)

_Task 1 followed TDD: RED (failing tests) then GREEN (implementation)._

## Files Created/Modified
- `ece324_tango/asce/trainers/base.py` - Added route_files: list[str] field to TrainConfig
- `ece324_tango/modeling/train.py` - Added --route-files CLI option with comma parsing and curriculum mode logging
- `ece324_tango/asce/trainers/local_mappo_backend.py` - Round-robin scenario_pool in parallel/sequential paths, _run_inline_eval multi-scenario wrapper, _run_single_eval extracted
- `tests/test_curriculum_config.py` - 23 tests for config, round-robin, and scenario ID derivation

## Decisions Made
- Scenario IDs derived from route file stems with .rou suffix stripped (Path.stem.removesuffix(".rou")) for clean labeling
- Inline eval returns worst-case (max) MAPPO/MP ratio across all scenarios for conservative best-model selection
- Refactored _run_inline_eval into a multi-scenario wrapper delegating to _run_single_eval for clean separation

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed double-extension scenario ID derivation**
- **Found during:** Task 1 (GREEN phase)
- **Issue:** Path(rf).stem only strips last extension (.xml), leaving .rou suffix (e.g., am_peak.rou instead of am_peak)
- **Fix:** Added .removesuffix(".rou") after .stem consistently in tests and backend code
- **Files modified:** tests/test_curriculum_config.py, ece324_tango/asce/trainers/local_mappo_backend.py
- **Verification:** test_scenario_id_derived_from_filename passes
- **Committed in:** c7d31a9, 7351fe3

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Necessary fix for correct scenario ID labeling. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Known Stubs
None - all data flows are wired.

## Next Phase Readiness
- Curriculum training code is ready; Plan 05-02 can generate demand files and launch training runs
- --route-files accepts comma-separated paths to any .rou.xml files
- Inline eval automatically covers all curriculum scenarios when route_files is set

---
*Phase: 05-curriculum-training-integration*
*Completed: 2026-03-31*
