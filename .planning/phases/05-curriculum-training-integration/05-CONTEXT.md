# Phase 5: Curriculum Training Integration - Context

**Gathered:** 2026-03-30
**Status:** Ready for planning

<domain>
## Phase Boundary

MAPPO trains across all 4 curriculum scenarios (am_peak, pm_peak, demand_surge, midday_multimodal) and achieves person-time-loss(MAPPO) <= 0.90 * person-time-loss(Max-Pressure) on at least one evaluation scenario.

This phase modifies the training pipeline to accept multiple route files. It does NOT add new scenarios, change the model architecture, or modify the eval/dataset pipeline (Phase 6).

</domain>

<decisions>
## Implementation Decisions

### Scenario Cycling (CUR-01, CUR-02)
- **D-01:** TrainConfig gets a `route_files: list[str]` field (replaces single `route_file` for curriculum mode). Single route_file kept for backward compat.
- **D-02:** Round-robin scenario assignment. With 8 workers per batch: each batch assigns 2 episodes per scenario (8 workers / 4 scenarios). Episode `i` uses `scenarios[i % len(scenarios)]`.
- **D-03:** Environment is rebuilt per episode with the selected scenario's route file. No persistent env across episodes.

### Reward Mode
- **D-04:** Use whichever reward mode (`objective` or `person_objective`) performs better in the Phase 4 comparison runs. If ambiguous, default to `objective` (proven MAPPO/MP=0.961 at ep 74).

### Warm-Start
- **D-05:** Curriculum starts from the best Phase 4 checkpoint via `--resume`. No cold-start. The base policy already beats MP on am_peak; curriculum fine-tunes generalization.
- **D-06:** Observation normalization carries forward from checkpoint. The normalizer will adapt to new scenarios' distributions during curriculum training.

### Eval Protocol (CUR-03)
- **D-07:** Inline eval every 25 episodes runs MAPPO vs MP vs FT on ALL 4 scenarios (not just current training scenario). This tracks per-scenario MAPPO/MP ratio during training.
- **D-08:** Final evaluation: 10 random seeds per scenario per controller (MAPPO, MP, FT, NEMA). Report mean +/- std for person-time-loss. This matches the proposal methodology.
- **D-09:** Success criterion: MAPPO/MP <= 0.90 on at least one scenario. Report all 4.

### Parallel Training
- **D-10:** Use `--num-workers 8` for curriculum runs. Each batch of 8 episodes covers all 4 scenarios twice.

### Claude's Discretion
- Episode count for curriculum (200-400 depending on convergence speed)
- Whether to also run NEMA baseline in inline eval (adds ~2min per eval point)
- Logging format for per-scenario metrics CSV

</decisions>

<canonical_refs>
## Canonical References

### Phase Requirements
- `.planning/REQUIREMENTS.md` — CUR-01, CUR-02, CUR-03 define the acceptance criteria

### Architecture
- `ece324_tango/asce/trainers/local_mappo_backend.py` — Training loop with parallel workers, inline eval, checkpoint/resume
- `ece324_tango/asce/trainers/base.py` — TrainConfig dataclass (needs route_files field)
- `ece324_tango/modeling/train.py` — CLI entry point (needs --route-files flag)

### Scenarios
- `sumo/demand/curriculum/am_peak.rou.xml`
- `sumo/demand/curriculum/pm_peak.rou.xml`
- `sumo/demand/curriculum/demand_surge.rou.xml`
- `sumo/demand/curriculum/midday_multimodal.rou.xml`

### Proposal
- `TANGO-proposal.tex` — KPI target: Delay_MAPPO <= 0.90 * Delay_Baseline (Section 2)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `_run_episode_worker()` already accepts `route_file` per episode — curriculum just varies this across workers in a batch
- `_train_parallel()` builds worker_args with per-episode config — route_file can be set per-worker
- `_run_inline_eval()` runs MAPPO/MP/FT/NEMA on a single scenario — needs loop over all scenarios
- Graceful checkpoint with SIGINT/SIGTERM already implemented

### Established Patterns
- TrainConfig is a frozen dataclass — add `route_files` alongside existing `route_file`
- CLI uses Typer Options — add `--route-files` as comma-separated string
- Parallel workers receive config as a dict — just swap `route_file` per worker

### Integration Points
- `train()` method: after `for ep_batch_start in range(...)`, select route_file for each worker from round-robin
- `_train_parallel()`: modify worker_args to include per-episode route_file
- `_run_inline_eval()`: loop over all route_files, run eval on each, log per-scenario metrics
- `num_workers=1` path: cycle route_file per episode

</code_context>

<specifics>
## Specific Ideas

- The 8-worker batch + 4 scenarios = exactly 2 episodes per scenario per batch. This is a clean division.
- Per-scenario metrics should be logged to the episode_metrics CSV with a `scenario_id` column so we can plot learning curves per scenario.
- The inline eval should print a table: `EVAL ep N: am_peak MAPPO/MP=X.XX, pm_peak MAPPO/MP=X.XX, ...`

</specifics>

<deferred>
## Deferred Ideas

- Weighted scenario sampling (e.g., more time on hard scenarios) — try uniform first, add weights if needed
- Phase 4.5 expansion scenarios (lane closure, safety-constrained, streetcar) — user deferred to Phase 7
- Scenario-specific reward weights — same weights across all scenarios for now

</deferred>

---

*Phase: 05-curriculum-training-integration*
*Context gathered: 2026-03-30*
