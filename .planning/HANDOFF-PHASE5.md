# Handoff Notes for Phase 5+ (Curriculum Training Integration)

**Date:** 2026-03-30
**From:** Session that did reward tuning, parallel training, PIRA dataset gen
**To:** Fresh agent starting Phase 5 planning + execution

## Read These First

1. `.planning/phases/05-curriculum-training-integration/05-CONTEXT.md` — All Phase 5 decisions are locked here
2. `.planning/ROADMAP.md` — Phase 5 success criteria (CUR-01, CUR-02, CUR-03)
3. `.planning/REQUIREMENTS.md` — Requirement definitions
4. `CLAUDE.md` — Project conventions, architecture, coding standards

## What Phase 5 Must Deliver

**CUR-01:** TrainConfig accepts a list of route files for curriculum training
**CUR-02:** Training loop cycles through route files across episodes (round-robin)
**CUR-03:** MAPPO achieves person-time-loss <= 0.90 × Max-Pressure on at least one scenario

## Critical Context

### Reward Mode
Use `--reward-mode person_objective`. This is the tested winner:
- `person_objective`: `-w_d * log(1 + person_delay/person_tp) + w_f * local_fairness`
- Peaked at MAPPO/MP = 0.897 (10.3% better than MP) on am_peak
- The old `objective` mode regressed and is not recommended

### Parallel Training
Use `--num-workers 8` for curriculum. Key details:
- 8 workers × 4 scenarios = 2 episodes per scenario per batch (clean round-robin)
- `--scale-lr` is ON by default — scales LR by 1/sqrt(8) ≈ 0.354
- Best-model checkpoint saves to `{model}_best.pt` when eval improves
- Eval/checkpoint triggers use boundary-crossing logic (not exact modulo)
- Workers use libsumo via LIBSUMO_AS_TRACI=1 (2-3x faster per step)

### Warm-Start
Start from the best Phase 4 checkpoint. Check which exists:
- `models/asce_mappo_person_obj_best.pt` (if best-model save was active)
- `models/asce_mappo_person_obj.pt` (final checkpoint)
The warm-start model was trained on am_peak only. Curriculum will fine-tune generalization.

### Scenarios (4 files)
```
sumo/demand/curriculum/am_peak.rou.xml
sumo/demand/curriculum/pm_peak.rou.xml
sumo/demand/curriculum/demand_surge.rou.xml
sumo/demand/curriculum/midday_multimodal.rou.xml
```

### What Needs to Change (Implementation)

**1. TrainConfig (base.py):**
- Add `route_files: list[str] = field(default_factory=list)` — list of route file paths
- Keep existing `route_file: str` for backward compat (single-scenario mode)
- If `route_files` is non-empty, use curriculum mode. If empty, fall back to `route_file`.

**2. CLI (train.py):**
- Add `--route-files` flag (comma-separated string → split into list)
- Example: `--route-files sumo/demand/curriculum/am_peak.rou.xml,sumo/demand/curriculum/pm_peak.rou.xml,...`

**3. Training Loop (local_mappo_backend.py):**

*Sequential path (num_workers=1):*
- Before each episode: `route = route_files[ep % len(route_files)]`
- Rebuild env with selected route file
- Close previous env, create new one

*Parallel path (num_workers>1):*
- Each worker gets a route_file based on `scenarios[worker_index % len(scenarios)]`
- Worker already creates its own env — just pass the right route_file per worker
- With 8 workers and 4 scenarios: workers 0,4 get scenario 0; workers 1,5 get scenario 1; etc.

**4. Inline Eval (critical change):**
- Currently evals on one scenario (the training scenario)
- Must eval on ALL 4 scenarios and report per-scenario MAPPO/MP ratios
- Print: `EVAL ep N: am_peak=0.XXX, pm_peak=0.XXX, demand_surge=0.XXX, midday=0.XXX`
- Best-model checkpoint should use the worst-case ratio (or mean across scenarios)

**5. Episode Metrics CSV:**
- Add `scenario_id` column to ep_metrics (currently missing in parallel path)
- Each row should identify which scenario that episode trained on

### What NOT to Change
- Model architecture (GatedActor, ResidualMAPPOTrainer) — no changes
- Reward function — use person_objective as-is
- Baselines (MP, FT, NEMA) — no changes
- Vendored sumo-rl — no changes
- Observation normalization — carries forward from checkpoint

### Gotchas and Lessons Learned

1. **Don't remove throughput from the reward AND add it back** — we tested this. Throughput dominated the reward when included (v2 result). person_objective without throughput is correct.

2. **Parallel training != sequential training** — 8 episodes per PPO update changes gradient dynamics. LR scaling is mandatory. Episode-for-episode comparison between parallel and sequential is not valid.

3. **Eval trigger with num_workers>1**: Uses boundary-crossing logic, not `(last_ep+1) % N == 0`. With 8 workers and eval_every=25, eval fires at eps 31, 55, 79, 103, 127, 151, 175.

4. **obs_norm in workers**: Workers update their LOCAL obs_norm during episodes (same as sequential). Main process also replays raw obs through its norm. Both are needed.

5. **libsumo global state**: Can't run multiple libsumo instances in one process. Workers use separate processes (multiprocessing spawn context), so each gets its own libsumo. Works fine.

6. **Import ece324_tango.sumo_rl**, never pip sumo_rl. The vendored copy preserves native TLS programs.

7. **Person-weighted metrics add ~25% overhead** per step (per-vehicle TraCI queries). Acceptable for the delay quality improvement.

## Phase 5 Estimated Command

```bash
PYTHONUNBUFFERED=1 pixi run python -m ece324_tango.modeling.train \
  --net-file sumo/network/osm.net.xml \
  --route-files sumo/demand/curriculum/am_peak.rou.xml,sumo/demand/curriculum/pm_peak.rou.xml,sumo/demand/curriculum/demand_surge.rou.xml,sumo/demand/curriculum/midday_multimodal.rou.xml \
  --scenario-id curriculum_v1 --episodes 400 --seconds 900 --delta-time 5 \
  --reward-mode person_objective --residual-mode action_gate --use-obs-norm \
  --model-path models/asce_mappo_curriculum.pt \
  --rollout-csv data/processed/asce_rollout_curriculum.csv \
  --episode-metrics-csv reports/results/asce_train_episode_metrics_curriculum.csv \
  --num-workers 8 --eval-every 25 --checkpoint-every 50 --resume \
  2>&1 | tee reports/results/curriculum_v1_run.log
```

Note: `--route-files` flag doesn't exist yet — Phase 5 implementation creates it.

## After Phase 5

**Phase 6: Eval Loop + Dataset Logging**
- Multi-seed evaluation (10 seeds × 4 scenarios × 4 controllers)
- Parquet output with controller column for PIRA
- Most of the infrastructure already exists (scripts/generate_baseline_dataset.py for MP/FT, inline eval for MAPPO)
- Main work: add MAPPO eval to dataset gen, add controller column, streaming Parquet writer

**Phase 4.5 (now Phase 7): Expansion Scenarios**
- Deferred due to time constraints
- Lane closure, safety-constrained, streetcar short-turn
- Only if initial 4 scenarios don't sufficiently differentiate MAPPO from MP

**PIRA Integration**
- Teammate handles PIRA training on his laptop
- Baseline datasets already generated at `data/pira/baseline_dataset.parquet`
- PIRA code has known integration issues (see memory notes) — teammate's responsibility
