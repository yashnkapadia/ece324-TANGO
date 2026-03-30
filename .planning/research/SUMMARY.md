# Project Research Summary

**Project:** TANGO / ASCE — Milestone 2 (Residual MAPPO + Curriculum Training)
**Domain:** Multi-agent reinforcement learning for adaptive traffic signal control (ATSC)
**Researched:** 2026-03-29
**Confidence:** HIGH (codebase-grounded; MEDIUM for demand studio end-to-end path)

## Executive Summary

TANGO's ASCE component must close a 25% performance gap to Max-Pressure on person-time-loss under the Toronto Dundas corridor benchmark. Research confirms this gap has two root causes that are both addressable within the compute budget (RTX 4070 Laptop, <1 day per experiment): the policy is structurally unable to learn *when* to follow or deviate from Max-Pressure because MP's recommendation is never visible as an input to the actor (only as a post-hoc reward penalty), and training on a single demand scenario for 30 episodes cannot produce a converged policy. The recommended approach is a binary action-gate residual MAPPO architecture (MP action as explicit actor input with a learned override gate) combined with curriculum training across 4-8 varied demand scenarios.

The key architectural insight is that reward-level residual (the existing `residual_mp` mode) is fundamentally insufficient: it teaches "don't deviate" but cannot teach "deviate when beneficial." Action-level residual, implemented as a two-head Actor (gate head + phase head) that receives the MP one-hot action as an observation feature, gives the policy the information needed to learn selective correction. This is a medium-complexity code change with well-defined boundaries — the MP call is already paid for in the training loop, obs_dim is parameterized, and the PPO update structure accommodates the joint log-probability without restructuring the core algorithm.

The highest-risk element is the demand studio headless integration path (confidence MEDIUM): `generate_scenario()` is confirmed callable and returns a typed result, but the function requires `NETWORK_CACHE` pre-population and pulls in Dash/Plotly as module-level imports, making the dependency chain non-trivial. All four required PyPI packages (dash, plotly, lxml, pyproj) are missing from `pixi.toml` and must be added before the headless CLI can be tested end-to-end. Fix the demand window truncation bug (demand exhausting at step 57 of 60 in 300s episodes) before launching any curriculum run, as it contaminates GAE advantage estimates.

## Key Findings

### Recommended Stack

The Milestone 2 additions require no new ML libraries. The existing custom MAPPO stack (`mappo.py`, `local_mappo_backend.py`) supports action-level residual as a pure code change: `obs_dim` is a constructor argument, MP is already called each step, and Welford normalization is dimension-agnostic. The only new dependency already integrated is `pyarrow 20.0.0` (conda-forge) for Parquet dataset logging, verified installed in the Pixi environment.

The demand studio headless path requires four PyPI additions to `pixi.toml`. These are not optional even for headless use because `app.py` imports Dash and Plotly at module level. The curriculum and action-gate architecture changes are pure Python refactors with no new library dependencies.

**Core technologies:**
- `pyarrow 20.0.0`: Parquet streaming write — only valid backend (pandas 3.x default; PIRA reads via `pd.read_parquet`); already installed
- `dash >=2.18` + `plotly >=5.24`: Required by `app.py` module-level imports — must add to `[pypi-dependencies]` in `pixi.toml`
- `lxml >=5.3`: XML generation inside `generate_scenario()` — must add to `[pypi-dependencies]`
- `pyproj >=3.6`: Coordinate projection in demand studio — must add to `[pypi-dependencies]`
- `local_mappo` backend: Only stable training path — do not attempt to stabilize BenchMARL or Xuance backends

### Expected Features

**Must have (table stakes — policy will not beat Max-Pressure without these):**
- Sufficient training episodes: minimum 200, ideally 500 — current 30 episodes cannot confirm convergence; highest-ROI single change requiring zero code changes
- Demand variability: minimum 3-5 route files covering off-peak/nominal/peak scales — single fixed demand is Max-Pressure's optimal regime; MAPPO can only win on varied demand

**Should have (differentiators — directly address structural gaps):**
- Action-level Max-Pressure inductive bias (D1): append MP one-hot to actor observation — closes the asymmetric residual gap documented in CONCERNS.md; estimated 1-2 days
- Pressure features in observation (D4): append phase pressure scores from existing `_phase_pressure()` — reuses computed values, adds gradient signal aligned with MP objective; <1 day
- Demand window fix (D5): change `end="300"` to `end="600"` in route files — eliminates FatalTraCIError at step 57 and enables cleaner GAE; prerequisite for all curriculum runs
- Demand curriculum with 3-5 scenarios (D2): headless `generate_scenario()` CLI + round-robin training loop — exploits Max-Pressure's known failure modes on non-stationary demand

**Defer to subsequent iterations:**
- Neighbor observation augmentation (D3): append adjacent agent state — implement only if action-gate + curriculum do not close gap
- Value normalization in critic (D6): implement after training is stable at 200+ episodes; Xuance history shows instability risk
- Full GNN/attention architecture rewrite: out of scope for 8-intersection linear corridor
- PIRA GNN surrogate implementation: Milestone 3 dependency; only dataset.parquet logging is needed from ASCE

### Architecture Approach

The recommended architecture is a Binary Override Gate: the Actor gains a `gate_head` (Linear → 2 logits) alongside the existing phase head. At each step, the gate decides {follow MP, override MP}; when overriding, the existing phase logits select the replacement action. The MP one-hot action is appended to the observation before the actor forward pass, letting the gate condition on what MP recommends. PPO update uses joint log-probability: `log P(gate) + (gate==1) * log P(phase | gate==1)`. This is interpretable — the gate fraction per episode directly answers "does MAPPO add value over MP?" — and backward-compatible when `residual_mode="none"`.

**Major components:**
1. `Actor` with `gate_head` (mappo.py) — adds 2-logit gate head; MLP body unchanged; branched on `residual_mode` config flag
2. `MAPPOTrainer.act_batch()` — accepts `mp_actions_list`, returns gate + final_action + joint logp; Transition dataclass gains `mp_action` and `gate` fields
3. `CurriculumManager` (new `asce/curriculum.py`) — wraps scenario pool selection; `next_route_file()` called per episode; env rebuilt per episode with selected route file
4. Headless demand CLI (new `modeling/generate_demand.py`) — imports `generate_scenario()` from `apps/demand_studio/app.py`; pre-populates NETWORK_CACHE; emits `.rou.xml` files to scenario pool
5. Parquet logger — `pyarrow.parquet.ParquetWriter` streaming writes; schema matches `ASCE_DATASET_COLUMNS` plus `controller` column addition

### Critical Pitfalls

1. **Asymmetric residual penalty trap (C1)** — The existing `residual_mp` reward penalty cannot teach beneficial deviations. Inject MP one-hot into actor observation. Do not rely on reward shaping alone for residual behavior.

2. **Demand exhaustion contaminates GAE (C4)** — FatalTraCIError at step ~57 is coded as `terminated=True`, causing zero bootstrap value for episodes that had 15s remaining. Fix `end="300"` to `end="600"` in all route files before any curriculum run.

3. **Reward proxy misaligns with proposal KPI (M4)** — Training reward uses `edge.getWaitingTime()` (stopped vehicles) but the proposal criterion is `person_time_loss_s` (includes slow-moving vehicles). Monitor `time_loss_s` ratio at every eval, not just training reward. A policy can improve training reward while the KPI worsens — this has happened empirically on this codebase.

4. **Curriculum from-scratch vs. warm-start checkpoint mismatch (M6)** — Adding MP one-hot to `obs_dim` makes existing checkpoints incompatible. Treat action-gate residual as a new training run. Do not load with `strict=False` (silently drops first-layer weights). Optionally transplant non-input layer weights manually.

5. **Welford normalizer poisoned by zero-padding (C2)** — Normalizer statistics for padded dimensions accumulate zeros across all short-obs agents, corrupting the actor's input for those positions. For current Toronto corridor (fixed topology, same padding structure), this is a latent risk rather than immediate blocker, but will cause silent failures if any curriculum scenario introduces different intersection topology. Apply normalizer to raw (unpadded) obs before padding.

## Implications for Roadmap

Based on dependencies and risk profile, 5 phases are recommended. Phases 1 and 2 can be parallelized by team members.

### Phase 1: Foundation Fixes + Scenario Pool

**Rationale:** Two blocking prerequisites must land before any training experiment is valid: the demand window truncation bug (C4) corrupts GAE for all subsequent runs, and the scenario pool is required by Phase 2 (curriculum) and Phase 4 (eval logging). Neither has code dependencies on the other phases.

**Delivers:** Fixed route file with `end="600"`; 6-8 `.rou.xml` scenario files (off-peak 0.6x, nominal 1.0x, peak 1.3x, AM/PM/incident variants); confirmed working headless demand CLI; `dash`, `plotly`, `lxml`, `pyproj` added to `pixi.toml`

**Addresses features:** D5 (demand window fix), D2 prereq (scenario generation)

**Avoids pitfalls:** C4 (GAE contamination), C3 (seed confounding — scenario IDs established here), M3 (global obs explosion — lock to 8-intersection network in all generated scenarios), m3 (unlabeled curriculum rollouts — validate `scenario_id` in schema before first write)

**Research flag:** MEDIUM confidence on headless demand path — requires end-to-end test with actual Pixi environment after adding deps.

### Phase 2: Action-Gate Residual MAPPO (Core Architecture)

**Rationale:** This is the primary structural fix for the 25% gap. Steps 2-6 in the build order form a single coherent unit (Actor gate_head → act_batch() changes → Transition extension → joint logp in update() → backend wiring). Must be built as a unit to avoid partial-state bugs.

**Delivers:** `residual_mode="action_gate"` flag trains without error; gate_fraction logged per episode; MP one-hot appended to observation; backward-compatible when flag is "none"

**Addresses features:** D1 (action-level MP bias), D4 (pressure features can be added here at low incremental cost)

**Avoids pitfalls:** C1 (asymmetric penalty trap — actor now sees MP recommendation), C5 (MP stale state — lock observe→MP-query→actor-forward order), M6 (checkpoint mismatch — start from scratch, document weight surgery option)

**Research flag:** Standard PPO factored action spaces — HIGH confidence, well-established pattern. No additional research needed. Unit test joint log-prob correctness before integration.

### Phase 3: Baseline Convergence Validation

**Rationale:** Before launching curriculum training, the policy must demonstrate stable learning at 200+ episodes with action-gate enabled. This phase is a validation gate, not a feature phase. Mistaking 30-episode variance for a convergence plateau (M5) is the primary risk for wasting the compute budget on curriculum before the base policy is stable.

**Delivers:** Training run of 200+ episodes with fixed route file (demand.rou.xml, fixed); convergence criterion met (rolling mean ratio <1.15x vs. MP for 3 consecutive 10-episode windows); gate_fraction trend confirms increasing override rate over training

**Addresses features:** "Sufficient training episodes" (table stakes gap #1)

**Avoids pitfalls:** M5 (convergence stagnation), M4 (monitor `time_loss_s` ratio not just training reward)

**Research flag:** No new research needed. Pass/fail gate based on empirical results.

### Phase 4: Curriculum Training Integration

**Rationale:** After the base policy converges at single-scenario, curriculum training exposes MAPPO to demand regimes where Max-Pressure is theoretically suboptimal (non-stationary, high-load, incident scenarios). This is the phase where MAPPO should structurally outperform MP.

**Delivers:** `CurriculumManager` class with random-schedule scenario sampling; per-episode env rebuild; training over 6-8 scenario pool; per-scenario `time_loss_s` ratio tracking; held-out test scenario evaluation

**Addresses features:** D2 (curriculum training), "Demand variability" (table stakes gap #2)

**Avoids pitfalls:** C3 (held-out test scenario enforced), C2 (Welford normalizer — monitor for padding corruption if topology differs), M1 (start with easy scenarios, gate on ratio <1.15x before adding hard scenarios), M3 (fixed 8-intersection topology in all scenarios)

**Research flag:** Standard patterns for the curriculum loop itself. Adaptive curriculum scheduling (weight by MAPPO/MP gap per scenario) is LOW confidence for this specific codebase — defer. Use random sampling over fixed pool.

### Phase 5: Eval Loop + Dataset Logging

**Rationale:** PIRA (Milestone 3 dependency) requires `dataset.parquet` across all controllers (MAPPO, MaxPressure, FixedTime). This phase finalizes evaluation infrastructure and produces the artifact that unblocks downstream work.

**Delivers:** Residual MAPPO added to eval loop as `residual_mappo` controller; `dataset.parquet` with streaming `ParquetWriter`; `controller` column added to schema; all 3 controllers evaluated across all curriculum scenarios; parquet validated against PIRA read pattern

**Addresses features:** "Parquet dataset logging" (PIRA dependency)

**Avoids pitfalls:** m3 (scenario_id present in all rows), m1 (checkpoint naming includes scenario_id + episode count), M4 (final eval uses `person_time_loss_s` as primary KPI)

**Research flag:** Standard patterns. pyarrow ParquetWriter already verified in Pixi env. No additional research needed.

### Phase Ordering Rationale

- Phase 1 must precede Phase 4 (curriculum needs the scenario pool) and must precede all training experiments (demand window fix is a correctness requirement for GAE).
- Phase 2 must precede Phase 3 (need the action-gate architecture to validate convergence with residual enabled).
- Phase 3 must precede Phase 4 (curriculum training confounded by unstable base policy).
- Phase 5 is additive and can begin in parallel with Phase 4 eval infrastructure, but requires Phase 4's training runs to produce the actual data.
- Phases 1 and 2 have no dependency on each other and can be parallelized.

### Research Flags

Phases requiring validation or additional investigation:

- **Phase 1 (demand headless):** MEDIUM confidence — end-to-end path not tested with actual Pixi env. Validate `generate_scenario()` returns a valid `.rou.xml` after installing all four deps.
- **Phase 2 (joint logp):** Write a unit test before integration that confirms gradient is zero on `phase_head` parameters for gate==0 transitions. This is the most likely silent correctness error.

Phases with standard patterns (no additional research needed):
- **Phase 2 (gate architecture):** Binary factored action space is a standard PPO pattern; HIGH confidence in formulation.
- **Phase 3 (convergence check):** Empirical gate, no research needed.
- **Phase 5 (parquet logging):** pyarrow ParquetWriter already verified in environment.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | pyarrow installed and tested; action-gate and curriculum are pure code changes; demand studio deps confirmed needed but not tested end-to-end |
| Features | HIGH | All findings grounded in codebase audit + known ATSC literature; web tools unavailable so PressLight/CoLight/MPLight citations are MEDIUM |
| Architecture | HIGH | Binary gate pattern derived directly from codebase source; joint logp formulation is mathematically derived; curriculum env-rebuild cost is an estimate |
| Pitfalls | HIGH | All critical pitfalls have direct codebase evidence (CONCERNS.md, prototype_log, source code line references) |

**Overall confidence:** HIGH for implementation decisions; MEDIUM for demand studio headless path until end-to-end tested.

### Gaps to Address

- **Demand studio end-to-end test:** Confirmed callable but not run with actual outputs. First task of Phase 1 should be: add deps to `pixi.toml`, run `generate_scenario()` headlessly, verify `.rou.xml` output is valid for SUMO. If this fails, fall back to manually authored demand files.
- **Optimal episode budget:** Literature says 100-500 episodes; project compute budget says <1 day on RTX 4070. The 200-episode target in Phase 3 needs empirical validation of wall-clock time on the Toronto corridor (SUMO-bound, not GPU-bound). If 200 episodes takes >8 hours, reduce to 100 with tighter convergence criterion.
- **Demand studio dependency versions:** `pixi.toml` on `data-setup` branch uses `"*"` for these packages. Minimum versions recommended (dash>=2.18, plotly>=5.24, lxml>=5.3, pyproj>=3.6) are based on matching known-working versions, not pinned to specific tested releases.
- **`controller` column in ASCE schema:** `schema.py`'s `ASCE_DATASET_COLUMNS` does not yet include `controller`. This column is required for PIRA to distinguish training data by controller type. Must be added before the first parquet write.

## Sources

### Primary (HIGH confidence — direct codebase access)

- `/home/as04/ece324-TANGO/ece324_tango/asce/mappo.py` — Actor/Critic/MAPPOTrainer; obs_dim parameterization, action masking, GAE
- `/home/as04/ece324-TANGO/ece324_tango/asce/trainers/local_mappo_backend.py` — Training loop; MP call at line 132; FatalTraCIError handling at line 158
- `/home/as04/ece324-TANGO/ece324_tango/asce/baselines.py` — MaxPressureController; `_phase_pressure()` method
- `/home/as04/ece324-TANGO/ece324_tango/asce/schema.py` — ASCE_DATASET_COLUMNS
- `/home/as04/ece324-TANGO/.planning/codebase/CONCERNS.md` — Welford normalizer padding bug, asymmetric residual gap (HIGH)
- `/home/as04/ece324-TANGO/docs/notes/prototype_log.md` — Empirical training run history including reward_mode experiments
- `/home/as04/ece324-TANGO/docs/notes/handoff_2026-03-02.md` — Root cause analysis
- `/home/as04/ece324-TANGO/docs/plans/2026-03-06-section4-modelling-design.md` — Interim report results, CityLight architecture reference
- `/home/as04/ece324-TANGO/.planning/PROJECT.md` — Scope constraints, compute budget, milestone targets
- `origin/PIRA:PIRA.ipynb` — PIRA read pattern; column expectations for parquet
- `origin/data-setup:apps/demand_studio/app.py` — `generate_scenario()` signature, `GenerationResult`, module-level imports
- `origin/data-setup:pixi.toml` — demand studio dependency list
- pyarrow 20.0.0 install verified in Pixi environment; ParquetWriter streaming write pattern tested

### Secondary (MEDIUM confidence — training knowledge, not web-verified this session)

- Varaiya 2013 — Max-Pressure throughput-optimality under stationary single-commodity demand (cited in project's own design doc — HIGH via that citation)
- PressLight (Wei et al. 2019) — pressure-as-observation approach
- CoLight (Wei et al. 2019) — graph attention for neighbor coordination
- MPLight (Chen et al. 2020) — FRAP phase-competition attention
- CityLight (Zeng et al. 2024) — MAPPO reference architecture; cited directly in project design doc
- Standard PPO factored action space joint log-probability formulation
- Curriculum ordering principles (easy-to-hard) from MARL literature

---
*Research completed: 2026-03-29*
*Ready for roadmap: yes*
