# Architecture Patterns: Action-Level Residual MAPPO

**Domain:** Multi-agent traffic signal control — residual RL over heuristic baseline
**Researched:** 2026-03-29
**Confidence:** HIGH (derived directly from codebase; limited external source access)

---

## Current Architecture Snapshot

The existing system operates in three separable layers during a training step:

```
Observation  →  [MAPPOTrainer.act_batch()]  →  actions  →  env.step()
                [MaxPressureController.actions()]  →  mp_actions
                                                          ↓
                                             rewards_from_metrics(mode="residual_mp")
                                             penalises deviation: reward -= w * |a != mp_a|
```

This is **reward-level** residual: MAPPO still outputs over the full action space independently;
the only coupling to Max-Pressure is a scalar penalty on the reward signal.
The policy is not structurally guided toward MP — it is only discouraged from diverging.

---

## Architectural Options for Action-Level Residual

Three patterns are viable. They differ in where MP information enters the decision.

### Option A — Binary Override Gate (Recommended)

MAPPO's output head is changed to two logits: **{keep MP, override MP}**.
When "keep MP" is selected, the MP action is used. When "override MP" is selected,
MAPPO emits a second head selecting among `n_actions` alternatives.

```
obs → Actor → [gate_logit_keep, gate_logit_override]  → Categorical → gate ∈ {0,1}
                                                           if gate==1:
                         obs → Actor → [phase_0..phase_k]  → Categorical → override_action
mp_action (from MaxPressureController)
final_action = mp_action if gate==0 else override_action
```

**Why recommended for this codebase:**
- MP action is already computed every step (line 132 in `local_mappo_backend.py`).
  The infrastructure exists; only the Actor head and action-selection logic change.
- Gate produces an interpretable signal: "fraction of steps where MAPPO overrides MP."
  This directly answers the research question of whether MAPPO adds value.
- Gradient flows through the gate Categorical independently of the phase head,
  giving PPO clean credit assignment for override decisions.
- Warm-start is natural: initialize gate head to bias toward gate==0 (MP follow).
  MAPPO then learns to override only when it improves objective.

**Implementation surface in current code:**
- `Actor` in `mappo.py`: add `gate_head` (Linear → 2 logits) alongside existing `net`.
- `MAPPOTrainer.act_batch()`: take `mp_actions_list` as new parameter; return both
  `gate_action` and `final_action`; store `gate_logp` + `phase_logp` in Transition.
- `Transition` dataclass: add `mp_action: int`, `gate: int` fields.
- `build_batch()` / `update()`: compute joint log-prob = gate_logp + (gate==1)*phase_logp;
  PPO ratio uses this joint logp.
- `local_mappo_backend.py` train loop: pass `mp_actions` dict into `act_batch()`.
- `TrainConfig`: add `residual_mode: str` field ("none" | "action_gate").

### Option B — Full Replacement with MP Prior (Not Recommended)

MAPPO outputs a distribution over all `n_actions` phases, but the logits are
initialized with or biased by the MP score. At test time, softmax over logits gives
a probability that is close to argmax(MP) initially but can diverge as training proceeds.

```
obs → Actor → raw_logits
mp_scores = [pressure(a) for a in phases]  → normalized as additive prior
final_logits = raw_logits + alpha * mp_scores
action = sample(Categorical(logits=final_logits))
```

**Why not recommended:**
- `alpha` is a hyperparameter requiring tuning. Its schedule over training adds another
  curriculum dimension.
- MP scores are computed per agent via TraCI calls (already done). Injecting them into
  logit space requires converting pressure values to a compatible scale — non-trivial.
- MAPPO cannot cleanly "turn off" the prior without architectural surgery later.
- The override/compliance ratio is not directly observable, making research interpretation harder.
- The current Actor is a plain MLP (`mappo.py` lines 14–27); adding a score-conditioned
  input path changes the obs dimension contract and invalidates existing checkpoints.

### Option C — Additive Correction Head (Overkill for This Project)

MAPPO outputs a correction vector of shape `[n_actions]`, added element-wise to MP's
one-hot or score vector. The corrected distribution is sampled.

```
mp_one_hot = one_hot(mp_action, n_actions)
correction = Actor(obs)  → shape [n_actions]
final_logits = mp_one_hot + correction
```

This is used in continuous-action residual RL (e.g., residual policy in robotics).
For discrete traffic phases (2–4 per intersection), it collapses to nearly the same
behavior as Option B but with worse gradient properties near the categorical boundary.
Not appropriate here.

---

## Recommended Architecture: Binary Override Gate

### Component Boundaries

| Component | What Changes | What Stays |
|-----------|-------------|------------|
| `Actor` (mappo.py) | Add `gate_head: nn.Linear(hidden_dim, 2)` alongside existing `net` | MLP body unchanged |
| `MAPPOTrainer.act_batch()` | Accept `mp_actions_list: list[int]`; return `gate`, `final_action`, joint `logp` | Normalization, batching, critic unchanged |
| `Transition` dataclass | Add `mp_action: int`, `gate: int` | All existing fields unchanged |
| `build_batch()` | Include `mp_action`, `gate` arrays in batch dict | GAE computation unchanged |
| `update()` in MAPPOTrainer | Compute joint log-prob for PPO ratio; apply action mask to gate+phase heads | PPO clip, value loss, entropy unchanged |
| `local_mappo_backend.py` train loop | Pass `mp_actions` into `act_batch()`; log `gate_fraction` per episode | Episode structure, reward pipeline unchanged |
| `TrainConfig` (base.py) | Add `residual_mode: str = "none"` | All existing config fields unchanged |
| `rewards_from_metrics()` | Add `"action_gate_mp"` mode that uses MP-objective reward without deviation penalty | Existing modes unchanged |

### Data Flow: Training Step with Action Gate

```
1. env state                → observations dict (unchanged)
2. MaxPressureController    → mp_actions dict         [no change: line 132 already does this]
3. Actor.net(obs)           → hidden representation   [no change to MLP body]
4. Actor.gate_head(hidden)  → gate_logits [N, 2]     [NEW]
5. Categorical(gate_logits) → gate ∈ {0=MP, 1=override} per agent [NEW]
6. Actor.net[-1](hidden)    → phase_logits [N, k]    [existing final layer]
   mask invalid phases      → masked_phase_logits    [no change to masking logic]
7. Categorical(masked)      → override_phase per agent [existing]
8. final_action = mp_actions[a] if gate==0 else override_phase [NEW merge step]
9. joint_logp = gate_logp + (gate==1) * phase_logp   [NEW; stored in Transition]
10. env.step(final_actions) → next_obs, rewards       [no change]
11. rewards_from_metrics()  → shaped rewards          [use "objective" mode — no deviation penalty]
12. Transition stored with  obs, global_obs, final_action, joint_logp,
                            mp_action, gate, reward, done, value, n_valid_actions
```

**Key invariant:** When `gate==0`, the phase log-prob term drops out. This is correct because
the agent's policy choice at that step was entirely the gate decision (MP action is deterministic,
not a policy output). PPO must only see gradient through the gate head for those transitions.

### Observation Augmentation (Optional but Recommended)

Add `mp_action` as an extra feature in each agent's observation. Concretely, append a
one-hot of size `n_actions` encoding the current MP recommendation to the padded obs vector.

```
augmented_obs = concat([padded_obs, one_hot(mp_action, n_actions)])
obs_dim_residual = obs_dim + n_actions
```

This lets the Actor see what MP recommends before deciding whether to override.
Without this, the gate head must infer MP's recommendation from queue features alone.
Since MP is deterministic given queue state, this is in principle learnable — but
explicit encoding accelerates convergence.

**Impact on existing components:**
- `obs_dim` increases by `n_actions` (max 4 for this network → negligible).
- `global_obs_dim` increases by `n_agents * n_actions` (8 agents * 4 = 32 extra features).
- `ObsRunningNorm` adapts automatically (Welford is dimension-parameterized at init).
- Existing checkpoints are incompatible; residual mode always trains from scratch.

### Checkpoint Backward Compatibility

The gate head is optional. When `residual_mode == "none"` (default), the `Actor` behaves
identically to today. When `residual_mode == "action_gate"`, a different `Actor` subclass
is instantiated. Save/load paths should branch on this flag, preserving full backward
compatibility with existing checkpoints in `models/`.

---

## Curriculum Training Architecture

### Purpose

Max-Pressure is provably optimal under stationary single-commodity demand.
The MAPPO gap (currently ~25%) narrows when demand is varied/irregular.
Curriculum training systematically exposes MAPPO to demand conditions where
MP is suboptimal, accelerating the learning of beneficial overrides.

### Component: CurriculumManager

A new class (suggested: `ece324_tango/asce/curriculum.py`) wraps demand scenario
selection and presents scenarios to the training loop in controlled order.

```python
@dataclass
class ScenarioCurriculum:
    scenario_pool: list[Path]       # route files from headless demand generation
    schedule: str                   # "random" | "ordered" | "adaptive"
    episodes_per_scenario: int      # how many episodes before cycling
    current_index: int = 0
```

The training loop calls `curriculum.next_route_file(ep_metrics)` instead of using a
fixed `cfg.route_file`. The backend reconstructs the environment per episode
(or per scenario block) with the selected route file.

### Curriculum Schedule Options

| Schedule | Behavior | When to Use |
|----------|----------|-------------|
| `ordered` | Nominal → peak → off-peak → multimodal, in sequence | Initial training; establishes baseline across conditions |
| `random` | Uniform sample from pool each episode | Prevents overfitting to any single condition |
| `adaptive` | Weight scenario selection by MAPPO/MP gap per scenario | Advanced; requires per-scenario KPI tracking overhead |

**Recommended: start with `random` over a pool of 4–8 scenarios.** Ordered curriculum
risks policy forgetting (catastrophic interference) if later scenarios dominate gradients.
Random sampling with a diverse pool is the established safe default.

### Scenario Pool Generation

The headless demand CLI (Active requirement in PROJECT.md) generates `.rou.xml` files
for varied conditions. Each scenario is characterized by:

- `demand_scale`: 0.6x (off-peak) / 1.0x (nominal) / 1.3x (peak)
- `time_of_day`: AM peak / PM peak / off-peak / overnight
- `modal_split`: baseline auto-heavy / transit-heavy / mixed

The pool should include at least one condition where MP is known to perform poorly
(e.g., asymmetric demand where one direction is saturated and MP oscillates).

### Integration with Training Loop

The current `LocalMappoBackend.train()` loop structure is:

```
for ep in range(cfg.episodes):
    obs = env.reset(seed=cfg.seed + ep)
    ...
    trainer.update(batch)
```

The environment is created once before the loop and reused. Curriculum requires
per-episode route variation. Two integration options:

**Option 1 — Rebuild env per episode (simpler, slower):**
Move `env = create_parallel_env(...)` inside the episode loop. Select route file from
curriculum manager. Close and reopen SUMO each episode. Cost: ~2–5s per episode
for SUMO startup. Acceptable for the compute budget (RTX 4070, <1 day target).

**Option 2 — Rebuild env per scenario block (faster):**
Group episodes by scenario. Rebuild env only when the scenario changes.
Requires `episodes_per_scenario` to be a multiple of `cfg.episodes` or managed
separately. More complex but reduces SUMO startup overhead by `episodes_per_scenario`x.

**Recommended: Option 1** for initial implementation. SUMO startup overhead is modest
and the simplicity avoids subtle state-reset bugs. Optimize to Option 2 if profiling
shows startup is a bottleneck.

### TrainConfig Changes for Curriculum

```python
@dataclass
class TrainConfig:
    ...
    # Curriculum additions
    curriculum_route_files: list[str] | None = None   # None = use single route_file
    curriculum_schedule: str = "random"               # "random" | "ordered"
    curriculum_episodes_per_scenario: int = 1         # 1 = resample every episode
```

When `curriculum_route_files` is set, the backend ignores `route_file` and uses the
curriculum manager. This is backward-compatible: existing configs without the new field
get the current behavior (single fixed route file).

---

## Build Order: Dependencies Between Components

The components have the following dependency chain. Build in this order to avoid
blocked work.

```
1. Headless demand CLI  →  generates scenario pool (no code deps, but blocks curriculum)
2. Action-gate Actor    →  new Actor subclass; no external deps; can be tested in isolation
3. Augmented obs        →  depends on knowing mp_action at act_batch() time
4. act_batch() changes  →  depends on 2 and 3; changes MAPPOTrainer interface
5. Transition/batch     →  depends on 4; extends data structures
6. Backend train loop   →  depends on 4 and 5; wires mp_actions into act_batch()
7. CurriculumManager    →  depends on 1 (route files); independent of 2–6
8. Backend curriculum   →  depends on 6 and 7; adds env-rebuild per episode
9. Eval loop update     →  depends on 6; adds residual_mappo controller to eval
10. dataset.parquet log →  depends on 6 and 9; PIRA data collection
```

Steps 2–6 are a single coherent unit (action-gate residual). Steps 7–8 are a second
coherent unit (curriculum). Step 10 is additive to either. These two units can be
parallelized across team members.

---

## What Changes vs. What Stays

### Unchanged

- `Critic` network and its forward pass
- GAE computation (`_compute_gae`)
- PPO clip logic in `update()` (structure unchanged; logp field changes meaning)
- `ObsRunningNorm` / `ObsRunningNorm.update()` (dimension-agnostic)
- `MaxPressureController._phase_pressure()` and `actions()` — called identically
- `rewards_from_metrics()` — use "objective" mode for residual training; existing
  "residual_mp" (penalty-based) mode stays available as a comparison baseline
- `KPITracker`, `IntersectionMetrics`, all metric computation
- Evaluation loop structure for FixedTime and MaxPressure controllers
- Error reporting, SUMO TraCI error handling, episode termination logic
- ASCE dataset columns schema (add `mp_action` and `gate` as optional columns)

### Changes Required

| File | Change | Risk |
|------|--------|------|
| `mappo.py` Actor | Add `gate_head` linear layer | Low; additive |
| `mappo.py` MAPPOTrainer.act_batch() | Accept `mp_actions_list`; return gate fields | Medium; interface change |
| `mappo.py` Transition | Add `mp_action`, `gate` fields with defaults | Low; backward-compatible defaults |
| `mappo.py` update() | Joint log-prob for PPO ratio | Medium; logic change in core training |
| `mappo.py` save/load | Branch on residual_mode; store gate_head weights | Low; additive |
| `trainers/base.py` TrainConfig | Add `residual_mode`, curriculum fields | Low; optional with defaults |
| `trainers/local_mappo_backend.py` train() | Pass mp_actions to act_batch; env rebuild for curriculum | Medium; loop restructuring |
| `asce/curriculum.py` | New file: CurriculumManager | Low; new isolated component |

---

## Pitfalls Specific to This Architecture

### Joint Log-Prob in PPO

The gate and phase are two sequential Categorical decisions. Their joint log-prob for
a transition where gate==1 (override) is:

```
logp_joint = log P(gate=1) + log P(phase=override_phase | gate=1)
```

For gate==0 (keep MP):

```
logp_joint = log P(gate=0)
```

The phase head never contributes when gate==0. During PPO update, when recomputing
logp from current policy, the same branching must apply. Mixing these is the most
likely implementation error. Test with a unit test that verifies gradient zero on
phase_head parameters for gate==0 transitions.

### MP Action Computational Cost

`MaxPressureController.actions()` iterates over all phases for all agents using TraCI
calls. It is already called once per step in the training loop (line 132). The residual
architecture does not increase this cost — the existing call is reused. Verify this
does not move inside a nested loop.

### Observation Dimension Contract

`obs_dim` is inferred at runtime from the first reset observation and used to initialize
`ObsRunningNorm` and the Actor. Adding the MP one-hot augmentation must happen
consistently: in the padded_obs_list construction, in `act_batch()`, and in the
`obs` field of Transition (so PPO update normalizes the same augmented vector).
If augmentation is added in act_batch but not in the stored Transition obs, the PPO
update normalizes different vectors than inference, corrupting training.

### Curriculum and Episode Seed

Currently, `env.reset(seed=cfg.seed + ep)` provides reproducibility. With per-episode
route file changes, the seed should also vary to avoid the same vehicle departure pattern
paired with every scenario. Recommend: `seed = cfg.seed + ep * 7 + scenario_hash`.

---

## Suggested Phase Structure (for Roadmap)

Based on dependencies and risk profile:

**Phase 1 — Headless Demand CLI + Scenario Pool**
- Prerequisite for curriculum; unblocks parallelism
- Deliverable: 6–8 route files covering demand_scale and time_of_day variants

**Phase 2 — Action-Gate Residual MAPPO (core)**
- Components: Actor gate_head, act_batch() changes, Transition extensions, PPO joint logp, backend wiring
- Deliverable: `residual_mode="action_gate"` flag trains without error; gate_fraction logged per episode
- Test: gate_fraction starts near 0 (MP-following), increases as policy learns overrides

**Phase 3 — Observation Augmentation**
- Add MP one-hot to obs; retrain from scratch with augmented obs_dim
- Can be done simultaneously with Phase 2 if gate_head and augmentation are landed together

**Phase 4 — Curriculum Training Integration**
- CurriculumManager + per-episode env rebuild
- Train over scenario pool; track per-scenario person-time-loss

**Phase 5 — Eval Loop + Dataset.parquet**
- Add residual_mappo controller to eval loop
- Log dataset.parquet across all controllers for PIRA

---

## Confidence Assessment

| Area | Confidence | Source |
|------|------------|--------|
| Current architecture boundaries | HIGH | Read from source files directly |
| Binary gate approach correctness | HIGH | Standard PPO factored action spaces; well-established pattern |
| Joint logp formulation | HIGH | Follows from Categorical distribution math |
| Obs augmentation recommendation | HIGH | Standard feature engineering; no external source needed |
| Curriculum env-rebuild cost | MEDIUM | SUMO startup time is project-specific; estimate from context |
| Adaptive curriculum (Option 3) | LOW | Not validated in this codebase context; defer |

---

*Architecture research: 2026-03-29*
