# Feature Landscape: Competitive MARL Traffic Signal Control

**Domain:** Multi-agent reinforcement learning for adaptive traffic signal control (ATSC)
**Researched:** 2026-03-29
**Milestone context:** ASCE trails Max-Pressure by 25% on person-time-loss under stationary car-only demand; policy not converged at 30 episodes.

---

## Confidence Notes

Web tools were unavailable during research. All findings below draw from:
- Training knowledge of the ATSC literature through August 2025 (HIGH for well-established patterns; MEDIUM for specific system details)
- Full codebase audit of TANGO `code-setup` branch (HIGH confidence for current-state claims)
- Internal design docs: `docs/plans/2026-03-06-section4-modelling-design.md`, `docs/notes/prototype_log.md`, `.planning/codebase/CONCERNS.md`

Claims marked [MEDIUM] or [LOW] should be verified against papers before citing.

---

## Framing: Why MAPPO Trails Max-Pressure

The gap has a known root cause documented in `docs/plans/2026-03-06-section4-modelling-design.md`:

> "Max-Pressure is provably throughput-optimal under stationary single-commodity demand, which is exactly this benchmark."

Varaiya (2013) proves this. The Toronto corridor with 70 uniform car flows is Max-Pressure's home regime. This means closing the gap requires either:
1. Making MAPPO competitive on **this** regime (harder — MP has theoretical optimality), or
2. Making MAPPO better on **harder** regimes (achievable — MP degrades on non-stationary, multi-commodity, or spillback-prone demand)

Both paths require features. They are enumerated below.

---

## Table Stakes

Features every competitive MARL ATSC system has. Their absence nearly guarantees failure to beat Max-Pressure. Systems without these features are used as ablation points in benchmarks, not as competitive methods.

| Feature | Why Expected | Complexity | Current Status | Notes |
|---------|--------------|------------|----------------|-------|
| **Sufficient training episodes** | Policy at 30 episodes shows improving curve but no convergence. The training curve in Panel A of the interim figures confirms this. Literature systems typically converge at 100-500 episodes depending on network size. | Low — just increase `--episodes` flag | MISSING — 30 episodes used | Highest-ROI single change. No code changes needed. |
| **Phase transition cost enforcement** | Switching phases mid-cycle has real-world cost (yellow/all-red clearance) and prevents oscillation. Without it, the policy degenerates to switching every step. sumo-rl's `TrafficSignal.min_green` and `yellow_time` handle this if the env is configured correctly. | Low | PRESENT via sumo-rl's built-in `min_green` / `delta_time` enforcement | Verify that `delta_time=5s` is functioning as the minimum green window |
| **Action masking for phase count heterogeneity** | Toronto intersections have 4-8 valid phases; shared actor has `n_actions=max`. Invalid actions must be masked to -inf to prevent selection. | Low | PRESENT — `n_valid_actions` masking in `mappo.py:126-127` and PPO update | No gap here |
| **Observation normalization** | Raw queue counts / vehicle counts span different scales; normalizer prevents gradient vanishing/exploding in actor/critic. | Low | PRESENT — Welford online per-feature in `obs_norm.py` | No gap here |
| **Bootstrapped value for truncated episodes** | Episodes that end by timeout (not terminal state) need value bootstrap in GAE to correctly estimate advantage. Without it, all truncated-episode returns are underestimated. | Medium | PRESENT — bootstrap logic in `local_mappo_backend.py:222-234` | No gap here |
| **Multi-objective reward with gradient signal** | Pure time-loss reward was tried and shown to be worse than Fixed-Time (8,434s vs 8,040s). A richer reward (delay + throughput + fairness) is necessary for stable gradient. | Medium | PRESENT — `objective` mode reward | No gap here |
| **Demand variability in training** | Single fixed demand scenario creates a policy that memorizes one distribution. Standard practice is 3-10+ scenario variants covering low/medium/peak demand to generalize. Max-Pressure does not need to generalize — it is heuristic — so MAPPO only beats it when the regime is harder than nominal. | Medium — requires curriculum demand file generation | MISSING — all training on single `demand.rou.xml` | Critical gap for beating MP in the milestone |

**Summary of table-stakes gaps:**
1. Training episodes: 30 → at minimum 200, ideally 500 (zero code change)
2. Demand curriculum: single scenario → 3-5 demand variants (medium effort; Demand Studio exists)

---

## Differentiators

Features present in competitive published systems (CityLight, CoLight, PressLight, MPLight) that provide measurable advantage over naive MARL and over Max-Pressure in realistic conditions.

### D1: Action-Level Max-Pressure Inductive Bias

**What:** Feed the Max-Pressure action as an input feature to the actor at inference time, rather than only penalizing deviations in the reward. The actor sees what MP would do and learns when to follow or deviate.

**Value proposition:** Residual learning from a strong prior. The policy's search space collapses — instead of learning signal coordination from scratch, it learns "follow MP except when condition X." This is the difference between the existing `residual_mp` reward mode (which only penalizes post-hoc) and true action-level residual policy.

**Evidence from codebase:** `CONCERNS.md` explicitly documents this gap: "The policy is not encouraged to propose actions based on max-pressure suggestion; it only gets penalized if it deviates. This is asymmetric — the policy never sees the max-pressure action as a suggested constraint during network forward pass."

**Implementation options (ordered by complexity):**
- Option A (Low): Append one-hot MP action to actor input vector. Actor sees [obs_features | mp_action_onehot]. No architecture change beyond input size increase.
- Option B (Medium): Warm-start actor from MP imitation (supervised loss for N_pretrain episodes, then PPO fine-tuning). Policy starts at MP performance, then improves from there.
- Option C (Medium): True delta action: actor outputs correction on top of MP suggestion rather than absolute phase index. Requires action space redefinition.

**Complexity:** Low-Medium. Option A is lowest risk.
**Depends on:** None (standalone feature)

| Confidence | Level |
|------------|-------|
| Mechanism validity | HIGH (documented in codebase concerns as a known gap) |
| Expected gain | MEDIUM (literature shows residual/imitation-warmstart consistently helps, magnitude unknown for this network) |

---

### D2: Curriculum Training Over Demand Scenarios

**What:** Train across a distribution of demand scenarios with systematically varying difficulty: base load, peak load (1.5x), off-peak (0.5x), demand spikes (step function at t=150s), and incident simulations (reduced capacity on one edge). Demand Studio already exists for generation.

**Value proposition:** Max-Pressure is locally optimal under stationary single-commodity demand. Its key failure modes are:
- Spillback: when downstream queues fill and pressure signals invert, MP can lock into a suboptimal phase
- Demand non-stationarity: sudden demand shifts violate the stationary assumption MP's optimality proof requires
- Multi-commodity demand: when transit vehicles (high person-occupancy) compete with car flows, simple queue counts misalign with person-time-loss

MAPPO trained on varied scenarios can learn to anticipate and adapt; MP cannot.

**Implementation:** `generate_scenario()` in Demand Studio is headlessable. Need a CLI wrapper that calls it to emit 3-5 `.rou.xml` files at varying scales, then cycle over them in the training loop.

**Complexity:** Medium. Demand generation is low-effort (Demand Studio exists). Training loop modification to cycle scenario files is straightforward.
**Depends on:** D1 (curriculum works better when policy starts near MP quality)

| Confidence | Level |
|------------|-------|
| That MP degrades on non-stationary demand | HIGH (theoretical result, well-documented in codebase) |
| That MAPPO will exploit this gap within milestone compute budget | MEDIUM |

---

### D3: Neighbor Observation Augmentation (Partial Cooperation)

**What:** Append the queue lengths and phase of immediately adjacent intersections to each agent's local observation. On the Toronto corridor (linear topology), each intersection has at most 2 neighbors. This gives the actor partial visibility into upstream/downstream state without full centralized execution.

**Value proposition:** Max-Pressure already uses local queue-differential pressure. MARL can exceed this by learning coordination signals — "my downstream neighbor is saturated, so I should hold green a bit longer to avoid pushing vehicles into a queue." CityLight (2024) and CoLight both use inter-agent communication/attention on adjacency graph; neighbor observation is the simplest approximation.

**Implementation:** During environment step, after each agent's observation is collected, append observations from adjacent agents. For 8 intersections on a linear corridor, adjacency is simple (i-1, i+1).

**Complexity:** Medium. Requires topology-aware observation assembly at step time. Actor input dimension increases by 2 * local_obs_dim at most. Welford normalizer already handles arbitrary input dim.
**Depends on:** Nothing (can be added independently)
**Caveat:** Increases obs_dim, which increases padding overhead and global_obs_dim for the critic. On 8 agents, this is bounded.

| Confidence | Level |
|------------|-------|
| That neighbor state improves coordination | HIGH (established in CoLight, MPLight literature) |
| Implementation complexity estimate | HIGH (straightforward given existing env structure) |

---

### D4: Pressure-Aware Observation Features

**What:** Add the computed Max-Pressure score per phase as an explicit observation feature, alongside the existing queue/speed/phase features.

**Value proposition:** The actor currently sees raw queue lengths and speeds. The Max-Pressure controller transforms these into a phase-ranked score internally. Giving the actor direct access to pressure scores shortcuts feature discovery — the actor doesn't need to rediscover the queue-differential relationship.

**Why this is different from D1:** D1 feeds the MP *action* (which phase MP would pick). D4 feeds the MP *scores* (numerical pressure for each phase). D4 gives richer gradient signal; D1 gives a stronger discrete inductive bias.

**Implementation:** Compute `_phase_pressure(env, ts_id, a)` for all a during observation assembly. Append as `n_actions` additional scalar features. O(n_phases) TraCI calls per agent per step — already being computed for the MP controller itself.

**Complexity:** Low. The pressure computation already exists in `baselines.py`. Extract it as a shared utility, call during obs assembly.
**Depends on:** Nothing.

| Confidence | Level |
|------------|-------|
| That pressure features help | MEDIUM (PressLight uses pressure as its primary observation; well-supported) |
| Implementation correctness | HIGH (leverages existing `_phase_pressure` method) |

---

### D5: Longer Episode Horizon / Demand Window Fix

**What:** Fix the demand exhaustion at ~285s so episodes run the full 300s (or extend to 600s). The `end="300"` in route flows and the 300s episode length create a truncated episode where the last 15s is essentially a no-op for training.

**Value proposition:** More training signal per episode. Also, longer episodes (600s) would capture peak-hour ramp-up dynamics that are invisible in 300s windows.

**Implementation:** Two options per `CONCERNS.md`:
- Change `end="300"` to `end="600"` in `sumo/demand/demand.rou.xml`
- Or add `--until 600` to SUMO config for longer simulation

**Complexity:** Low. Route file edit.
**Depends on:** Nothing.

| Confidence | Level |
|------------|-------|
| That this improves training data quality | HIGH (current truncation is a known defect) |

---

### D6: Value Normalization in Critic

**What:** Normalize the critic's return targets to zero mean and unit variance using running statistics, separate from the observation normalizer. This is distinct from advantage normalization (which is already done in `update()` with `adv = (adv - adv.mean()) / (adv.std() + 1e-8)`).

**Value proposition:** The value function learns faster when targets are normalized. Multi-objective rewards produce returns that span multiple orders of magnitude; unnormalized targets slow critic convergence and cause instability in the actor via poor advantage estimates.

**Implementation:** Add a `ReturnRunningNorm` using the same `ObsRunningNorm` class. Apply before critic MSE loss computation. Note: Xuance's implementation of this was unstable on the Toronto network (`TANGO_XUANCE_USE_VALUE_NORM` history in prototype_log) — implement independently in `local_mappo_backend.py` with correct handling.

**Complexity:** Medium (existing `ObsRunningNorm` infrastructure can be reused; need to verify numerical stability).
**Depends on:** Nothing.

| Confidence | Level |
|------------|-------|
| That return normalization helps | MEDIUM (common in PPO implementations; MAPPO paper includes it) |
| Stability risk | MEDIUM (Xuance version was unstable — implement carefully) |

---

## What CityLight, CoLight, and PressLight Actually Do

This section maps known published system features to the table above.

### PressLight (Wei et al. 2019)
[MEDIUM confidence — training knowledge, not verified against paper text during this session]

- **Core insight:** Use pressure (queue differential across intersection) as the primary observation feature rather than raw queue lengths. This gives the policy the same information Max-Pressure uses, but as a learned signal.
- **Observation:** Pressure per phase (number of vehicles on incoming minus outgoing lanes for each movement) + current phase
- **Reward:** Negative total pressure at intersection (directly mirrors MP objective)
- **Architecture:** Single-agent DQN applied independently (IPPO variant), not MAPPO
- **Relevance to TANGO:** D4 (pressure features) is the lightweight adaptation of PressLight's core idea for MAPPO. The pressure reward is less relevant — our multi-objective reward already showed improvement over single-objective.

### CoLight (Wei et al. 2019)
[MEDIUM confidence]

- **Core insight:** Graph attention network to aggregate neighbor information. Each intersection attends over adjacent intersections' states.
- **Architecture:** GNN with attention over topological neighbors. Shared weights across intersections.
- **Observation:** Local queue + phase + weighted neighbor queues via attention
- **Relevance to TANGO:** D3 (neighbor observation) is a simplified version of CoLight's attention mechanism. Full GNN is out of scope for this milestone; manual neighbor concatenation captures most of the benefit.

### MPLight (Chen et al. 2020)
[MEDIUM confidence]

- **Core insight:** Combine PressLight's pressure observation with full MARL coordination. Uses FRAP architecture (phase-level attention over movements).
- **Architecture:** Phase-competition attention — learns which phases are "competing" for the same movements and weights their pressure accordingly.
- **Relevance to TANGO:** D4 captures MPLight's primary input signal. The FRAP attention architecture is the differentiator but requires a full architecture rewrite — out of scope.

### CityLight (Zeng et al. 2024)
[MEDIUM confidence — cited as reference in project's own design doc]

- **Core insight:** Regional coordination at scale. Uses a hierarchical structure with local agents and regional coordinators.
- **Architecture:** Standard MAPPO per intersection + centralized regional value function. Heterogeneous intersection support via graph-based observation encoding.
- **Relevance to TANGO:** The project already cites CityLight as its architecture reference (`docs/plans/2026-03-06-section4-modelling-design.md`: "Standard MAPPO following CityLight"). The architecture is already implemented. The gap is in training procedure and observation features, not the base MAPPO structure.

---

## Anti-Features

Features to explicitly NOT build for this milestone.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| **Full GNN / attention-based architecture rewrite** | CityLight/CoLight attention architectures require replacing the entire Actor class. High implementation risk, long debugging time, uncertain gain on 8-intersection linear corridor. Corridor topology does not benefit from general graph attention. | Use manual neighbor concatenation (D3) — same info, no arch change |
| **Imitation pre-training as primary approach** | Supervised imitation from Max-Pressure produces a policy that is bounded by MP performance. We need to exceed MP, not copy it. If used, use only as warm-start, not as the training objective. | Use MP action as input feature (D1 Option A) not as imitation target |
| **Pedestrian / cyclist optimization** | Out of scope per `PROJECT.md`. Vehicle-only focus is the current requirement. Adding non-vehicle demand types multiplies the scenario space significantly. | Out of scope for milestone 2 |
| **PIRA GNN surrogate implementation** | Deferred to milestone 3 per `PROJECT.md`. Only dependency from ASCE milestone is `dataset.parquet` logging. | Log dataset.parquet, but do not implement PIRA model |
| **Multi-city generalization / transfer learning** | Out of scope. 8-intersection Dundas corridor is the entire evaluation domain. Generalization research requires multiple network topologies and longer timelines. | Focus on single corridor performance |
| **BenchMARL / Xuance backend stabilization** | Both backends are known unstable on Toronto network (`CONCERNS.md`). Fixing them is not on the critical path. `local_mappo` is the only stable training path. | Keep local_mappo as production backend |
| **Hyperparameter search** | RTX 4070 Laptop, <1 day per experiment constraint. Hyperparameter sweeps consume compute budget without guaranteed return. Curriculum + action-level residual are higher-ROI changes. | Fix known architectural gaps first, tune after gap closes |
| **Per-agent heterogeneous networks (distinct actor per intersection type)** | High implementation complexity, not needed at 8-intersection scale. Zero-padding + masking is already functional. | GNN encoding if corridor scales to 50+ intersections in future milestone |

---

## Feature Dependencies

```
[Sufficient episodes] — independent, highest ROI
[Demand curriculum] — independent but pairs with [Action-level MP bias] for best effect
[Action-level MP bias (D1)] — independent
[Pressure features (D4)] — independent, low effort
[Neighbor observations (D3)] — independent, medium effort
[Demand window fix (D5)] — independent, prerequisite for longer episodes
[Value normalization (D6)] — independent, implement carefully given Xuance instability history

Recommended order:
1. [Demand window fix (D5)] — enables longer/cleaner episodes, unblocks all others
2. [Sufficient episodes] — zero-code-change, immediate training gain signal
3. [Action-level MP bias (D1 Option A)] — closes the structural gap between residual reward and residual action
4. [Pressure features (D4)] — low effort, directly targets the MP observation advantage
5. [Demand curriculum] — enables MAPPO to exploit regimes where MP is suboptimal
6. [Neighbor observations (D3)] — coordination signal, useful if above steps don't close gap
7. [Value normalization (D6)] — implement last; risk of instability
```

---

## MVP Recommendation

Prioritize for closing the 25% Max-Pressure gap:

1. **Fix demand window + extend to 200+ episodes** — Zero-code fix + config change. The policy shows improving curve at 30 episodes; insufficient training is a plausible primary contributor to the gap before any architecture changes.

2. **Action-level Max-Pressure inductive bias (D1 Option A)** — Append MP one-hot action to actor input. This directly addresses the documented asymmetry in the existing `residual_mp` mode. Estimated 1-2 days implementation.

3. **Pressure features in observation (D4)** — Reuse existing `_phase_pressure` computation, append to obs vector. Estimated <1 day implementation.

4. **Demand curriculum with 3-5 scenarios** — Use Demand Studio's `generate_scenario()` headlessly to emit peak, off-peak, and spike scenarios. Train in round-robin. This is where MAPPO should structurally outperform MP.

Defer to subsequent iterations if gap remains:
- Neighbor observations (D3): adds complexity, implement if above do not close gap
- Value normalization (D6): implement only after basic training is stable at 200+ episodes

**Expected outcome:** Items 1-3 should measurably reduce the gap. Item 4 may actually invert the result on non-nominal demand (MAPPO better than MP on spike scenarios), satisfying the proposal criterion that MAPPO generalizes better under irregular demand.

---

## Sources

- `docs/plans/2026-03-06-section4-modelling-design.md` — Interim report results, 10-seed benchmark, training curve evidence
- `docs/notes/prototype_log.md` — Experiment history, Xuance instability record
- `docs/notes/handoff_2026-03-02.md` — Root cause analysis, proposal alignment clarification
- `.planning/codebase/CONCERNS.md` — Known gaps including asymmetric residual reward, observation padding limitations
- `.planning/PROJECT.md` — Scope constraints, compute budget, milestone targets
- `ece324_tango/asce/mappo.py` — Current actor/critic architecture (confirmed: 2-layer MLP, no attention)
- `ece324_tango/asce/traffic_metrics.py` — Current reward function (confirmed: log delay + log throughput + Jain)
- `ece324_tango/asce/baselines.py` — Max-Pressure implementation (confirmed: lane-level halting pressure)
- Training-knowledge of PressLight (Wei et al. 2019), CoLight (Wei et al. 2019), MPLight (Chen et al. 2020), CityLight (Zeng et al. 2024) [MEDIUM confidence — not web-verified in this session]
- Varaiya 2013 — Max Pressure optimality under stationary demand [HIGH confidence — cited in project's own design doc]
