# Domain Pitfalls: MARL Traffic Signal Control with Residual MAPPO

**Domain:** Multi-agent reinforcement learning for signalized intersection control (SUMO/TraCI)
**Project:** TANGO ASCE — closing 25% gap to Max-Pressure via action-level residual MAPPO and curriculum training
**Researched:** 2026-03-29
**Confidence:** HIGH for codebase-grounded pitfalls (direct evidence); MEDIUM for curriculum/residual implementation patterns (well-established RL literature applied to this specific context)

---

## Critical Pitfalls

Mistakes that cause silent training failure, policy collapse, or wasted experiments.

---

### Pitfall C1: Reward-Level vs. Action-Level Residual — The Asymmetric Penalty Trap

**What goes wrong:**
The current `residual_mp` mode only penalizes the MAPPO agent after the fact (`reward -= residual_weight * 1[action != mp_action]`). The actor network never sees the Max-Pressure suggestion as input during the forward pass — it cannot learn *when* to follow MP and when to deviate. The result is a policy that minimizes the penalty by drifting toward MP behavior without learning a principled deviation model.

**Why it happens:**
Reward shaping can encourage or discourage behaviors, but it cannot inject structured information into the policy's input space. A post-hoc penalty teaches "don't deviate," not "deviate when it is worth it." The asymmetry is that the policy has no positive gradient signal for *beneficial* deviations — it can only learn that deviating costs penalty weight units regardless of outcome.

**Consequences:**
- Training converges to a near-Max-Pressure clone rather than a policy that supplements MP.
- The residual weight becomes a hyperparameter that controls how much the policy is regularized toward MP, not a mechanism for learning adaptive deviations.
- If `residual_weight` is too high, gradient dominance drives the policy toward MP imitation with zero exploration. If too low, the residual term has no effect.
- Experiments with `residual_mp` reward mode will not achieve the 0.78x remaining gap without an action-level signal.

**Prevention:**
Implement action-level residual by concatenating the Max-Pressure action (as a one-hot or integer-encoded feature) directly into the actor's input observation. This gives the actor the information it needs to condition its output on MP's suggestion. Two concrete approaches:
1. **Feature injection (lowest effort):** Append `mp_action_onehot` (shape `[n_actions]`) to the padded observation before the actor forward pass. Requires updating `obs_dim` at training initialization.
2. **True residual output (stronger prior):** Train the actor to output a delta over the MP action in log-probability space. The policy distribution starts centered on MP and learns adjustments.

**Warning signs:**
- `residual_mp` training produces entropy collapse (entropy drops toward 0 early, faster than `objective` mode).
- Eval `time_loss_s` ratio with `residual_mp` is not meaningfully better than `objective` mode after the same episode budget.
- Actor loss is driven almost entirely by the residual penalty term rather than the advantage signal.

**Phase:** Action-level residual MAPPO implementation (Phase 1 of current milestone).

---

### Pitfall C2: Welford Normalizer Contaminated by Zero-Padding

**What goes wrong:**
The `ObsRunningNorm` (Welford online per-feature) is applied to padded observations where features beyond the true observation length are always zero. The normalizer accumulates these zero values across all agents at every step, pulling the running mean toward zero and collapsing the running variance for padded dimensions. During normalization, these padded positions get divided by near-zero standard deviation (or clamped by the `1e-8` epsilon), producing large non-zero normalized values where the input was zero. This corrupts the actor's input when the same actor is shared across heterogeneous agents.

**Why it happens:**
Zero-padding to `max(obs_dims)` is applied before the normalizer update in the training loop (`trainer.obs_norm.update(padded_obs)`). Agents with shorter observations contribute zeros to the normalizer for their padded dimensions at every step. Since there are more short-observation agents (or equal numbers), the normalizer statistics for padded dimensions are polluted.

**Evidence in codebase:**
`CONCERNS.md` explicitly flags: *"Welford normalizer treats padded zeros as legitimate observations and converges them to near-zero mean/std, creating discontinuity in normalized feature space across networks."* `local_mappo_backend.py` lines 122-126 confirm normalizer update is called on padded observations.

**Consequences:**
- The effective observation seen by the actor differs significantly between agents with short vs. long observations.
- Transfer learning to any network with different topology will fail: normalizer statistics are tuned to the specific padding pattern of the Toronto corridor.
- Curriculum training across multiple demand scenarios (same network, same topology) is unaffected — but switching networks would be catastrophic.

**Prevention:**
Update the normalizer with unpadded observations only, or maintain separate per-agent normalizers (expensive but correct). The practical fix within parameter-sharing MAPPO is to apply the normalizer update before padding, then re-apply padding after normalization:
```python
# Normalize with raw obs, then pad the normalized vector
obs_normalized = normalizer.normalize(raw_obs)  # on original dims
padded_normalized = pad_observation(obs_normalized, target_dim=obs_dim)
```
This requires that normalization stats are maintained per true observation size, not per padded size. Given the Toronto corridor has fixed topology for this milestone, the current behavior is a latent risk rather than an active blocker — but it will cause silent failures if the network topology changes between curriculum scenarios.

**Warning signs:**
- Actor loss or eval metrics diverge suddenly when a new scenario with different intersection topology is introduced.
- Normalizer statistics for features beyond index 9 (the minimum obs dim) show mean near 0 and std near `1e-8` or similar epsilon floor.

**Phase:** Curriculum training introduction; also relevant during action-level residual (if MP one-hot is appended, obs_dim changes).

---

### Pitfall C3: Single-Scenario Seed Confounding — Training on `seed + ep` With One Route File

**What goes wrong:**
The current training loop increments episode seeds (`seed + ep`) but runs the same `demand.rou.xml` for every episode. SUMO's seeded randomness affects vehicle departure time jitter and route variation, but the underlying demand structure (70 TMC-calibrated flows with specific OD pairs and volumes) does not change meaningfully between seeds on a single route file. The policy sees minor stochastic variation but is fundamentally optimizing a single demand scenario repeated 30 times.

**Why it happens:**
Curriculum training is a planned improvement, not yet implemented. Using `seed + ep` on a fixed demand file is the correct single-scenario training pattern — but it creates a false sense of generalization when comparing curriculum-trained vs. single-scenario MAPPO.

**Consequences:**
- A policy that converges on single-scenario training is not the same as a curriculum-trained policy. It may overfit to the specific flow volumes and OD patterns of the TMC-calibrated scenario.
- When evaluating "generalization" claims for the proposal, comparing `single_scenario` vs. `curriculum` requires holding out a scenario that was not in training — failing to do this inflates generalization metrics.
- Max-Pressure is not stochastic; it does not improve under seed variation. MAPPO's apparent benefit from more seeds may reflect overfitting to seeded noise rather than genuine improvement.

**Prevention:**
Distinguish clearly between single-scenario convergence experiments (same demand file, many seeds) and curriculum generalization experiments (multiple demand files, held-out test scenario). Label checkpoints and eval CSVs with `scenario_id` and `demand_file_hash` to prevent accidental comparison across training modes.

**Warning signs:**
- MAPPO performs well on `demand.rou.xml` after curriculum training but its ratio vs MP on a held-out scenario (e.g., incident demand or non-peak timing) does not improve over the single-scenario-trained policy.
- Eval seeds are all drawn from the same demand file that was used in training.

**Phase:** Curriculum demand file generation and multi-scenario training.

---

### Pitfall C4: SUMO Demand Exhaustion Creates Truncated Episodes That Contaminate GAE

**What goes wrong:**
The known `FatalTraCIError` at ~step 57 (~285 s into a 300 s episode) is treated as `terminated=True` (see `local_mappo_backend.py` line 158: `terminated = True`). This is subtly incorrect: the simulation ended because demand was exhausted, not because the MDP reached a natural terminal state. Under GAE, `terminated=True` causes `bootstrap_value = 0.0` — the critic assigns no future value to the truncated state. In reality, the environment would have continued for ~15 s more with residual vehicles still in network.

**Why it matters more in curriculum training:**
If some curriculum scenarios reliably exhaust demand earlier than others (e.g., low-volume off-peak scenarios finishing at step 30), the bias in advantage estimates will differ systematically across scenarios. The critic will learn that low-volume scenarios have "terminal" states at unusual timesteps, corrupting value estimates.

**Evidence in codebase:**
`local_mappo_backend.py` line 158 sets `terminated = True` for `FatalTraCIError`. The `_compute_gae` function uses `1.0 - dones[t]` where `done` maps to `terminated` in the Transition store. Truncation bootstrapping is only applied when `episode_truncated and not episode_terminated` — demand exhaustion is coded as terminated, bypassing the bootstrap path.

**Prevention:**
Fix the route file before extended training: change `end="300"` to `end="600"` (or match `--seconds`) in `demand.rou.xml`. This eliminates the exhaustion trigger. Alternatively, re-classify demand exhaustion as `truncated=True` (not terminated) so the critic bootstraps from the last valid observation. The latter is semantically more correct — demand exhaustion is an environment truncation, not a true terminal state.

**Warning signs:**
- `error_events.jsonl` shows `FatalTraCIError` entries consistently at step ~57 across episodes.
- Critic loss is systematically higher for episodes that triggered demand exhaustion vs. those that completed normally.
- Training on longer `--seconds` values (e.g., 600) fails silently or produces degraded performance without obvious explanation.

**Phase:** Fix before any curriculum training run. Fixing the route file is the correct resolution — it also enables longer episode horizons needed for peak-hour dynamics.

---

### Pitfall C5: Max-Pressure Baseline Computed Twice Per Step During Training — Stale State Risk

**What goes wrong:**
In `local_mappo_backend.py`, `mp_actions = max_pressure.actions(obs, env=env)` is called immediately after `batch_out = trainer.act_batch(...)` (line 132), but before `env.step(actions)` is called. Max-Pressure reads live lane halting counts from TraCI at the time of the call. If the order of operations shifts (e.g., if MP is called post-step during residual reward calculation), the MP action will be computed from post-step state rather than pre-step state, creating an off-by-one misalignment between the MP baseline action and the MAPPO action it is being compared against.

During eval, `mp_actions = max_pressure.actions(obs, env=env)` is called again at line 383, but the MP baseline controller is also evaluated separately as `controller_name == "max_pressure"` in its own loop. The MP actions computed inside the MAPPO loop are only used for residual reward computation and are not recorded — this creates a latent correctness risk if someone adds residual reward to eval metrics.

**Prevention:**
When implementing action-level residual MAPPO, the MP action used as a feature input to the actor must be computed from the same pre-step environment state that produced the observation. Lock the computation order: `observe → compute MP action → feed [obs; mp_action] to actor → env.step()`. Add an assertion that `mp_action` is computed before `env.step()` in the action injection path.

**Warning signs:**
- Residual actor shows higher performance on even-numbered steps than odd-numbered steps (symptom of off-by-one in MP query timing).
- Unit tests comparing `mp_action` with post-step TraCI state produce different results than pre-step queries.

**Phase:** Action-level residual MAPPO implementation.

---

## Moderate Pitfalls

Mistakes that degrade performance or create confusing results without necessarily causing training failure.

---

### Pitfall M1: Curriculum Difficulty Ordering — Starting Too Hard Kills Early Learning

**What goes wrong:**
Curriculum training in RL requires that the policy encounter tractable problems before hard ones. In traffic signal control, "hard" scenarios (incidents, spillback, surges) have high variance rewards and long-horizon dependencies. If the curriculum starts with hard scenarios, the MAPPO critic cannot form meaningful value estimates early in training, entropy collapses, and the policy degenerates.

**Prevention:**
Order curriculum scenarios from low-complexity to high-complexity:
1. Off-peak, single time-of-day, uniform demand (close to current single-scenario baseline)
2. Peak-hour demand with higher volumes (same OD structure, scaled up)
3. Mixed time-of-day demand (morning vs. evening peak within one episode)
4. Incident scenarios (one lane blocked, demand surge at specific intersections)

Validate that the policy achieves near-Max-Pressure ratio on difficulty level 1 before advancing to level 2. If ratio does not reach 1.05x by episode 30 on level 1, the baseline convergence is not stable enough to benefit from harder scenarios.

**Warning signs:**
- Actor entropy drops below 0.3 nats before episode 10 of any curriculum stage.
- Performance on the current easy scenario regresses when a new hard scenario is added to the mix.

**Phase:** Curriculum demand file generation and training.

---

### Pitfall M2: Shared Actor With Heterogeneous Action Spaces Learns Phase Preference Artifacts

**What goes wrong:**
The parameter-sharing actor is trained on all 8 agents simultaneously. Agents with fewer valid phases (some intersections have 6, others 8) contribute transitions where certain action logits are always masked to `-inf`. The shared actor can develop an implicit preference against those logit positions even when they are valid for larger-action-space agents, because the gradient for those positions is dominated by agents where they are masked.

**Evidence in codebase:**
Action masking is applied per-agent in `act_batch()` and during `update()` via `invalid_mask`. The masking is correct. The risk is gradient contamination in the shared MLP from always-masked positions being updated to produce near-`-inf` logits.

**Prevention:**
This is the known limitation of parameter-sharing under heterogeneous action spaces. Monitor the distribution of logit values per position across agents. If agents with 6 phases consistently produce logits below -10 for positions 6-7, the shared actor has learned a spurious bias. The fix is either separate per-agent actors (breaks parameter sharing efficiency) or using a learned phase-count embedding as an additional input feature.

**Warning signs:**
- When adding a new intersection with a different phase count (e.g., via a new network topology in curriculum), eval performance degrades sharply.
- Logit values for positions `>= min(action_dims)` are consistently very negative even before masking is applied.

**Phase:** Action-level residual MAPPO (if one-hot MP action is appended and changes effective action dimensionality).

---

### Pitfall M3: Global Observation Dimension Explosion Under Curriculum Multi-Scenario

**What goes wrong:**
The centralized critic takes `global_obs = concatenate(all_agent_obs)` as input (see `flatten_obs_by_agent` in `env.py`). For the Toronto corridor with 8 active agents and `max_obs_dim = 38`, the global obs dimension is `8 * 38 = 304`. If curriculum introduces a scenario where a 9th intersection becomes active (e.g., a demand scenario covering additional streets), the global obs dimension jumps and the critic network becomes incompatible with the saved checkpoint.

This is not just a scaling concern — loading a checkpoint trained at `global_obs_dim=304` and running eval at `global_obs_dim=342` will fail silently if the load path does not validate network dimensions (PyTorch `load_state_dict` will raise a shape mismatch error if strict=True, but may silently drop keys if strict=False).

**Prevention:**
Lock the set of controlled intersections to a fixed list for the duration of training. Generate all curriculum demand scenarios for the same 8-intersection network topology. Do not introduce new intersections between curriculum stages. If expanding the network is needed, treat it as a new training run from scratch (or from a separate initialization).

**Warning signs:**
- A demand scenario causes a new agent ID to appear in `obs.keys()` that was not present in the original training initialization.
- `global_obs_dim` at checkpoint save time differs from `global_obs_dim` at eval time.

**Phase:** Headless demand generation CLI and curriculum demand file generation.

---

### Pitfall M4: Reward Shaping Misalignment Between Training Signal and Proposal KPI

**What goes wrong:**
The proposal success criterion is `person_time_loss_s(MAPPO) <= 0.90 * person_time_loss_s(Max-Pressure)`. The training reward (`objective` mode) uses `delay = edge.getWaitingTime()` as its delay proxy, which is accumulated waiting time (vehicles stopped). This is not the same as `time_loss_s`, which is the difference between actual travel time and free-flow travel time (includes moving-slow vehicles). A policy optimized on `edge.getWaitingTime()` can appear to improve on the training reward while `person_time_loss_s` stays flat or worsens.

**Evidence in codebase:**
The 2026-03-02 log entry shows that `reward_mode=time_loss` training (which uses `-log1p(delay)` where delay is waiting time) actually *worsened* `time_loss_s` ratio to 1.49x vs. 1.24x for `objective` mode — demonstrating the misalignment is real and measurable.

**Prevention:**
Use `reward_mode=time_loss` with `delay` computed from `edge.getLastStepMeanSpeed()` compared to free-flow speed, or compute the surrogate more carefully from `timeLoss` TraCI fields if available. Alternatively, add a KPI-aligned reward component: after each episode, compute the episode-level `person_time_loss_s` and add it as a terminal reward bonus. This aligns the dense per-step signal with the sparse KPI metric.

**Warning signs:**
- Training reward improves monotonically but eval `time_loss_s` ratio vs. Max-Pressure stays flat or regresses.
- `objective_mean_reward` increases while `person_time_loss_s` worsens.

**Phase:** All training phases. Monitor `time_loss_s` ratio in every eval run, not just training reward.

---

### Pitfall M5: Convergence Stagnation at 30 Episodes — Mistaking Variance for Trend

**What goes wrong:**
The 30-episode training curve (prototype log 2026-03-01) shows high variance rewards: episode 22 reaches `+0.0189`, episode 16 reaches `-0.2843`. The trend is upward but the signal-to-noise ratio is low enough that 30 episodes cannot confirm convergence. Declaring a policy "converged" at 30 episodes and moving to curriculum training means the starting policy for curriculum is not yet a stable baseline — curriculum training will be confounded by residual learning from scratch.

**Prevention:**
Train the single-scenario baseline to at least 100 episodes (RTX 4070 is SUMO-limited, not GPU-limited; this should complete in under 2 hours) before evaluating convergence. Compute a rolling mean over the last 10 episodes as the convergence criterion. Target: rolling mean ratio `< 1.15x` vs. Max-Pressure for 3 consecutive 10-episode windows. Only then is the policy stable enough to bootstrap curriculum training from.

**Warning signs:**
- Episode-to-episode reward variance is comparable in magnitude to the total improvement over 30 episodes.
- Eval `time_loss_s` ratio changes by more than 0.1x between consecutive 10-episode eval windows on the same checkpoint.

**Phase:** All training phases. Pre-requisite check before launching curriculum.

---

### Pitfall M6: Action-Level Residual Warm-Start vs. From-Scratch Initialization

**What goes wrong:**
If action-level residual MAPPO is initialized from the existing `objective`-mode checkpoint (which never saw the MP action as input), the actor weights will have converged for a different input dimensionality. Appending the MP one-hot vector increases `obs_dim`, making the existing checkpoint incompatible. Re-using the checkpoint by padding new input weights with zeros is technically possible but creates an initialization where the MP one-hot signal starts at zero weight — the network effectively ignores it at first, negating the warm-start benefit.

**Prevention:**
Treat action-level residual MAPPO as a new initialization. The correct warm-start for residual RL is to initialize the policy's non-MP-related weights from the existing checkpoint and initialize the weights corresponding to the new MP one-hot input features from scratch (small random values). This requires manual weight surgery on the first Linear layer of the actor. Document this explicitly in the implementation so the checkpoint loading code handles the dimension change without silent shape mismatch.

Alternatively, start from scratch and compensate by running more episodes — the policy should learn MP imitation rapidly because the task is shaped to encourage it.

**Warning signs:**
- `trainer.load(checkpoint)` raises `RuntimeError: size mismatch` when the new actor has expanded input dim.
- Loading with `strict=False` silently drops the first layer weights, starting from random initialization with no warm-start benefit.

**Phase:** Action-level residual MAPPO implementation.

---

## Minor Pitfalls

Mistakes that create friction without causing correctness failures.

---

### Pitfall m1: Checkpoint Scenario ID Not Persisted — Curriculum Checkpoints Overwrite Each Other

**What goes wrong:**
All training runs save to the same path (e.g., `models/asce_mappo_toronto_demand.pt`). If curriculum training runs multiple scenarios sequentially in one session, each scenario's final checkpoint overwrites the previous. An interrupted curriculum run cannot resume from mid-curriculum.

**Prevention:**
Include `scenario_id` in the checkpoint filename for curriculum runs: `models/asce_mappo_{scenario_id}_ep{episode}.pt`. Save an intermediate checkpoint every 20 episodes (configurable via `--checkpoint-interval`).

**Phase:** Curriculum training.

---

### Pitfall m2: Max-Pressure Called With `env=env` After TraCI Socket Closes

**What goes wrong:**
In the `FatalTraCIError` handler, `mp_actions = max_pressure.actions(obs, env=env)` has already been called earlier in the loop iteration (line 132). But if the error handling path is changed to re-query MP after the exception, the TraCI socket is closed and `max_pressure.actions()` will raise a second `FatalTraCIError`, causing an unhandled exception rather than a clean episode termination.

**Prevention:**
The current code is correct (MP is called before `env.step()` and before the exception can occur). Any refactor that moves the MP call to a post-step position must account for the socket-closed case and use the last known `mp_actions` dict as a fallback.

**Phase:** Residual MAPPO implementation (likely to refactor the MP call order).

---

### Pitfall m3: `dataset.parquet` Logging Mixed with Curriculum Rollouts

**What goes wrong:**
If PIRA dataset logging is added during curriculum training, rollout rows from different scenarios will be concatenated into one parquet file without a `scenario_id` column differentiation. PIRA training data without scenario labels is unlabeled and cannot be used for scenario-level prediction.

**Prevention:**
Ensure every rollout row includes `scenario_id` (already in `IntersectionMetrics`) and that the parquet write path partitions by scenario. Verify the `ASCE_DATASET_COLUMNS` schema includes `scenario_id` before the first curriculum run.

**Phase:** Dataset.parquet logging.

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| Action-level residual MAPPO | C1: Asymmetric penalty trap | Inject MP one-hot into actor input; don't rely on reward penalty alone |
| Action-level residual MAPPO | C5: MP computed from stale state | Lock observation-gather → MP-query → actor-forward order |
| Action-level residual MAPPO | M6: Warm-start checkpoint mismatch | New initialization with manual weight surgery on first layer |
| Curriculum demand generation | C3: Seed confounding vs. scenario variation | Log `demand_file_hash` per episode; enforce held-out test scenario |
| Curriculum demand generation | M3: Global obs dim explosion | Lock to 8-intersection network; never introduce new agents mid-curriculum |
| Curriculum training | C4: Demand exhaustion / GAE contamination | Fix `end="300"` → `end="600"` in route files before any curriculum run |
| Curriculum training | C2: Welford normalizer poisoned by padding | Apply normalizer to raw obs before padding; or mask update to valid dims |
| Curriculum training | M1: Starting curriculum too hard | Stage difficulty; validate ratio < 1.15x on easy scenario before hard |
| Curriculum training | M5: 30 episodes is insufficient baseline | Train to 100+ episodes before bootstrapping curriculum |
| Training reward vs. KPI | M4: Reward proxy misaligns with proposal KPI | Monitor `time_loss_s` ratio at every eval; not just training reward |
| Dataset.parquet logging | m3: Unlabeled curriculum rollouts | Validate `scenario_id` column present before first write |
| All training runs | m1: Checkpoint overwrites | Include `scenario_id` and episode count in checkpoint filename |

---

## Sources

All findings are grounded in direct codebase evidence:

- `/home/as04/ece324-TANGO/.planning/codebase/CONCERNS.md` — Known bugs, tech debt, and scaling limits (HIGH confidence)
- `/home/as04/ece324-TANGO/docs/notes/prototype_log.md` — Empirical evidence from training runs on Toronto corridor (HIGH confidence)
- `/home/as04/ece324-TANGO/ece324_tango/asce/mappo.py` — GAE implementation and action masking (HIGH confidence)
- `/home/as04/ece324-TANGO/ece324_tango/asce/trainers/local_mappo_backend.py` — Training loop, FatalTraCIError handling, MP call order (HIGH confidence)
- `/home/as04/ece324-TANGO/ece324_tango/asce/traffic_metrics.py` — Reward computation and fallback chain (HIGH confidence)
- `/home/as04/ece324-TANGO/.planning/PROJECT.md` — Milestone requirements and constraints (HIGH confidence)
- Established MARL and residual RL theory: asymmetric reward shaping, curriculum ordering, parameter-sharing under heterogeneous action spaces (MEDIUM confidence — training knowledge, not web-verified)
