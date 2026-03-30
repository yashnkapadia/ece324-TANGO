# Codebase Concerns

**Analysis Date:** 2026-03-29

## Known Bugs

**SUMO Demand Exhaustion / FatalTraCIError:**
- Symptom: `traci.exceptions.FatalTraCIError: Connection closed by SUMO` at approximately step 57 (~285 s into a 300 s episode)
- Files: `ece324_tango/asce/trainers/local_mappo_backend.py` (lines 137-149), `ece324_tango/asce/env.py`
- Root cause: `demand.rou.xml` defines finite vehicle flows with `end="300"` that complete before the simulation clock reaches `--seconds`. SUMO has no more vehicles to generate and exits, closing the TraCI TCP socket.
- Current mitigation: `env.step()` is wrapped in `try/except FatalTraCIError`; the episode is treated as done and training continues. A warning is logged via `error_reporting.py`.
- Fix approach:
  - Extend flow departure window in route file: change `end="300"` → `end="600"` (or match `--seconds`).
  - Or keep SUMO alive with `--until <seconds>` in `sumo/config/baseline.sumocfg`.
  - Test `random_trips.rou.xml` for the same issue before long runs.
- Impact: Limits training/eval episodes to ~285 s practical duration; workaround prevents training failure but truncates data collection.

**Non-Local Backend Failures on Toronto Demand:**
- Files: `ece324_tango/asce/trainers/benchmarl_backend.py`, `ece324_tango/asce/trainers/xuance_backend.py`
- Symptom 1 (BenchMARL): `FatalTraCIError: Connection closed by SUMO` during TorchRL/PettingZoo step path on Toronto demand (even at `seconds=60`).
- Symptom 2 (Xuance): Heterogeneous observation packing error (`ValueError: setting an array element with a sequence`) including retry with `TANGO_XUANCE_USE_VALUE_NORM=0`.
- Status: Both failures occur in short-horizon runs, suggesting root causes beyond simple demand exhaustion timing.
- Impact: BenchMARL and Xuance backends are not production-stable for real corridor topologies; `local_mappo` is the only stable path. See `docs/notes/prototype_log.md` 2026-03-02 entry for evidence.
- Recommendation: Keep as spike/experimental paths pending fixes; do not use for production runs on Toronto network until failures are resolved.

## Performance Bottlenecks

**MAPPO Substantially Underperforms Max-Pressure Baseline:**
- Problem: MAPPO underperforms Max-Pressure by ~25% on person-time-loss in nominal single-mode (car-only) traffic on Toronto corridor
- Files: `ece324_tango/asce/mappo.py`, `ece324_tango/asce/trainers/local_mappo_backend.py`, `ece324_tango/asce/traffic_metrics.py`
- Cause: Max-Pressure is provably throughput-optimal for single-commodity demand. The current Toronto benchmark (70 uniform car flows on a simple corridor) is exactly the regime where Max-Pressure excels with its local queue-balancing heuristic.
- Evidence (from `docs/notes/prototype_log.md` 2026-03-02 objective retest):
  - MAPPO `time_loss_s=5405.39`, ratio vs best baseline `1.2417` (22.4% worse)
  - Previous 30-episode training: ratio `1.68x` (68% worse)
  - Expected gap to proposal target (≤0.90x): 0.78x remaining
- Improvement path:
  - Test MAPPO under demand perturbation scenarios where Max-Pressure's local heuristic breaks (incidents, spillback, non-stationary surges, network-level backlogs).
  - Introduce multi-modal demand (transit, pedestrians) on the `data-setup` branch.
  - Consider residual policy approach (see **residual_mp Reward Mode** concern below) to learn selective deviations from Max-Pressure.
  - Extend training horizon and tune MAPPO hyperparameters.

**Training Throughput Limited by SUMO CPU, Not GPU:**
- Problem: All N-agent observations are batched into a single GPU forward pass per step (`act_batch()`), yet training throughput remains CPU-limited by SUMO simulation
- Files: `ece324_tango/asce/trainers/local_mappo_backend.py` (line 127), `ece324_tango/asce/mappo.py` (lines 139-165)
- Evidence (from `docs/notes/runbook.md`): RTX 4070 Laptop GPU, CUDA 12.6. Longer episodes give more data. To maximize data per wall-clock minute, increase `--episodes` rather than `--seconds`.
- Impact: GPU is underutilized; no further optimization of neural network throughput will help until SUMO integration is addressed.

## Tech Debt

**Residual Max-Pressure Reward Mode Not Fully Implemented:**
- Issue: `residual_mp` reward mode exists but only applies reward-level penalization, not action-level learning
- Files: `ece324_tango/asce/traffic_metrics.py` (lines 296-303), `ece324_tango/asce/trainers/local_mappo_backend.py` (lines 187-191, 416-420)
- Current implementation: `reward -= weights.residual * float(mp_deviation_by_agent.get(agent_id, 0.0))` penalizes deviations from max-pressure *post-facto* in the reward function
- Limitation: The policy is not encouraged to *propose* actions based on max-pressure suggestion; it only gets penalized if it deviates. This is asymmetric — the policy never sees the max-pressure action as a suggested constraint during network forward pass.
- Fix approach:
  - Add an action-level inductive bias: concatenate max-pressure action (one-hot) into actor input or create a separate baseline policy output head.
  - Or use explicit action imitation loss during training phase to warm-start from max-pressure behavior.
  - Or frame as a true residual policy: train on `(action_rl - action_mp)` deltas rather than absolute actions.
- Impact: Current residual mode underutilizes the max-pressure heuristic; potential improvement path for corridor scenarios.

**Heterogeneous Observation Dimensions Across Intersections:**
- Problem: Per-intersection observation lengths range from 9–38 features depending on network topology
- Files: `ece324_tango/asce/env.py` (lines 59-70, `pad_observation`), `ece324_tango/asce/trainers/local_mappo_backend.py` (lines 64-66)
- Current mitigation: Zero-padding to `max(obs_dims)` before actor forward pass
- Limitation: Padding masks actual feature variance; Welford normalizer (in `obs_norm.py`) treats padded zeros as legitimate observations and converges them to near-zero mean/std, creating discontinuity in normalized feature space across networks
- Risk: Transfer learning across networks with different topologies will fail; padding overhead scales linearly with network complexity
- Fix approach:
  - Implement explicit masking in actor network (e.g., `nn.MultiheadAttention` with mask or graph convolution per incoming edges).
  - Or standardize observation extraction to fixed-size state (e.g., always 9 features) by lossy aggregation.
  - Test padding vs. masking impact on convergence speed and final reward.
- Impact: Moderate. Current approach works but is inelegant and may limit scalability to larger networks.

**Observation Normalization Checkpoint Mismatch Risk:**
- Problem: If `use_obs_norm` flag differs between training and eval, or checkpoint is loaded with wrong flag, training/eval diverges
- Files: `ece324_tango/asce/trainers/local_mappo_backend.py` (eval path), `ece324_tango/asce/obs_norm.py`, `ece324_tango/asce/mappo.py`
- Current safeguard: Local eval validates `use_obs_norm` parity and fails fast on mismatch (see `docs/notes/prototype_log.md` 2026-03-02)
- Limitation: Safeguard only in local backend; BenchMARL and Xuance backends may not have equivalent checks
- Fix approach:
  - Persist `use_obs_norm` as mandatory checkpoint metadata field.
  - Add runtime assertion in all backend eval paths before critic/actor forward.
  - Add integration test to catch mismatch across all backends.
- Impact: Low risk if local backend is used (safeguard in place); medium risk if spike backends are used in production.

## Fragile Areas

**Traffic Metrics Extraction Fallback Chain:**
- Files: `ece324_tango/asce/traffic_metrics.py` (lines 148-236)
- Fragility: Three-level fallback when TraCI edge lookup fails:
  1. Primary path: `_incoming_edges_for_ts()` → lane axis detection via `getShape()`
  2. Secondary path: If any lane-axis query fails, log exception and continue with majority vote
  3. Tertiary path: If full edge extraction fails, fall back to observation-based queue/arrival proxies
- Risk: Fallback heuristic (`split_ns_ew_from_obs()`) divides observation vector naively: assumes first half = NS, second half = EW. This is fragile if observation schema changes or edge topology is asymmetric.
- Evidence: Fixed in 2026-03-01 session (see `docs/notes/prototype_log.md`) by switching from `traci.edge.getShape` (unavailable) to `traci.lane.getShape` with lane-to-edge voting.
- Safe modification: Before changing observation schema or network topology, verify `compute_metrics_for_agent()` still finds edges correctly via logs; add a unit test for your specific network that validates edge detection.
- Test coverage: `tests/test_traffic_metrics.py` covers happy path; fallback paths are tested via integration tests on Toronto network.

**Max-Pressure Action Validity Checking:**
- Files: `ece324_tango/asce/baselines.py` (lines 66-101), `ece324_tango/asce/mappo.py` (line 126-127)
- Fragility: Max-Pressure selects best phase by iterating `range(n_actions)` where `n_actions = max(action_dims)`. If an agent's true action space is smaller, the selected action can be invalid.
- Fix applied (2026-03-01, see `docs/notes/prototype_log.md`): Action masking in actor forward pass (`logits[0, n_valid_actions:] = float("-inf")`) prevents MAPPO from selecting invalid actions. Max-Pressure baseline uses raw phase indices, not filtered.
- Safe modification: Max-Pressure actions are validated during rollout construction; ensure `action < action_dims[agent]` before using in environment. Currently safe because `baselines.py` iterates only valid phases and `env.step()` enforces action bounds.
- Test coverage: `tests/test_baseline_max_pressure.py` validates action space parity.

**Exception Logging Without Silent Swallow:**
- Files: `ece324_tango/error_reporting.py`, usage throughout `traffic_metrics.py`, `runtime.py`
- Fragility: Fallback paths now log exceptions to `reports/results/error_events.jsonl` instead of swallowing silently. If logging fails (missing directory, disk full), training continues with the fallback result.
- Risk: Silent recovery with incomplete error context could mask real bugs if error is unexpected (e.g., network topology change mid-run).
- Safe modification: Before changes to network topology or SUMO interface, verify that `error_reporting.jsonl` is being written and review its contents. Unexpected fallback triggers should be investigated.
- Test coverage: Integration tests on Toronto network trigger demand-exhaustion and fallback logging; verify `error_events.jsonl` is populated.

## Scaling Limits

**Observation Dimension Scaling with Network Size:**
- Limit: Per-agent observation length grows with number of incoming edges; current Toronto network (12 intersections) generates 9–38 feature observations
- Scaling factor: Rough O(n) with network size (more edges → more lanes → larger observations)
- Limit reached: No explicit bounds; padding overhead grows linearly. Actor network size is fixed (hardcoded 128 hidden), so input dimension explosion will eventually exceed reasonable batch sizes.
- Scaling path:
  - For networks > 50 intersections: move to graph-based feature aggregation (GNN) rather than dense padding.
  - Or implement hierarchical control: learn per-region coordinators that aggregate sub-network states.
  - Test threshold by profiling memory and wall-clock time on progressively larger test networks (20, 30, 50 intersections).

**Episode Duration and Demand Window Exhaustion:**
- Limit: Current route files define finite vehicle flows (end="300" s); practical episodes capped at ~285 s
- Scaling impact: Longer-horizon optimization (e.g., 600 s episodes for peak-hour dynamics) requires route file regeneration per scenario
- Scaling path:
  - Generate route files with longer departure windows or wrap-around periodic demand (Poisson arrivals).
  - Or implement on-the-fly vehicle generation via TraCI flow insertion.
  - Test on Toronto demand with `end="600"` before committing to longer horizons.

## Security Considerations

**No Threat Model Defined:**
- Status: ASCE is a research/prototype codebase; no explicit security hardening for production deployment
- Risk: Potential exposure if deployed in public-facing traffic system
  - Adversarial inputs: malformed TraCI commands, network topology injection
  - Model extraction: checkpoint files are unencrypted, weights are readable
  - Availability: SUMO simulation can be starved by resource-intensive policies
- Recommendations:
  - Do not expose ASCE to untrusted network input without input validation wrapper.
  - Treat model checkpoints (`models/*.pt`) as sensitive assets; restrict access.
  - Add timeout/resource limits for SUMO simulation to prevent runaway episodes.
  - Consider checkpoint encryption if deploying beyond research environment.

**Dependency Supply Chain Risk:**
- Files: `pixi.toml` (lines 25-28)
- Risk: Three framework dependencies (BenchMARL, Xuance, sumo-rl) are external and may have security updates
- Mitigation: Pixi lock file (`pixi.lock`) pins exact versions; updates are explicit
- Recommendations:
  - Review security advisories for `pytorch-gpu`, `sumo-rl`, `benchmarl`, `xuance` before major version upgrades
  - Consider removing unused frameworks (BenchMARL, Xuance) if stability issues are not resolved in next sprint

## Missing Critical Features

**PIRA (Scenario Planning GNN Surrogate) Not Implemented:**
- Problem: PIRA module is listed in project roadmap but not yet implemented
- Blocks: Scenario-level label generation, GNN surrogate model, KPI inference on unseen scenarios
- Impact: Cannot evaluate ASCE policy on hypothetical scenarios (e.g., "construction on Main St") without re-running full simulation
- Files: Placeholder mentioned in `docs/notes/prototype_log.md` (2026-02-20 notes, task 3); no implementation under `ece324_tango/`
- Recommendation: Defer to next phase; current focus is hardening ASCE baseline performance

**LibSignal Backend Not Integrated:**
- Problem: LibSignal is listed as potential backend but marked as deferred
- Files: `ece324_tango/asce/trainers/libsignal_backend.py` (placeholder), `docs/notes/libsignal_backend_assessment.md`
- Blocker: Adapter must map `net_file`/`route_file` to LibSignal format and translate schema/KPI outputs
- Recommendation: Defer; prioritize stability of existing backends

## Test Coverage Gaps

**Non-Local Backend Integration Testing Incomplete:**
- What's not tested: Full train/eval lifecycle on Toronto network for BenchMARL and Xuance
- Files: `tests/test_backend_integration_slow.py`
- Risk: BenchMARL and Xuance failures on real topologies are not caught in CI; only local backend is reliable
- Priority: High if spike backends are to be production-ready; low if they remain experimental
- Recommendation: Mark BenchMARL and Xuance as experimental in docs; block production use until integration test passes on Toronto network

**Residual Max-Pressure Training Not Tested:**
- What's not tested: `reward_mode=residual_mp` training does not have regression tests validating convergence or reward trajectory
- Files: `ece324_tango/asce/traffic_metrics.py`, `ece324_tango/asce/trainers/local_mappo_backend.py`
- Risk: Changes to reward computation could silently break residual mode behavior
- Test needed: Unit test comparing `residual_mp` vs. `objective` rewards under fixed max-pressure actions
- Priority: Medium; residual mode is not yet primary optimization target

**Action Masking Edge Cases Not Fully Covered:**
- What's not tested: Intersection with only 1 valid action; intersection with all actions masked (degenerate case)
- Files: `ece324_tango/asce/mappo.py` (line 126-127), `tests/test_baseline_max_pressure.py`
- Risk: Edge case could cause silent NaN/Inf in Categorical distribution if logits are all `-inf`
- Test needed: Unit test with `n_valid_actions=1` and verify actor samples deterministically
- Priority: Low; current Toronto network has 6-8 valid actions per intersection (not edge case-prone)

---

*Concerns audit: 2026-03-29*
