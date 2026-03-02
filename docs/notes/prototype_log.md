# Prototype Log

## 2026-02-19
- Implemented Phase 0 scaffolding under `ece324_tango` package.
- Added pixi tasks for ASCE train/eval/schema validation.
- Implemented Phase 1 minimal ASCE MAPPO pipeline:
  - shared actor MLP and centralized critic,
  - sample SUMO network bootstrap via sumo-rl package data,
  - fixed-time and queue-greedy baseline eval script,
  - rollout logging to the agreed dataset schema.
- Added schema tests in `tests/test_data.py`.
- Added GPU device flag support in training/eval (`--device auto|cuda|cpu`).
- Migrated pixi GPU setup to conda-forge style with `pytorch-gpu`, `cuda-version = 12.8.*`, and `[system-requirements] cuda = \"12\"`.
- Verified in pixi env: `torch.cuda.is_available() == True` on RTX 4070 Laptop GPU.
- Replaced max-pressure proxy baseline with edge-level max-pressure controller using controlled TLS links and per-lane halting counts from TraCI.

## Follow-up
- Replace proxy max-pressure with true edge-level pressure from traci edge graph.
- Replace proxy NS/EW queue/arrival/speed values with direct per-approach aggregations.
- Add person-weighted delay and timeLoss extraction for proposal-aligned KPI reporting.

## 2026-02-19 (backend abstraction pass)
- Added trainer backend architecture under `ece324_tango/asce/trainers/`.
- Rewired `ece324_tango/modeling/train.py` and `ece324_tango/modeling/predict.py` to use backend selection:
  - `trainer_backend=local_mappo|benchmarl|xuance`
- Added runtime utility helpers in `ece324_tango/asce/runtime.py`.
- Added spike backend modules for BenchMARL/Xuance with package gating and parity fallback to local pipeline.
- Added test coverage for backend factory selection, max-pressure action validity, and device resolution.
- Added strict handoff memory docs:
  - `docs/notes/runbook.md`
  - `docs/notes/adr/ADR-0001-backend-strategy.md`

## Next 3 tasks
1. Implement native Xuance custom MARL env adapter (replace fallback).
2. Add integration smoke tests for backend-specific runs under optional test marks.
3. Reduce BenchMARL runtime overhead and noisy logging in sample-network mode.

## 2026-02-20 (pixi workflow correction)
- Corrected dependency-management policy:
  - Use `pixi add` for conda dependencies.
  - Use `pixi add --pypi` for PyPI dependencies.
  - Use `pixi install` to sync/update the environment from manifest + lock.
- Explicitly banned `pixi run pip install ...` for project dependency management in runbook.
- Added `benchmarl` and `xuance` via `pixi add --pypi ...`, updating both `pixi.toml` and `pixi.lock`.

## 2026-02-20 (native BenchMARL adapter)
- Replaced BenchMARL fallback with a native path:
  - Added `SumoParallelAdapter` + `SumoBenchmarlTask` under `ece324_tango/asce/trainers/benchmarl_task.py`.
  - BenchMARL backend now runs `benchmarl` MAPPO on SUMO through TorchRL PettingZoo wrapper.
  - BenchMARL train now saves a backend-tagged checkpoint payload and writes rollout/schema CSV + episode metrics CSV.
  - BenchMARL eval now loads checkpoint payload, evaluates MAPPO policy, and reports alongside fixed-time and max-pressure baselines.
- Added adapter unit tests in `tests/test_benchmarl_adapter.py`.
- Smoke validated:
  - `train --trainer-backend benchmarl` succeeds on sample network.
  - `predict --trainer-backend benchmarl` succeeds and writes metrics CSV.

## Remaining from plan
1. Add backend-marked integration tests (optional/slow marker) for benchmarl and xuance train/eval CLI runs.
2. Reduce BenchMARL training noise/runtime overhead and tighten env close lifecycle handling.
3. Improve Xuance adapter robustness so value normalization/GAE can be safely re-enabled for stronger baseline parity.

## 2026-02-20 (native Xuance adapter)
- Added OpenMPI to pixi environment to satisfy Xuance runtime dependency (`libmpi.so`).
- Replaced Xuance fallback with native path:
  - Added custom SUMO environment registration for Xuance in `ece324_tango/asce/trainers/xuance_env.py`.
  - Implemented native Xuance MAPPO train/eval backend in `ece324_tango/asce/trainers/xuance_backend.py`.
  - Training now exports both a direct model artifact (`models/*.pt`) and Xuance-compatible checkpoint folder (`*_xuance/seed_export`).
  - Evaluation loads the Xuance checkpoint folder and reports MAPPO against fixed-time and max-pressure baselines.
- Validated via short smoke runs on sample SUMO network for:
  - `--trainer-backend xuance` training CLI.
  - `--trainer-backend xuance` evaluation CLI.

## 2026-02-20 (integration tests)
- Added backend integration test module: `tests/test_backend_integration_slow.py`.
- Coverage includes both `benchmarl` and `xuance` CLI training/evaluation smoke paths.
- Tests are marked `slow` + `integration` and gated by env var:
  - run with `RUN_SLOW_INTEGRATION=1 pixi run pytest tests/test_backend_integration_slow.py`
  - default `pixi run pytest tests` remains fast (integration tests skip unless enabled).

## 2026-02-20 (noise controls + Xuance re-check)
- Added config-gated backend noise control:
  - CLI flag: `--backend-verbose` (default quiet mode).
  - Backends now pass quiet SUMO settings by default (`sumo_warnings=False`, `--no-step-log true`).
  - Added helper: `ece324_tango/asce/trainers/noise_control.py`.
- Re-checked Xuance settings matrix on SUMO adapter:
  - `use_gae=True, use_value_norm=False`: stable.
  - `use_value_norm=True` (with or without GAE): unstable shape/type failures in Xuance on-policy buffer path.
- Updated Xuance backend defaults accordingly and exposed env-var gates:
  - `TANGO_XUANCE_USE_GAE` (default `1`)
  - `TANGO_XUANCE_USE_VALUE_NORM` (now default `1` with local compat patch)
  - `TANGO_XUANCE_USE_ADVNORM` (default `0`)

## 2026-02-20 (local Xuance compat patch + fair compare)
- Added local runtime patch for Xuance on-policy buffer value normalization shape handling:
  - `ece324_tango/asce/trainers/xuance_compat.py`
  - toggle: `TANGO_XUANCE_PATCH_VALUE_NORM` (default `1`)
- Verified Xuance train works with `TANGO_XUANCE_USE_VALUE_NORM=1` and `TANGO_XUANCE_USE_GAE=1`.
- Fair micro-benchmark (cpu, 1 episode, 30s sim, 1 eval episode):
  - BenchMARL: train 66.267s, eval 37.809s, MAPPO mean_reward -0.000208
  - Xuance (patched): train 41.256s, eval 41.003s, MAPPO mean_reward -0.000208

## 2026-02-20 (Xuance custom-config path, no MPE bootstrap)
- Switched Xuance backend config load from framework MPE default bootstrap to project-owned config file:
  - `ece324_tango/asce/trainers/configs/xuance_mappo_sumo.yaml`
  - backend now calls `get_arguments(..., config_path=<project yaml>)`.
- Kept runtime overrides for scenario-specific values (seed, device, SUMO files/timing, quiet mode).
- Revalidated with smoke runs:
  - `train --trainer-backend xuance --episodes 1 --seconds 30 --delta-time 5`
  - `predict --trainer-backend xuance --episodes 1 --seconds 30 --delta-time 5`

## 2026-02-20 (objective reward + TraCI metrics)
- Added shared traffic metrics/reward module: `ece324_tango/asce/traffic_metrics.py`.
  - Computes per-intersection metrics from TraCI controlled links and edge-level aggregations.
  - Falls back to observation-based proxy splitting if TraCI edge mapping fails.
- Added objective reward mode with config gates:
  - CLI: `--reward-mode objective|sumo`
  - Weights: `--reward-delay-weight`, `--reward-throughput-weight`, `--reward-fairness-weight`
- Wired objective reward into:
  - local MAPPO train/eval loops,
  - Xuance custom env wrapper,
  - BenchMARL SUMO adapter,
  - baseline comparisons in backend eval paths.
- Local and Xuance rollout logging now uses TraCI-derived metrics path (with fallback).

## 2026-02-20 (objective-mode backend micro-compare)
- Objective-mode micro run (1 episode, 30s, delta=5, cpu):
  - BenchMARL train: 41.86s, eval: 26.13s
  - Xuance train: 25.98s, eval: 35.17s
- Eval outputs were numerically aligned between BenchMARL and Xuance in this micro run:
  - MAPPO mean_reward: `0.116181`
  - fixed_time mean_reward: `0.116181`
  - max_pressure mean_reward: `0.132357`
- Note: in this sandbox, both BenchMARL and Xuance required unsandboxed execution for some runs due `/dev/shm` and network permission constraints.

## 2026-02-20 (benchmarl rollout realism + multi-seed compare)
- BenchMARL rollout export path upgraded:
  - replaced observation-proxy CSV conversion with action-replay in raw SUMO env + TraCI metric extraction.
  - file: `ece324_tango/asce/trainers/benchmarl_backend.py` (`_rollout_to_schema_rows_from_replay`).
- Verified BenchMARL rollout CSV now reports non-placeholder fields (`avg_speed_*`, `current_phase`) consistent with TraCI path.
- Multi-seed objective-mode compare (backend in {benchmarl, xuance}, seeds {7,17,27}, episodes=1, seconds=30, delta=5, cpu):
  - BenchMARL mean: train 49.377s, eval 27.468s, mean_reward 0.118996, throughput_proxy 11.423643, fairness 0.188181
  - Xuance mean: train 26.083s, eval 39.349s, mean_reward 0.121229, throughput_proxy 11.638025, fairness 0.187422
- Raw summary artifact: `/tmp/tango_backend_sweep/summary.csv`

## 2026-02-20 (benchmark automation)
- Added reproducible benchmark CLI:
  - module: `ece324_tango/modeling/benchmark_backends.py`
  - pixi task: `pixi run benchmark-backends`
- CLI writes per-seed and aggregate outputs under:
  - `reports/results/backend_compare/<run_id>/summary.csv`
  - `reports/results/backend_compare/<run_id>/aggregate.csv`

## Next 3 tasks (proposal-aligned)
1. Run 10-seed ASCE significance evaluation using newly added proposal KPI primitives (`timeLoss`, occupancy-weighted person delay, per-trip travel time) against max-pressure/fixed-time.
2. Integrate Toronto corridor SUMO assets + TMC calibration into canonical dataset generation command (scenario IDs: baseline/construction/transit) with schema-complete exports.
3. Start PIRA data pipeline: generate scenario-level labels from trained ASCE rollouts and implement first GNN surrogate baseline with MAPE/R2/inference tracking.

## 2026-02-20 (proposal KPI extraction)
- Added KPI tracker module: `ece324_tango/asce/kpi.py`
  - network `timeLoss` accumulation from TraCI (`vehicle.getTimeLoss` deltas),
  - occupancy-weighted person delay (`person_time_loss_s`),
  - average trip time for arrived vehicles (`avg_trip_time_s`).
- Wired KPI fields into eval records for:
  - `local_mappo`,
  - `xuance`,
  - `benchmarl` (including rollout-action replay path).

## 2026-02-20 (exception handling hardening)
- Added shared error reporting utility: `ece324_tango/error_reporting.py`.
- Non-fatal exceptions and fallback paths now:
  - emit warning logs,
  - persist structured error events to `reports/results/error_events.jsonl`.
- Removed silent swallow behavior in current broad-exception fallback sites (runtime phase fallback, traffic-metric fallback, KPI lookup fallbacks, Xuance runner close path, optional tqdm setup).

## 2026-02-20 (LibSignal backend assessment)
- Reviewed LibSignal codebase (`run.py`, `trainer/tsc_trainer.py`, registry/config pipeline) for backend fit.
- Added `libsignal` backend name to factory as an explicit placeholder:
  - file: `ece324_tango/asce/trainers/libsignal_backend.py`
  - behavior: fail-fast runtime error with integration blocker context.
- Added assessment note:
  - `docs/notes/libsignal_backend_assessment.md`
  - decision: treat LibSignal as deferred backend until adapter can map our `net_file`/`route_file` and schema/KPI outputs.

## 2026-03-01 (Toronto asset integration + correctness fixes)
- Integrated teammate SUMO corridor assets from `origin/data-setup` without merging the whole branch (to avoid overwriting current ASCE backend code):
  - `sumo/network/osm.net.xml.gz` + unzipped `sumo/network/osm.net.xml`
  - `sumo/demand/demand.rou.xml`
  - `sumo/demand/random_trips.rou.xml`
  - `sumo/config/baseline.sumocfg`
- Added pixi tasks for Toronto corridor smoke runs:
  - `train-asce-toronto-demand`
  - `eval-asce-toronto-demand`
  - `train-asce-toronto-random`
- Removed unused scaffold package directory: `ece324_tango_model/`.
- Investigated train/eval warnings using local runtime + upstream `sumo-rl` source:
  - root cause confirmed: `traci.edge.getShape` is unavailable; lane geometry must be read from `traci.lane.getShape`.
- Fixed ASCE metric extraction correctness:
  - Replaced edge-shape axis inference with lane-shape voting per incoming edge in `traffic_metrics.py`.
  - Stopped swallowing unexpected programming errors in `compute_metrics_for_agent` (fallback now only for expected runtime/TraCI-style extraction failures).
- Fixed local MAPPO for real corridor topology:
  - root cause: heterogeneous per-intersection observation lengths caused actor shape mismatch.
  - added zero-padding to shared actor input (`pad_observation`) and wired local train/eval to use max per-agent observation length.
- Hardened eval failure path:
  - `LocalMappoBackend.evaluate` now checks model existence before creating SUMO env.
  - wrapped per-controller env lifecycle in `try/finally` close.
- Validation evidence:
  - `pixi run pytest tests` => pass (`26 passed, 2 skipped`)
  - local MAPPO train/eval on sample SUMO network succeeds with KPI fields in eval CSV.
  - local MAPPO train/eval on Toronto `demand.rou.xml` succeeds and writes rollout/train/eval artifacts.

## 2026-03-01 (30-episode obs-norm training on Toronto demand)
- Trained local MAPPO for 30 episodes × 300s on Toronto TMC demand with --use-obs-norm.
- Batch actor inference (act_batch) reduces GPU round-trips from 12 to 1 per step.
- Two bugs fixed in this session:
  - `FatalTraCIError` (SUMO demand exhaustion at ~285s into 300s episode) now caught per-step and treated as done=True, allowing episode to close cleanly and training to continue.
  - `torch.load` called with `weights_only=False` to handle numpy arrays in obs-norm checkpoint payload (PyTorch 2.6 default changed to `weights_only=True`).
- Episode reward progression (mean_global_reward per episode):
  - ep0: -0.2804, ep1: -0.1439, ep2: -0.1332, ep3: -0.1598, ep4: -0.2080
  - ep5: -0.1892, ep6: -0.1618, ep7: -0.0427, ep8: -0.1910, ep9: -0.1448
  - ep10: -0.1084, ep11: -0.1516, ep12: -0.2604, ep13: -0.1863, ep14: -0.1827
  - ep15: -0.2315, ep16: -0.2843, ep17: -0.0653, ep18: -0.0954, ep19: -0.2449
  - ep20: -0.0949, ep21: -0.1625, ep22: +0.0189, ep23: -0.1128, ep24: +0.0020
  - ep25: -0.0211, ep26: -0.0527, ep27: +0.0497, ep28: -0.0618, ep29: -0.0163
- Eval results vs baselines (seed 17, 1 episode, 120s):
  - MAPPO:        time_loss_s=2993.75, arrived=42, mean_reward=-0.0888
  - Fixed-time:   time_loss_s=2244.61, arrived=45
  - Max-pressure: time_loss_s=1783.50, arrived=47
- time_loss ratio (MAPPO / max_pressure): 1.68x (previous: 1.82x from 10-episode run)
- Remaining gap to proposal target (≤0.90x): 0.78x

## 2026-03-02 (MAPPO correctness hardening + parity checks)
- Fixed a core PPO correctness issue in local MAPPO:
  - action masking is now applied consistently in both action collection and PPO update.
  - `n_valid_actions` is now tracked per transition and used in `MAPPOTrainer.update()`.
- Fixed truncation handling semantics:
  - runtime now exposes `extract_step_details()` with `terminated`/`truncated` flags.
  - local training now bootstraps critic values on truncation using final observations.
  - GAE terminal mask now uses true termination only (not generic done).
- Fixed backend evaluation seed parity:
  - BenchMARL MAPPO eval now builds per-episode experiments with `seed + ep`.
  - Xuance MAPPO eval now passes `episode_seed=seed + ep` through rollout execution.
- Observation normalization operational safety:
  - `use_obs_norm` is now persisted in checkpoints.
  - local eval now validates checkpoint `use_obs_norm` parity before env creation.
  - CLI default switched to obs normalization on (`--use-obs-norm/--no-use-obs-norm`).
- Added new regression tests:
  - `tests/test_mappo_core.py`
  - `tests/test_local_backend_bootstrap.py`
  - `tests/test_runtime_step_flags.py`
  - `tests/test_backend_eval_seeding.py`
  - `tests/test_obs_norm_parity.py`
  - `tests/test_cli_obs_norm_defaults.py`
  - `tests/test_local_eval_fallback_observation_alignment.py`

## 2026-03-02 (time-loss reward mode + objective-scored baseline eval)
- Added new reward mode:
  - `reward_mode=time_loss` in `rewards_from_metrics()`
  - semantics: per-agent reward `= -reward_delay_weight * delay` (delay-only surrogate to align with proposal `time_loss_s`)
  - unsupported reward-mode strings now fail fast with a clear `ValueError`.
- Added cross-controller objective-scored eval outputs:
  - local/benchmarl/xuance eval CSV rows now include:
    - `objective_mean_reward`
    - `objective_delay_proxy`
    - `objective_throughput_proxy`
    - `objective_fairness_jain`
  - this allows direct inspection of baseline (`max_pressure`) performance on MAPPO-style objective shaping, independent of active reward mode.
- CLI updates:
  - `train.py`, `predict.py`, and `benchmark_backends.py` now expose and validate `objective|sumo|time_loss` reward modes.
- New tests:
  - `tests/test_local_eval_objective_scoring.py`
  - extended `tests/test_traffic_metrics.py` for `time_loss` semantics + invalid-mode guard
  - extended `tests/test_cli_obs_norm_defaults.py` to lock reward-mode availability.
- Validation:
  - `pixi run pytest tests -q` => `50 passed, 2 skipped`.
- Toronto demand experiment (local backend):
  - Train: 30 episodes, 300s, `--reward-mode time_loss`, checkpoint:
    - `models/asce_mappo_toronto_demand_time_loss.pt`
  - Eval: 10 episodes, 300s, CSV:
    - `reports/results/asce_eval_metrics_toronto_demand_time_loss_e10.csv`
  - Aggregate means:
    - `time_loss_s`:
      - MAPPO: `6013.92`
      - Fixed-time: `6184.94`
      - Max-pressure: `4353.33` (best baseline)
      - ratio `MAPPO / best_baseline = 1.3815` (target `<= 0.90` not met)
    - `objective_mean_reward`:
      - MAPPO: `-0.0094`
      - Fixed-time: `0.1804`
      - Max-pressure: `0.4077`
      - max-pressure leads on objective shaping as well.
- Warnings observed:
  - repeated per-episode `FatalTraCIError` closure at step ~57 due demand exhaustion; handled as episode end (non-fatal, expected in this demand file).

## 2026-03-02 (time-loss normalization rerun)
- Updated `reward_mode=time_loss` to use normalized delay:
  - `-reward_delay_weight * log1p(delay)` (instead of raw `-delay`) for PPO stability.
- Validation:
  - `pixi run pytest tests -q` => `50 passed, 2 skipped`.
- Reran Toronto demand training/eval:
  - Train: 30 episodes, 300s, local backend, obs-norm on
  - Eval: 10 episodes, 300s
- Stability improvement:
  - `critic_loss` mean dropped from ~205k (raw-delay run) to ~425 (normalized run).
- KPI outcome:
  - `time_loss_s` means:
    - MAPPO: `6487.80`
    - Fixed-time: `6184.94`
    - Max-pressure: `4353.33`
  - ratio `MAPPO / best_baseline = 1.4903` (worse than previous normalized-objective and raw-delay runs; still above target `<=0.90`).

## 2026-03-02 (objective-mode retest after regression concern)
- Trigger:
  - user concern that recent behavior changes may have broken objective-mode correctness.
- Check performed:
  - reran 10-episode objective eval with existing objective-trained checkpoint:
    - `models/asce_mappo_toronto_demand.pt`
    - output: `reports/results/asce_eval_metrics_toronto_demand_objective_retest_e10.csv`
- Result:
  - objective-mode behavior is consistent with prior baseline run:
    - old objective run: MAPPO `time_loss_s=5463.34`, ratio vs best baseline `1.255`
    - retest objective run: MAPPO `time_loss_s=5405.39`, ratio vs best baseline `1.2417`
  - fixed-time and max-pressure metrics matched prior values in this setting.
- Explicit max-pressure-on-objective comparison:
  - now available directly in eval CSV via `objective_mean_reward`.
  - retest mean values:
    - MAPPO `objective_mean_reward=0.0748`
    - max-pressure `objective_mean_reward=0.4077` (higher)
  - conclusion: max-pressure still leads MAPPO on both `time_loss_s` and objective-shaped reward for this corridor setup.
