# TANGO Runbook

## Last Updated
2026-03-01

## Environment
- Package manager: pixi
- GPU path: `pytorch-gpu` + `cuda-version=12.8.*` + `[system-requirements] cuda = "12"`

## Pixi Dependency Rules
- Always add/edit dependencies via Pixi manifest operations, not ad-hoc pip:
  - Conda package: `pixi add <package>`
  - PyPI package: `pixi add --pypi <package>`
  - Install/sync lockfile: `pixi install`
- Never use `pixi run pip install ...` for project dependencies.
- If a package must come from a non-default conda channel, use channel-qualified specs (for example `pixi add "pytorch [channel='pytorch']"`).
- Commit both `pixi.toml` and `pixi.lock` together whenever dependencies change.

## Canonical Commands
- Install env: `pixi install`
- Run tests: `pixi run pytest tests`
- Train ASCE: `pixi run python -m ece324_tango.modeling.train --device auto`
- Eval ASCE: `pixi run python -m ece324_tango.modeling.predict --device auto`
- Train Toronto (TMC demand): `pixi run train-asce-toronto-demand`
- Eval Toronto (TMC demand): `pixi run eval-asce-toronto-demand`
- Train Toronto (random trips): `pixi run train-asce-toronto-random`
- Validate schema: `pixi run python -m ece324_tango.dataset`
- Backend verbose logs (optional): add `--backend-verbose`
- Use objective reward shaping (default): `--reward-mode objective`
- Use SUMO-native rewards: `--reward-mode sumo`
- Use delay-only surrogate for proposal KPI alignment: `--reward-mode time_loss`
- Use residual objective around Max-Pressure: `--reward-mode residual_mp`

## Backend
- The training and evaluation CLI now run only the local MAPPO backend.
- There is no backend selection flag anymore; correctness fixes land in the local path directly.

## Artifact Paths
- Model checkpoint: `models/asce_mappo.pt`
- Toronto demand checkpoint: `models/asce_mappo_toronto_demand.pt`
- Toronto random checkpoint: `models/asce_mappo_toronto_random.pt`
- Rollout dataset: `data/processed/asce_rollout_samples.csv`
- Toronto demand rollout: `data/processed/asce_rollout_toronto_demand.csv`
- Toronto random rollout: `data/processed/asce_rollout_toronto_random.csv`
- Train metrics: `reports/results/asce_train_episode_metrics.csv`
- Eval metrics: `reports/results/asce_eval_metrics.csv`
- Toronto demand eval metrics: `reports/results/asce_eval_metrics_toronto_demand.csv`
- Non-fatal exception/fallback log: `reports/results/error_events.jsonl`

## Toronto SUMO Assets
- Integrated from `origin/data-setup` under:
  - `sumo/network/osm.net.xml.gz` (source archive)
  - `sumo/network/osm.net.xml` (unzipped net file used by training/eval)
  - `sumo/demand/demand.rou.xml` (TMC-derived flows)
  - `sumo/demand/random_trips.rou.xml` (random flow baseline)
- Use `sumo/network/osm.net.xml` as `--net-file` for both route files.
- If `osm.net.xml` is missing locally, regenerate it with:
  - `gzip -dk -f sumo/network/osm.net.xml.gz`

## Known Risks
- Third-party deprecation warnings from `torchrl` appear during pytest collection. They do not currently affect run correctness but should be tracked for dependency updates.

## Reward Objective
- `reward_mode`: `objective` (default), `sumo`, `time_loss`, or `residual_mp`
- Objective reward:
  - `- reward_delay_weight * log1p(delay)`
  - `+ reward_throughput_weight * log1p(throughput)`
  - `+ reward_fairness_weight * Jain(throughput across intersections)`
- Time-loss mode reward:
  - `- reward_delay_weight * log1p(delay)` (normalized for stability)
- Residual Max-Pressure mode reward:
  - `objective_reward - reward_residual_weight * 1[action != action_max_pressure]`
  - Encourages ASCE to learn selective deviations instead of fully replacing Max-Pressure.

## Observation Normalization
- Enabled by default (`--use-obs-norm`).
- Disable explicitly with `--no-use-obs-norm`.
- `ObsRunningNorm` (Welford online, per-feature) is applied to padded local obs and global obs.
- Stats are saved under keys `"obs_norm"` / `"gobs_norm"` in the model checkpoint.
- Eval path validates `use_obs_norm` parity with checkpoint metadata and fails fast on mismatch.

## GPU Notes
- RTX 4070 Laptop, CUDA 12.6. Use `--device auto` (default) to pick GPU automatically.
- All N-agent observations are batched into a single GPU forward pass per step (`act_batch()`).
- Training throughput is SUMO-limited (CPU), not GPU-limited. Longer episodes give more data.
- To maximize data per wall-clock minute: increase `--episodes` rather than `--seconds`
  until the demand exhaustion issue is resolved (see Known Issue below).

## Known Bugs Fixed (2026-03-01)
- **Action space crippled**: `n_actions` was `min(action_dims)` = 2; 10/12 agents could
  never select phases 2-5. Fixed: use `max` + `-inf` masking in `act()` / `act_batch()`.
- **GAE bias on truncation**: `last_value` was hardcoded 0.0; now bootstrapped from critic.
- **Credit assignment lost**: all agents received global mean reward; now each gets own reward.

## Known Issue: SUMO Demand Exhaustion (FatalTraCIError)
- **Symptom**: `traci.exceptions.FatalTraCIError: Connection closed by SUMO` at ~step 57
  (~285 s into a 300 s episode).
- **Root cause**: `demand.rou.xml` defines finite vehicle flows that all complete their trips
  before the simulation clock reaches `--seconds`. SUMO has nothing left to simulate and
  exits, closing the TraCI TCP socket. The next `env.step()` call hits the closed socket.
- **Current mitigation**: `env.step()` is wrapped in `try/except FatalTraCIError`; the
  episode is treated as done and training continues normally. A warning is logged.
- **To fix properly** (needed for `--seconds 600` runs):
  - Extend flow departure window in the route file: change `end="300"` → `end="600"` (or
    match `--seconds`).
  - Or keep SUMO alive with `--until <seconds>` in `sumo/config/baseline.sumocfg`.
  - `random_trips.rou.xml` may or may not have the same issue — verify before long runs.

## Proposal KPI Path
- Evaluation CSV now includes proposal-aligned fields:
  - `time_loss_s`
  - `person_time_loss_s`
  - `avg_trip_time_s`
  - `arrived_vehicles`
  - `objective_mean_reward` / `objective_delay_proxy` / `objective_throughput_proxy` / `objective_fairness_jain`
- Current occupancy heuristic:
  - transit-like IDs (`bus|tram|streetcar|ttc`) => 30 persons
  - otherwise => 1.3 persons

## Handoff Checklist
1. Confirm latest good commit hash.
2. Confirm local backend command used for latest run.
3. Attach latest command line used for train/eval.
4. Record open risks and next 3 tasks in `docs/notes/prototype_log.md`.
