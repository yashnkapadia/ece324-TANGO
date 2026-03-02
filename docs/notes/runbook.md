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
- Train ASCE: `pixi run python -m ece324_tango.modeling.train --trainer-backend local_mappo --device auto`
- Eval ASCE: `pixi run python -m ece324_tango.modeling.predict --trainer-backend local_mappo --device auto`
- Train Toronto (TMC demand): `pixi run train-asce-toronto-demand`
- Eval Toronto (TMC demand): `pixi run eval-asce-toronto-demand`
- Train Toronto (random trips): `pixi run train-asce-toronto-random`
- Validate schema: `pixi run python -m ece324_tango.dataset`
- Benchmark backends: `pixi run benchmark-backends`
- Backend verbose logs (optional): add `--backend-verbose`
- Use objective reward shaping (default): `--reward-mode objective`
- Optional baseline reward path: `--reward-mode sumo`

## Backend Selection
- Supported values: `local_mappo`, `benchmarl`, `xuance`, `libsignal`
- Current production backend: `local_mappo`
- BenchMARL backend is native via custom SUMO PettingZoo adapter + BenchMARL MAPPO.
- Xuance backend is native via custom SUMO env registration + Xuance MAPPO.
- LibSignal backend is currently a planning placeholder (fails fast by design); see `docs/notes/libsignal_backend_assessment.md`.
- Xuance now loads a project-owned config file: `ece324_tango/asce/trainers/configs/xuance_mappo_sumo.yaml`.

## Xuance Stability Toggles
- `TANGO_XUANCE_USE_GAE` (default `1`)
- `TANGO_XUANCE_USE_VALUE_NORM` (default `1`, via local compat patch)
- `TANGO_XUANCE_USE_ADVNORM` (default `0`)
- `TANGO_XUANCE_PATCH_VALUE_NORM` (default `1`)

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
- BenchMARL runs are currently noisy (SUMO/torchrl logs) and slower to train than Xuance in current micro setup.
- Xuance value normalization is enabled via local compat patch. If regressions appear, disable with `TANGO_XUANCE_USE_VALUE_NORM=0`.
- Third-party deprecation warnings from `torchrl` appear during pytest collection. They do not currently affect run correctness but should be tracked for dependency updates.

## Reward Objective
- `reward_mode`: `objective` (default) or `sumo`
- Objective reward:
  - `- reward_delay_weight * log1p(delay)`
  - `+ reward_throughput_weight * log1p(throughput)`
  - `+ reward_fairness_weight * Jain(throughput across intersections)`

## Observation Normalization
- Enabled with `--use-obs-norm` flag (default: off).
- `ObsRunningNorm` (Welford online, per-feature) is applied to padded local obs and global obs.
- Stats are saved under keys `"obs_norm"` / `"gobs_norm"` in the model checkpoint.
- Eval path loads and applies the same stats without updating them.

## GPU Notes
- RTX 4070 Laptop, CUDA 12.6. Use `--device auto` (default) to pick GPU automatically.
- All N-agent observations are batched into a single GPU forward pass per step (`act_batch()`).
- Training throughput is SUMO-limited (CPU), not GPU-limited. Longer episodes give more data.
- To maximize data per wall-clock minute: increase `--episodes` rather than `--seconds`
  if the long-horizon SUMO crash recurs.

## Known Bugs Fixed (2026-03-01)
- **Action space crippled**: `n_actions` was `min(action_dims)` = 2; 10/12 agents could
  never select phases 2-5. Fixed: use `max` + `-inf` masking in `act()` / `act_batch()`.
- **GAE bias on truncation**: `last_value` was hardcoded 0.0; now bootstrapped from critic.
- **Credit assignment lost**: all agents received global mean reward; now each gets own reward.

## Proposal KPI Path
- Evaluation CSV now includes proposal-aligned fields:
  - `time_loss_s`
  - `person_time_loss_s`
  - `avg_trip_time_s`
  - `arrived_vehicles`
- Current occupancy heuristic:
  - transit-like IDs (`bus|tram|streetcar|ttc`) => 30 persons
  - otherwise => 1.3 persons

## Handoff Checklist
1. Confirm latest good commit hash.
2. Confirm backend used for latest run.
3. Attach latest command line used for train/eval.
4. Record open risks and next 3 tasks in `docs/notes/prototype_log.md`.
