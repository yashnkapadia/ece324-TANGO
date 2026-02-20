# TANGO Runbook

## Last Updated
2026-02-19

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
- Validate schema: `pixi run python -m ece324_tango.dataset`

## Backend Selection
- Supported values: `local_mappo`, `benchmarl`, `xuance`
- Current production backend: `local_mappo`
- BenchMARL backend is native via custom SUMO PettingZoo adapter + BenchMARL MAPPO.
- Xuance backend is still spike/fallback-gated.

## Artifact Paths
- Model checkpoint: `models/asce_mappo.pt`
- Rollout dataset: `data/processed/asce_rollout_samples.csv`
- Train metrics: `reports/results/asce_train_episode_metrics.csv`
- Eval metrics: `reports/results/asce_eval_metrics.csv`

## Known Risks
- Xuance native adapter is not implemented yet; backend remains a gated fallback.
- BenchMARL runs are currently noisy (SUMO/torchrl logs) and slower than local MAPPO on sample network.
- Delay/throughput/fairness are still proxy metrics in sample-network mode.

## Handoff Checklist
1. Confirm latest good commit hash.
2. Confirm backend used for latest run.
3. Attach latest command line used for train/eval.
4. Record open risks and next 3 tasks in `docs/notes/prototype_log.md`.
