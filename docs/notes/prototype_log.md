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
