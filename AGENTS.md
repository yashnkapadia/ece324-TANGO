# AGENTS

## Scope
This file tracks implementation decisions and data contracts for the TANGO ML prototype.

## Active Focus
- Owner focus: core ML code (ASCE MAPPO and baseline evaluation).
- Teammate focus: Toronto SUMO generation pipeline.

## Prototype Status
- [x] Phase 0: package scaffolding and pixi-first commands.
- [x] Phase 1: minimal ASCE MAPPO training/evaluation on sample SUMO network.
- [x] GPU support path added (`--device auto|cuda|cpu`) for ASCE train/eval.
- [x] Baseline upgraded from queue proxy to edge-level max-pressure (TraCI controlled links).
- [x] Backend abstraction added (`local_mappo`, `benchmarl`, `xuance`) with CLI selection.
- [x] `libsignal` backend candidate registered as explicit placeholder with assessment notes.
- [x] BenchMARL/Xuance package dependencies tracked in pixi (`pypi-dependencies`).
- [x] Handoff rigor upgraded with runbook + ADR.
- [x] Native BenchMARL adapter (custom SUMO PettingZoo adapter + BenchMARL MAPPO path).
- [x] Native Xuance adapter (custom SUMO env registration + Xuance MAPPO path).
- [x] Xuance switched to project-owned custom MAPPO config (no MPE bootstrap defaults).
- [x] Slow integration tests for BenchMARL/Xuance CLI train+eval paths.
- [x] Backend noise controls added (`backend_verbose` flag, quiet SUMO defaults).
- [x] Objective reward mode added (`reward_mode=objective|sumo`) with delay/throughput/fairness weights.
- [x] TraCI-driven metrics extraction added for rollout logging in local + Xuance paths.
- [x] BenchMARL rollout export switched to TraCI-derived replay logging (no proxy placeholders).
- [x] Multi-seed backend comparison rerun under objective reward preset.
- [x] Reproducible backend benchmark CLI/task added (`ece324_tango.modeling.benchmark_backends`, `pixi run benchmark-backends`).
- [x] Proposal KPI tracking added in eval outputs (`time_loss_s`, `person_time_loss_s`, `avg_trip_time_s`, `arrived_vehicles`).
- [x] Local Xuance value_norm compatibility patch added for fair MAPPO settings.
- [ ] Phase 2: PIRA scenario dataset generation.
- [ ] Phase 3: PIRA GNN surrogate.

## Canonical Notes
- Dataset schema: `docs/notes/data_schema.md`
- Prototype log: `docs/notes/prototype_log.md`
- Runbook: `docs/notes/runbook.md`
- ADRs: `docs/notes/adr/`
- LibSignal assessment: `docs/notes/libsignal_backend_assessment.md`
