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
- [ ] Phase 2: PIRA scenario dataset generation.
- [ ] Phase 3: PIRA GNN surrogate.

## Canonical Notes
- Dataset schema: `docs/notes/data_schema.md`
- Prototype log: `docs/notes/prototype_log.md`
