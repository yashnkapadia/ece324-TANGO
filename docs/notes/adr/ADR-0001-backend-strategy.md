# ADR-0001: ASCE Trainer Backend Strategy

- Date: 2026-02-19
- Status: Accepted

## Context
TANGO ASCE started with a local MAPPO implementation for speed. We need integration points for framework-backed MARL stacks (BenchMARL, Xuance) while preserving current data contracts and CLI.

## Decision
Introduce a backend abstraction with three names:
- `local_mappo` (production path)
- `benchmarl` (spike path)
- `xuance` (spike path)

The CLI must expose backend selection while keeping output paths/schema unchanged.

## Consequences
Positive:
- Stable public interfaces for train/eval commands.
- Easier migration from local implementation to framework-native trainers.
- Explicit package gating for optional frameworks.

Negative:
- Spike backends still need native adapter implementation work.
- Slight increase in module complexity.

## Follow-up
Implement native BenchMARL and Xuance adapters that replace local fallback behavior while preserving dataset and metrics contracts.
