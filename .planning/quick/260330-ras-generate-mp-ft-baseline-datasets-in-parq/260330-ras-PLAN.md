---
phase: quick
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - ece324_tango/asce/schema.py
  - scripts/generate_baseline_dataset.py
autonomous: true
---

# Quick Task: Generate MP/FT Baseline Datasets for PIRA

## Task 1: Update ASCE schema with person fields

**Files:** `ece324_tango/asce/schema.py`
**Action:** Add person_delay, person_throughput, person_delay_ns, person_delay_ew to ASCE_DATASET_COLUMNS

## Task 2: Create baseline dataset generation script

**Files:** `scripts/generate_baseline_dataset.py`
**Action:** Script that runs MP and FT controllers on each curriculum scenario, logs per-step IntersectionMetrics to Parquet. Uses existing compute_metrics_for_agents + baselines infrastructure.

Output: `data/pira/{controller}_{scenario}.parquet` per controller × scenario combination.

## Task 3: Run dataset generation for all 4 scenarios × 2 controllers
