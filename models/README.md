# `models/` — Trained Checkpoints

This directory holds the trained ASCE (MAPPO) checkpoints used in the
final report. Stale experimental checkpoints have been moved to
`_archive/` to keep the canonical artifacts easy to find.

## Canonical checkpoints

| File | Purpose |
|---|---|
| `asce_mappo_curriculum_best.pt` | **Headline model.** Best curriculum checkpoint (curriculum episode 432, total episode 632), selected by minimum worst-case MAPPO/MP ratio across the four scenarios. This is the model reported in Table 1 of the final report. |
| `asce_mappo_curriculum.pt` | Final curriculum model (episode 600). Cited in the report's "Comparison with the final model" paragraph as evidence that best-checkpointing is essential — it underperforms the best checkpoint on every scenario due to gradient interference late in training. |
| `asce_mappo_curriculum_train_state.json` | Training-state metadata (episode counter, best-eval bookkeeping) for resuming the curriculum run. |
| `asce_mappo_curriculum_best_am_peak.pt` | Per-scenario best checkpoint — best AM Peak performance during the curriculum, retained for ablation. |
| `asce_mappo_curriculum_best_pm_peak.pt` | Per-scenario best for PM Peak. |
| `asce_mappo_curriculum_best_demand_surge.pt` | Per-scenario best for Demand Surge. |
| `asce_mappo_curriculum_best_midday_multimodal.pt` | Per-scenario best for Midday Multimodal. |

To reproduce the headline numbers, evaluate `asce_mappo_curriculum_best.pt`
with `pixi run eval-asce-curriculum-best` (see `pixi.toml`) or via
`scripts/eval_matrix.sh`.

## `_archive/20260406_pre_final_report/`

Earlier experimental checkpoints kept for reference but not used in the
report:

- `asce_mappo.pt` — initial single-scenario model, pre-vendored sumo-rl.
- `asce_mappo_objective_e{5,10,20}_s17.pt` — early `objective`-reward iterations (interim report era).
- `asce_mappo_phase4.pt`, `asce_mappo_reward_v2.pt`, `asce_mappo_person_obj.pt` — intermediate reward-design iterations (see Appendix D of the final report).
- `asce_mappo_sumo_e5_s17.pt`, `asce_mappo_toronto_*.pt` — early Toronto-corridor runs before curriculum training.
- `speed_test_*w.pt`, `test_parallel_*w.pt` — parallel-worker scaling smoke tests.
- `quick_backend_compare/`, `quick_backend_compare_s60e2/` — backend-comparison sanity runs.
- `asce_mappo_curriculum_copy.pt`, `asce_mappo_curriculum_best_copy.pt` — accidental Windows-style file duplicates.

These are kept under version control for traceability of the reward-design
ablation history; do not load them for inference.
