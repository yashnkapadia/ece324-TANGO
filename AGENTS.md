# AGENTS — Codebase Orientation

## Project
TANGO: Traffic Adaptive Network Guidance & Optimization.
This repo implements the ASCE (Adaptive Signal Control Engine) MAPPO pipeline
for real-time traffic signal control on a Toronto corridor SUMO network.

## Quick-start commands (run all via pixi)
| Command | What it does |
|---|---|
| `pixi run pytest tests -q` | Full unit test suite (fast, no SUMO) |
| `pixi run train-asce-toronto-demand` | Train MAPPO on Toronto TMC demand |
| `pixi run eval-asce-toronto-demand` | Evaluate checkpoint vs baselines |
| `pixi run train-asce-toronto-random` | Train on random-trip demand |
| `pixi run benchmark-backends` | Multi-seed backend comparison (slow) |

See `docs/notes/runbook.md` for all CLI flags and artifact paths.

## Package layout

```
ece324_tango/
  asce/
    mappo.py              # Actor, Critic, MAPPOTrainer (PPO update, GAE, build_batch)
    obs_norm.py           # ObsRunningNorm: Welford online normalizer, saved in checkpoint
    env.py                # create_parallel_env(), pad_observation(), flatten_obs_by_agent()
    traffic_metrics.py    # TraCI metric extraction, reward shaping (objective/sumo modes)
    baselines.py          # FixedTimeController, MaxPressureController
    runtime.py            # extract_reset_obs/step, safe_done, jain_index
    kpi.py                # KPITracker: episode-level time_loss, person_time_loss, trip time
    schema.py             # ASCE_DATASET_COLUMNS — canonical CSV column list
    trainers/
      base.py             # TrainConfig, EvalConfig dataclasses (all CLI args live here)
      local_mappo_backend.py  # Main train/eval loop (LocalMappoBackend)
      benchmarl_backend.py    # BenchMARL MAPPO path
      xuance_backend.py       # Xuance MAPPO path
      libsignal_backend.py    # Placeholder (fail-fast)
      noise_control.py        # Quiet SUMO helper
  modeling/
    train.py              # CLI entry point: python -m ece324_tango.modeling.train
    predict.py            # CLI entry point: python -m ece324_tango.modeling.predict
    benchmark_backends.py # CLI: multi-seed backend comparison
  config.py               # PROJ_ROOT, MODELS_DIR, RESULTS_DIR, DATA_DIR
  dataset.py              # Schema validation CLI
  error_reporting.py      # report_exception() — non-fatal fallback logger
```

## Critical design decisions (read before changing anything)

### Action space (FIXED 2026-03-01)
`n_actions = max(action_dims.values())` — the actor outputs logits for the largest
action space. Agents with fewer phases get invalid logits masked to `-inf` before
sampling (see `MAPPOTrainer.act(n_valid_actions=...)`). **Never use `min`.**

### Value bootstrap (FIXED 2026-03-01)
Episodes end by time truncation. `build_batch(last_values=...)` takes a dict of
per-agent bootstrap values computed from the critic at the final observation.
`_compute_gae(last_value=0.0)` is only correct for truly terminal states.

### Per-agent rewards (FIXED 2026-03-01)
Each agent's `Transition.reward` is its own shaped reward, NOT the global mean.
`rewards_from_metrics()` returns per-agent values — use them directly.

### Observation normalization
`ObsRunningNorm` (Welford) normalizes per-feature across all observations seen
during training. Saved in checkpoint under key `"obs_norm"`. Applied to padded
local obs and to global obs before every actor/critic call. Padded-zero dimensions
stay 0 after normalization (their running mean → 0, std → ε). Enable with `--use-obs-norm`.

### Reward modes
- `objective` (default): `-w_d * log1p(delay) + w_t * log1p(throughput) + w_f * fairness`
- `sumo`: raw sumo-rl rewards (usually negative accumulated waiting time)

### Toronto SUMO network
- 12 intersections with heterogeneous obs dims (9-38) and action spaces (2-6 phases).
- Net file: `sumo/network/osm.net.xml` (regenerate with `gzip -dk sumo/network/osm.net.xml.gz`).
- Demand: `sumo/demand/demand.rou.xml` (TMC-calibrated) or `random_trips.rou.xml`.
- Long-horizon runs (>300s) may hit `FatalTraCIError` on this network — known issue,
  root cause is in sumo-rl controllable TLS interaction.

### Proposal success criterion
`time_loss_s(MAPPO) <= 0.90 * time_loss_s(best_baseline)` over 10 seeds.
Use `person_time_loss_s` for the person-weighted version. Report from eval CSV.

## Key file cross-references
- `TrainConfig` / `EvalConfig`: `ece324_tango/asce/trainers/base.py` — add new CLI flags here.
- Backend registration: `ece324_tango/asce/trainers/__init__.py` — `get_backend()` factory.
- Dataset schema: `docs/notes/data_schema.md` + `ece324_tango/asce/schema.py`.
- Prototype log (chronological changes): `docs/notes/prototype_log.md`.
- ADRs: `docs/notes/adr/`.

## Test suite
`pixi run pytest tests -q` — 26 tests, 2 skipped (slow integration).
Slow integration tests: `RUN_SLOW_INTEGRATION=1 pixi run pytest tests/test_backend_integration_slow.py`
