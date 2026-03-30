# Deferred Items — Phase 3

## Pre-existing Test Failures (Out of Scope)

1. **test_action_gate_mappo.py** — ImportError: `GatedActor` and `ResidualMAPPOTrainer` not yet implemented (Phase 2 work)
2. **test_local_backend_bootstrap.py** — `_DummyEnv` missing `traffic_signals` attribute needed by `MaxPressureController`
3. **test_local_eval_fallback_observation_alignment.py** — Mock `rewards_from_metrics` missing `mp_deviation_by_agent` kwarg
4. **test_local_eval_objective_scoring.py** — Same mock signature mismatch as #3

These failures predate Phase 3 changes and are not caused by FIX-01, FIX-02, DEM-01, or DEM-02.
