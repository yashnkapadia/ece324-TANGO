# LibSignal Backend Assessment

## What LibSignal Provides
- Mature traffic-signal-control library with SUMO + CityFlow support.
- Large set of traffic-control algorithms and benchmark configs.
- Own training stack (`run.py` + Registry + trainer/task abstractions).

## Integration Reality for This Repo
- LibSignal is not a drop-in backend for our current ASCE interface.
- Our contract uses direct `net_file`/`route_file` inputs and writes schema-aligned rollout/eval CSVs.
- LibSignal expects named simulator configs from its own `configs/sim/*.cfg` and its own output layout.

## Decision (Current)
- Keep `local_mappo`, `benchmarl`, and `xuance` as executable backends for ASCE parity runs.
- Register `libsignal` in backend factory as an explicit planning placeholder that fails fast with a clear message.
- Do not block ASCE/PIRA progress on a deep LibSignal adapter until Toronto scenarios and KPI acceptance tests are stable.

## Viable Path to Full Backend
1. Build an adapter that translates our `TrainConfig`/`EvalConfig` into LibSignal config files.
2. Add metric extraction layer mapping LibSignal logs to our canonical schema + proposal KPIs.
3. Add integration smoke tests gated by optional env var and installed LibSignal deps.
4. Add benchmark run in `benchmark-backends` only after parity tests pass.
