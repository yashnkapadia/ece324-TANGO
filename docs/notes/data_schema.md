# ASCE Dataset Schema (Teammate Proposal)

Date noted: 2026-02-19
Status: adopted as target contract for rollout logging.

| Column | Type | Meaning | Source | Computation |
| --- | --- | --- | --- | --- |
| intersection_id | str | SUMO TLS/junction ID | `traci.trafficlight.getIDList()` | Direct read from simulation |
| time_step | float | Simulation time in seconds | `traci.simulation.getTime()` | Direct read |
| queue_ns | int | Halting vehicles on NS approaches | `traci.edge.getLastStepHaltingNumber()` | Sum over NS approach edges |
| queue_ew | int | Halting vehicles on EW approaches | Same as above | Sum over EW approach edges |
| arrivals_ns | int | Vehicles present on NS approaches | `traci.edge.getLastStepVehicleNumber()` | Sum over NS edges |
| arrivals_ew | int | Vehicles present on EW approaches | Same | Sum over EW edges |
| avg_speed_ns | float (m/s) | Mean speed on NS approaches | `traci.edge.getLastStepMeanSpeed()` | Average over NS edges |
| avg_speed_ew | float (m/s) | Mean speed on EW approaches | Same | Average over EW edges |
| current_phase | int | Active TLS phase index at log time | `traci.trafficlight.getPhase()` | Direct read |
| time_of_day | float [0,1] | Fraction of 24h day | `sim_time + start_hour` | `(start_hour*3600 + sim_time) / 86400` |
| action_phase | int | Phase selected by controller | `Controller.get_actions()` | From baseline/controller output |
| action_green_dur | float (s) | Green duration assigned | `Controller.get_actions()` | From baseline/controller output |
| delay | float (s) | Total waiting time on all approaches | `traci.edge.getWaitingTime()` | Sum over all approach edges |
| queue_total | int | Total halting vehicles, all approaches | `traci.edge.getLastStepHaltingNumber()` | Sum over all approach edges |
| throughput | int | Vehicles on approach edges (proxy) | `traci.edge.getLastStepVehicleNumber()` | Sum over all approach edges |
| scenario_id | str | Scenario identifier | CLI argument `--scenario-id` | `baseline`, `construction_01`, `transit_01` |

## Prototype Note
Current Phase 1 logger writes this exact column set.
- `local_mappo` and `xuance` now use TraCI controlled-link edge aggregations with fallback-safe proxy mode.
- `benchmarl` training rollout CSV now replays rollout actions in SUMO and logs TraCI-derived metrics (with fallback-safe proxy mode only on TraCI mapping failures).
