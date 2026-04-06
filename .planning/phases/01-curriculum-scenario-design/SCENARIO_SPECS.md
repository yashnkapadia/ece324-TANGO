# Curriculum Scenario Specifications

**Locked:** 2026-03-29
**Revised:** 2026-03-29 (updated to match implemented multimodal specs)
**Status:** Implemented — 4 initial scenarios generated and SUMO-validated

## Design Principles

- Max-Pressure is provably optimal under stationary single-commodity demand (Varaiya 2013)
- Scenarios target specific MP failure modes: non-stationarity, multimodal conflict, demand surge
- Residual MAPPO starts from MP floor (gate biased to 0) — can only improve
- Safety constraints (min green 7s, max green 45s) enforced via hard action masking
- Episode duration: 900s default, up to 1200s for complex scenarios
- **All scenarios are fully multimodal** — cars, trucks, buses, pedestrians from TMC; streetcars from TTC schedule

## Data Sources

| Data | Source | How Used |
|------|--------|----------|
| Car/truck/bus volumes | City of Toronto TMC (`tmc_parsed.csv`, 41K rows) | `generate_scenario()` Layer 1 via demand studio |
| Pedestrian volumes | City of Toronto TMC (same dataset, `n/s/e/w_appr_peds` columns) | `generate_scenario()` Layer 1 with `pedestrians` in `included_modes` |
| 505 Dundas streetcar headways | TTC schedule (6-min planned 2026, 7am-7pm) | Layer 2: `StreetcarInjection(headway_seconds=360)` |
| 511 Bathurst streetcar headways | TTC schedule (~10-min) | Not yet included; candidate for expansion |

## Implementation Layers

**Layer 1 — Demand Studio:** `generate_scenario()` with TMC data filtering
- Time window, mode mix, volume scaling, simulation timing, location selection
- Pedestrians generate `<personFlow>` elements; vehicles generate `<flow>` elements
- Streetcars excluded from Layer 1 (TMC undercounts them); handled by Layer 2

**Layer 2 — SUMO-native post-processing:**
- `StreetcarInjection`: appends streetcar `<flow>` elements based on TTC headways
- `SurgeConfig`: appends directionally-filtered overlay flows for a time window
- Future: lane closures (TraCI), capacity reduction

## Initial 4 Scenarios (Implemented)

### Scenario 1: AM Peak
**Rationale:** MP's home turf — near-stationary, car-dominated. Includes real AM pedestrians + 505 streetcar at 6-min headway. Control scenario.
**MP failure mode:** None expected. Baseline for comparison.
**Layer 1 params:**
- time_window_start: 420 (07:00)
- time_window_duration: 120 min
- simulation_seconds: 900
- included_modes: {cars, trucks, buses, pedestrians}
- global_demand_scale: 1.0
- mode_scales: {cars: 1.0, trucks: 1.0, buses: 1.0, pedestrians: 1.0}
**Layer 2 mods:**
- StreetcarInjection: 360s headway, eastbound + westbound (505 Dundas, TTC 2026)
**Generated output:** `sumo/demand/curriculum/am_peak.rou.xml`
- 176 vehicle flows, 31 pedestrian flows
- 13,144 cars, 388 trucks, 168 buses, 4 streetcars, 7,598 pedestrians
**Expected pattern:** Strong westbound tidal flow on Dundas (WE dominance from TMC data), moderate cross-street, ~600 peds/hr at Spadina.

### Scenario 2: PM Peak (Tidal Reversal)
**Rationale:** Demand direction reverses from AM. Heavy pedestrians (~4,000/hr Spadina). MP responds to current queues but can't anticipate the directional shift.
**MP failure mode:** Non-stationary demand — queue pressure flips direction as PM peak builds. Spadina EW=552-588 at 16:00-17:00 vs WE=276-329.
**Layer 1 params:**
- time_window_start: 960 (16:00)
- time_window_duration: 120 min
- simulation_seconds: 900
- included_modes: {cars, trucks, buses, pedestrians}
- global_demand_scale: 1.0
- mode_scales: {cars: 1.0, trucks: 1.0, buses: 1.0, pedestrians: 1.0}
**Layer 2 mods:**
- StreetcarInjection: 360s headway, eastbound + westbound (505 Dundas, still within 7am-7pm window)
**Generated output:** `sumo/demand/curriculum/pm_peak.rou.xml`
- 158 vehicle flows, 31 pedestrian flows
- 18,810 cars, 239 trucks, 114 buses, 4 streetcars, 28,125 pedestrians

### Scenario 3: Midday Multimodal + Heavy Pedestrian
**Rationale:** Peak multimodal conflict zone — Spadina/Dundas = ~3,800 peds/hr (Chinatown/Kensington). Full vehicle mix. MP ignores person-weighting (bus=30 people, car=1.3).
**MP failure mode:** Treats all queue counts equally; doesn't weight by person-delay. Pedestrian phases eat into vehicle green time.
**Layer 1 params:**
- time_window_start: 660 (11:00)
- time_window_duration: 180 min
- simulation_seconds: 1200
- included_modes: {cars, trucks, buses, pedestrians}
- global_demand_scale: 1.0
- mode_scales: {cars: 1.0, trucks: 1.0, buses: 1.0, pedestrians: 1.0}
**Layer 2 mods:**
- StreetcarInjection: 360s headway, eastbound + westbound (505 Dundas midday, TTC 2026)
**Generated output:** `sumo/demand/curriculum/midday_multimodal.rou.xml`
- 179 vehicle flows, 31 pedestrian flows
- 20,928 cars, 679 trucks, 179 buses, 6 streetcars, 32,511 pedestrians

### Scenario 4: Demand Surge (Event Dispersal)
**Rationale:** Sudden 2x eastbound spike on PM base demand simulating event dispersal. MP over-allocates green to surge, starves cross-streets, causes downstream spillback.
**MP failure mode:** Over-reacts to surge queue imbalance; can't anticipate surge end; starves cross-streets during surge.
**Layer 1 params:**
- time_window_start: 960 (16:00) — same as PM peak base
- time_window_duration: 120 min
- simulation_seconds: 1200 (longer to capture surge + recovery)
- included_modes: {cars, trucks, buses, pedestrians}
- global_demand_scale: 1.0
- mode_scales: {cars: 1.0, trucks: 1.0, buses: 1.0, pedestrians: 1.0}
**Layer 2 mods:**
- StreetcarInjection: 360s headway, eastbound + westbound
- SurgeConfig: direction=eastbound, multiplier=2.0, begin=300s, end=600s
  - 37 directionally-filtered overlay flows, 1,056 additional vehicles
**Generated output:** `sumo/demand/curriculum/demand_surge.rou.xml`
- 195 vehicle flows, 31 pedestrian flows
- 19,838 cars, 253 trucks, 127 buses, 7 streetcars, 28,125 pedestrians (includes surge overlay)

## Expansion Scenarios (Phase 4.5 — Conditional)

### Scenario 5: Lane Closure (Construction)
**Rationale:** Reduced downstream capacity that MP can't see — pushes vehicles into bottleneck.
**MP failure mode:** Optimizes per-intersection independently. Can't meter traffic upstream of closure.
**Implementation:** Reduce speed on Dundas edge between Spadina and Denison to 5 km/h via TraCI at sim start. Or close 1 lane. Base demand: AM peak multimodal.
**Duration:** 900s

### Scenario 6: Safety-Constrained Baseline
**Rationale:** Same demand as Scenario 1 but with hard safety constraints. Tests whether MAPPO works within constraints naturally vs MP having them clamped post-hoc.
**Implementation:** Identical to Scenario 1 + min_green=7s, max_green=45s, ped_recall_every=2 cycles in action masking.
**Duration:** 900s

### Scenario 7: Streetcar Short-Turn (505 Dundas)
**Rationale:** Toronto-specific — 505 Dundas bunching at Spadina. Compounds non-stationarity + multimodal + asymmetric demand.
**Implementation:** Midday multimodal base + inject 3-4 streetcars within 60s at Spadina (normally 6-min headway), then 15-min gap. 30m tram vtype blocks turns.
**Duration:** 1200s

## Scenario-to-MP-Failure Matrix

| Scenario | Non-stationary | Multimodal | Surge/Asymmetric | Capacity Reduction | Safety |
|----------|:-:|:-:|:-:|:-:|:-:|
| 1. AM Peak | | X | | | |
| 2. PM Peak | X | X | | | |
| 3. Midday Multi | | X | | | |
| 4. Demand Surge | X | X | X | | |
| 5. Lane Closure | | X | | X | |
| 6. Safety Base | | X | | | X |
| 7. Streetcar | X | X | X | | |

Note: All scenarios are now multimodal (cars + trucks + buses + peds + streetcars), so the Multimodal column is X for all. The distinction is degree — Scenario 3 has the heaviest multimodal mix with 32K peds.

## TMC Locations

All scenarios use the same 8 corridor intersections matched to SUMO TLS:
- University Ave / Dundas St W
- Dundas St W / St Patrick St
- Dundas St W / McCaul St
- Dundas St W / Beverley St
- Dundas St W / Huron St
- Spadina Ave / Dundas St W
- Dundas St W / Denison Ave
- Bathurst St / Dundas St W

## Phase 3 Contract

Phase 3 (Simulation Alignment + Headless Demand CLI) must:
1. Ensure flow `end` times extend **past** `simulation_seconds` (currently `end == simulation_seconds`; FIX-01 requires buffer)
2. The `_audit_scenario` check should flag `end == simulation_seconds` as a warning, not just `end > simulation_seconds`
3. Regenerate all 4 scenarios with the buffer applied
4. Formalize `generate_curriculum.py` as a proper pixi task
