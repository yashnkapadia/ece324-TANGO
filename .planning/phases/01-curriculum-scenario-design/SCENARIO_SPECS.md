# Curriculum Scenario Specifications

**Locked:** 2026-03-29
**Status:** Ready for implementation

## Design Principles

- Max-Pressure is provably optimal under stationary single-commodity demand (Varaiya 2013)
- Scenarios target specific MP failure modes: non-stationarity, multimodal conflict, demand surge
- Residual MAPPO starts from MP floor (gate biased to 0) — can only improve
- Safety constraints (min green 7s, max green 45s) enforced via hard action masking
- Episode duration: 900s default, up to 1200s for complex scenarios

## Implementation Layers

**Layer 1 — Demand Studio:** `generate_scenario()` with TMC data filtering
- Time window, mode mix, volume scaling, simulation timing, location selection

**Layer 2 — SUMO-native:** Post-processing or TraCI runtime modifications
- Multi-phase flows (overlapping begin/end for surges)
- Lane closures (edge speed reduction)
- Streetcar platoon injection

## Initial 4 Scenarios

### Scenario 1: AM Peak (Nominal Baseline)
**Rationale:** MP's home turf — stationary single-mode demand. MAPPO must at least match.
**MP failure mode:** None expected. This is the control scenario.
**Layer 1 params:**
- time_window_start: 420 (07:00)
- time_window_duration: 120 (2 hours)
- simulation_seconds: 900
- included_modes: {cars}
- global_demand_scale: 1.0
- mode_scales: {cars: 1.0}
**Layer 2 mods:** None
**Expected pattern:** Strong westbound tidal flow on Dundas (WE dominance from TMC data), moderate cross-street. Bathurst EW:NS ratio ~0.7-0.9x.

### Scenario 2: PM Peak (Tidal Reversal)
**Rationale:** Demand direction reverses from AM. MP responds to current queues but can't anticipate the shift.
**MP failure mode:** Non-stationary demand. Queue pressure flips direction mid-episode as PM peak builds.
**Layer 1 params:**
- time_window_start: 960 (16:00)
- time_window_duration: 120 (2 hours)
- simulation_seconds: 900
- included_modes: {cars}
- global_demand_scale: 1.0
- mode_scales: {cars: 1.0}
**Layer 2 mods:** None
**Expected pattern:** Strong eastbound tidal flow on Dundas (EW dominance at Spadina: 552-588 cars/hr at 16:00-17:00). Reversal from AM pattern.

### Scenario 3: Midday Multimodal + Heavy Pedestrian
**Rationale:** Multiple vehicle classes with different dynamics + extreme pedestrian volumes at Spadina.
**MP failure mode:** Treats all queue counts equally. A bus with 30 people stuck > 1 car, but MP doesn't weight by person-delay. Pedestrian phases eat into vehicle green.
**Layer 1 params:**
- time_window_start: 660 (11:00)
- time_window_duration: 180 (3 hours)
- simulation_seconds: 1200 (longer to see multimodal interactions develop)
- included_modes: {cars, trucks, buses, streetcars}
- global_demand_scale: 1.0
- mode_scales: {cars: 1.0, trucks: 1.0, buses: 1.0, streetcars: 1.0}
- streetcar_share_from_bus: 0.35
**Layer 2 mods:** None
**Expected pattern:** Spadina/Dundas: 3000-4000 peds/hour. Mixed vehicle types with different acceleration profiles. Bus/streetcar platoons creating variable phase demand.

### Scenario 4: Demand Surge (Event Dispersal)
**Rationale:** Sudden asymmetric volume spike simulating event dispersal (concert, game, etc.)
**MP failure mode:** Over-allocates green to surge direction, starves cross-streets, causes downstream spillback. Cannot anticipate surge end.
**Layer 1 params (base demand):**
- time_window_start: 960 (16:00)
- time_window_duration: 120 (2 hours)
- simulation_seconds: 1200 (longer to capture surge + recovery)
- included_modes: {cars}
- global_demand_scale: 1.0
- mode_scales: {cars: 1.0}
**Layer 2 mods:**
- Surge overlay: additional eastbound flows on Dundas approaches at 2x volume
- Surge window: begin=300, end=600 (minutes 5-10 of simulation)
- Implementation: append extra `<flow>` elements to .rou.xml with restricted time window and eastbound-only from/to edges

## Expansion Scenarios (Phase 2 of curriculum)

### Scenario 5: Lane Closure (Construction)
**Rationale:** Reduced downstream capacity that MP can't see — pushes vehicles into bottleneck.
**MP failure mode:** Optimizes per-intersection independently. Can't meter traffic upstream of closure.
**Implementation:** Reduce speed on Dundas edge between Spadina and Denison to 5 km/h (from ~50) via TraCI at simulation start. Or close one lane.
**Duration:** 900s

### Scenario 6: Safety-Constrained Baseline
**Rationale:** Same demand as Scenario 1 but with hard safety constraints. Tests whether MAPPO learns to work within constraints naturally.
**Implementation:** Identical to Scenario 1 but with min_green=7, max_green=45, ped_recall_every=2 cycles enforced in action masking.
**Duration:** 900s

### Scenario 7: Streetcar Short-Turn (505 Dundas)
**Rationale:** Toronto-specific event — streetcar bunching at Spadina creates multimodal burst + gap.
**MP failure mode:** Multiple failure modes intersect: non-stationary, multimodal, asymmetric.
**Implementation:** Base midday demand + inject 3-4 streetcar flows arriving within 60s at Spadina (normally 5-8 min spacing), followed by 15-min gap. Use tram vtype (length=30m, slow accel).
**Duration:** 1200s

## Scenario-to-MP-Failure Matrix

| Scenario | Non-stationary | Multimodal | Surge/Asymmetric | Capacity Reduction | Safety |
|----------|:-:|:-:|:-:|:-:|:-:|
| 1. AM Peak | | | | | |
| 2. PM Peak | X | | | | |
| 3. Midday Multi | | X | | | |
| 4. Demand Surge | X | | X | | |
| 5. Lane Closure | | | | X | |
| 6. Safety Base | | | | | X |
| 7. Streetcar | X | X | X | | |

## TMC Locations for All Scenarios

All scenarios use the same 8 corridor intersections matched to SUMO TLS:
- University Ave / Dundas St W
- Dundas St W / St Patrick St
- Dundas St W / McCaul St
- Dundas St W / Beverley St
- Dundas St W / Huron St
- Spadina Ave / Dundas St W
- Dundas St W / Denison Ave
- Bathurst St / Dundas St W
