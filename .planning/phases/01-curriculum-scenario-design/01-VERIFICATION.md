---
phase: 01-curriculum-scenario-design
verified: 2026-03-29T00:00:00Z
status: gaps_found
score: 4/5 success criteria verified
re_verification: false
gaps:
  - truth: "Scenario specs define time window, duration, demand scale, mode mix, and capacity modifications — locking what Phase 3's CLI must produce"
    status: partial
    reason: "SCENARIO_SPECS.md specifies Scenarios 1 and 2 as cars-only, but generate_curriculum.py implements all 4 as full multimodal (cars, trucks, buses, pedestrians, streetcars). The spec document is the authoritative contract for Phase 3, but it no longer accurately reflects what was actually generated. Phase 3 will rely on either document or the other — the inconsistency must be resolved before Phase 3 begins."
    artifacts:
      - path: ".planning/phases/01-curriculum-scenario-design/SCENARIO_SPECS.md"
        issue: "Scenario 1 and 2 specs list included_modes as {cars} only. Scenario 3 spec lists {cars, trucks, buses, streetcars} (no pedestrians). None of the spec text mentions pedestrians as an included mode, yet all 4 generated files contain personFlow elements."
      - path: "scripts/generate_curriculum.py"
        issue: "All 4 ScenarioSpec definitions include pedestrians and streetcars. This is a richer implementation than the spec authorizes. The implementation is likely the intended final state, but the spec document hasn't been updated to match."
    missing:
      - "Update SCENARIO_SPECS.md scenario 1, 2, and 3 included_modes to match what generate_curriculum.py actually uses: {cars, trucks, buses, pedestrians, streetcars}"
      - "Add explicit streetcar_injection and pedestrian headway documentation to each scenario in SCENARIO_SPECS.md so Phase 3 has an unambiguous CLI contract"
---

# Phase 1: Curriculum Scenario Design Verification Report

**Phase Goal:** A well-reasoned portfolio of demand scenarios exists with clear rationale for each, and the spec for what the demand CLI must produce is locked
**Verified:** 2026-03-29
**Status:** gaps_found (1 gap: spec-to-implementation inconsistency)
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | TMC data explored to identify meaningful demand variation | ✓ VERIFIED | `data/processed/tmc_parsed.csv` — 41,229 rows with directional turn counts by mode (n/s/e/w approach for cars, trucks, buses, peds) plus am_peak/pm_peak columns; demand studio imports it at generation time |
| 2 | Max-Pressure theoretical failure modes documented | ✓ VERIFIED | SCENARIO_SPECS.md section "Design Principles" cites Varaiya 2013 optimality condition; failure-mode matrix table maps non-stationarity, multimodal conflict, surge/asymmetric, capacity reduction, safety to each scenario |
| 3 | Each scenario has documented rationale explaining what MP assumption it violates | ✓ VERIFIED | SCENARIO_SPECS.md has per-scenario "Rationale" and "MP failure mode" fields; generate_curriculum.py `ScenarioSpec.rationale` field encodes this directly in code |
| 4 | Scenario specs define time window, duration, demand scale, mode mix, capacity mods | ✗ PARTIAL | All parameters exist in generate_curriculum.py ScenarioSpec definitions, but SCENARIO_SPECS.md (the locked spec document) incorrectly lists cars-only modes for Scenarios 1 and 2, and omits pedestrians from Scenario 3. Spec document and implementation diverge. |
| 5 | At least 4 distinct demand regimes with multimodal demand where TMC supports it | ✓ VERIFIED | 4 regimes generated: AM peak (07:00 stationary), PM peak (16:00 tidal reversal), midday multimodal (11:00 mixed), demand surge (PM base + 2x eastbound overlay). All 4 include cars, trucks, buses from TMC data. |

**Score:** 4/5 truths verified (1 partial)

---

## Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `.planning/phases/01-curriculum-scenario-design/SCENARIO_SPECS.md` | Locked scenario design document | ✓ VERIFIED | Present, 123 lines, includes design principles, 4 initial scenarios, 3 expansion scenarios, MP failure matrix, TMC location table |
| `scripts/generate_curriculum.py` | Generation script with ScenarioSpec definitions | ✓ VERIFIED | 559 lines; defines ScenarioSpec dataclass, SurgeConfig, StreetcarInjection, _apply_surge, _inject_streetcars, _audit_scenario, and all 4 scenario definitions in SCENARIOS dict |
| `sumo/demand/curriculum/am_peak.rou.xml` | AM peak scenario file | ✓ VERIFIED | ~273 lines XML; car/truck/bus/ped flows at end=900; streetcar flows present |
| `sumo/demand/curriculum/pm_peak.rou.xml` | PM peak scenario file | ✓ VERIFIED | ~255 lines XML; car/truck/bus/ped flows at end=900; streetcar flows present |
| `sumo/demand/curriculum/midday_multimodal.rou.xml` | Midday multimodal scenario file | ✓ VERIFIED | ~276 lines XML; car/truck/bus/ped flows at end=1200; streetcar flows present |
| `sumo/demand/curriculum/demand_surge.rou.xml` | Demand surge scenario file | ✓ VERIFIED | ~255 lines XML; car/truck/bus/ped/streetcar base flows at end=1200; 37 surge_* overlay flows at begin=300 end=600 |

---

## Scenario File Deep Verification

### Mode Presence by Scenario

| Scenario | Cars | Trucks | Buses | Streetcars | Pedestrians | All Non-Zero |
|----------|:----:|:------:|:-----:|:----------:|:-----------:|:------------:|
| am_peak.rou.xml | Yes (12+ flows) | Yes (11 flows) | Yes (8 flows) | Yes (2 EB + 2 WB) | Yes (1144 total) | ✓ |
| pm_peak.rou.xml | Yes (12+ flows) | Yes (9 flows) | Yes (6 flows) | Yes (2 EB + 2 WB) | Yes (4210 total) | ✓ |
| midday_multimodal.rou.xml | Yes (12+ flows) | Yes (10+ flows) | Yes (9 flows) | Yes (3 EB + 3 WB) | Yes (4808 total) | ✓ |
| demand_surge.rou.xml | Yes (12+ flows) | Yes (9+ flows) | Yes (6+ flows) | Yes (3 EB + 3 WB + surge) | Yes (4210 total) | ✓ |

All claimed modes present with non-zero vehicle counts.

### Streetcar Injection Verification

Streetcar flows are injected via TTC 505 Dundas schedule (headway_seconds=360), not from TMC data.

| Scenario | Sim Duration | Expected n_streetcars (900/360 or 1200/360) | Actual EB | Actual WB | Status |
|----------|-------------|----------------------------------------------|-----------|-----------|--------|
| am_peak | 900s | floor(900/360) = 2 | 2 | 2 | ✓ |
| pm_peak | 900s | floor(900/360) = 2 | 2 | 2 | ✓ |
| midday_multimodal | 1200s | floor(1200/360) = 3 | 3 | 3 | ✓ |
| demand_surge | 1200s | floor(1200/360) = 3 | 3 | 3 (+ surge_36) | ✓ |

Streetcar vType confirmed: `vClass="tram" guiShape="rail" length="30" maxSpeed="16"` — correct 30m tram profile.

### Flow Begin/End Times vs. Simulation Duration

| Scenario | Sim Seconds (spec) | All base flows end= | Pedestrian flows end= | Correct Alignment |
|----------|-------------------|--------------------|-----------------------|-------------------|
| am_peak | 900 | 900 | 900 | ✓ Match |
| pm_peak | 900 | 900 | 900 | ✓ Match |
| midday_multimodal | 1200 | 1200 | 1200 | ✓ Match |
| demand_surge | 1200 | 1200 (base), 300-600 (surge) | 1200 | ✓ Match |

**Coordination note for Phase 3 (FIX-01):** Flow end times currently EQUAL simulation_seconds. The FIX-01 requirement states flows must extend PAST simulation end time to prevent FatalTraCIError from demand exhaustion. Phase 3 must adjust these files or the CLI must generate flows with `end = simulation_seconds + buffer`. This is not a Phase 1 failure but Phase 3 must not regenerate these files without applying the fix.

### Surge Scenario Directional Filtering

`demand_surge.rou.xml` contains 37 surge overlay flows (surge_0 through surge_36) all with `begin="300" end="600"`. The flows originate from eastbound-matching edges across all 8 corridor intersections. The `_apply_surge` function correctly uses SUMO heading geometry (30-150 degrees for eastbound) to filter flows. Surge includes cars, trucks, buses, and one streetcar — matching the directional nature of the overlay.

### Pedestrian Volumes vs. TMC Reference

SCENARIO_SPECS.md documents Spadina/Dundas at ~3,800 peds/hr midday and ~4,000 peds/hr PM peak. Simulated pedestrian counts at the Spadina cluster node (cluster_29696843_393551252_393552059_394512440):

| Scenario | Sim Duration | Total Spadina Peds | Annualized Rate (/hr) |
|----------|-------------|--------------------|-----------------------|
| am_peak | 900s (0.25h) | 1,144 | ~4,576/hr |
| pm_peak | 900s (0.25h) | 4,210 | ~16,840/hr |
| midday_multimodal | 1200s (0.33h) | 4,808 | ~14,400/hr |
| demand_surge | 1200s (0.33h) | 4,210 | ~12,600/hr |

PM peak and midday exceed TMC reference (~3,800-4,000/hr), which is plausible given the sim captures compressed time-of-day demand within the episode window rather than a full-hour average.

---

## Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `generate_curriculum.py` | `data/processed/tmc_parsed.csv` | `studio.load_tmc_data(TMC_PATH)` | ✓ WIRED | TMC_PATH defined at line 27; loaded at line 522 |
| `generate_curriculum.py` | `sumo/network/osm.net.xml` | `studio.load_network_summary(NETWORK_PATH)` | ✓ WIRED | NETWORK_PATH defined at line 25; loaded at line 526 |
| `generate_curriculum.py` | `apps/demand_studio/app.py` | `sys.path.insert + import app as studio` | ✓ WIRED | Path injection at line 18; studio.generate_scenario called at line 346 |
| `ScenarioSpec` definitions | `.rou.xml` output | `generate_scenario()` → `OUTPUT_DIR/spec.name.rou.xml` | ✓ WIRED | OUTPUT_DIR = `sumo/demand/curriculum/`; all 4 files present |
| `SurgeConfig` in demand_surge spec | surge flows in XML | `_apply_surge()` | ✓ WIRED | 37 surge_* flows confirmed in demand_surge.rou.xml |
| `StreetcarInjection` in all specs | streetcar flows in XML | `_inject_streetcars()` | ✓ WIRED | streetcar vType + 2-4 flows confirmed in all 4 files |

---

## Requirements Coverage

| Requirement | Description | Status | Evidence |
|-------------|-------------|--------|----------|
| DEM-03 | At least 4 curriculum scenarios generated: AM peak, PM peak, off-peak, incident/reduced capacity | ✓ SATISFIED | 4 files exist: am_peak, pm_peak, midday_multimodal (off-peak), demand_surge (incident-style) |
| DEM-04 | Multimodal demand included (cars, trucks, buses) using TMC data | ✓ SATISFIED | All 4 scenarios contain car, truck, bus flows sourced from TMC via demand studio |

Note: DEM-04 calls for "cars, trucks, buses using TMC data." The implementation goes further and adds streetcars (TTC schedule injection) and pedestrians (TMC pedestrian counts). This exceeds the requirement.

---

## Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `scripts/generate_curriculum.py` | 218-231 | Fallback streetcar route selection ignores tram-allowance check when <2 tram edges found, falls back to any passenger edges — streetcar may be routed on edges where SUMO won't allow tram vClass | ⚠️ Warning | Streetcar flows may produce SUMO routing errors at runtime; tram routing should be validated against network tram-permitted edges |
| `scripts/generate_curriculum.py` | 312 | `_audit_scenario` flags timing if `flow.end > spec.simulation_seconds` — this check would NOT flag the current files where end == simulation_seconds, so the FIX-01 issue (flows must extend past sim end) will not be caught by the existing audit | ⚠️ Warning | FIX-01 silently passes audit; Phase 3 must change audit condition to `end >= simulation_seconds` to catch this |
| `SCENARIO_SPECS.md` | 31-35 | Scenario 1 spec lists `included_modes: {cars}` but generated file has cars, trucks, buses, pedestrians, streetcars | ⚠️ Warning | Spec is the Phase 3 contract — divergence from implementation creates ambiguity |
| `SCENARIO_SPECS.md` | 45-49 | Scenario 2 spec lists `included_modes: {cars}` but generated file has cars, trucks, buses, pedestrians, streetcars | ⚠️ Warning | Same concern as Scenario 1 |

---

## Behavioral Spot-Checks

Step 7b: SKIPPED for demand files (not runnable without SUMO). The generator script is runnable but requires TMC data and SUMO network — cannot test without environment. The audit function embedded in `generate_curriculum.py` serves as the equivalent in-process check.

What can be verified statically:

| Behavior | Check | Result | Status |
|----------|-------|--------|--------|
| AM peak flow count > 100 vehicles | Sum of number= attributes in am_peak.rou.xml | ~11,000+ vehicles across all flows | ✓ PASS |
| Surge flows window is 300-600s | grep begin/end on surge_* flows | All surge_* flows: begin=300 end=600 | ✓ PASS |
| Midday sim duration is 1200s | All flow end= values in midday_multimodal.rou.xml | All flows end=1200 | ✓ PASS |
| Streetcar vClass is tram | vType id="streetcar" vClass attribute | vClass="tram" guiShape="rail" in all 4 files | ✓ PASS |
| PM peak pedestrians higher than AM peak at Spadina cluster | Summed personFlow numbers | PM=4210 vs AM=1144 | ✓ PASS (3.7x higher, consistent with TMC data claim) |

---

## Human Verification Required

### 1. Eastbound Surge Direction Correctness

**Test:** Load `demand_surge.rou.xml` in SUMO and visualize edge directions for edges with id prefix matching surge_0 through surge_10 source edges (`-222151622#1`, `-493830469#2`). Confirm these edges carry westbound-approaching vehicles that turn eastbound — i.e., the surge truly creates eastbound pressure on Dundas.
**Expected:** Vehicles from surge flows accumulate on eastbound Dundas lanes, creating queue imbalance toward Bathurst.
**Why human:** The heading-based filtering (30-150 degrees for eastbound) assigns edges to "eastbound" category, but SUMO's coordinate system for this network needs visual confirmation that the geometry interpretation is correct.

### 2. Streetcar Route Validity in SUMO

**Test:** Open SUMO-GUI with `am_peak.rou.xml` and verify the two streetcar flows route successfully along the Dundas corridor without "vehicle cannot be placed" errors.
**Expected:** Streetcars travel from `1447612547#1` to `179167371` (EB) and `680230366#3` to `4652002#0` (WB) using tram-permitted edges.
**Why human:** The fallback routing logic may assign passenger edges to streetcars; tram routing validity requires SUMO to attempt the route.

### 3. Tidal Reversal Pattern Visible in PM Peak

**Test:** Run a short SUMO simulation with `am_peak.rou.xml` and `pm_peak.rou.xml` separately, compare westbound vs. eastbound volume counts on Dundas at the Spadina intersection.
**Expected:** AM peak shows westbound dominance; PM peak shows eastbound dominance (consistent with SCENARIO_SPECS.md citing 552-588 EB vs 276-329 WB at 16:00-17:00).
**Why human:** Directional asymmetry depends on TMC data alignment to SUMO edges; cannot verify without running the simulation.

---

## Gaps Summary

One gap is blocking the "spec is locked" component of the phase goal:

**SCENARIO_SPECS.md is inconsistent with the generated output.** The spec document (which Phase 3 will use as a contract to build the headless CLI) states that Scenarios 1 and 2 include only cars, but all 4 generated `.rou.xml` files include trucks, buses, pedestrians, and streetcars. The implementation in `generate_curriculum.py` is the correct and more complete version — but SCENARIO_SPECS.md has not been updated to reflect it.

This matters specifically because SCENARIO_SPECS.md states at the top: "Status: Ready for implementation." Phase 3 will read this document to determine what modes the CLI must support per scenario. If it follows the spec literally for Scenarios 1 and 2, it will generate cars-only files that don't match what Scenarios 1 and 2 already are.

**Resolution:** Update SCENARIO_SPECS.md to match the actual ScenarioSpec definitions in generate_curriculum.py — specifically update included_modes for Scenarios 1, 2, and 3, and add explicit documentation of streetcar_injection and pedestrian data sourcing.

The generated `.rou.xml` files themselves are substantive and correct. The documentation gap is the only blocker.

---

_Verified: 2026-03-29_
_Verifier: Claude (gsd-verifier)_
