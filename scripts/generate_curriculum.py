"""Generate curriculum training scenario files from TMC data via demand studio.

Usage:
    pixi run python scripts/generate_curriculum.py
    pixi run python scripts/generate_curriculum.py --scenarios am_peak pm_peak
    pixi run python scripts/generate_curriculum.py --list
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# demand studio lives in apps/demand_studio/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps" / "demand_studio"))

import app as studio  # noqa: E402
import sumolib  # noqa: E402
from lxml import etree  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
NETWORK_PATH = REPO_ROOT / "sumo" / "network" / "osm.net.xml"
TMC_PATH = REPO_ROOT / "data" / "processed" / "tmc_parsed.csv"
OUTPUT_DIR = REPO_ROOT / "sumo" / "demand" / "curriculum"

# 8 corridor intersections matched to SUMO TLS
CORRIDOR_LOCATIONS = [
    "University Ave / Dundas St W",
    "Dundas St W / St Patrick St",
    "Dundas St W / McCaul St",
    "Dundas St W / Beverley St",
    "Dundas St W / Huron St",
    "Spadina Ave / Dundas St W",
    "Dundas St W / Denison Ave",
    "Bathurst St / Dundas St W",
]


@dataclass
class SurgeConfig:
    """Extra flows overlaid on base demand for a time window."""

    direction: str  # "eastbound" or "westbound"
    multiplier: float  # e.g., 2.0 = double the base volume
    begin_s: int  # surge start (simulation seconds)
    end_s: int  # surge end (simulation seconds)


@dataclass
class StreetcarInjection:
    """Inject streetcar flows based on TTC schedule (not TMC data).

    The 505 Dundas streetcar runs every 5-8 min during peak, 8-12 min off-peak.
    TMC grossly undercounts streetcars. We inject based on realistic headways.
    """

    headway_seconds: int  # time between streetcars (e.g., 360 = 6 min)
    directions: list[str]  # ["eastbound", "westbound"]


@dataclass
class ScenarioSpec:
    """Complete specification for one curriculum scenario."""

    name: str
    rationale: str

    # Layer 1: demand studio params
    time_window_start: int  # minutes from midnight
    time_window_duration: int  # minutes
    simulation_seconds: int
    included_modes: set[str]
    global_demand_scale: float = 1.0
    mode_scales: dict[str, float] = field(default_factory=lambda: {"cars": 1.0})
    streetcar_share_from_bus: float = 0.0  # default 0 — use injection instead

    # Layer 2: SUMO-native modifiers
    surge: SurgeConfig | None = None
    streetcar_injection: StreetcarInjection | None = None


def _get_dundas_edges(net: Any) -> dict[str, list[str]]:
    """Find eastbound and westbound through-edges on Dundas corridor.

    Returns dict with 'eastbound' and 'westbound' lists of edge IDs
    that run along Dundas St (the main arterial, roughly east-west).
    """
    eastbound = []
    westbound = []

    for edge in net.getEdges():
        if edge.isSpecial():
            continue
        shape = edge.getShape()
        if len(shape) < 2:
            continue
        dx = shape[-1][0] - shape[0][0]
        dy = shape[-1][1] - shape[0][1]
        heading = math.degrees(math.atan2(dy, dx)) % 360

        # Dundas runs roughly east-west in this network
        # Eastbound: heading ~350-10 or ~70-110 depending on exact geometry
        # Use the approach heading ranges from demand studio
        length = math.sqrt(dx * dx + dy * dy)
        if length < 20:  # skip very short edges
            continue

        # Check if this edge connects to any of our TLS junctions
        to_node = edge.getToNode()
        from_node = edge.getFromNode()

        # Eastbound: heading roughly 250-310 (demand studio "e" approach means
        # traffic coming FROM the east, but edge heading is direction of travel)
        # In SUMO coords for this network, eastbound Dundas has heading ~60-120
        if 30 <= heading <= 150:
            eastbound.append(edge.getID())
        elif 210 <= heading <= 330:
            westbound.append(edge.getID())

    return {"eastbound": eastbound, "westbound": westbound}


def _apply_surge(rou_path: Path, surge: SurgeConfig, net: Any) -> None:
    """Append directionally-filtered surge flows to .rou.xml.

    Only duplicates flows whose from-edge heading matches the surge direction.
    """
    tree = etree.parse(str(rou_path))
    root = tree.getroot()

    dundas_edges = _get_dundas_edges(net)
    target_edges = set(dundas_edges.get(surge.direction, []))

    if not target_edges:
        print(f"  WARNING: No {surge.direction} edges found for surge filtering")
        return

    surge_id = 0
    total_surge_vehicles = 0

    for flow_elem in list(root.findall("flow")):
        from_edge = flow_elem.get("from", "")
        if from_edge not in target_edges:
            continue

        original_number = int(flow_elem.get("number", "0"))
        if original_number < 1:
            continue

        # Surge adds extra vehicles proportional to the base flow
        # but only during the surge window
        sim_end = int(flow_elem.get("end", "900"))
        window_fraction = (surge.end_s - surge.begin_s) / max(sim_end, 1)
        surge_number = max(1, int(original_number * (surge.multiplier - 1.0) * window_fraction))

        surge_elem = etree.SubElement(
            root,
            "flow",
            id=f"surge_{surge_id}",
            type=flow_elem.get("type", "car"),
            begin=str(surge.begin_s),
            end=str(surge.end_s),
            number=str(surge_number),
        )
        surge_elem.set("from", from_edge)
        surge_elem.set("to", flow_elem.get("to", ""))
        surge_elem.set("departLane", "best")
        surge_elem.set("departSpeed", "max")

        surge_id += 1
        total_surge_vehicles += surge_number

    tree.write(str(rou_path), pretty_print=True, xml_declaration=True, encoding="UTF-8")
    print(
        f"  Surge: {surge_id} overlay flows, {total_surge_vehicles} vehicles "
        f"({surge.direction}, {surge.multiplier}x, {surge.begin_s}-{surge.end_s}s)"
    )


def _inject_streetcars(rou_path: Path, injection: StreetcarInjection, net: Any,
                       sim_seconds: int) -> None:
    """Inject streetcar flows based on TTC schedule headways.

    Creates through-running streetcar flows on Dundas corridor edges.
    TMC data grossly undercounts streetcars; this uses realistic TTC headways.
    """
    tree = etree.parse(str(rou_path))
    root = tree.getroot()

    # Ensure streetcar vtype exists
    existing_types = {vt.get("id") for vt in root.findall("vType")}
    if "streetcar" not in existing_types:
        studio.add_vehicle_type(
            root, "streetcar", studio.DEFAULT_VTYPE["streetcar"],
            {"vClass": "tram", "guiShape": "rail"},
        )

    dundas_edges = _get_dundas_edges(net)
    flow_id = 0

    for direction in injection.directions:
        edges = dundas_edges.get(direction, [])
        if not edges:
            print(f"  WARNING: No {direction} edges for streetcar injection")
            continue

        # Pick a long through-edge pair for the streetcar route
        # Streetcars run the full corridor, so pick edges near the extremes
        # Sort by x-coordinate (east-west position)
        edge_positions = []
        for eid in edges:
            edge = net.getEdge(eid)
            if edge and edge.allows("tram"):
                shape = edge.getShape()
                mid_x = (shape[0][0] + shape[-1][0]) / 2
                edge_positions.append((mid_x, eid))

        if len(edge_positions) < 2:
            # Fallback: use any edges that allow the streetcar vClass
            # If none allow tram, use passenger edges (streetcar will use road)
            for eid in edges:
                edge = net.getEdge(eid)
                if edge:
                    shape = edge.getShape()
                    mid_x = (shape[0][0] + shape[-1][0]) / 2
                    edge_positions.append((mid_x, eid))

        if len(edge_positions) < 2:
            print(f"  WARNING: Not enough edges for {direction} streetcar route")
            continue

        edge_positions.sort()
        if direction == "eastbound":
            from_edge = edge_positions[0][1]  # westmost
            to_edge = edge_positions[-1][1]  # eastmost
        else:
            from_edge = edge_positions[-1][1]  # eastmost
            to_edge = edge_positions[0][1]  # westmost

        # Number of streetcars = simulation_time / headway
        n_streetcars = max(1, sim_seconds // injection.headway_seconds)

        etree.SubElement(
            root,
            "flow",
            id=f"streetcar_{direction}_{flow_id}",
            type="streetcar",
            begin="0",
            end=str(sim_seconds),
            number=str(n_streetcars),
            departLane="best",
            departSpeed="max",
            **{"from": from_edge, "to": to_edge},
        )
        flow_id += 1
        print(f"  Streetcar: {n_streetcars} {direction} (headway={injection.headway_seconds}s)")

    tree.write(str(rou_path), pretty_print=True, xml_declaration=True, encoding="UTF-8")


def _audit_scenario(rou_path: Path, spec: ScenarioSpec) -> dict[str, Any]:
    """Audit a generated scenario against its spec. Returns issues found."""
    tree = etree.parse(str(rou_path))
    root = tree.getroot()
    issues = []

    flows = root.findall("flow")
    person_flows = root.findall("personFlow")

    # Count by type
    from collections import Counter
    type_vehicles = Counter()
    type_flows = Counter()
    for f in flows:
        vtype = f.get("type", "unknown")
        count = int(f.get("number", "0"))
        type_flows[vtype] += 1
        type_vehicles[vtype] += count

    # Count person flows
    total_peds = 0
    for pf in person_flows:
        total_peds += int(pf.get("number", "0"))

    # Check expected modes are present
    mode_to_vtype = {"cars": "car", "trucks": "truck", "buses": "bus", "streetcars": "streetcar"}
    for mode in spec.included_modes:
        if mode == "pedestrians":
            if total_peds == 0:
                issues.append("MISSING: pedestrians has 0 person flows")
            continue
        vtype = mode_to_vtype.get(mode, mode)
        if type_vehicles[vtype] == 0:
            issues.append(f"MISSING: {mode} ({vtype}) has 0 vehicles")

    # Check streetcar injection if specified
    if spec.streetcar_injection and type_vehicles["streetcar"] == 0:
        issues.append("MISSING: streetcar injection produced 0 vehicles")

    # Check total vehicle count is reasonable (not too low)
    total = sum(type_vehicles.values())
    if total < 100:
        issues.append(f"LOW VOLUME: only {total} total vehicles")

    # Check simulation duration alignment
    for f in flows:
        end = int(f.get("end", "0"))
        if end > spec.simulation_seconds:
            issues.append(f"TIMING: flow {f.get('id')} ends at {end}s > sim {spec.simulation_seconds}s")
            break

    # Check surge was applied if specified
    if spec.surge:
        surge_flows = [f for f in flows if f.get("id", "").startswith("surge_")]
        if not surge_flows:
            issues.append("MISSING: surge flows not found")

    audit = {
        "total_flows": len(flows),
        "person_flow_count": len(person_flows),
        "total_pedestrians": total_peds,
        "by_type": dict(type_vehicles),
        "flow_counts": dict(type_flows),
        "total_vehicles": total,
        "issues": issues,
    }

    return audit


def generate_scenario(spec: ScenarioSpec, net: Any) -> Path | None:
    """Generate a single scenario .rou.xml from a ScenarioSpec."""
    print(f"\n{'='*60}")
    print(f"Generating: {spec.name}")
    print(f"  Rationale: {spec.rationale}")
    print(f"  Time: {spec.time_window_start // 60:02d}:{spec.time_window_start % 60:02d} "
          f"for {spec.time_window_duration} min, {spec.simulation_seconds}s sim")
    print(f"  Modes: {spec.included_modes}")

    network_key = str(NETWORK_PATH.resolve())

    # Generate base demand via demand studio
    result = studio.generate_scenario(
        network_store={"network_key": network_key},
        selected_locations=CORRIDOR_LOCATIONS,
        max_match_distance=50.0,
        date_policy="latest",
        selected_date=None,
        date_range_start=None,
        date_range_end=None,
        time_window_start=spec.time_window_start,
        time_window_duration=spec.time_window_duration,
        simulation_begin=0,
        simulation_end=spec.simulation_seconds,
        strict_route_check=False,
        min_count_threshold=1,
        global_demand_scale=spec.global_demand_scale,
        included_modes=spec.included_modes - {"streetcars"},  # handle streetcars separately
        mode_scales=spec.mode_scales,
        streetcar_share_from_bus=0.0,  # don't split bus counts
        controller_mode="mappo",
        include_cluster_signals=True,
        fixed_signal=studio.DEFAULT_FIXED_SIGNAL,
        max_pressure_signal=studio.DEFAULT_MAX_PRESSURE_SIGNAL,
        mappo_hyperparams=studio.DEFAULT_MAPPO,
        heading_ranges=studio.APPROACH_HEADING_RANGES_DEFAULT,
        vtype_settings=studio.DEFAULT_VTYPE,
        run_validation=False,
    )

    if not result.success:
        print(f"  FAILED: {result.message}")
        return None

    print(f"  Base demand: {result.stats['vehicle_flows_created']} flows, "
          f"{result.stats['vehicles_by_mode']}")

    # Move to curriculum directory
    src = result.demand_path
    dst = OUTPUT_DIR / f"{spec.name}.rou.xml"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    src.rename(dst)

    # Apply Layer 2 modifiers
    if spec.streetcar_injection is not None:
        _inject_streetcars(dst, spec.streetcar_injection, net, spec.simulation_seconds)

    if spec.surge is not None:
        _apply_surge(dst, spec.surge, net)

    # Audit
    audit = _audit_scenario(dst, spec)
    if audit["issues"]:
        print(f"  AUDIT ISSUES:")
        for issue in audit["issues"]:
            print(f"    - {issue}")
    else:
        print(f"  Audit: PASS")
    print(f"  Final: {audit['total_flows']} vehicle flows, "
          f"{audit['person_flow_count']} ped flows ({audit['total_pedestrians']} peds)")
    print(f"  By type: {audit['by_type']}")
    print(f"  Output: {dst}")

    return dst


# ─── Scenario Definitions ───────────────────────────────────────

SCENARIOS: dict[str, ScenarioSpec] = {
    # ── Scenario 1: AM Peak ──────────────────────────────────────
    # MP's home turf. Demand is relatively stationary and car-dominated.
    # Moderate pedestrians (~600/hr at Spadina). 505 Dundas streetcar runs
    # at 6-min headways (TTC 2026 plan). Buses from TMC (cross-street routes
    # 7 Bathurst, 511 Bathurst captured in TMC bus counts).
    "am_peak": ScenarioSpec(
        name="am_peak",
        rationale="MP's home turf — near-stationary, car-dominated. "
                  "Includes real AM pedestrians + 505 streetcar at 6-min headway.",
        time_window_start=420,  # 07:00
        time_window_duration=120,
        simulation_seconds=900,
        included_modes={"cars", "trucks", "buses", "pedestrians", "streetcars"},
        mode_scales={"cars": 1.0, "trucks": 1.0, "buses": 1.0, "pedestrians": 1.0},
        # 505 Dundas: 6-min headways (TTC 2026 plan, 7am-7pm)
        streetcar_injection=StreetcarInjection(
            headway_seconds=360,
            directions=["eastbound", "westbound"],
        ),
    ),
    # ── Scenario 2: PM Peak (Tidal Reversal) ─────────────────────
    # Strong eastbound tidal flow on Dundas reverses from AM pattern.
    # Spadina: EW=552 at 16:00, 588 at 17:00 vs WE=276-329.
    # Heavy pedestrians (~4,000/hr at Spadina). MP responds to current
    # queues but can't anticipate the directional ramp-up.
    "pm_peak": ScenarioSpec(
        name="pm_peak",
        rationale="Tidal reversal — eastbound dominance + heavy peds (~4k/hr Spadina). "
                  "MP can't anticipate directional shift.",
        time_window_start=960,  # 16:00
        time_window_duration=120,
        simulation_seconds=900,
        included_modes={"cars", "trucks", "buses", "pedestrians", "streetcars"},
        mode_scales={"cars": 1.0, "trucks": 1.0, "buses": 1.0, "pedestrians": 1.0},
        streetcar_injection=StreetcarInjection(
            headway_seconds=360,  # 6-min (still within 7am-7pm window)
            directions=["eastbound", "westbound"],
        ),
    ),
    # ── Scenario 3: Midday Multimodal + Heavy Pedestrian ─────────
    # Peak pedestrian conflict zone: Spadina/Dundas = ~3,800 peds/hr.
    # Chinatown/Kensington area. Full multimodal: cars, trucks, buses
    # (TMC), streetcars (TTC 505 at 6-min, 511 Bathurst at 10-min).
    # MP ignores person-weighting (bus=30 people, car=1.3).
    "midday_multimodal": ScenarioSpec(
        name="midday_multimodal",
        rationale="Peak multimodal conflict — ~3,800 peds/hr at Spadina, "
                  "505+buses, MP ignores person-weighting.",
        time_window_start=660,  # 11:00
        time_window_duration=180,
        simulation_seconds=1200,
        included_modes={"cars", "trucks", "buses", "pedestrians", "streetcars"},
        mode_scales={"cars": 1.0, "trucks": 1.0, "buses": 1.0, "pedestrians": 1.0},
        streetcar_injection=StreetcarInjection(
            headway_seconds=360,  # 505 Dundas: 6-min midday (TTC 2026)
            directions=["eastbound", "westbound"],
        ),
    ),
    # ── Scenario 4: Demand Surge (Event Dispersal) ───────────────
    # PM base demand + sudden 2x eastbound spike (300-600s window)
    # simulating event dispersal (concert, Raptors game, etc.).
    # MP over-allocates green to surge, starves cross-streets, causes
    # downstream spillback. Safety constraints matter here — MAPPO
    # should maintain minimum green for cross-streets during surge.
    "demand_surge": ScenarioSpec(
        name="demand_surge",
        rationale="Event dispersal — 2x eastbound surge on PM base. "
                  "MP over-allocates to surge, starves cross-streets.",
        time_window_start=960,  # 16:00 base
        time_window_duration=120,
        simulation_seconds=1200,
        included_modes={"cars", "trucks", "buses", "pedestrians", "streetcars"},
        mode_scales={"cars": 1.0, "trucks": 1.0, "buses": 1.0, "pedestrians": 1.0},
        streetcar_injection=StreetcarInjection(
            headway_seconds=360,
            directions=["eastbound", "westbound"],
        ),
        surge=SurgeConfig(
            direction="eastbound",
            multiplier=2.0,
            begin_s=300,
            end_s=600,
        ),
    ),
}

INITIAL_4 = ["am_peak", "pm_peak", "midday_multimodal", "demand_surge"]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate curriculum training scenarios")
    parser.add_argument("--scenarios", nargs="*", default=None,
                        help="Scenario names to generate (default: all initial 4)")
    parser.add_argument("--list", action="store_true", help="List available scenarios and exit")
    args = parser.parse_args()

    if args.list:
        print("Available scenarios:")
        for name, spec in SCENARIOS.items():
            initial = " [initial]" if name in INITIAL_4 else ""
            print(f"  {name:25s} — {spec.rationale[:60]}{initial}")
        return

    # Init demand studio
    print("Loading TMC data...")
    studio.TMC_DF = studio.load_tmc_data(TMC_PATH)
    print(f"  {len(studio.TMC_DF)} rows loaded")

    print("Loading SUMO network...")
    summary = studio.load_network_summary(NETWORK_PATH)
    net = summary["net"]
    print("  Network ready")

    # Generate selected scenarios
    targets = args.scenarios if args.scenarios else INITIAL_4
    results = {}
    for name in targets:
        if name not in SCENARIOS:
            print(f"\nERROR: Unknown scenario '{name}'. Use --list to see available.")
            continue
        path = generate_scenario(SCENARIOS[name], net)
        results[name] = path

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    all_ok = True
    for name, path in results.items():
        status = f"OK -> {path}" if path else "FAILED"
        if not path:
            all_ok = False
        print(f"  {name:25s} {status}")

    ok_count = sum(1 for p in results.values() if p is not None)
    print(f"\n{ok_count}/{len(results)} scenarios generated in {OUTPUT_DIR}")
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
