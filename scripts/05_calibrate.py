"""
Generates calibrated demand (demand.rou.xml) from TMC turning-movement counts.

Reads the most-recent AM-peak (08:00-09:00) 15-minute bin data for each
study intersection, sums counts into hourly approach totals per movement
(right / through / left), maps TMC approach directions (N/S/E/W) to SUMO
incoming edges via edge heading angles, resolves outgoing edges by turn
type, verifies route reachability, and writes one SUMO <flow> element per
valid approach-movement combination.

After writing the demand file, a GEH validation summary is printed
comparing assigned flow totals against the original TMC approach volumes
for every study intersection.

Inputs:
  - data/processed/tmc_parsed.csv
  - data/processed/intersection_map.csv
  - sumo/network/osm.net.xml.gz

Outputs:
  - sumo/demand/demand.rou.xml
  - data/processed/calibration_report.csv

Running:
    python scripts/05_calibrate.py
"""

import sumolib
import pandas as pd
import numpy as np
import os
import math
from lxml import etree

# TMC location names corresponding to each study intersection
TMC_NAME_MAP = {
    "Dundas_University": "University Ave / Dundas St W",
    "Dundas_StPatrick": "Dundas St W / St Patrick St",
    "Dundas_McCaul": "Dundas St W / McCaul St",
    "Dundas_Beverly": "Dundas St W / Beverley St",
    "Dundas_Huron": "Dundas St W / Huron St",
    "Dundas_Spadina": "Spadina Ave / Dundas St W",
    "Dundas_Denison": "Dundas St W / Denison Ave",
    "Dundas_Bathurst": "Bathurst St / Dundas St W",
}

APPROACH_HEADING_RANGES = {
    "n": (160, 230),
    "s": (340, 40),
    "e": (250, 310),
    "w": (70, 130),
}

SIM_START = 0
SIM_END = 3600


def edge_heading(edge):
    """Return the heading of an edge in degrees [0, 360), based on the last segment of its shape."""
    shape = edge.getShape()
    if len(shape) < 2:
        return None
    dx = shape[-1][0] - shape[-2][0]
    dy = shape[-1][1] - shape[-2][1]
    return math.degrees(math.atan2(dy, dx)) % 360


def heading_in_range(heading, lo, hi):
    """Check if heading falls within [lo, hi], handling wrap-around at 360."""
    if lo <= hi:
        return lo <= heading <= hi
    return heading >= lo or heading <= hi


def classify_approach(heading):
    """Map an edge heading to a TMC approach direction (n/s/e/w) or None."""
    for direction, (lo, hi) in APPROACH_HEADING_RANGES.items():
        if heading_in_range(heading, lo, hi):
            return direction
    return None


def get_outgoing_edges_by_turn(net, node, incoming_edge):
    """
    Given an incoming edge to a junction, classify each outgoing edge as
    right, thru, or left based on the angle difference.
    Returns dict: {"r": edge_or_None, "t": edge_or_None, "l": edge_or_None}
    """
    in_heading = edge_heading(incoming_edge)
    if in_heading is None:
        return {}

    outgoing = [
        e for e in node.getOutgoing()
        if not e.isSpecial()
        and e.allows("passenger")
        and e.getID() != incoming_edge.getID()
        and (
            len(incoming_edge.getOutgoing().get(e, []))
            if isinstance(incoming_edge.getOutgoing(), dict)
            else True
        )
    ]

    if not outgoing:
        return {}

    candidates = []
    for out_edge in outgoing:
        out_heading = edge_heading(out_edge)
        if out_heading is None:
            continue
        diff = (out_heading - in_heading) % 360
        candidates.append((out_edge, diff))

    if not candidates:
        return {}

    result = {}
    for out_edge, diff in candidates:
        if 150 <= diff <= 210:
            # ~180 degree turn = U-turn, skip
            continue
        elif 30 <= diff <= 150:
            result.setdefault("l", (out_edge, abs(diff - 90)))
        elif 210 <= diff <= 330:
            result.setdefault("r", (out_edge, abs(diff - 270)))
        else:
            result.setdefault("t", (out_edge, abs(diff)))

    # If multiple edges map to the same turn type, pick the closest angle match
    final = {}
    for movement in ["r", "t", "l"]:
        best = None
        best_diff = 999
        for out_edge, diff in candidates:
            if 150 <= diff <= 210:
                continue
            if movement == "l" and 30 <= diff <= 150 and abs(diff - 90) < best_diff:
                best, best_diff = out_edge, abs(diff - 90)
            elif movement == "r" and 210 <= diff <= 330 and abs(diff - 270) < best_diff:
                best, best_diff = out_edge, abs(diff - 270)
            elif movement == "t" and (diff < 30 or diff > 330) and min(diff, 360 - diff) < best_diff:
                best, best_diff = out_edge, min(diff, 360 - diff)
        if best is not None:
            final[movement] = best

    return final


def get_am_peak_volumes(tmc_df, tmc_location_name):
    """
    Extract the AM peak (08:00-09:00) turning movement volumes from the most
    recent count date for a given TMC location.
    Returns dict: {("n","r"): volume, ("n","t"): volume, ...} for all 12 combos.
    """
    sub = tmc_df[
        (tmc_df["location_name"] == tmc_location_name)
        & (tmc_df["n_appr_cars_t"].notna())
    ]

    if len(sub) == 0:
        return {}

    dates = sorted(sub["count_date"].dropna().unique(), reverse=True)
    latest = sub[sub["count_date"] == dates[0]]

    am_bins = latest[latest["start_time"].str.contains("T08:", na=False)]
    if len(am_bins) == 0:
        am_bins = latest[latest["start_time"].str.contains("T07:", na=False)]
    if len(am_bins) == 0:
        return {}

    volumes = {}
    for direction in ["n", "s", "e", "w"]:
        for movement in ["r", "t", "l"]:
            col = f"{direction}_appr_cars_{movement}"
            vol = am_bins[col].sum()
            if pd.notna(vol) and vol > 0:
                volumes[(direction, movement)] = int(vol)

    return volumes


def compute_geh(assigned, observed):
    """GEH statistic for comparing assigned vs observed hourly volumes."""
    if assigned + observed == 0:
        return 0.0
    return math.sqrt(2 * (assigned - observed) ** 2 / (assigned + observed))


def verify_route(net, from_edge, to_edge):
    """Check that a route exists between two edges."""
    route = net.getShortestPath(from_edge, to_edge)
    return route[0] is not None


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    net_file = os.path.join(base_dir, "sumo", "network", "osm.net.xml.gz")
    net = sumolib.net.readNet(net_file)

    tmc = pd.read_csv(
        os.path.join(base_dir, "data", "processed", "tmc_parsed.csv"),
        low_memory=False,
    )
    int_map = pd.read_csv(
        os.path.join(base_dir, "data", "processed", "intersection_map.csv")
    )

    root = etree.Element("routes")
    etree.SubElement(
        root, "vType",
        id="car", accel="2.6", decel="4.5",
        sigma="0.5", length="5.0", maxSpeed="13.89",
    )

    flow_id = 0
    total_skipped = 0
    total_created = 0
    geh_records = []

    for _, row in int_map.iterrows():
        int_name = row["intersection_name"]
        jid = row["sumo_junction_id"]
        tmc_location = TMC_NAME_MAP.get(int_name)

        if tmc_location is None:
            print(f"  {int_name}: no TMC mapping defined, skipping")
            continue

        volumes = get_am_peak_volumes(tmc, tmc_location)
        if not volumes:
            print(f"  {int_name}: no TMC AM-peak data found for '{tmc_location}'")
            continue

        node = net.getNode(jid)
        if node is None:
            print(f"  {int_name}: junction '{jid}' not found in network")
            continue

        incoming_passenger = [
            e for e in node.getIncoming()
            if not e.isSpecial() and e.allows("passenger")
        ]

        approach_to_edge = {}
        for edge in incoming_passenger:
            h = edge_heading(edge)
            if h is None:
                continue
            direction = classify_approach(h)
            if direction is None:
                continue
            if direction in approach_to_edge:
                existing = approach_to_edge[direction]
                if edge.getLaneNumber() > existing.getLaneNumber():
                    approach_to_edge[direction] = edge
            else:
                approach_to_edge[direction] = edge

        print(f"\n{int_name} ({tmc_location}):")
        for direction, edge in approach_to_edge.items():
            print(f"  {direction.upper()} approach -> edge {edge.getID()}")

        assigned_by_approach = {d: 0 for d in ["n", "s", "e", "w"]}
        tmc_by_approach = {d: 0 for d in ["n", "s", "e", "w"]}
        for (d, _), v in volumes.items():
            tmc_by_approach[d] += v

        for (direction, movement), vph in volumes.items():
            if direction not in approach_to_edge:
                total_skipped += 1
                continue

            in_edge = approach_to_edge[direction]
            turn_edges = get_outgoing_edges_by_turn(net, node, in_edge)

            if movement not in turn_edges:
                total_skipped += 1
                continue

            out_edge = turn_edges[movement]

            if not verify_route(net, in_edge, out_edge):
                print(f"    SKIP {direction.upper()}-{movement}: no route {in_edge.getID()} -> {out_edge.getID()}")
                total_skipped += 1
                continue

            etree.SubElement(
                root, "flow",
                id=f"flow_{flow_id}",
                type="car",
                begin=str(SIM_START),
                end=str(SIM_END),
                vehsPerHour=str(vph),
                **{"from": in_edge.getID()},
                to=out_edge.getID(),
                departLane="best",
                departSpeed="max",
            )
            print(f"    flow_{flow_id}: {direction.upper()}-{movement} = {vph} veh/hr ({in_edge.getID()} -> {out_edge.getID()})")
            assigned_by_approach[direction] += vph
            flow_id += 1
            total_created += 1

        for d in ["n", "s", "e", "w"]:
            tmc_vol = tmc_by_approach[d]
            asgn_vol = assigned_by_approach[d]
            if tmc_vol == 0 and asgn_vol == 0:
                continue
            geh = compute_geh(asgn_vol, tmc_vol)
            geh_records.append({
                "intersection": int_name,
                "approach": d.upper(),
                "tmc_vph": tmc_vol,
                "assigned_vph": asgn_vol,
                "geh": round(geh, 2),
                "pass": geh < 5.0,
            })

    out_path = os.path.join(base_dir, "sumo", "demand", "demand.rou.xml")
    tree = etree.ElementTree(root)
    tree.write(out_path, pretty_print=True, xml_declaration=True, encoding="UTF-8")

    print(f"\nCreated {total_created} flows, skipped {total_skipped} unmapped movements")
    print(f"Written to {out_path}")

    # GEH validation summary
    geh_df = pd.DataFrame(geh_records)
    report_path = os.path.join(base_dir, "data", "processed", "calibration_report.csv")
    geh_df.to_csv(report_path, index=False)

    n_pass = geh_df["pass"].sum()
    n_total = len(geh_df)
    pct = n_pass / n_total * 100 if n_total else 0

    print(f"\n{'='*60}")
    print("GEH Validation: Assigned Flow vs TMC Approach Totals")
    print(f"{'='*60}")
    print(f"{'Intersection':<22} {'Dir':>3} {'TMC':>6} {'Asgn':>6} {'GEH':>6} {'Pass':>5}")
    print("-" * 60)
    for _, r in geh_df.iterrows():
        mark = "OK" if r["pass"] else "FAIL"
        print(f"{r['intersection']:<22} {r['approach']:>3} {r['tmc_vph']:>6} {r['assigned_vph']:>6} {r['geh']:>6.2f} {mark:>5}")
    print("-" * 60)
    print(f"{n_pass}/{n_total} approaches pass GEH < 5  ({pct:.0f}%)")
    print(f"Calibration report saved to {report_path}")


if __name__ == "__main__":
    main()
