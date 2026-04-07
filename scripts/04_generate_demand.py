"""
Purpose: Generates sample demand (demand.rou.xml) using heuristic flow rates.

Logic: Identifies boundary entry and exit edges in the network, filters to edges
that allow passenger vehicles and have valid outgoing connections, verifies
route reachability, and writes one flow per entry-exit pair. Flow rates are
estimated from lane count and speed limit — this is NOT calibrated to TMC
data.  Use 05_calibrate.py to produce TMC-based demand instead.

Inputs:
  - sumo/network/osm.net.xml.gz

Outputs:
  - sumo/demand/demand.rou.xml (heuristic flows, overwritten by 05_calibrate.py)

Running:
    python scripts\04_generate_demand.py
"""

# HEURISTIC DEMAND FOR SMOKE-TESTING THE NETWORK BEFORE TMC CALIBRATION.
# RUN 05_CALIBRATE.PY TO REPLACE THIS WITH REAL TMC-BASED DEMAND.

# Interim phase of the project only focuses on demand flows of passenger vehicles.

import sumolib
import pandas as pd
import os
from lxml import etree


def get_entry_edges(net):
    """Find all edges that enter the network from the boundary (dead-end sources).
    Only returns edges that allow passenger vehicles and have outgoing connections."""
    entry_edges = []
    for edge in net.getEdges():
        if edge.isSpecial():
            continue
        if not edge.allows("passenger"):
            continue
        from_node = edge.getFromNode()
        # If the from-node has no incoming edges (other than this edge's reverse),
        # it's a boundary entry point
        incoming = from_node.getIncoming()
        if len(incoming) <= 1:  # only this edge or nothing
            # Verify the edge has at least one outgoing connection for passenger vehicles
            to_node = edge.getToNode()
            has_connection = False
            for outgoing in to_node.getOutgoing():
                if outgoing.allows("passenger") and outgoing.getID() != edge.getID():
                    has_connection = True
                    break
            if has_connection:
                entry_edges.append(edge)
    return entry_edges


def get_exit_edges(net):
    """Find all edges that exit the network at the boundary.
    Only returns edges that allow passenger vehicles."""
    exit_edges = []
    for edge in net.getEdges():
        if edge.isSpecial():
            continue
        if not edge.allows("passenger"):
            continue
        to_node = edge.getToNode()
        outgoing = to_node.getOutgoing()
        if len(outgoing) <= 1:
            exit_edges.append(edge)
    return exit_edges


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    net_file = os.path.join(base_dir, "sumo", "network", "osm.net.xml.gz")
    net = sumolib.net.readNet(net_file)

    # Load TMC + intersection map to verify calibration inputs are present; heuristic
    # flows below don't consume them (05_calibrate.py does the real fit).
    pd.read_csv(os.path.join(base_dir, "data", "processed", "tmc_parsed.csv"))
    pd.read_csv(os.path.join(base_dir, "data", "processed", "intersection_map.csv"))

    entry_edges = get_entry_edges(net)
    exit_edges = get_exit_edges(net)

    print(f"Entry edges: {len(entry_edges)}")
    print(f"Exit edges: {len(exit_edges)}")

    # Assign heuristic flows between entry-exit edge pairs.
    # For TMC-calibrated demand, run 05_calibrate.py instead.

    # For each entry edge, create flows to a few exit edges
    # with volume estimated from lane count and speed.

    root = etree.Element("routes")

    # Add default vehicle type
    etree.SubElement(
        root,
        "vType",
        id="car",
        accel="2.6",
        decel="4.5",
        sigma="0.5",
        length="5.0",
        maxSpeed="13.89",
    )

    # AM peak hour (8:00-9:00) as default period
    # Simulation runs 0-3600s
    SIM_START = 0
    SIM_END = 3600

    # Create flows for each entry edge
    flow_id = 0
    skipped = 0
    for entry_edge in entry_edges:
        # Default flow rate: 200 veh/hr for major roads, 50 for minor
        n_lanes = entry_edge.getLaneNumber()
        speed = entry_edge.getSpeed()  # m/s

        # Heuristic: more lanes and higher speed = more volume
        if n_lanes >= 2 and speed > 11:  # > ~40 km/h
            vph = 300  # vehicles per hour (initial estimate, calibrated later)
        elif n_lanes >= 2:
            vph = 150
        else:
            vph = 50

        # Convert vph to SUMO's vehsPerHour
        # Pick a random exit edge (SUMO will route via shortest path)
        for exit_edge in exit_edges[:3]:  # distribute across 3 exits
            if exit_edge.getID() == entry_edge.getID():
                continue
            # Verify a route exists between entry and exit
            route = net.getShortestPath(entry_edge, exit_edge)
            if route[0] is None:
                skipped += 1
                continue
            partial_vph = max(1, vph // 3)
            etree.SubElement(
                root,
                "flow",
                id=f"flow_{flow_id}",
                type="car",
                begin=str(SIM_START),
                end=str(SIM_END),
                vehsPerHour=str(partial_vph),
                **{"from": entry_edge.getID()},
                to=exit_edge.getID(),
                departLane="best",
                departSpeed="max",
            )
            flow_id += 1
    print(f"Skipped {skipped} unreachable entry-exit pairs")

    # Write demand file
    out_path = os.path.join(base_dir, "sumo", "demand", "demand.rou.xml")
    tree = etree.ElementTree(root)
    tree.write(out_path, pretty_print=True, xml_declaration=True, encoding="UTF-8")
    print(f"\nWrote {flow_id} flows to {out_path}")


if __name__ == "__main__":
    main()
