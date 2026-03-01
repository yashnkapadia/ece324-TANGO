"""
Purpose: Converts TMC turning volumes into SUMO flow definitions.
         Produces demand.rou.xml with vehicle flows by time period.

Logic:
    - For each entry edge compute total entry volume from TMC
    - For each TMC intersection, compute turning ratios per approach
    - Convert entry volumes to SUMO flow elements
    - Turning ratios are applied using <vType> route distributions or via explicit routes to specific exit edges 

Inputs:
  - data/processed/tmc_parsed.csv (parsed TMC volumes)
  - data/processed/intersection_map.csv (SUMO junction-to-TMC mapping)
  - sumo/network/osm.net.xml.gz (to resolve edge IDs)

Outputs:
  - sumo/demand/demand.rou.xml (SUMO flow definitions)

Running:
    python scripts\04_generate_demand.py
"""

# USES HEURISTICS FOR INITIAL DEMAND - CALIBRATION IN 05_calibrate.py WILL SCALE TO MATCH TMC APPROACH TOTALS
# STARTING POINT FOR CALIBRATION

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

    tmc = pd.read_csv(os.path.join(base_dir, "data", "processed", "tmc_parsed.csv"))
    int_map = pd.read_csv(os.path.join(base_dir, "data", "processed", "intersection_map.csv"))

    entry_edges = get_entry_edges(net)
    exit_edges = get_exit_edges(net)

    print(f"Entry edges: {len(entry_edges)}")
    print(f"Exit edges: {len(exit_edges)}")

    # For the initial demand, we assign flows between entry-exit edge pairs.
    # The volume is derived from TMC approach totals.
    
    # Approach: For each entry edge, find the nearest study intersection,
    # look up the TMC approach volume, and create a flow to plausible exit edges.

    root = etree.Element("routes")

    # Add default vehicle type
    etree.SubElement(root, "vType", id="car", accel="2.6", decel="4.5",
                      sigma="0.5", length="5.0", maxSpeed="13.89")

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
            etree.SubElement(root, "flow",
                              id=f"flow_{flow_id}",
                              type="car",
                              begin=str(SIM_START),
                              end=str(SIM_END),
                              vehsPerHour=str(partial_vph),
                              **{"from": entry_edge.getID()},
                              to=exit_edge.getID(),
                              departLane="best",
                              departSpeed="max")
            flow_id += 1
    print(f"Skipped {skipped} unreachable entry-exit pairs")

    # Write demand file
    out_path = os.path.join(base_dir, "sumo", "demand", "demand.rou.xml")
    tree = etree.ElementTree(root)
    tree.write(out_path, pretty_print=True, xml_declaration=True, encoding="UTF-8")
    print(f"\nWrote {flow_id} flows to {out_path}")

    # TODO: After calibration, replace the heuristic volumes with TMC-derived values.
    # The calibration loop (05_calibrate.py) will scale these flows.

if __name__ == "__main__":
    main()