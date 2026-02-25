"""
Purpose: Maps Toronto intersection names to SUMO junction IDs by matching
         coordinates from the SUMO network to known intersection locations.
         Creates the definitive intersection_map.csv used by all downstream scripts.

Inputs:
  - sumo/network/osm.net.xml.gz (SUMO network file from OSM Web Wizard)
  
Outputs:
  - data/processed/intersection_map.csv
    Columns: intersection_name, sumo_junction_id, x, y, has_tls, tmc_station_id
"""

import sumolib
import pandas as pd
import os

# Known intersection approximate WGS84 coords (lat, lon)
# Convert to SUMO's projected coords using sumolib
KNOWN_INTERSECTIONS = [
    ("Dundas_University",    43.654830044932815, -79.3883154570028),
    ("Dundas_Simcoe",        43.65468474113212, -79.38921534771256),
    ("Dundas_McCaul",        43.65431844096621, -79.3914424285752),
    ("Dundas_Beverly",       43.653822834909064, -79.39384930321549),
    ("Dundas_Huron",         43.653346620950664, -79.39615287529742),
    ("Dundas_StPatrick",     43.65449208352238, -79.39033230968097),
    ("Dundas_Spadina",       43.65293576180883, -79.39804988289666),
    ("Dundas_Augusta",       43.65227579321216, -79.40113064068092),
    ("Dundas_Bellevue",      43.65214898379732, -79.40482185756817),
    ("Dundas_Denison",       43.65204326403906, -79.40225356856256),
    ("Dundas_Bathurst",      43.65230935563683, -79.4060238013276),
]

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    net_file = os.path.join(base_dir, "sumo", "network", "osm.net.xml.gz")
    net = sumolib.net.readNet(net_file)

    rows = []
    tls_ids = {tls.getID() for tls in net.getTrafficLights()}

    for name, lat, lon in KNOWN_INTERSECTIONS:
        # Convert lat/lon to SUMO x,y
        x, y = net.convertLonLat2XY(lon, lat)

        # Find nearest junction
        # sumolib uses (x, y) in projected coordinates
        min_dist = float('inf')
        best_node = None
        for node in net.getNodes():
            nx, ny = node.getCoord()
            dist = ((nx - x)**2 + (ny - y)**2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                best_node = node

        if best_node and min_dist < 100:  # within 100m
            jid = best_node.getID()
            has_tls = jid in tls_ids or best_node.getType() == "traffic_light"
            rows.append({
                "intersection_name": name,
                "sumo_junction_id": jid,
                "x": best_node.getCoord()[0],
                "y": best_node.getCoord()[1],
                "has_tls": has_tls,
                "tmc_station_id": "",  # filled in Step 3
                "match_distance_m": round(min_dist, 1),
            })
            print(f"  {name:30s} -> {jid:40s} (dist={min_dist:.1f}m, tls={has_tls})")
        else:
            print(f"  WARNING: {name} — no junction within 100m (closest: {min_dist:.1f}m)")

    df = pd.DataFrame(rows)
    out_path = os.path.join(base_dir, "data", "processed", "intersection_map.csv")
    df.to_csv(out_path, index=False)
    print(f"\nWrote {len(df)} intersections to {out_path}")

if __name__ == "__main__":
    main()