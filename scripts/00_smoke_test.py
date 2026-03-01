"""
Purpose: Minimal smoke test that verifies SUMO installation, loads the network,
         runs a 60-second simulation with random trips, and confirms TraCI works.

Inputs:
  - sumo/network/osm.net.xml.gz

Outputs:
  - Console output confirming success
  - sumo/demand/random_trips.rou.xml (auto-generated random trips for test only)

Running:
    python scripts\00_smoke_test.py
"""

import os
import sys
import subprocess

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    net_file = os.path.join(base_dir, "sumo", "network", "osm.net.xml.gz")

    if not os.path.exists(net_file):
        print("ERROR: Network file not found. Run Step 1 first.")
        sys.exit(1)

    # Generate random trips for smoke test
    sumo_home = os.environ.get("SUMO_HOME", "")
    random_trips_py = os.path.join(sumo_home, "tools", "randomTrips.py")

    trip_file = os.path.join(base_dir, "sumo", "demand", "random_trips.rou.xml")
    print("Generating random trips for smoke test...")
    subprocess.run([
        sys.executable, random_trips_py,
        "-n", net_file,
        "-o", trip_file,
        "-e", "60",        # 60 seconds
        "-p", "2.0",       # one vehicle every 2 seconds
        "--route-file", trip_file,
        "--validate",
    ], check=True)

    # Run SUMO headless for 60 seconds
    print("\nRunning SUMO headless for 60 seconds...")
    import traci
    
    sumo_cmd = [
        "sumo",  # headless (no GUI)
        "-n", net_file,
        "-r", trip_file,
        "--additional-files", os.path.join(base_dir, "sumo", "network", "tls_overrides.add.xml.gz"),
        "--begin", "0",
        "--end", "60",
        "--step-length", "1.0",
        "--no-step-log", "true",
    ]

    traci.start(sumo_cmd)
    step = 0
    while step < 60:
        traci.simulationStep()
        if step % 10 == 0:
            n_vehicles = traci.vehicle.getIDCount()
            n_tls = len(traci.trafficlight.getIDList())
            print(f"  Step {step:4d}: {n_vehicles:3d} vehicles, {n_tls} traffic lights")
        step += 1
    
    traci.close()
    print("\nSmoke test PASSED. SUMO and TraCI working correctly.")

if __name__ == "__main__":
    main()