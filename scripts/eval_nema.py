"""Run NEMA baseline evaluation across 4 scenarios, 5 seeds each.

NEMA = native SUMO signal programs (fixed_ts=True). These are the default
actuated programs from the OSM network with proper phase sequences,
yellow/all-red clearance, and min-green constraints.

Usage:
    PYTHONUNBUFFERED=1 pixi run python scripts/eval_nema.py
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("LIBSUMO_AS_TRACI", "1")

PROJ_ROOT = Path(__file__).resolve().parent.parent
NET_FILE = str(PROJ_ROOT / "sumo" / "network" / "osm.net.xml.gz")
SCENARIOS = {
    "am_peak": str(PROJ_ROOT / "sumo" / "demand" / "curriculum" / "am_peak.rou.xml"),
    "pm_peak": str(PROJ_ROOT / "sumo" / "demand" / "curriculum" / "pm_peak.rou.xml"),
    "demand_surge": str(PROJ_ROOT / "sumo" / "demand" / "curriculum" / "demand_surge.rou.xml"),
    "midday_multimodal": str(
        PROJ_ROOT / "sumo" / "demand" / "curriculum" / "midday_multimodal.rou.xml"
    ),
}
DEFAULT_SEED_BASE = 1000
DEFAULT_NUM_SEEDS = 5
OUT_DIR = PROJ_ROOT / "reports" / "results" / "eval_matrix"
DELTA_TIME = 5


def run_nema_episode(net_file: str, route_file: str, seconds: int, seed: int) -> dict:
    """Run one NEMA episode with KPI tracking."""
    from ece324_tango.sumo_rl.environment.env import SumoEnvironment
    from ece324_tango.asce.kpi import KPITracker

    env = SumoEnvironment(
        net_file=net_file,
        route_file=route_file,
        use_gui=False,
        num_seconds=seconds,
        delta_time=DELTA_TIME,
        sumo_seed=seed,
        single_agent=False,
        sumo_warnings=False,
        fixed_ts=True,
        additional_sumo_cmd="--no-step-log true",
    )

    tracker = KPITracker()
    obs = env.reset(seed=seed)
    done = False
    steps = 0

    while not done:
        tracker.update(env)
        # NEMA: pass empty actions, signals follow native program
        actions = {agent_id: 0 for agent_id in env.ts_ids}
        # SumoEnvironment with fixed_ts uses old Gym API (4 returns)
        step_result = env.step(actions)
        if len(step_result) == 5:
            obs, rewards, terminated, truncated, info = step_result
            done = terminated.get("__all__", False) or truncated.get("__all__", False)
        else:
            obs, rewards, dones, info = step_result
            done = dones.get("__all__", False) if isinstance(dones, dict) else dones
        steps += 1

    # Final collection
    tracker.update(env)
    kpi = tracker.summary()
    env.close()

    return {
        "steps": steps,
        "time_loss_s": kpi.time_loss_s,
        "person_time_loss_s": kpi.person_time_loss_s,
        "avg_trip_time_s": kpi.avg_trip_time_s,
        "arrived_vehicles": kpi.arrived_vehicles,
        "vehicle_delay_jain": kpi.vehicle_delay_jain,
    }


def _run_single(args):
    """Run a single NEMA eval in a subprocess to avoid libsumo state leaks."""
    scenario_name, route_file, seed, ep_idx = args
    import json
    try:
        result = run_nema_episode(NET_FILE, route_file, 900, seed)
        return json.dumps({"ok": True, "scenario": scenario_name, "seed": seed,
                           "ep": ep_idx, **result})
    except Exception as e:
        return json.dumps({"ok": False, "scenario": scenario_name, "seed": seed,
                           "ep": ep_idx, "error": str(e)})


def main():
    import argparse
    import json
    import subprocess

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed-base",
        type=int,
        default=DEFAULT_SEED_BASE,
        help="First seed (subsequent seeds are seed-base+1, +2, ...).",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=DEFAULT_NUM_SEEDS,
        help="Number of seeds per scenario.",
    )
    parser.add_argument(
        "--out-suffix",
        type=str,
        default="",
        help="Suffix for output CSV name (e.g. '_v2'); default writes nema__<scenario>.csv.",
    )
    args = parser.parse_args()

    SEEDS = [args.seed_base + i for i in range(args.num_seeds)]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "controller", "episode", "seed", "steps",
        "time_loss_s", "person_time_loss_s", "avg_trip_time_s",
        "arrived_vehicles", "vehicle_delay_jain",
    ]

    for scenario_name, route_file in SCENARIOS.items():
        out_csv = OUT_DIR / f"nema{args.out_suffix}__{scenario_name}.csv"
        print(f"=== NEMA eval: {scenario_name} ===")

        rows = []
        for ep_idx, seed in enumerate(SEEDS):
            print(f"  Seed {seed} ({ep_idx+1}/5)...", end=" ", flush=True)
            # Run each seed in a subprocess to avoid libsumo state issues
            cmd = [
                sys.executable, "-c",
                f"""
import os, sys, json
sys.path.insert(0, '{PROJ_ROOT}')
os.environ['LIBSUMO_AS_TRACI'] = '1'
from scripts.eval_nema import run_nema_episode
try:
    r = run_nema_episode('{NET_FILE}', '{route_file}', 900, {seed})
    print(json.dumps({{"ok": True, **r}}))
except Exception as e:
    print(json.dumps({{"ok": False, "error": str(e)}}))
"""
            ]
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=120,
                    env={**os.environ, "LIBSUMO_AS_TRACI": "1"},
                )
                # Find the JSON line in stdout
                result_line = None
                for line in proc.stdout.strip().split("\n"):
                    line = line.strip()
                    if line.startswith("{"):
                        result_line = line
                        break

                if result_line:
                    result = json.loads(result_line)
                    if result.get("ok"):
                        row = {
                            "controller": "nema", "episode": ep_idx, "seed": seed,
                            "steps": result["steps"],
                            "time_loss_s": result["time_loss_s"],
                            "person_time_loss_s": result["person_time_loss_s"],
                            "avg_trip_time_s": result["avg_trip_time_s"],
                            "arrived_vehicles": result["arrived_vehicles"],
                            "vehicle_delay_jain": result["vehicle_delay_jain"],
                        }
                        rows.append(row)
                        print(f"PTL={result['person_time_loss_s']:,.0f}  "
                              f"Jain={result['vehicle_delay_jain']:.3f}  "
                              f"arrived={result['arrived_vehicles']}")
                    else:
                        print(f"FAILED: {result.get('error', 'unknown')}")
                else:
                    stderr_tail = proc.stderr[-200:] if proc.stderr else ""
                    print(f"FAILED: no JSON output. stderr: {stderr_tail}")
            except subprocess.TimeoutExpired:
                print("TIMEOUT")
            except Exception as e:
                print(f"FAILED: {e}")

        if rows:
            with open(out_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            print(f"  Saved: {out_csv} ({len(rows)}/5 seeds)")
        print()

    print("All NEMA evals complete.")


if __name__ == "__main__":
    main()
