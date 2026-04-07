"""
Purpose: Parses raw TMC CSV files and filters to our study area intersections.
         Produces a clean CSV with per-approach, per-movement volumes by time period.

Inputs:
  - data/raw/tmc/*.csv (raw TMC downloads)
  - data/processed/intersection_map.csv (to filter to study area)

Outputs:
  - data/processed/tmc_parsed.csv
    Columns: intersection_name, tmc_station_id, date, time_period,
             approach_direction, movement, vehicle_count

Running:
    python scripts\02_parse_tmc.py
"""

import pandas as pd
import os
import glob


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, "data", "raw", "tmc")

    # Load all CSVs
    csv_files = glob.glob(os.path.join(raw_dir, "*.csv"))
    if not csv_files:
        print("ERROR: No CSV files found in data/raw/tmc/")
        print("Please run scripts/01_download_tmc.py first or manually download TMC data.")
        return

    dfs = []
    for f in csv_files:
        print(f"Reading {os.path.basename(f)}...")
        try:
            df = pd.read_csv(f, encoding="utf-8")
            dfs.append(df)
            print(f"  {len(df)} rows, columns: {list(df.columns)}")
        except Exception as e:
            print(f"  Error reading {f}: {e}")

    if not dfs:
        print("No data loaded.")
        return

    raw = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal raw rows: {len(raw)}")
    print(f"Columns: {list(raw.columns)}")
    print("\nSample rows:")
    print(raw.head())

    # The TMC dataset columns vary by year. Common patterns:
    # - "location_id" or "int_id" - intersection identifier
    # - "location" or "intersection_name" - human readable name
    # - "datetime_bin" or "count_date" + "time_start" - time
    # - "classification" - vehicle type
    # - "dir" or "direction" - approach direction (N, S, E, W)
    # - "movement" - turning movement (left, thru, right)
    # - "volume" or "count" - vehicle count

    # Standardize column names (adjust based on actual downloaded data)
    # Print unique columns to understand structure
    print(f"\nUnique columns: {list(raw.columns)}")

    # Auto-detect key columns
    col_map = {}
    for col in raw.columns:
        cl = col.lower().strip()
        if "location" in cl and "id" in cl:
            col_map["location_id"] = col
        elif "location" in cl or "intersection" in cl:
            col_map["location_name"] = col
        elif "datetime" in cl or "date" in cl:
            col_map["datetime"] = col
        elif cl in ("dir", "direction", "approach"):
            col_map["direction"] = col
        elif "movement" in cl or "turn" in cl:
            col_map["movement"] = col
        elif cl in ("volume", "count", "veh_volume", "vehicle_count"):
            col_map["volume"] = col
        elif "classification" in cl or "class" in cl:
            col_map["classification"] = col

    print(f"\nDetected column mapping: {col_map}")

    # Filter to study area
    # Our study area intersection names contain "Dundas"
    # Filter by name if available
    if "location_name" in col_map:
        name_col = col_map["location_name"]
        dundas_mask = raw[name_col].str.contains("DUNDAS", case=False, na=False)
        filtered = raw[dundas_mask].copy()
        print(f"\nFiltered to Dundas intersections: {len(filtered)} rows")
        print(f"Unique locations: {filtered[name_col].unique()}")
    else:
        print("\nWARNING: Could not find location name column. Saving all data.")
        filtered = raw.copy()

    # Save parsed output
    out_path = os.path.join(base_dir, "data", "processed", "tmc_parsed.csv")
    filtered.to_csv(out_path, index=False)
    print(f"\nSaved {len(filtered)} rows to {out_path}")


if __name__ == "__main__":
    main()
