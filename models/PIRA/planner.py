"""
planner.py — PIRA Scenario Planner
===================================
A friendly interface for running PIRA "what-if" scenarios without touching
the model internals.

QUICK START
-----------
There are two ways to use this file:

  (A) Run it directly from the terminal:

        python planner.py

      It will walk you through finding an intersection and running a scenario
      interactively.

  (B) Import PIRAPlanner in your own script or notebook:

        from planner import PIRAPlanner

        planner = PIRAPlanner(
            model_path      = 'pira_model.pt',
            network_path    = 'osm.net.xml',
            data_path       = 'data/final/dataset.parquet',
            # Optional — maps readable names to junction IDs:
            intersection_map = 'intersection_map.csv',
        )

        # See what intersections are available
        planner.list_intersections()

        # Run a scenario
        result = planner.run(
            disruption_type   = 'lane_closure',
            from_intersection = 'Spadina Ave / Dundas St W',
            to_intersection   = 'Bathurst St / Dundas St W',
            capacity          = 0.5,   # 0 = full closure, 1 = no change
            demand            = 1.0,   # 1.0 = normal, 1.3 = 30% more traffic
        )

WHAT THE PARAMETERS MEAN
-------------------------
  disruption_type:
      One of: 'none', 'construction', 'lane_closure', 'transit_line',
              'major_event'

  from_intersection / to_intersection:
      The road segment you are closing runs FROM one intersection TO the next.
      Direction matters — "Spadina -> Bathurst" is westbound on Dundas.
      Use planner.list_intersections() to see the exact names to type.

  capacity:
      How much road capacity remains after the disruption.
        0.0  — road is fully closed (edge removed from the graph)
        0.5  — one lane blocked, ~half capacity remains
        1.0  — no change (useful for demand-only scenarios)

  demand:
      Multiplier on normal traffic volume.
        1.0  — unchanged
        1.3  — 30% more traffic (e.g. rerouted vehicles)
        0.8  — 20% less traffic (e.g. people avoiding the area)

OUTPUT
------
  result['impact']  — DataFrame: predicted delay, throughput, queue per
                      intersection, sorted worst-first by delay.
  result['timing']  — DataFrame: recommended green durations (seconds) for
                      NS and EW phases at each intersection.
  result['elapsed_ms'] — how long inference took (should be < 5000 ms).
"""

import sys
import textwrap
import pandas as pd
import torch

# Everything we need lives in pira.py, which must be in the same directory.
try:
    from pira import (
        parse_sumo_network,
        build_edge_lookup,
        ScenarioDescriptor,
        predict_scenario,
        PIRAModel,
        NODE_FEATURE_COLUMNS,
        DISRUPTION_TYPES,
        SCENARIO_FEAT_DIM,
    )
except ImportError as exc:
    sys.exit(
        "Could not import pira.py. Make sure planner.py and pira.py are in "
        f"the same directory.\n  Error: {exc}"
    )


# ===========================================================================
# PIRAPlanner — the main class
# ===========================================================================
class PIRAPlanner:
    """One-stop interface for PIRA scenario prediction.

    Parameters
    ----------
    model_path : str
        Path to a saved ``pira_model.pt`` checkpoint.
    network_path : str
        Path to the SUMO ``osm.net.xml`` road network file.
    data_path : str
        Path to the Parquet dataset (``data/final/dataset.parquet``).
        Used to pull the baseline traffic snapshot for inference.
    intersection_map : str, optional
        Path to the intersection-map CSV produced by the data pipeline.
        Without this, you must use raw SUMO junction IDs instead of
        readable intersection names.
    device : str
        ``'cpu'``, ``'cuda'``, or ``'auto'`` (default — picks GPU if available).
    """

    def __init__(self, model_path, network_path, data_path, intersection_map=None, device="auto"):

        self._device = (
            ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        )

        print("Loading PIRA planner ...")

        # ── Network ──────────────────────────────────────────────────────────
        # Parse the SUMO road network to get the graph structure.
        self._edge_index, self._node_map, self._edge_meta = parse_sumo_network(network_path)

        num_nodes = len(self._node_map)
        num_edges = self._edge_index.shape[1]
        print(f"  Network : {num_nodes} intersections, {num_edges} road segments")

        # ── Edge lookup ───────────────────────────────────────────────────────
        # Lets us find edge indices by intersection name (if the map is given)
        # or by raw junction ID.
        self._lookup = build_edge_lookup(self._edge_meta, intersection_map)
        self._has_names = intersection_map is not None

        # ── Baseline traffic snapshot ─────────────────────────────────────────
        # We use the latest logged time-step of the 'baseline' scenario as the
        # starting traffic state for all predictions.
        df = pd.read_parquet(data_path)
        self._baseline_df = self._extract_baseline(df)
        print(f"  Baseline: {len(self._baseline_df)} intersection rows loaded")

        # ── Model ─────────────────────────────────────────────────────────────
        checkpoint = torch.load(model_path, map_location=self._device)
        cfg = checkpoint["config"]
        self._model = PIRAModel(**cfg)
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model.eval()
        print(f"  Model   : loaded from '{model_path}'")
        print("Ready.\n")

    # -----------------------------------------------------------------------
    # Public helpers
    # -----------------------------------------------------------------------

    def list_intersections(self):
        """Print every intersection name (or junction ID) known to the planner.

        Use these exact strings in the ``from_intersection`` and
        ``to_intersection`` arguments of :meth:`run`.
        """
        if self._has_names:
            print("\nAvailable intersections (use these exact names in run()):")
            print("-" * 60)
            self._lookup.list_named()
        else:
            print("\nNo intersection map loaded — showing raw junction IDs.")
            print("Pass intersection_map='intersection_map.csv' to PIRAPlanner")
            print("to see human-readable names instead.")
            print("-" * 60)
            self._lookup.list_all()
        print()

    def find_edge(self, from_intersection, to_intersection):
        """Look up the road segment between two intersections.

        Returns the list of edge indices — you normally don't need to call
        this directly; :meth:`run` calls it for you.

        Parameters
        ----------
        from_intersection : str
            Origin intersection name (or junction ID if no map is loaded).
        to_intersection : str
            Destination intersection name (or junction ID if no map is loaded).
        """
        if self._has_names:
            indices = self._lookup.find_by_name(from_intersection, to_intersection)
        else:
            indices = self._lookup.find(from_intersection, to_intersection)

        if not indices:
            raise ValueError(
                f"No road segment found from '{from_intersection}' to "
                f"'{to_intersection}'.\n"
                "  Tips:\n"
                "  - Direction matters: 'A -> B' is different from 'B -> A'.\n"
                "  - Run planner.list_intersections() to see valid names.\n"
                "  - If you don't have an intersection_map.csv yet, pass\n"
                "    raw SUMO junction IDs (e.g. 'cluster_123_456')."
            )

        print(f"  Found {len(indices)} road segment(s): edge index/indices {indices}")
        return indices

    def run(
        self,
        disruption_type,
        from_intersection,
        to_intersection,
        capacity=0.5,
        demand=1.0,
        scenario_id=None,
        verbose=True,
    ):
        """Predict the traffic impact of a disruption scenario.

        Parameters
        ----------
        disruption_type : str
            What kind of disruption this is.
            Choices: ``'none'``, ``'construction'``, ``'lane_closure'``,
            ``'transit_line'``, ``'major_event'``.
        from_intersection : str
            The disruption starts at this intersection (road origin).
            For a westbound lane closure, this is the *eastern* end.
        to_intersection : str
            The disruption ends at this intersection (road destination).
            For a westbound lane closure, this is the *western* end.
        capacity : float
            Remaining road capacity on the affected segment.
              - ``0.0`` = fully closed (edge removed from graph)
              - ``0.5`` = one lane removed, half capacity
              - ``1.0`` = no capacity change
        demand : float
            Traffic demand multiplier.
              - ``1.0`` = normal volume
              - ``1.3`` = 30 percent more (e.g. rerouted vehicles)
              - ``0.8`` = 20 percent less
        scenario_id : str, optional
            A label for this scenario. Auto-generated if omitted.
        verbose : bool
            Print a formatted summary of results.

        Returns
        -------
        dict
            ``impact``      – DataFrame of delay/throughput/queue per junction.
            ``timing``      – DataFrame of recommended green durations (s).
            ``elapsed_ms``  – Inference time in milliseconds.
        """
        # ── Validate disruption type ─────────────────────────────────────────
        if disruption_type not in DISRUPTION_TYPES:
            raise ValueError(
                f"Unknown disruption_type '{disruption_type}'.\n"
                f"  Valid options: {DISRUPTION_TYPES}"
            )

        # ── Validate capacity ────────────────────────────────────────────────
        if not (0.0 <= capacity <= 1.0):
            raise ValueError(f"capacity must be between 0.0 and 1.0, got {capacity}.")

        # ── Validate demand ──────────────────────────────────────────────────
        if demand <= 0.0:
            raise ValueError(f"demand must be positive, got {demand}.")

        # ── Find affected edges ──────────────────────────────────────────────
        print(f"\nScenario: {disruption_type}  |  " f"capacity={capacity}  |  demand={demand}x")
        print(f"  Segment : '{from_intersection}'  ->  '{to_intersection}'")

        affected_edges = self.find_edge(from_intersection, to_intersection)

        # ── Build descriptor ─────────────────────────────────────────────────
        sid = scenario_id or (f"{disruption_type}_{from_intersection[:20].replace(' ', '_')}")
        scenario = ScenarioDescriptor(
            scenario_id=sid,
            disruption_type=disruption_type,
            affected_edge_indices=affected_edges,
            capacity_reduction=capacity,
            demand_multiplier=demand,
        )

        # ── Run inference ────────────────────────────────────────────────────
        result = predict_scenario(
            model=self._model,
            scenario=scenario,
            baseline_traffic_state=self._baseline_df,
            edge_index=self._edge_index,
            node_id_to_idx=self._node_map,
            device=self._device,
        )

        # ── Print summary ────────────────────────────────────────────────────
        if verbose:
            self._print_summary(result, disruption_type, capacity, demand)

        return result

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _extract_baseline(self, df):
        """Pull the latest time-step of the baseline scenario as the snapshot."""
        # Prefer 'baseline' scenario; fall back to the first scenario available.
        baseline_ids = [s for s in df["scenario_id"].unique() if "baseline" in str(s).lower()]
        sid = baseline_ids[0] if baseline_ids else df["scenario_id"].iloc[0]

        subset = df[df["scenario_id"] == sid]
        latest_step = subset["time_step"].max()
        return subset[subset["time_step"] == latest_step].copy()

    def _print_summary(self, result, disruption_type, capacity, demand):
        """Print a formatted, human-readable results summary."""
        impact = result["impact"].copy()
        timing = result["timing"].copy()
        ms = result["elapsed_ms"]

        # Sort intersections by predicted delay so worst ones are first.
        impact_sorted = impact.sort_values("delay", ascending=False)

        bar = "=" * 62
        print(f"\n{bar}")
        print("  PIRA Scenario Impact Summary")
        print(bar)
        print(f"  Type     : {disruption_type}")
        print(f"  Capacity : {capacity:.0%} remaining")
        print(f"  Demand   : {demand:.1f}x baseline")
        print(f"  Latency  : {ms:.1f} ms")
        print()

        # Impact table
        print(f"  {'Junction':<35} {'Delay(s)':>9} {'Throughput':>11} {'Queue':>6}")
        print(f"  {'-'*35} {'-'*9} {'-'*11} {'-'*6}")
        for jid, row in impact_sorted.iterrows():
            # Truncate long junction IDs so the table stays readable
            label = (jid[:33] + "..") if len(jid) > 35 else jid
            print(
                f"  {label:<35} {row['delay']:>9.1f} "
                f"{row['throughput']:>11.1f} {row['queue_total']:>6.1f}"
            )

        print()

        # Timing recommendations
        print(f"  {'Junction':<35} {'Green NS (s)':>13} {'Green EW (s)':>13}")
        print(f"  {'-'*35} {'-'*13} {'-'*13}")
        for jid, row in timing.iterrows():
            label = (jid[:33] + "..") if len(jid) > 35 else jid
            print(f"  {label:<35} {row['green_ns_s']:>13.1f} {row['green_ew_s']:>13.1f}")

        print(bar)
        print()


# ===========================================================================
# Interactive CLI — runs when you do: python planner.py
# ===========================================================================
def _interactive():
    """Walk the user through a scenario step by step."""

    print(textwrap.dedent("""
    ╔══════════════════════════════════════════════════════════╗
    ║           PIRA Interactive Scenario Planner             ║
    ╚══════════════════════════════════════════════════════════╝

    This tool predicts how a lane closure, construction project,
    or new transit line will affect traffic on the Dundas corridor.

    You will need:
      1. pira_model.pt          — trained PIRA model
      2. osm.net.xml            — SUMO road network
      3. data/final/dataset.parquet  — ASCE rollout logs
      4. intersection_map.csv   — (optional) readable intersection names
    """))

    # ── File paths ───────────────────────────────────────────────────────────
    model_path = input("Path to model     [pira_model.pt]: ").strip() or "pira_model.pt"
    network_path = input("Path to network   [osm.net.xml]: ").strip() or "osm.net.xml"
    data_path = (
        input("Path to dataset   [data/final/dataset.parquet]: ").strip()
        or "data/final/dataset.parquet"
    )
    map_path = input("Intersection map  [intersection_map.csv, Enter to skip]: ").strip() or None

    try:
        planner = PIRAPlanner(
            model_path=model_path,
            network_path=network_path,
            data_path=data_path,
            intersection_map=map_path,
        )
    except Exception as exc:
        sys.exit(f"Failed to load planner:\n  {exc}")

    # ── Show available intersections ─────────────────────────────────────────
    show = input("Show available intersections? [Y/n]: ").strip().lower()
    if show != "n":
        planner.list_intersections()

    # ── Scenario loop ────────────────────────────────────────────────────────
    while True:
        print("\n" + "-" * 50)
        print("Define a scenario (Ctrl-C to quit)")
        print("-" * 50)

        # Disruption type
        print(f"Disruption types: {', '.join(DISRUPTION_TYPES)}")
        while True:
            dtype = input("Disruption type [lane_closure]: ").strip() or "lane_closure"
            if dtype in DISRUPTION_TYPES:
                break
            print(f"  Invalid. Choose from: {DISRUPTION_TYPES}")

        # Intersections
        from_int = input("From intersection (road origin ): ").strip()
        to_int = input("To   intersection (road destination): ").strip()

        # Capacity
        while True:
            raw = input("Remaining capacity (0=closed, 0.5=half, 1=open) [0.5]: ").strip()
            try:
                cap = float(raw) if raw else 0.5
                if 0.0 <= cap <= 1.0:
                    break
                print("  Must be between 0.0 and 1.0.")
            except ValueError:
                print("  Enter a number like 0.5.")

        # Demand
        while True:
            raw = input("Demand multiplier (1.0=normal, 1.3=30% more) [1.0]: ").strip()
            try:
                dem = float(raw) if raw else 1.0
                if dem > 0:
                    break
                print("  Must be positive.")
            except ValueError:
                print("  Enter a number like 1.0.")

        # Run
        try:
            planner.run(
                disruption_type=dtype,
                from_intersection=from_int,
                to_intersection=to_int,
                capacity=cap,
                demand=dem,
            )
        except (ValueError, KeyError) as exc:
            print(f"\nError: {exc}")

        again = input("\nRun another scenario? [Y/n]: ").strip().lower()
        if again == "n":
            break

    print("\nDone.")


if __name__ == "__main__":
    _interactive()
