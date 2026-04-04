"""
TANGO Demand Studio

A local web interface for connecting a SUMO network file and generating
scenario-specific demand route files directly from data/processed/tmc_parsed.csv.

This app does not touch the existing hardcoded demand file. It writes generated
outputs to dedicated generated folders under sumo/.
"""

from __future__ import annotations

import gzip
import json
import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sumolib
from dash import Dash, Input, Output, State, callback, dcc, html
from dash.exceptions import PreventUpdate
from lxml import etree

REPO_ROOT = Path(__file__).resolve().parents[2]
TMC_PATH = REPO_ROOT / "data" / "processed" / "tmc_parsed.csv"

DEMAND_OUTPUT_DIR = REPO_ROOT / "sumo" / "demand" / "generated"
SCENARIO_OUTPUT_DIR = REPO_ROOT / "sumo" / "scenarios" / "generated"
SUMOCFG_OUTPUT_DIR = REPO_ROOT / "sumo" / "config" / "generated"
TLS_OUTPUT_DIR = REPO_ROOT / "sumo" / "network" / "generated"

ALLOWED_NET_SUFFIXES = (
    ".net.xml",
    ".net.xml.gz",
    ".network.xml",
    ".network.xml.gz",
)

DIRECTIONS = ("n", "s", "e", "w")
MOVEMENTS = ("r", "t", "l")

APPROACH_HEADING_RANGES_DEFAULT = {
    "n": (160.0, 230.0),
    "s": (340.0, 40.0),
    "e": (250.0, 310.0),
    "w": (70.0, 130.0),
}

DEFAULT_FIXED_SIGNAL = {
    "ew_green": 30,
    "ns_green": 20,
    "yellow": 3,
    "all_red": 2,
    "offset": 0,
}

DEFAULT_MAX_PRESSURE_SIGNAL = {
    "min_green": 10,
    "max_green": 45,
    "yellow": 3,
    "all_red": 2,
    "pressure_exponent": 1.0,
    "queue_exponent": 1.0,
}

DEFAULT_MAPPO = {
    "learning_rate_actor": 3e-4,
    "learning_rate_critic": 5e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_ratio": 0.2,
    "entropy_coef": 0.01,
    "value_loss_coef": 0.5,
    "max_grad_norm": 0.5,
    "rollout_steps": 1024,
    "ppo_epochs": 10,
    "minibatch_size": 256,
    "num_envs": 8,
    "hidden_dim": 256,
    "hidden_layers": 2,
    "target_kl": 0.03,
    "seed": 42,
    "shared_critic": True,
    "normalize_advantages": True,
}

DEFAULT_VTYPE = {
    "car": {
        "accel": 2.6,
        "decel": 4.5,
        "sigma": 0.5,
        "length": 5.0,
        "maxSpeed": 13.89,
    },
    "truck": {
        "accel": 1.2,
        "decel": 3.5,
        "sigma": 0.5,
        "length": 10.5,
        "maxSpeed": 12.5,
    },
    "bus": {
        "accel": 1.3,
        "decel": 4.0,
        "sigma": 0.5,
        "length": 12.0,
        "maxSpeed": 12.5,
    },
    "streetcar": {
        "accel": 1.0,
        "decel": 2.5,
        "sigma": 0.4,
        "length": 30.0,
        "maxSpeed": 16.0,
    },
}

MODE_COLUMN_PREFIX = {
    "cars": "cars",
    "trucks": "truck",
    "buses": "bus",
}

MODE_TO_VTYPE_AND_CLASS = {
    "cars": ("car", "passenger"),
    "trucks": ("truck", "truck"),
    "buses": ("bus", "bus"),
    "streetcars": ("streetcar", "tram"),
}

NETWORK_CACHE: dict[str, dict[str, Any]] = {}


@dataclass
class GenerationResult:
    success: bool
    message: str
    demand_path: Path | None = None
    scenario_json_path: Path | None = None
    sumocfg_path: Path | None = None
    tls_path: Path | None = None
    stats: dict[str, Any] | None = None
    warnings: list[str] | None = None
    validation: dict[str, Any] | None = None
    scenario_payload: dict[str, Any] | None = None


def to_relpath(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def sanitize_tag(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_")
    return cleaned.lower() or "scenario"


def parse_iso_date(date_str: str | None) -> date | None:
    if not date_str:
        return None
    try:
        return date.fromisoformat(str(date_str))
    except ValueError:
        return None


def clamp_float(
    value: Any, fallback: float, lo: float | None = None, hi: float | None = None
) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = float(fallback)
    if lo is not None:
        parsed = max(lo, parsed)
    if hi is not None:
        parsed = min(hi, parsed)
    return parsed


def clamp_int(value: Any, fallback: int, lo: int | None = None, hi: int | None = None) -> int:
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        parsed = int(fallback)
    if lo is not None:
        parsed = max(lo, parsed)
    if hi is not None:
        parsed = min(hi, parsed)
    return parsed


def discover_network_files() -> list[str]:
    files: set[str] = set()
    patterns = [
        "*.net.xml",
        "*.net.xml.gz",
        "*.network.xml",
        "*.network.xml.gz",
        "*.osm.net.xml",
        "*.osm.net.xml.gz",
    ]
    for pattern in patterns:
        for path in REPO_ROOT.rglob(pattern):
            if ".ipynb_checkpoints" in path.parts:
                continue
            files.add(to_relpath(path))
    return sorted(files)


def select_default_network(network_files: list[str]) -> str | None:
    preferred = "sumo/network/osm.net.xml.gz"
    if preferred in network_files:
        return preferred
    return network_files[0] if network_files else None


def resolve_network_path(dropdown_value: str | None, custom_value: str | None) -> Path:
    if custom_value and str(custom_value).strip():
        candidate = Path(str(custom_value).strip().strip('"')).expanduser()
        if not candidate.is_absolute():
            candidate = (REPO_ROOT / candidate).resolve()
    elif dropdown_value:
        candidate = (REPO_ROOT / dropdown_value).resolve()
    else:
        raise ValueError("No network path provided")

    if not candidate.exists():
        raise FileNotFoundError(f"Network file not found: {candidate}")

    lowered = candidate.name.lower()
    if not any(lowered.endswith(suffix) for suffix in ALLOWED_NET_SUFFIXES):
        raise ValueError(
            "Network file must end with .net.xml, .net.xml.gz, .network.xml, or .network.xml.gz"
        )

    return candidate


def load_tmc_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"TMC parsed file not found: {path}")

    df = pd.read_csv(path, low_memory=False)
    if "location_name" not in df.columns:
        raise ValueError("tmc_parsed.csv does not include location_name column")

    df = df[df["location_name"].notna()].copy()

    if "count_date" in df.columns:
        df["count_date"] = pd.to_datetime(df["count_date"], errors="coerce").dt.date
    else:
        df["count_date"] = pd.NaT

    if "start_time" in df.columns:
        start_time = pd.to_datetime(df["start_time"], errors="coerce")
        df["time_minutes"] = start_time.dt.hour.fillna(-1).astype(
            int
        ) * 60 + start_time.dt.minute.fillna(0).astype(int)
    else:
        df["time_minutes"] = -1

    numeric_columns = [
        "longitude",
        "latitude",
    ]

    for mode_prefix in MODE_COLUMN_PREFIX.values():
        for direction in DIRECTIONS:
            for movement in MOVEMENTS:
                numeric_columns.append(f"{direction}_appr_{mode_prefix}_{movement}")

    for direction in DIRECTIONS:
        numeric_columns.append(f"{direction}_appr_peds")

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)

    return df


def available_dates(df: pd.DataFrame) -> list[str]:
    unique_dates = sorted(
        [d for d in df["count_date"].dropna().unique() if isinstance(d, date)], reverse=True
    )
    return [d.isoformat() for d in unique_dates]


def minutes_to_label(minutes: int) -> str:
    hours = (minutes // 60) % 24
    mins = minutes % 60
    return f"{hours:02d}:{mins:02d}"


def heading_in_range(heading: float, lo: float, hi: float) -> bool:
    if lo <= hi:
        return lo <= heading <= hi
    return heading >= lo or heading <= hi


def edge_heading(edge) -> float | None:
    shape = edge.getShape()
    if len(shape) < 2:
        return None
    dx = shape[-1][0] - shape[-2][0]
    dy = shape[-1][1] - shape[-2][1]
    return math.degrees(math.atan2(dy, dx)) % 360


def classify_approach(
    heading: float, heading_ranges: dict[str, tuple[float, float]]
) -> str | None:
    for direction, (lo, hi) in heading_ranges.items():
        if heading_in_range(heading, lo, hi):
            return direction
    return None


def verify_route(net, from_edge, to_edge, mode_class: str) -> bool:
    try:
        route = net.getShortestPath(from_edge, to_edge, vClass=mode_class)
    except TypeError:
        route = net.getShortestPath(from_edge, to_edge)
    except Exception:
        return False
    return route[0] is not None


def get_incoming_edges_by_approach(
    node, mode_class: str, heading_ranges: dict[str, tuple[float, float]]
):
    best: dict[str, Any] = {}

    for edge in node.getIncoming():
        if mode_class != "pedestrian" and edge.isSpecial():
            continue
        if not edge.allows(mode_class):
            continue

        heading = edge_heading(edge)
        if heading is None:
            continue

        direction = classify_approach(heading, heading_ranges)
        if direction is None:
            continue

        score = float(edge.getLaneNumber()) * max(float(edge.getSpeed()), 1.0)
        current = best.get(direction)
        if current is None or score > current[1]:
            best[direction] = (edge, score)

    return {direction: entry[0] for direction, entry in best.items()}


def get_outgoing_edges_by_turn(net, node, incoming_edge, mode_class: str):
    in_heading = edge_heading(incoming_edge)
    if in_heading is None:
        return {}

    outgoing_edges = []
    for edge in node.getOutgoing():
        if mode_class != "pedestrian" and edge.isSpecial():
            continue
        if not edge.allows(mode_class):
            continue
        if edge.getID() == incoming_edge.getID():
            continue
        outgoing_edges.append(edge)

    candidates = []
    for out_edge in outgoing_edges:
        out_heading = edge_heading(out_edge)
        if out_heading is None:
            continue
        diff = (out_heading - in_heading) % 360
        candidates.append((out_edge, diff))

    if not candidates:
        return {}

    best_for_turn: dict[str, tuple[Any, float]] = {}

    for out_edge, diff in candidates:
        if 150 <= diff <= 210:
            continue
        if 30 <= diff <= 150:
            turn = "l"
            quality = abs(diff - 90)
        elif 210 <= diff <= 330:
            turn = "r"
            quality = abs(diff - 270)
        else:
            turn = "t"
            quality = min(diff, 360 - diff)

        existing = best_for_turn.get(turn)
        if existing is None or quality < existing[1]:
            best_for_turn[turn] = (out_edge, quality)

    return {turn: edge_quality[0] for turn, edge_quality in best_for_turn.items()}


def apply_date_policy(
    rows: pd.DataFrame,
    date_policy: str,
    selected_date: str | None,
    range_start: str | None,
    range_end: str | None,
) -> pd.DataFrame:
    if rows.empty:
        return rows

    if date_policy == "selected_date":
        selected = parse_iso_date(selected_date)
        if selected is None:
            return rows.iloc[0:0]
        return rows[rows["count_date"] == selected]

    if date_policy == "date_range":
        start_date = parse_iso_date(range_start)
        end_date = parse_iso_date(range_end)
        if start_date is None or end_date is None:
            return rows.iloc[0:0]
        if start_date > end_date:
            start_date, end_date = end_date, start_date
        return rows[(rows["count_date"] >= start_date) & (rows["count_date"] <= end_date)]

    latest = rows["count_date"].dropna()
    if latest.empty:
        return rows.iloc[0:0]
    return rows[rows["count_date"] == latest.max()]


def apply_time_window(
    rows: pd.DataFrame, start_minute: int, duration_minutes: int
) -> pd.DataFrame:
    if rows.empty:
        return rows

    start = start_minute % (24 * 60)
    duration = max(15, min(duration_minutes, 24 * 60))

    valid_rows = rows[rows["time_minutes"] >= 0]
    if valid_rows.empty:
        return valid_rows

    if duration >= 24 * 60:
        return valid_rows

    end = (start + duration) % (24 * 60)
    if start + duration <= 24 * 60:
        return valid_rows[
            (valid_rows["time_minutes"] >= start) & (valid_rows["time_minutes"] < start + duration)
        ]

    return valid_rows[(valid_rows["time_minutes"] >= start) | (valid_rows["time_minutes"] < end)]


def aggregate_turn_counts(rows: pd.DataFrame, mode: str) -> dict[tuple[str, str], float]:
    prefix = MODE_COLUMN_PREFIX[mode]
    counts: dict[tuple[str, str], float] = {}

    for direction in DIRECTIONS:
        for movement in MOVEMENTS:
            column = f"{direction}_appr_{prefix}_{movement}"
            if column not in rows.columns:
                continue
            value = float(rows[column].sum())
            if value > 0:
                counts[(direction, movement)] = value

    return counts


def aggregate_ped_counts(rows: pd.DataFrame) -> dict[str, float]:
    counts: dict[str, float] = {}
    for direction in DIRECTIONS:
        column = f"{direction}_appr_peds"
        if column not in rows.columns:
            continue
        value = float(rows[column].sum())
        if value > 0:
            counts[direction] = value
    return counts


def map_locations_to_network(
    net, node_coords: np.ndarray, node_ids: list[str], node_signalized: list[bool]
):
    if node_coords.size == 0:
        return []

    location_rows = (
        TMC_DF[["location_name", "longitude", "latitude"]]
        .dropna(subset=["location_name", "longitude", "latitude"])
        .drop_duplicates(subset=["location_name"])
        .copy()
    )

    mappings: list[dict[str, Any]] = []

    for row in location_rows.itertuples(index=False):
        location_name = str(row.location_name)
        lon = float(row.longitude)
        lat = float(row.latitude)

        try:
            x, y = net.convertLonLat2XY(lon, lat)
        except ModuleNotFoundError as exc:
            if "pyproj" in str(exc).lower():
                raise RuntimeError(
                    "pyproj is required for mapping TMC latitude/longitude to network coordinates. "
                    "Install pyproj and reconnect the network"
                ) from exc
            raise
        delta = node_coords - np.array([x, y])
        dist_sq = np.einsum("ij,ij->i", delta, delta)
        nearest_idx = int(np.argmin(dist_sq))
        dist_m = float(math.sqrt(float(dist_sq[nearest_idx])))

        mappings.append(
            {
                "location_name": location_name,
                "junction_id": node_ids[nearest_idx],
                "distance_m": round(dist_m, 2),
                "signalized": bool(node_signalized[nearest_idx]),
            }
        )

    mappings.sort(key=lambda item: (item["distance_m"], item["location_name"]))
    return mappings


def load_network_summary(network_path: Path) -> dict[str, Any]:
    key = str(network_path.resolve())
    cached = NETWORK_CACHE.get(key)
    if cached is not None:
        return cached

    net = sumolib.net.readNet(
        str(network_path),
        withInternal=True,
        withPrograms=True,
        withLatestPrograms=True,
        withPedestrianConnections=True,
    )

    tls_ids = {tls.getID() for tls in net.getTrafficLights()}

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    edge_count = 0

    for edge in net.getEdges():
        if edge.isSpecial():
            continue
        shape = edge.getShape()
        if len(shape) < 2:
            continue
        for x, y in shape:
            edge_x.append(float(x))
            edge_y.append(float(y))
        edge_x.append(None)
        edge_y.append(None)
        edge_count += 1

    node_ids: list[str] = []
    node_x: list[float] = []
    node_y: list[float] = []
    node_signalized: list[bool] = []
    node_type: list[str] = []
    incoming_count: list[int] = []
    outgoing_count: list[int] = []

    for node in net.getNodes():
        x, y = node.getCoord()
        nid = node.getID()
        ntype = str(node.getType() or "priority")
        signalized = nid in tls_ids or ntype.startswith("traffic_light")

        node_ids.append(nid)
        node_x.append(float(x))
        node_y.append(float(y))
        node_signalized.append(bool(signalized))
        node_type.append(ntype)
        incoming_count.append(len(node.getIncoming()))
        outgoing_count.append(len(node.getOutgoing()))

    node_coords = (
        np.column_stack((np.array(node_x), np.array(node_y))) if node_x else np.empty((0, 2))
    )
    location_mappings = map_locations_to_network(net, node_coords, node_ids, node_signalized)

    summary = {
        "key": key,
        "net": net,
        "network_path": network_path,
        "network_relpath": to_relpath(network_path),
        "edge_x": edge_x,
        "edge_y": edge_y,
        "edge_count": edge_count,
        "node_ids": node_ids,
        "node_x": node_x,
        "node_y": node_y,
        "node_signalized": node_signalized,
        "node_type": node_type,
        "incoming_count": incoming_count,
        "outgoing_count": outgoing_count,
        "location_mappings": location_mappings,
        "tls_count": int(sum(node_signalized)),
    }

    NETWORK_CACHE[key] = summary
    return summary


def empty_network_figure() -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#f8f5ef",
        plot_bgcolor="#f8f5ef",
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[
            dict(
                text="Connect a network file to view junctions and signal metadata",
                showarrow=False,
                font=dict(size=16, color="#4f5d63"),
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
            )
        ],
    )
    return fig


def build_network_figure(
    network_key: str,
    show_signalized_only: bool,
    marker_size: int,
    selected_locations: list[str],
    max_match_distance: float,
) -> go.Figure:
    summary = NETWORK_CACHE.get(network_key)
    if summary is None:
        return empty_network_figure()

    node_ids = summary["node_ids"]
    node_x = summary["node_x"]
    node_y = summary["node_y"]
    node_signalized = summary["node_signalized"]
    node_type = summary["node_type"]
    incoming_count = summary["incoming_count"]
    outgoing_count = summary["outgoing_count"]

    include_index = [
        idx for idx, sig in enumerate(node_signalized) if (sig if show_signalized_only else True)
    ]

    marker_x = [node_x[idx] for idx in include_index]
    marker_y = [node_y[idx] for idx in include_index]
    marker_color = ["#d26d47" if node_signalized[idx] else "#6a7f87" for idx in include_index]
    marker_custom = [
        [
            node_ids[idx],
            "yes" if node_signalized[idx] else "no",
            incoming_count[idx],
            outgoing_count[idx],
            node_type[idx],
        ]
        for idx in include_index
    ]

    selected_junctions: set[str] = set()
    if selected_locations:
        for mapping in summary["location_mappings"]:
            if (
                mapping["distance_m"] <= max_match_distance
                and mapping["location_name"] in selected_locations
            ):
                selected_junctions.add(mapping["junction_id"])

    selected_x = []
    selected_y = []
    selected_custom = []
    node_index_by_id = {nid: idx for idx, nid in enumerate(node_ids)}
    for junction_id in sorted(selected_junctions):
        if junction_id not in node_index_by_id:
            continue
        idx = node_index_by_id[junction_id]
        selected_x.append(node_x[idx])
        selected_y.append(node_y[idx])
        selected_custom.append(
            [
                node_ids[idx],
                "yes" if node_signalized[idx] else "no",
                incoming_count[idx],
                outgoing_count[idx],
                node_type[idx],
            ]
        )

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=summary["edge_x"],
            y=summary["edge_y"],
            mode="lines",
            line=dict(width=1.1, color="#b7c3c8"),
            hoverinfo="skip",
            name="Network edges",
        )
    )

    fig.add_trace(
        go.Scattergl(
            x=marker_x,
            y=marker_y,
            mode="markers",
            marker=dict(size=max(4, marker_size), color=marker_color, opacity=0.85),
            customdata=marker_custom,
            name="Junctions",
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Signalized: %{customdata[1]}<br>"
                "Incoming edges: %{customdata[2]}<br>"
                "Outgoing edges: %{customdata[3]}<br>"
                "Node type: %{customdata[4]}"
                "<extra></extra>"
            ),
        )
    )

    if selected_x:
        fig.add_trace(
            go.Scattergl(
                x=selected_x,
                y=selected_y,
                mode="markers",
                marker=dict(size=max(marker_size + 5, 10), color="#193441", symbol="diamond-open"),
                customdata=selected_custom,
                name="Selected TMC mappings",
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Signalized: %{customdata[1]}<br>"
                    "Incoming edges: %{customdata[2]}<br>"
                    "Outgoing edges: %{customdata[3]}<br>"
                    "Node type: %{customdata[4]}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#f8f5ef",
        plot_bgcolor="#f8f5ef",
        margin=dict(l=16, r=16, t=16, b=16),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
        uirevision=network_key,
        dragmode="pan",
    )

    return fig


def _expected_state_size(tls_obj) -> int | None:
    programs = tls_obj.getPrograms() or {}
    if programs:
        program = next(iter(programs.values()))
        phases = getattr(program, "getPhases", lambda: [])()
        lengths = [
            len(phase.state)
            for phase in phases
            if hasattr(phase, "state") and phase.state is not None
        ]
        if lengths:
            return max(lengths)

    links = tls_obj.getLinks() or {}
    if links:
        keys = list(links.keys())
        if keys:
            return max(keys) + 1

    connections = tls_obj.getConnections() or []
    if connections:
        indices = [link_index for (_, _, link_index) in connections]
        return max(indices) + 1 if indices else None

    return None


def _classify_signal_indices(tls_obj) -> tuple[set[int], set[int], set[int]]:
    east_west: set[int] = set()
    north_south: set[int] = set()
    excluded: set[int] = set()
    links = tls_obj.getLinks() or {}

    for signal_idx, conn_list in links.items():
        if not conn_list:
            excluded.add(signal_idx)
            continue

        in_lane = conn_list[0][0]
        edge = in_lane.getEdge()
        function = edge.getFunction()

        if function in ("crossing", "walkingarea"):
            excluded.add(signal_idx)
            continue

        heading = edge_heading(edge)
        if heading is None:
            east_west.add(signal_idx)
            continue

        if heading < 45 or heading > 315 or 135 < heading < 225:
            east_west.add(signal_idx)
        else:
            north_south.add(signal_idx)

    return east_west, north_south, excluded


def _state_string(n_links: int, active_idx: set[int], active_char: str) -> str:
    state = ["r"] * n_links
    for idx in active_idx:
        if 0 <= idx < n_links:
            state[idx] = active_char
    return "".join(state)


def generate_fixed_time_tls_file(
    net,
    output_path: Path,
    ew_green: int,
    ns_green: int,
    yellow: int,
    all_red: int,
    offset: int,
    include_cluster_signals: bool,
) -> dict[str, Any]:
    root = etree.Element("additional")
    created = 0
    skipped = 0

    for tls_obj in net.getTrafficLights():
        tls_id = tls_obj.getID()

        if not include_cluster_signals and tls_id.startswith("cluster_"):
            skipped += 1
            continue

        n_links = _expected_state_size(tls_obj)
        if not n_links or n_links <= 0:
            skipped += 1
            continue

        ew, ns, excluded = _classify_signal_indices(tls_obj)

        all_indices = set(range(n_links))
        candidate = sorted(all_indices - excluded)

        if not ew or not ns:
            if len(candidate) < 2:
                skipped += 1
                continue
            half = max(1, len(candidate) // 2)
            ew = set(candidate[:half])
            ns = set(candidate[half:]) if candidate[half:] else set(candidate[:half])

        all_red_state = "r" * n_links
        ped_green = list(all_red_state)
        for idx in excluded:
            if 0 <= idx < n_links:
                ped_green[idx] = "G"
        ped_green_state = "".join(ped_green)

        phases = [
            (ew_green, _state_string(n_links, ew, "g")),
            (yellow, _state_string(n_links, ew, "y")),
            (all_red, ped_green_state),
            (ns_green, _state_string(n_links, ns, "g")),
            (yellow, _state_string(n_links, ns, "y")),
            (all_red, ped_green_state),
        ]

        tl_elem = etree.SubElement(
            root,
            "tlLogic",
            id=tls_id,
            type="static",
            programID="ui_fixed_time",
            offset=str(offset),
        )

        for duration, state in phases:
            etree.SubElement(tl_elem, "phase", duration=str(duration), state=state)

        created += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    xml_bytes = etree.tostring(root, pretty_print=True, xml_declaration=True, encoding="UTF-8")
    with gzip.open(output_path, "wb") as handle:
        handle.write(xml_bytes)

    return {
        "created_tls": created,
        "skipped_tls": skipped,
        "path": output_path,
    }


def write_sumocfg(
    output_path: Path,
    net_path: Path,
    route_path: Path,
    additional_path: Path | None,
    begin: int,
    end: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    root = etree.Element("configuration")
    input_elem = etree.SubElement(root, "input")

    etree.SubElement(
        input_elem,
        "net-file",
        value=os.path.relpath(net_path, output_path.parent).replace("\\", "/"),
    )
    etree.SubElement(
        input_elem,
        "route-files",
        value=os.path.relpath(route_path, output_path.parent).replace("\\", "/"),
    )

    if additional_path is not None and additional_path.exists():
        etree.SubElement(
            input_elem,
            "additional-files",
            value=os.path.relpath(additional_path, output_path.parent).replace("\\", "/"),
        )

    time_elem = etree.SubElement(root, "time")
    etree.SubElement(time_elem, "begin", value=str(begin))
    etree.SubElement(time_elem, "end", value=str(end))
    etree.SubElement(time_elem, "step-length", value="1.0")

    processing_elem = etree.SubElement(root, "processing")
    etree.SubElement(processing_elem, "time-to-teleport", value="300")
    etree.SubElement(processing_elem, "ignore-route-errors", value="true")

    report_elem = etree.SubElement(root, "report")
    etree.SubElement(report_elem, "no-step-log", value="true")

    tree = etree.ElementTree(root)
    tree.write(str(output_path), pretty_print=True, xml_declaration=True, encoding="UTF-8")


def run_sumo_validation(
    net_path: Path,
    route_path: Path,
    additional_path: Path | None,
    begin: int,
    end: int,
) -> dict[str, Any]:
    sumo_binary = shutil.which("sumo")
    if sumo_binary is None:
        return {
            "ran": False,
            "ok": False,
            "message": "SUMO executable was not found on PATH. Demand file was generated but no simulation validation was run.",
        }

    check_end = max(begin + 60, min(end, begin + 240))

    command = [
        sumo_binary,
        "-n",
        str(net_path),
        "-r",
        str(route_path),
        "--begin",
        str(begin),
        "--end",
        str(check_end),
        "--step-length",
        "1.0",
        "--no-step-log",
        "true",
        "--ignore-route-errors",
        "true",
    ]

    if additional_path is not None and additional_path.exists():
        command.extend(["--additional-files", str(additional_path)])

    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=180, check=False)
    except subprocess.TimeoutExpired:
        return {
            "ran": True,
            "ok": False,
            "message": "SUMO validation timed out after 180 seconds",
        }

    stderr_tail = "\n".join(result.stderr.splitlines()[-6:]) if result.stderr else ""

    if result.returncode == 0:
        msg = "SUMO validation run completed successfully"
        if stderr_tail:
            msg = f"{msg}\n{stderr_tail}"
        return {
            "ran": True,
            "ok": True,
            "message": msg,
        }

    msg = f"SUMO validation failed with return code {result.returncode}"
    if stderr_tail:
        msg = f"{msg}\n{stderr_tail}"

    return {
        "ran": True,
        "ok": False,
        "message": msg,
    }


def add_vehicle_type(
    root, type_id: str, settings: dict[str, float], extra_attributes: dict[str, str] | None = None
) -> None:
    attrs = {
        "id": type_id,
        "accel": f"{settings['accel']:.6g}",
        "decel": f"{settings['decel']:.6g}",
        "sigma": f"{settings['sigma']:.6g}",
        "length": f"{settings['length']:.6g}",
        "maxSpeed": f"{settings['maxSpeed']:.6g}",
    }
    if extra_attributes:
        attrs.update(extra_attributes)
    etree.SubElement(root, "vType", **attrs)


def generate_scenario(
    network_store: dict[str, Any],
    selected_locations: list[str],
    max_match_distance: float,
    date_policy: str,
    selected_date: str | None,
    date_range_start: str | None,
    date_range_end: str | None,
    time_window_start: int,
    time_window_duration: int,
    simulation_begin: int,
    simulation_end: int,
    strict_route_check: bool,
    min_count_threshold: int,
    global_demand_scale: float,
    included_modes: set[str],
    mode_scales: dict[str, float],
    streetcar_share_from_bus: float,
    controller_mode: str,
    include_cluster_signals: bool,
    fixed_signal: dict[str, int],
    max_pressure_signal: dict[str, float],
    mappo_hyperparams: dict[str, Any],
    heading_ranges: dict[str, tuple[float, float]],
    vtype_settings: dict[str, dict[str, float]],
    run_validation: bool,
) -> GenerationResult:
    if not network_store:
        return GenerationResult(False, "No network is connected")

    network_key = network_store.get("network_key")
    summary = NETWORK_CACHE.get(str(network_key))
    if summary is None:
        return GenerationResult(
            False,
            "Connected network was not found in memory cache. Reconnect the network and try again",
        )

    if simulation_end <= simulation_begin:
        return GenerationResult(False, "Simulation end must be greater than simulation begin")

    if not included_modes:
        return GenerationResult(False, "Select at least one demand mode")

    for out_dir in [DEMAND_OUTPUT_DIR, SCENARIO_OUTPUT_DIR, SUMOCFG_OUTPUT_DIR, TLS_OUTPUT_DIR]:
        out_dir.mkdir(parents=True, exist_ok=True)

    net = summary["net"]
    network_path: Path = summary["network_path"]

    selected_set = set(selected_locations or [])
    mapping_candidates = [
        item for item in summary["location_mappings"] if item["distance_m"] <= max_match_distance
    ]

    if selected_set:
        active_mappings = [
            item for item in mapping_candidates if item["location_name"] in selected_set
        ]
    else:
        active_mappings = mapping_candidates

    if not active_mappings:
        return GenerationResult(
            False,
            "No TMC locations are available under the current match distance and selection filters",
        )

    root = etree.Element("routes")

    if "cars" in included_modes:
        add_vehicle_type(root, "car", vtype_settings["car"], {"vClass": "passenger"})
    if "trucks" in included_modes:
        add_vehicle_type(root, "truck", vtype_settings["truck"], {"vClass": "truck"})
    if "buses" in included_modes:
        add_vehicle_type(root, "bus", vtype_settings["bus"], {"vClass": "bus"})
    if "streetcars" in included_modes:
        add_vehicle_type(
            root, "streetcar", vtype_settings["streetcar"], {"vClass": "tram", "guiShape": "rail"}
        )

    flow_id = 0
    person_flow_id = 0

    stats = {
        "locations_considered": len(active_mappings),
        "locations_with_output": 0,
        "vehicle_flows_created": 0,
        "person_flows_created": 0,
        "flows_by_mode": {"cars": 0, "trucks": 0, "buses": 0, "streetcars": 0, "pedestrians": 0},
        "vehicles_by_mode": {
            "cars": 0,
            "trucks": 0,
            "buses": 0,
            "streetcars": 0,
            "pedestrians": 0,
        },
        "skipped_no_data": 0,
        "skipped_no_node": 0,
        "skipped_unmapped_approach": 0,
        "skipped_missing_turn": 0,
        "skipped_no_route": 0,
        "skipped_below_threshold": 0,
    }

    warnings: list[str] = []

    streetcar_share = clamp_float(streetcar_share_from_bus, 0.35, 0.0, 1.0)

    for mapping in active_mappings:
        location_name = mapping["location_name"]
        node_id = mapping["junction_id"]

        rows = TMC_DF[TMC_DF["location_name"] == location_name]
        rows = apply_date_policy(
            rows, date_policy, selected_date, date_range_start, date_range_end
        )
        rows = apply_time_window(rows, time_window_start, time_window_duration)

        if rows.empty:
            stats["skipped_no_data"] += 1
            continue

        node = net.getNode(node_id)
        if node is None:
            stats["skipped_no_node"] += 1
            warnings.append(f"Node {node_id} not found for location {location_name}")
            continue

        had_output_for_location = False

        counts_by_mode = {
            "cars": aggregate_turn_counts(rows, "cars"),
            "trucks": aggregate_turn_counts(rows, "trucks"),
            "buses": aggregate_turn_counts(rows, "buses"),
        }

        # Streetcar demand is derived from bus turn counts so demand remains tied
        # directly to tmc_parsed.csv without random generation.
        if "streetcars" in included_modes:
            counts_by_mode["streetcars"] = {
                key: value * streetcar_share for key, value in counts_by_mode["buses"].items()
            }
            if "buses" in included_modes:
                counts_by_mode["buses"] = {
                    key: value * (1.0 - streetcar_share)
                    for key, value in counts_by_mode["buses"].items()
                }
        else:
            counts_by_mode["streetcars"] = {}

        approach_cache: dict[str, dict[str, Any]] = {}

        for mode in ["cars", "trucks", "buses", "streetcars"]:
            if mode not in included_modes:
                continue

            _, mode_class = MODE_TO_VTYPE_AND_CLASS[mode]
            approach_map = approach_cache.get(mode_class)
            if approach_map is None:
                approach_map = get_incoming_edges_by_approach(node, mode_class, heading_ranges)
                approach_cache[mode_class] = approach_map

            for (direction, movement), raw_count in counts_by_mode[mode].items():
                scaled = raw_count * global_demand_scale * mode_scales.get(mode, 1.0)
                volume = int(round(scaled))

                if volume < min_count_threshold:
                    stats["skipped_below_threshold"] += 1
                    continue

                in_edge = approach_map.get(direction)
                if in_edge is None:
                    stats["skipped_unmapped_approach"] += 1
                    continue

                turn_map = get_outgoing_edges_by_turn(net, node, in_edge, mode_class)
                out_edge = turn_map.get(movement)
                if out_edge is None:
                    stats["skipped_missing_turn"] += 1
                    continue

                if strict_route_check and not verify_route(net, in_edge, out_edge, mode_class):
                    stats["skipped_no_route"] += 1
                    continue

                etree.SubElement(
                    root,
                    "flow",
                    id=f"{mode}_{flow_id}",
                    type=MODE_TO_VTYPE_AND_CLASS[mode][0],
                    begin=str(simulation_begin),
                    end=str(simulation_end),
                    number=str(volume),
                    **{"from": in_edge.getID()},
                    to=out_edge.getID(),
                    departLane="best",
                    departSpeed="max",
                )

                flow_id += 1
                stats["vehicle_flows_created"] += 1
                stats["flows_by_mode"][mode] += 1
                stats["vehicles_by_mode"][mode] += volume
                had_output_for_location = True

        if "pedestrians" in included_modes:
            ped_counts = aggregate_ped_counts(rows)
            ped_scale = mode_scales.get("pedestrians", 1.0)
            ped_approaches = get_incoming_edges_by_approach(node, "pedestrian", heading_ranges)

            for direction, raw_count in ped_counts.items():
                scaled = raw_count * global_demand_scale * ped_scale
                volume = int(round(scaled))

                if volume < min_count_threshold:
                    stats["skipped_below_threshold"] += 1
                    continue

                in_edge = ped_approaches.get(direction)
                if in_edge is None:
                    stats["skipped_unmapped_approach"] += 1
                    continue

                turn_map = get_outgoing_edges_by_turn(net, node, in_edge, "pedestrian")
                out_edge = turn_map.get("t") or turn_map.get("r") or turn_map.get("l")
                if out_edge is None:
                    stats["skipped_missing_turn"] += 1
                    continue

                if strict_route_check and not verify_route(net, in_edge, out_edge, "pedestrian"):
                    stats["skipped_no_route"] += 1
                    continue

                person_flow = etree.SubElement(
                    root,
                    "personFlow",
                    id=f"ped_{person_flow_id}",
                    begin=str(simulation_begin),
                    end=str(simulation_end),
                    number=str(volume),
                )
                etree.SubElement(
                    person_flow,
                    "walk",
                    **{"from": in_edge.getID(), "to": out_edge.getID()},
                )

                person_flow_id += 1
                stats["person_flows_created"] += 1
                stats["flows_by_mode"]["pedestrians"] += 1
                stats["vehicles_by_mode"]["pedestrians"] += volume
                had_output_for_location = True

        if had_output_for_location:
            stats["locations_with_output"] += 1

    if stats["vehicle_flows_created"] == 0 and stats["person_flows_created"] == 0:
        return GenerationResult(
            False,
            "Generation finished with zero flows. Try increasing match distance or reducing strict filters",
            stats=stats,
            warnings=warnings,
        )

    network_tag = sanitize_tag(
        summary["network_path"].name.replace(".xml.gz", "").replace(".xml", "")
    )
    scenario_tag = f"{network_tag}_{sanitize_tag(controller_mode)}"

    demand_path = DEMAND_OUTPUT_DIR / f"{scenario_tag}.rou.xml"
    scenario_json_path = SCENARIO_OUTPUT_DIR / f"{scenario_tag}.json"
    sumocfg_path = SUMOCFG_OUTPUT_DIR / f"{scenario_tag}.sumocfg"

    tree = etree.ElementTree(root)
    tree.write(str(demand_path), pretty_print=True, xml_declaration=True, encoding="UTF-8")

    tls_path: Path | None = None
    tls_summary: dict[str, Any] | None = None

    if controller_mode == "baseline_fixed_time":
        tls_path = TLS_OUTPUT_DIR / f"{scenario_tag}_fixed_tls.add.xml.gz"
        tls_summary = generate_fixed_time_tls_file(
            net=net,
            output_path=tls_path,
            ew_green=fixed_signal["ew_green"],
            ns_green=fixed_signal["ns_green"],
            yellow=fixed_signal["yellow"],
            all_red=fixed_signal["all_red"],
            offset=fixed_signal["offset"],
            include_cluster_signals=include_cluster_signals,
        )
    else:
        default_tls = REPO_ROOT / "sumo" / "network" / "tls_overrides.add.xml.gz"
        if default_tls.exists():
            tls_path = default_tls

    write_sumocfg(
        output_path=sumocfg_path,
        net_path=network_path,
        route_path=demand_path,
        additional_path=tls_path,
        begin=simulation_begin,
        end=simulation_end,
    )

    scenario_payload = {
        "generated_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "network_file": to_relpath(network_path),
        "demand_file": to_relpath(demand_path),
        "sumocfg_file": to_relpath(sumocfg_path),
        "signal_file": to_relpath(tls_path) if tls_path is not None else None,
        "controller_mode": controller_mode,
        "signal_settings": {
            "fixed_time": fixed_signal,
            "max_pressure": max_pressure_signal,
            "include_cluster_signals": include_cluster_signals,
        },
        "mappo_hyperparameters": mappo_hyperparams,
        "demand_settings": {
            "tmc_source": to_relpath(TMC_PATH),
            "date_policy": date_policy,
            "selected_date": selected_date,
            "date_range_start": date_range_start,
            "date_range_end": date_range_end,
            "time_window_start": minutes_to_label(time_window_start),
            "time_window_duration_min": time_window_duration,
            "simulation_begin": simulation_begin,
            "simulation_end": simulation_end,
            "strict_route_check": strict_route_check,
            "min_count_threshold": min_count_threshold,
            "global_demand_scale": global_demand_scale,
            "mode_scales": mode_scales,
            "streetcar_share_from_bus": streetcar_share,
            "heading_ranges": heading_ranges,
            "included_modes": sorted(list(included_modes)),
            "vtype_settings": vtype_settings,
            "matched_locations": [
                {
                    "location_name": item["location_name"],
                    "junction_id": item["junction_id"],
                    "distance_m": item["distance_m"],
                }
                for item in active_mappings
            ],
        },
        "stats": stats,
        "warnings": warnings,
        "tls_generation": (
            {
                **tls_summary,
                "path": to_relpath(tls_summary["path"]),
            }
            if tls_summary is not None
            else None
        ),
    }

    scenario_json_path.write_text(json.dumps(scenario_payload, indent=2), encoding="utf-8")

    validation = None
    if run_validation:
        validation = run_sumo_validation(
            net_path=network_path,
            route_path=demand_path,
            additional_path=tls_path,
            begin=simulation_begin,
            end=simulation_end,
        )

    return GenerationResult(
        success=True,
        message="Demand generation completed",
        demand_path=demand_path,
        scenario_json_path=scenario_json_path,
        sumocfg_path=sumocfg_path,
        tls_path=tls_path,
        stats=stats,
        warnings=warnings,
        validation=validation,
        scenario_payload=scenario_payload,
    )


def field_number(
    label: str,
    field_id: str,
    value: float | int,
    step: float | int,
    min_value: float | int | None = None,
    max_value: float | int | None = None,
) -> html.Div:
    return html.Div(
        className="field-block",
        children=[
            html.Label(label, className="field-label", htmlFor=field_id),
            dcc.Input(
                id=field_id,
                className="field-input",
                type="number",
                value=value,
                step=step,
                min=min_value,
                max=max_value,
            ),
        ],
    )


def mode_vtype_block(mode_title: str, key: str) -> html.Div:
    defaults = DEFAULT_VTYPE[key]
    return html.Div(
        className="sub-panel",
        children=[
            html.Div(mode_title, className="sub-panel-title"),
            html.Div(
                className="field-grid five",
                children=[
                    field_number("Accel", f"{key}-accel", defaults["accel"], 0.1, 0.1, 10),
                    field_number("Decel", f"{key}-decel", defaults["decel"], 0.1, 0.1, 10),
                    field_number("Sigma", f"{key}-sigma", defaults["sigma"], 0.05, 0.0, 1.0),
                    field_number("Length", f"{key}-length", defaults["length"], 0.1, 0.5, 100),
                    field_number("Max Speed", f"{key}-maxspeed", defaults["maxSpeed"], 0.1, 1, 60),
                ],
            ),
        ],
    )


def build_network_meta(store: dict[str, Any] | None) -> html.Div:
    if not store:
        return html.Div("No network connected", className="meta-line")

    return html.Div(
        className="meta-wrap",
        children=[
            html.Div(f"Connected network: {store['network_relpath']}", className="meta-line mono"),
            html.Div(f"Edges shown: {store['edge_count']}", className="meta-line"),
            html.Div(f"Junctions: {store['node_count']}", className="meta-line"),
            html.Div(f"Signalized junctions: {store['tls_count']}", className="meta-line"),
            html.Div(
                f"TMC locations with nearest-junction mapping: {store['mapped_location_count']}",
                className="meta-line",
            ),
        ],
    )


def build_generation_status(result: GenerationResult) -> html.Div:
    if not result.success:
        return html.Div(
            className="status-card error",
            children=[
                html.Div("Generation failed", className="status-title"),
                html.Div(result.message, className="status-line"),
            ],
        )

    stats = result.stats or {}
    warnings = result.warnings or []

    lines = [
        html.Div("Demand generation complete", className="status-title"),
        html.Div(
            f"Vehicle flows: {stats.get('vehicle_flows_created', 0)} | Person flows: {stats.get('person_flows_created', 0)}",
            className="status-line",
        ),
        html.Div(
            f"Locations with generated output: {stats.get('locations_with_output', 0)} / {stats.get('locations_considered', 0)}",
            className="status-line",
        ),
    ]

    mode_totals = stats.get("vehicles_by_mode", {})
    mode_line = (
        f"Assigned counts by mode -> cars {mode_totals.get('cars', 0)}, trucks {mode_totals.get('trucks', 0)}, "
        f"buses {mode_totals.get('buses', 0)}, streetcars {mode_totals.get('streetcars', 0)}, "
        f"pedestrians {mode_totals.get('pedestrians', 0)}"
    )
    lines.append(html.Div(mode_line, className="status-line"))

    if result.validation:
        state = "passed" if result.validation.get("ok") else "reported issues"
        lines.append(html.Div(f"SUMO validation {state}", className="status-line"))
        lines.append(html.Div(result.validation.get("message", ""), className="status-line mono"))

    if warnings:
        lines.append(html.Div("Warnings", className="status-subtitle"))
        lines.append(
            html.Ul(
                className="status-list",
                children=[html.Li(msg) for msg in warnings[:10]],
            )
        )

    return html.Div(className="status-card success", children=lines)


def build_output_paths(result: GenerationResult) -> html.Div:
    if not result.success:
        return html.Div(
            className="status-card error",
            children=[
                html.Div("No files created", className="status-title"),
                html.Div("Fix the generation settings and try again", className="status-line"),
            ],
        )

    path_lines = [
        html.Div("Files were written to fixed repo paths", className="status-title"),
        html.Div(
            f"Demand route file: {to_relpath(result.demand_path)}", className="status-line mono"
        ),
        html.Div(
            f"Scenario settings: {to_relpath(result.scenario_json_path)}",
            className="status-line mono",
        ),
        html.Div(f"SUMO config: {to_relpath(result.sumocfg_path)}", className="status-line mono"),
    ]

    if result.tls_path is not None:
        path_lines.append(
            html.Div(f"Signal file: {to_relpath(result.tls_path)}", className="status-line mono")
        )

    path_lines.append(
        html.Div(
            "Use the download buttons to export demand.xml and the settings confirmation .txt",
            className="status-line",
        )
    )

    return html.Div(className="status-card", children=path_lines)


DATE_POLICY_LABELS = {
    "latest_per_location": "Latest per location",
    "selected_date": "One exact date",
    "date_range": "Date range",
}

CONTROLLER_MODE_LABELS = {
    "mappo_custom": "MAPPO custom",
    "baseline_fixed_time": "Baseline fixed time",
    "baseline_max_pressure": "Baseline max pressure",
}


def format_setting_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "none"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)):
        if not value:
            return "(none)"
        return ", ".join(format_setting_value(item) for item in value)
    return str(value)


def build_confirmation_sections(
    payload: dict[str, Any],
) -> tuple[list[tuple[str, list[tuple[str, str]]]], list[dict[str, Any]]]:
    demand_settings = payload.get("demand_settings", {}) if isinstance(payload, dict) else {}
    signal_settings = payload.get("signal_settings", {}) if isinstance(payload, dict) else {}
    mappo_hyperparams = (
        payload.get("mappo_hyperparameters", {}) if isinstance(payload, dict) else {}
    )
    stats = payload.get("stats", {}) if isinstance(payload, dict) else {}

    if not isinstance(demand_settings, dict):
        demand_settings = {}
    if not isinstance(signal_settings, dict):
        signal_settings = {}
    if not isinstance(mappo_hyperparams, dict):
        mappo_hyperparams = {}
    if not isinstance(stats, dict):
        stats = {}

    fixed_time = signal_settings.get("fixed_time", {})
    max_pressure = signal_settings.get("max_pressure", {})
    mode_scales = demand_settings.get("mode_scales", {})
    heading_ranges = demand_settings.get("heading_ranges", {})
    vtype_settings = demand_settings.get("vtype_settings", {})
    matched_locations_raw = demand_settings.get("matched_locations", [])

    if not isinstance(fixed_time, dict):
        fixed_time = {}
    if not isinstance(max_pressure, dict):
        max_pressure = {}
    if not isinstance(mode_scales, dict):
        mode_scales = {}
    if not isinstance(heading_ranges, dict):
        heading_ranges = {}
    if not isinstance(vtype_settings, dict):
        vtype_settings = {}

    matched_locations = [
        item
        for item in (matched_locations_raw if isinstance(matched_locations_raw, list) else [])
        if isinstance(item, dict)
    ]

    files_and_source_rows: list[tuple[str, str]] = [
        ("generated_utc", format_setting_value(payload.get("generated_utc"))),
        ("network_file", format_setting_value(payload.get("network_file"))),
        ("demand_file", format_setting_value(payload.get("demand_file"))),
        ("sumocfg_file", format_setting_value(payload.get("sumocfg_file"))),
        ("signal_file", format_setting_value(payload.get("signal_file"))),
        ("tmc_source", format_setting_value(demand_settings.get("tmc_source"))),
    ]

    demand_rows: list[tuple[str, str]] = [
        (
            "date_policy",
            DATE_POLICY_LABELS.get(
                str(demand_settings.get("date_policy")),
                format_setting_value(demand_settings.get("date_policy")),
            ),
        ),
        ("selected_date", format_setting_value(demand_settings.get("selected_date"))),
        ("date_range_start", format_setting_value(demand_settings.get("date_range_start"))),
        ("date_range_end", format_setting_value(demand_settings.get("date_range_end"))),
        ("time_window_start", format_setting_value(demand_settings.get("time_window_start"))),
        (
            "time_window_duration_min",
            format_setting_value(demand_settings.get("time_window_duration_min")),
        ),
        ("simulation_begin", format_setting_value(demand_settings.get("simulation_begin"))),
        ("simulation_end", format_setting_value(demand_settings.get("simulation_end"))),
        ("strict_route_check", format_setting_value(demand_settings.get("strict_route_check"))),
        ("min_count_threshold", format_setting_value(demand_settings.get("min_count_threshold"))),
        ("global_demand_scale", format_setting_value(demand_settings.get("global_demand_scale"))),
        (
            "streetcar_share_from_bus",
            format_setting_value(demand_settings.get("streetcar_share_from_bus")),
        ),
        ("included_modes", format_setting_value(demand_settings.get("included_modes"))),
        ("matched_location_count", format_setting_value(len(matched_locations))),
    ]

    for mode_name, scale in sorted(mode_scales.items()):
        demand_rows.append((f"mode_scale.{mode_name}", format_setting_value(scale)))

    signal_rows: list[tuple[str, str]] = [
        (
            "controller_mode",
            CONTROLLER_MODE_LABELS.get(
                str(payload.get("controller_mode")),
                format_setting_value(payload.get("controller_mode")),
            ),
        ),
        (
            "include_cluster_signals",
            format_setting_value(signal_settings.get("include_cluster_signals")),
        ),
    ]

    for key, value in sorted(fixed_time.items()):
        signal_rows.append((f"fixed_time.{key}", format_setting_value(value)))
    for key, value in sorted(max_pressure.items()):
        signal_rows.append((f"max_pressure.{key}", format_setting_value(value)))

    mappo_rows: list[tuple[str, str]] = []
    for key, value in sorted(mappo_hyperparams.items()):
        mappo_rows.append((key, format_setting_value(value)))

    heading_rows: list[tuple[str, str]] = []
    for approach, bounds in sorted(heading_ranges.items()):
        if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
            heading_rows.append(
                (
                    approach,
                    f"{format_setting_value(bounds[0])} -> {format_setting_value(bounds[1])}",
                )
            )
        else:
            heading_rows.append((approach, format_setting_value(bounds)))

    vtype_rows: list[tuple[str, str]] = []
    for vehicle_type, settings in sorted(vtype_settings.items()):
        if isinstance(settings, dict):
            for key, value in sorted(settings.items()):
                vtype_rows.append((f"{vehicle_type}.{key}", format_setting_value(value)))

    stats_rows: list[tuple[str, str]] = []
    for key in [
        "locations_considered",
        "locations_with_output",
        "vehicle_flows_created",
        "person_flows_created",
        "skipped_no_data",
        "skipped_no_node",
        "skipped_unmapped_approach",
        "skipped_missing_turn",
        "skipped_no_route",
        "skipped_below_threshold",
    ]:
        if key in stats:
            stats_rows.append((key, format_setting_value(stats.get(key))))

    return (
        [
            ("Files and Source", files_and_source_rows),
            ("Demand Settings", demand_rows),
            ("Signal Settings", signal_rows),
            ("MAPPO Hyperparameters", mappo_rows),
            ("Approach Heading Ranges", heading_rows),
            ("Vehicle Type Parameters", vtype_rows),
            ("Generation Stats", stats_rows),
        ],
        matched_locations,
    )


def build_confirmation_text(payload: dict[str, Any]) -> str:
    sections, matched_locations = build_confirmation_sections(payload)

    lines = [
        "TANGO Demand Generation Confirmation",
        "This report was created from the exact settings payload used to generate the downloaded demand.xml.",
    ]

    for title, rows in sections:
        lines.append("")
        lines.append(f"[{title}]")
        for key, value in rows:
            lines.append(f"{key}: {value}")

    lines.append("")
    lines.append("[Matched Locations]")
    if matched_locations:
        for idx, item in enumerate(matched_locations, start=1):
            location_name = format_setting_value(item.get("location_name"))
            junction_id = format_setting_value(item.get("junction_id"))
            distance = format_setting_value(item.get("distance_m"))
            lines.append(f"{idx}. {location_name} -> {junction_id} ({distance} m)")
    else:
        lines.append("none")

    return "\n".join(lines)


def build_generation_artifacts(result: GenerationResult) -> dict[str, Any] | None:
    if not result.success or result.demand_path is None or result.scenario_json_path is None:
        return None

    payload: dict[str, Any] | None = (
        result.scenario_payload if isinstance(result.scenario_payload, dict) else None
    )
    if payload is None and result.scenario_json_path.exists():
        try:
            payload = json.loads(result.scenario_json_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}

    if payload is None:
        payload = {}

    confirmation_text = build_confirmation_text(payload)
    confirmation_filename = f"{sanitize_tag(result.demand_path.stem)}_settings_confirmation.txt"

    return {
        "demand_abs": str(result.demand_path.resolve()),
        "demand_rel": to_relpath(result.demand_path),
        "scenario_abs": str(result.scenario_json_path.resolve()),
        "scenario_rel": to_relpath(result.scenario_json_path),
        "sumocfg_rel": (
            to_relpath(result.sumocfg_path) if result.sumocfg_path is not None else None
        ),
        "scenario_payload": payload,
        "confirmation_text": confirmation_text,
        "confirmation_filename": confirmation_filename,
    }


def build_confirmation_table(rows: list[tuple[str, str]]) -> html.Table:
    return html.Table(
        className="confirm-table",
        children=[
            html.Tbody(
                children=[
                    html.Tr(
                        children=[
                            html.Th(key),
                            html.Td(value),
                        ]
                    )
                    for key, value in rows
                ]
            )
        ],
    )


def build_confirmation_screen(artifacts: dict[str, Any] | None, downloaded: bool) -> html.Div:
    if not artifacts:
        return html.Div(
            className="status-card",
            children=[
                html.Div("No confirmation yet", className="status-title"),
                html.Div(
                    "Generate demand first. After downloading demand.xml, the full settings confirmation will appear here.",
                    className="status-line",
                ),
            ],
        )

    payload = artifacts.get("scenario_payload", {})
    if not isinstance(payload, dict):
        payload = {}
    demand_settings = payload.get("demand_settings", {})
    if not isinstance(demand_settings, dict):
        demand_settings = {}

    if not downloaded:
        return html.Div(
            className="confirm-card",
            children=[
                html.Div("Settings Confirmation", className="confirm-title"),
                html.Div(
                    "Demand file generation is complete. Download demand.xml to unlock the full settings confirmation screen and .txt export.",
                    className="confirm-subtitle",
                ),
                build_confirmation_table(
                    [
                        ("demand_file", format_setting_value(artifacts.get("demand_rel"))),
                        (
                            "scenario_settings_file",
                            format_setting_value(artifacts.get("scenario_rel")),
                        ),
                        (
                            "tmc_source",
                            format_setting_value(demand_settings.get("tmc_source")),
                        ),
                    ]
                ),
            ],
        )

    sections, matched_locations = build_confirmation_sections(payload)

    section_blocks = [
        html.Div(
            className="confirm-section",
            children=[
                html.Div(title, className="confirm-section-title"),
                build_confirmation_table(rows),
            ],
        )
        for title, rows in sections
    ]

    if matched_locations:
        locations_block = html.Div(
            className="confirm-section",
            children=[
                html.Div("Matched Locations Used", className="confirm-section-title"),
                html.Ul(
                    className="confirm-location-list",
                    children=[
                        html.Li(
                            f"{format_setting_value(item.get('location_name'))} -> "
                            f"{format_setting_value(item.get('junction_id'))} "
                            f"({format_setting_value(item.get('distance_m'))} m)",
                        )
                        for item in matched_locations
                    ],
                ),
            ],
        )
    else:
        locations_block = html.Div(
            className="confirm-section",
            children=[
                html.Div("Matched Locations Used", className="confirm-section-title"),
                html.Div("none", className="status-line"),
            ],
        )

    return html.Div(
        className="confirm-card",
        children=[
            html.Div("Settings Confirmation", className="confirm-title"),
            html.Div(
                "This screen is generated from the exact payload used to produce the downloaded demand.xml file.",
                className="confirm-subtitle",
            ),
            *section_blocks,
            locations_block,
        ],
    )


TMC_DF = load_tmc_data(TMC_PATH)
DATE_OPTIONS = available_dates(TMC_DF)
NETWORK_FILES = discover_network_files()
DEFAULT_NETWORK = select_default_network(NETWORK_FILES)

TIME_WINDOW_START_OPTIONS = [
    {"label": minutes_to_label(minutes), "value": minutes} for minutes in range(0, 24 * 60, 15)
]

TIME_WINDOW_DURATION_OPTIONS = [
    {"label": f"{minutes} min", "value": minutes} for minutes in [15, 30, 45, 60, 90, 120, 180]
]


app = Dash(
    __name__,
    title="TANGO Demand Studio",
    assets_folder=str(Path(__file__).resolve().parent / "assets"),
    external_stylesheets=[
        "https://fonts.googleapis.com/css2?family=Roboto+Flex:opsz,wght@8..144,300..800&family=IBM+Plex+Mono:wght@400;500;600&display=swap"
    ],
)


app.layout = html.Div(
    className="tango-page",
    children=[
        dcc.Store(id="network-store"),
        dcc.Store(id="generated-artifacts-store"),
        dcc.Store(id="demand-download-state", data={"downloaded": False}),
        dcc.Download(id="demand-file-download"),
        dcc.Download(id="settings-txt-download"),
        html.Div(
            className="top-strip",
            children=[
                html.Div(
                    children=[
                        html.Div("TANGO Demand Studio", className="top-title"),
                        html.Div(
                            "Simple local interface for generating SUMO demand and scenario files from real TMC data",
                            className="top-subtitle",
                        ),
                    ]
                ),
                html.Div("Source data: data/processed/tmc_parsed.csv", className="top-note mono"),
            ],
        ),
        html.Div(
            className="main-shell",
            children=[
                html.Div(
                    className="viewer-pane",
                    children=[
                        html.Div(
                            className="pane-head",
                            children=[
                                html.Div("Network Viewer", className="pane-title"),
                                html.Div(
                                    "Hover junctions for ID, signalized state, and edge counts",
                                    className="pane-subtitle",
                                ),
                            ],
                        ),
                        html.Div(
                            className="viewer-toolbar",
                            children=[
                                html.Div(
                                    className="field-block compact",
                                    children=[
                                        html.Label("Junction display", className="field-label"),
                                        dcc.Checklist(
                                            id="show-signalized-only",
                                            options=[
                                                {
                                                    "label": "Only signalized junctions",
                                                    "value": "signalized_only",
                                                }
                                            ],
                                            value=[],
                                            className="check-row",
                                        ),
                                    ],
                                ),
                                field_number(
                                    "Junction marker size", "junction-marker-size", 6, 1, 2, 20
                                ),
                            ],
                        ),
                        dcc.Graph(
                            id="network-graph",
                            figure=empty_network_figure(),
                            className="network-graph",
                        ),
                        html.Div(
                            className="viewer-foot",
                            children=[
                                html.Div(id="junction-hover-details", className="status-inline"),
                                html.Div(id="network-meta", children=build_network_meta(None)),
                            ],
                        ),
                    ],
                ),
                html.Div(
                    className="settings-pane",
                    children=[
                        dcc.Tabs(
                            id="settings-tabs",
                            value="tab-setup",
                            parent_className="settings-tabs-parent",
                            className="settings-tabs",
                            content_style={"height": "100%", "overflow": "auto"},
                            children=[
                                dcc.Tab(
                                    label="Setup",
                                    value="tab-setup",
                                    className="settings-tab",
                                    selected_className="settings-tab-selected",
                                    children=[
                                        html.Div(
                                            className="tab-panel",
                                            children=[
                                                html.Div(
                                                    "Connect Network", className="panel-title"
                                                ),
                                                html.Div(
                                                    className="field-block",
                                                    children=[
                                                        html.Label(
                                                            "Network file in repo",
                                                            className="field-label",
                                                        ),
                                                        dcc.Dropdown(
                                                            id="network-file-dropdown",
                                                            options=[
                                                                {"label": path, "value": path}
                                                                for path in NETWORK_FILES
                                                            ],
                                                            value=DEFAULT_NETWORK,
                                                            placeholder="Select *.net.xml or *.net.xml.gz",
                                                            className="dropdown",
                                                        ),
                                                    ],
                                                ),
                                                html.Div(
                                                    className="field-block",
                                                    children=[
                                                        html.Label(
                                                            "Or enter custom path",
                                                            className="field-label",
                                                        ),
                                                        dcc.Input(
                                                            id="network-custom-path",
                                                            type="text",
                                                            className="field-input",
                                                            placeholder="Optional absolute or repo relative path",
                                                        ),
                                                    ],
                                                ),
                                                html.Div(
                                                    className="inline-actions",
                                                    children=[
                                                        html.Button(
                                                            "Connect Network",
                                                            id="connect-network-button",
                                                            className="action-button",
                                                        ),
                                                    ],
                                                ),
                                                html.Div(
                                                    id="network-status", className="status-inline"
                                                ),
                                                html.Div(
                                                    className="sub-panel",
                                                    children=[
                                                        html.Div(
                                                            "Fixed output folders",
                                                            className="sub-panel-title",
                                                        ),
                                                        html.Div(
                                                            f"Demand: {to_relpath(DEMAND_OUTPUT_DIR)}",
                                                            className="mono tiny",
                                                        ),
                                                        html.Div(
                                                            f"SUMO cfg: {to_relpath(SUMOCFG_OUTPUT_DIR)}",
                                                            className="mono tiny",
                                                        ),
                                                        html.Div(
                                                            f"Scenario JSON: {to_relpath(SCENARIO_OUTPUT_DIR)}",
                                                            className="mono tiny",
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        )
                                    ],
                                ),
                                dcc.Tab(
                                    label="Demand",
                                    value="tab-demand",
                                    className="settings-tab",
                                    selected_className="settings-tab-selected",
                                    children=[
                                        html.Div(
                                            className="tab-panel",
                                            children=[
                                                html.Div(
                                                    "Demand Data Settings", className="panel-title"
                                                ),
                                                html.Div(
                                                    className="field-block",
                                                    children=[
                                                        html.Label(
                                                            "Mapped TMC locations",
                                                            className="field-label",
                                                        ),
                                                        dcc.Dropdown(
                                                            id="location-selector",
                                                            options=[],
                                                            value=[],
                                                            multi=True,
                                                            className="dropdown",
                                                            placeholder="Connect a network first",
                                                        ),
                                                    ],
                                                ),
                                                html.Div(
                                                    className="field-grid three",
                                                    children=[
                                                        field_number(
                                                            "Max TMC to junction match distance (m)",
                                                            "max-match-distance",
                                                            180,
                                                            5,
                                                            1,
                                                            2500,
                                                        ),
                                                        field_number(
                                                            "Min flow count threshold",
                                                            "min-count-threshold",
                                                            1,
                                                            1,
                                                            0,
                                                            1000,
                                                        ),
                                                        field_number(
                                                            "Global demand scale",
                                                            "global-demand-scale",
                                                            1.0,
                                                            0.05,
                                                            0.0,
                                                            10.0,
                                                        ),
                                                    ],
                                                ),
                                                html.Div(
                                                    className="field-block",
                                                    children=[
                                                        html.Label(
                                                            "Date selection policy",
                                                            className="field-label",
                                                        ),
                                                        dcc.RadioItems(
                                                            id="date-policy",
                                                            options=[
                                                                {
                                                                    "label": "Latest per location",
                                                                    "value": "latest_per_location",
                                                                },
                                                                {
                                                                    "label": "One exact date",
                                                                    "value": "selected_date",
                                                                },
                                                                {
                                                                    "label": "Date range",
                                                                    "value": "date_range",
                                                                },
                                                            ],
                                                            value="latest_per_location",
                                                            className="radio-row",
                                                        ),
                                                    ],
                                                ),
                                                html.Div(
                                                    id="selected-date-wrapper",
                                                    className="field-block",
                                                    children=[
                                                        html.Label(
                                                            "Selected date",
                                                            className="field-label",
                                                        ),
                                                        dcc.Dropdown(
                                                            id="selected-date",
                                                            options=[
                                                                {"label": d, "value": d}
                                                                for d in DATE_OPTIONS
                                                            ],
                                                            value=(
                                                                DATE_OPTIONS[0]
                                                                if DATE_OPTIONS
                                                                else None
                                                            ),
                                                            className="dropdown",
                                                        ),
                                                    ],
                                                ),
                                                html.Div(
                                                    id="date-range-wrapper",
                                                    className="field-grid two",
                                                    children=[
                                                        html.Div(
                                                            className="field-block",
                                                            children=[
                                                                html.Label(
                                                                    "Range start",
                                                                    className="field-label",
                                                                ),
                                                                dcc.Dropdown(
                                                                    id="date-range-start",
                                                                    options=[
                                                                        {"label": d, "value": d}
                                                                        for d in DATE_OPTIONS
                                                                    ],
                                                                    value=(
                                                                        DATE_OPTIONS[-1]
                                                                        if DATE_OPTIONS
                                                                        else None
                                                                    ),
                                                                    className="dropdown",
                                                                ),
                                                            ],
                                                        ),
                                                        html.Div(
                                                            className="field-block",
                                                            children=[
                                                                html.Label(
                                                                    "Range end",
                                                                    className="field-label",
                                                                ),
                                                                dcc.Dropdown(
                                                                    id="date-range-end",
                                                                    options=[
                                                                        {"label": d, "value": d}
                                                                        for d in DATE_OPTIONS
                                                                    ],
                                                                    value=(
                                                                        DATE_OPTIONS[0]
                                                                        if DATE_OPTIONS
                                                                        else None
                                                                    ),
                                                                    className="dropdown",
                                                                ),
                                                            ],
                                                        ),
                                                    ],
                                                ),
                                                html.Div(
                                                    className="field-grid four",
                                                    children=[
                                                        html.Div(
                                                            className="field-block",
                                                            children=[
                                                                html.Label(
                                                                    "TMC time window start",
                                                                    className="field-label",
                                                                ),
                                                                dcc.Dropdown(
                                                                    id="time-window-start",
                                                                    options=TIME_WINDOW_START_OPTIONS,
                                                                    value=8 * 60,
                                                                    className="dropdown",
                                                                ),
                                                            ],
                                                        ),
                                                        html.Div(
                                                            className="field-block",
                                                            children=[
                                                                html.Label(
                                                                    "TMC time window duration",
                                                                    className="field-label",
                                                                ),
                                                                dcc.Dropdown(
                                                                    id="time-window-duration",
                                                                    options=TIME_WINDOW_DURATION_OPTIONS,
                                                                    value=60,
                                                                    className="dropdown",
                                                                ),
                                                            ],
                                                        ),
                                                        field_number(
                                                            "Simulation begin (s)",
                                                            "sim-begin",
                                                            0,
                                                            1,
                                                            0,
                                                            1000000,
                                                        ),
                                                        field_number(
                                                            "Simulation end (s)",
                                                            "sim-end",
                                                            3600,
                                                            1,
                                                            1,
                                                            1000000,
                                                        ),
                                                    ],
                                                ),
                                                html.Div(
                                                    className="field-grid two",
                                                    children=[
                                                        html.Div(
                                                            className="field-block",
                                                            children=[
                                                                html.Label(
                                                                    "Include modes",
                                                                    className="field-label",
                                                                ),
                                                                dcc.Checklist(
                                                                    id="mode-checklist",
                                                                    options=[
                                                                        {
                                                                            "label": "Cars",
                                                                            "value": "cars",
                                                                        },
                                                                        {
                                                                            "label": "Trucks",
                                                                            "value": "trucks",
                                                                        },
                                                                        {
                                                                            "label": "Buses",
                                                                            "value": "buses",
                                                                        },
                                                                        {
                                                                            "label": "Streetcars",
                                                                            "value": "streetcars",
                                                                        },
                                                                        {
                                                                            "label": "Pedestrians",
                                                                            "value": "pedestrians",
                                                                        },
                                                                    ],
                                                                    value=[
                                                                        "cars",
                                                                        "trucks",
                                                                        "buses",
                                                                    ],
                                                                    className="check-grid",
                                                                ),
                                                            ],
                                                        ),
                                                        html.Div(
                                                            className="field-block",
                                                            children=[
                                                                html.Label(
                                                                    "Demand scaling by mode",
                                                                    className="field-label",
                                                                ),
                                                                html.Div(
                                                                    className="field-grid two",
                                                                    children=[
                                                                        field_number(
                                                                            "Cars",
                                                                            "car-scale",
                                                                            1.0,
                                                                            0.05,
                                                                            0.0,
                                                                            10.0,
                                                                        ),
                                                                        field_number(
                                                                            "Trucks",
                                                                            "truck-scale",
                                                                            1.0,
                                                                            0.05,
                                                                            0.0,
                                                                            10.0,
                                                                        ),
                                                                        field_number(
                                                                            "Buses",
                                                                            "bus-scale",
                                                                            1.0,
                                                                            0.05,
                                                                            0.0,
                                                                            10.0,
                                                                        ),
                                                                        field_number(
                                                                            "Streetcars",
                                                                            "streetcar-scale",
                                                                            1.0,
                                                                            0.05,
                                                                            0.0,
                                                                            10.0,
                                                                        ),
                                                                        field_number(
                                                                            "Pedestrians",
                                                                            "pedestrian-scale",
                                                                            1.0,
                                                                            0.05,
                                                                            0.0,
                                                                            10.0,
                                                                        ),
                                                                        field_number(
                                                                            "Streetcar share from bus",
                                                                            "streetcar-share",
                                                                            0.35,
                                                                            0.05,
                                                                            0.0,
                                                                            1.0,
                                                                        ),
                                                                    ],
                                                                ),
                                                            ],
                                                        ),
                                                    ],
                                                ),
                                                html.Div(
                                                    className="field-block",
                                                    children=[
                                                        html.Label(
                                                            "Route constraints",
                                                            className="field-label",
                                                        ),
                                                        dcc.Checklist(
                                                            id="strict-route-check",
                                                            options=[
                                                                {
                                                                    "label": "Strict route check with SUMO shortest path",
                                                                    "value": "strict",
                                                                }
                                                            ],
                                                            value=["strict"],
                                                            className="check-row",
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        )
                                    ],
                                ),
                                dcc.Tab(
                                    label="Signal and MAPPO",
                                    value="tab-control",
                                    className="settings-tab",
                                    selected_className="settings-tab-selected",
                                    children=[
                                        html.Div(
                                            className="tab-panel",
                                            children=[
                                                html.Div(
                                                    "Signal and MAPPO Settings",
                                                    className="panel-title",
                                                ),
                                                html.Div(
                                                    className="field-block",
                                                    children=[
                                                        html.Label(
                                                            "Controller mode",
                                                            className="field-label",
                                                        ),
                                                        dcc.RadioItems(
                                                            id="controller-mode",
                                                            options=[
                                                                {
                                                                    "label": "MAPPO custom",
                                                                    "value": "mappo_custom",
                                                                },
                                                                {
                                                                    "label": "Baseline fixed time",
                                                                    "value": "baseline_fixed_time",
                                                                },
                                                                {
                                                                    "label": "Baseline max pressure",
                                                                    "value": "baseline_max_pressure",
                                                                },
                                                            ],
                                                            value="mappo_custom",
                                                            className="radio-row",
                                                        ),
                                                    ],
                                                ),
                                                html.Div(
                                                    id="fixed-time-group",
                                                    className="sub-panel",
                                                    children=[
                                                        html.Div(
                                                            "Fixed time settings",
                                                            className="sub-panel-title",
                                                        ),
                                                        html.Div(
                                                            className="field-grid four",
                                                            children=[
                                                                field_number(
                                                                    "EW green",
                                                                    "fixed-ew-green",
                                                                    DEFAULT_FIXED_SIGNAL[
                                                                        "ew_green"
                                                                    ],
                                                                    1,
                                                                    1,
                                                                    240,
                                                                ),
                                                                field_number(
                                                                    "NS green",
                                                                    "fixed-ns-green",
                                                                    DEFAULT_FIXED_SIGNAL[
                                                                        "ns_green"
                                                                    ],
                                                                    1,
                                                                    1,
                                                                    240,
                                                                ),
                                                                field_number(
                                                                    "Yellow",
                                                                    "fixed-yellow",
                                                                    DEFAULT_FIXED_SIGNAL["yellow"],
                                                                    1,
                                                                    1,
                                                                    20,
                                                                ),
                                                                field_number(
                                                                    "All red",
                                                                    "fixed-all-red",
                                                                    DEFAULT_FIXED_SIGNAL[
                                                                        "all_red"
                                                                    ],
                                                                    1,
                                                                    0,
                                                                    20,
                                                                ),
                                                            ],
                                                        ),
                                                        html.Div(
                                                            className="field-grid two",
                                                            children=[
                                                                field_number(
                                                                    "Offset",
                                                                    "fixed-offset",
                                                                    DEFAULT_FIXED_SIGNAL["offset"],
                                                                    1,
                                                                    0,
                                                                    3600,
                                                                ),
                                                                html.Div(
                                                                    className="field-block",
                                                                    children=[
                                                                        html.Label(
                                                                            "TLS selection",
                                                                            className="field-label",
                                                                        ),
                                                                        dcc.Checklist(
                                                                            id="include-cluster-signals",
                                                                            options=[
                                                                                {
                                                                                    "label": "Include cluster_* traffic lights",
                                                                                    "value": "include",
                                                                                }
                                                                            ],
                                                                            value=[],
                                                                            className="check-row",
                                                                        ),
                                                                    ],
                                                                ),
                                                            ],
                                                        ),
                                                    ],
                                                ),
                                                html.Div(
                                                    id="max-pressure-group",
                                                    className="sub-panel",
                                                    children=[
                                                        html.Div(
                                                            "Max pressure settings",
                                                            className="sub-panel-title",
                                                        ),
                                                        html.Div(
                                                            className="field-grid three",
                                                            children=[
                                                                field_number(
                                                                    "Min green",
                                                                    "mp-min-green",
                                                                    DEFAULT_MAX_PRESSURE_SIGNAL[
                                                                        "min_green"
                                                                    ],
                                                                    1,
                                                                    1,
                                                                    180,
                                                                ),
                                                                field_number(
                                                                    "Max green",
                                                                    "mp-max-green",
                                                                    DEFAULT_MAX_PRESSURE_SIGNAL[
                                                                        "max_green"
                                                                    ],
                                                                    1,
                                                                    2,
                                                                    360,
                                                                ),
                                                                field_number(
                                                                    "Yellow",
                                                                    "mp-yellow",
                                                                    DEFAULT_MAX_PRESSURE_SIGNAL[
                                                                        "yellow"
                                                                    ],
                                                                    1,
                                                                    1,
                                                                    20,
                                                                ),
                                                                field_number(
                                                                    "All red",
                                                                    "mp-all-red",
                                                                    DEFAULT_MAX_PRESSURE_SIGNAL[
                                                                        "all_red"
                                                                    ],
                                                                    1,
                                                                    0,
                                                                    20,
                                                                ),
                                                                field_number(
                                                                    "Pressure exponent",
                                                                    "mp-pressure-exp",
                                                                    DEFAULT_MAX_PRESSURE_SIGNAL[
                                                                        "pressure_exponent"
                                                                    ],
                                                                    0.1,
                                                                    0.0,
                                                                    5.0,
                                                                ),
                                                                field_number(
                                                                    "Queue exponent",
                                                                    "mp-queue-exp",
                                                                    DEFAULT_MAX_PRESSURE_SIGNAL[
                                                                        "queue_exponent"
                                                                    ],
                                                                    0.1,
                                                                    0.0,
                                                                    5.0,
                                                                ),
                                                            ],
                                                        ),
                                                    ],
                                                ),
                                                html.Div(
                                                    id="mappo-group",
                                                    className="sub-panel",
                                                    children=[
                                                        html.Div(
                                                            "MAPPO hyperparameters",
                                                            className="sub-panel-title",
                                                        ),
                                                        html.Div(
                                                            className="field-grid three",
                                                            children=[
                                                                field_number(
                                                                    "LR actor",
                                                                    "mappo-lr-actor",
                                                                    DEFAULT_MAPPO[
                                                                        "learning_rate_actor"
                                                                    ],
                                                                    0.0001,
                                                                    1e-06,
                                                                    1.0,
                                                                ),
                                                                field_number(
                                                                    "LR critic",
                                                                    "mappo-lr-critic",
                                                                    DEFAULT_MAPPO[
                                                                        "learning_rate_critic"
                                                                    ],
                                                                    0.0001,
                                                                    1e-06,
                                                                    1.0,
                                                                ),
                                                                field_number(
                                                                    "Gamma",
                                                                    "mappo-gamma",
                                                                    DEFAULT_MAPPO["gamma"],
                                                                    0.01,
                                                                    0.0,
                                                                    1.0,
                                                                ),
                                                                field_number(
                                                                    "GAE lambda",
                                                                    "mappo-gae-lambda",
                                                                    DEFAULT_MAPPO["gae_lambda"],
                                                                    0.01,
                                                                    0.0,
                                                                    1.0,
                                                                ),
                                                                field_number(
                                                                    "Clip ratio",
                                                                    "mappo-clip-ratio",
                                                                    DEFAULT_MAPPO["clip_ratio"],
                                                                    0.01,
                                                                    0.01,
                                                                    1.0,
                                                                ),
                                                                field_number(
                                                                    "Entropy coef",
                                                                    "mappo-entropy-coef",
                                                                    DEFAULT_MAPPO["entropy_coef"],
                                                                    0.001,
                                                                    0.0,
                                                                    1.0,
                                                                ),
                                                                field_number(
                                                                    "Value loss coef",
                                                                    "mappo-value-coef",
                                                                    DEFAULT_MAPPO[
                                                                        "value_loss_coef"
                                                                    ],
                                                                    0.01,
                                                                    0.0,
                                                                    10.0,
                                                                ),
                                                                field_number(
                                                                    "Max grad norm",
                                                                    "mappo-max-grad-norm",
                                                                    DEFAULT_MAPPO["max_grad_norm"],
                                                                    0.1,
                                                                    0.0,
                                                                    10.0,
                                                                ),
                                                                field_number(
                                                                    "Target KL",
                                                                    "mappo-target-kl",
                                                                    DEFAULT_MAPPO["target_kl"],
                                                                    0.01,
                                                                    0.0,
                                                                    1.0,
                                                                ),
                                                                field_number(
                                                                    "Rollout steps",
                                                                    "mappo-rollout-steps",
                                                                    DEFAULT_MAPPO["rollout_steps"],
                                                                    1,
                                                                    16,
                                                                    1000000,
                                                                ),
                                                                field_number(
                                                                    "PPO epochs",
                                                                    "mappo-ppo-epochs",
                                                                    DEFAULT_MAPPO["ppo_epochs"],
                                                                    1,
                                                                    1,
                                                                    500,
                                                                ),
                                                                field_number(
                                                                    "Minibatch size",
                                                                    "mappo-minibatch-size",
                                                                    DEFAULT_MAPPO[
                                                                        "minibatch_size"
                                                                    ],
                                                                    1,
                                                                    1,
                                                                    1000000,
                                                                ),
                                                                field_number(
                                                                    "Num envs",
                                                                    "mappo-num-envs",
                                                                    DEFAULT_MAPPO["num_envs"],
                                                                    1,
                                                                    1,
                                                                    10000,
                                                                ),
                                                                field_number(
                                                                    "Hidden dim",
                                                                    "mappo-hidden-dim",
                                                                    DEFAULT_MAPPO["hidden_dim"],
                                                                    1,
                                                                    4,
                                                                    4096,
                                                                ),
                                                                field_number(
                                                                    "Hidden layers",
                                                                    "mappo-hidden-layers",
                                                                    DEFAULT_MAPPO["hidden_layers"],
                                                                    1,
                                                                    1,
                                                                    16,
                                                                ),
                                                                field_number(
                                                                    "Seed",
                                                                    "mappo-seed",
                                                                    DEFAULT_MAPPO["seed"],
                                                                    1,
                                                                    0,
                                                                    1000000,
                                                                ),
                                                            ],
                                                        ),
                                                        html.Div(
                                                            className="field-grid two",
                                                            children=[
                                                                html.Div(
                                                                    className="field-block",
                                                                    children=[
                                                                        html.Label(
                                                                            "Shared critic",
                                                                            className="field-label",
                                                                        ),
                                                                        dcc.Checklist(
                                                                            id="mappo-shared-critic",
                                                                            options=[
                                                                                {
                                                                                    "label": "Enabled",
                                                                                    "value": "enabled",
                                                                                }
                                                                            ],
                                                                            value=["enabled"],
                                                                            className="check-row",
                                                                        ),
                                                                    ],
                                                                ),
                                                                html.Div(
                                                                    className="field-block",
                                                                    children=[
                                                                        html.Label(
                                                                            "Normalize advantages",
                                                                            className="field-label",
                                                                        ),
                                                                        dcc.Checklist(
                                                                            id="mappo-normalize-advantages",
                                                                            options=[
                                                                                {
                                                                                    "label": "Enabled",
                                                                                    "value": "enabled",
                                                                                }
                                                                            ],
                                                                            value=["enabled"],
                                                                            className="check-row",
                                                                        ),
                                                                    ],
                                                                ),
                                                            ],
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        )
                                    ],
                                ),
                                dcc.Tab(
                                    label="Advanced",
                                    value="tab-advanced",
                                    className="settings-tab",
                                    selected_className="settings-tab-selected",
                                    children=[
                                        html.Div(
                                            className="tab-panel",
                                            children=[
                                                html.Div(
                                                    "Advanced Mapping and Vehicle Settings",
                                                    className="panel-title",
                                                ),
                                                html.Div(
                                                    className="sub-panel",
                                                    children=[
                                                        html.Div(
                                                            "Approach heading ranges (degrees)",
                                                            className="sub-panel-title",
                                                        ),
                                                        html.Div(
                                                            className="field-grid four",
                                                            children=[
                                                                field_number(
                                                                    "N low",
                                                                    "heading-n-lo",
                                                                    APPROACH_HEADING_RANGES_DEFAULT[
                                                                        "n"
                                                                    ][
                                                                        0
                                                                    ],
                                                                    1,
                                                                    0,
                                                                    360,
                                                                ),
                                                                field_number(
                                                                    "N high",
                                                                    "heading-n-hi",
                                                                    APPROACH_HEADING_RANGES_DEFAULT[
                                                                        "n"
                                                                    ][
                                                                        1
                                                                    ],
                                                                    1,
                                                                    0,
                                                                    360,
                                                                ),
                                                                field_number(
                                                                    "S low",
                                                                    "heading-s-lo",
                                                                    APPROACH_HEADING_RANGES_DEFAULT[
                                                                        "s"
                                                                    ][
                                                                        0
                                                                    ],
                                                                    1,
                                                                    0,
                                                                    360,
                                                                ),
                                                                field_number(
                                                                    "S high",
                                                                    "heading-s-hi",
                                                                    APPROACH_HEADING_RANGES_DEFAULT[
                                                                        "s"
                                                                    ][
                                                                        1
                                                                    ],
                                                                    1,
                                                                    0,
                                                                    360,
                                                                ),
                                                                field_number(
                                                                    "E low",
                                                                    "heading-e-lo",
                                                                    APPROACH_HEADING_RANGES_DEFAULT[
                                                                        "e"
                                                                    ][
                                                                        0
                                                                    ],
                                                                    1,
                                                                    0,
                                                                    360,
                                                                ),
                                                                field_number(
                                                                    "E high",
                                                                    "heading-e-hi",
                                                                    APPROACH_HEADING_RANGES_DEFAULT[
                                                                        "e"
                                                                    ][
                                                                        1
                                                                    ],
                                                                    1,
                                                                    0,
                                                                    360,
                                                                ),
                                                                field_number(
                                                                    "W low",
                                                                    "heading-w-lo",
                                                                    APPROACH_HEADING_RANGES_DEFAULT[
                                                                        "w"
                                                                    ][
                                                                        0
                                                                    ],
                                                                    1,
                                                                    0,
                                                                    360,
                                                                ),
                                                                field_number(
                                                                    "W high",
                                                                    "heading-w-hi",
                                                                    APPROACH_HEADING_RANGES_DEFAULT[
                                                                        "w"
                                                                    ][
                                                                        1
                                                                    ],
                                                                    1,
                                                                    0,
                                                                    360,
                                                                ),
                                                            ],
                                                        ),
                                                    ],
                                                ),
                                                mode_vtype_block("Car vType", "car"),
                                                mode_vtype_block("Truck vType", "truck"),
                                                mode_vtype_block("Bus vType", "bus"),
                                                mode_vtype_block("Streetcar vType", "streetcar"),
                                                html.Div(
                                                    className="field-block",
                                                    children=[
                                                        html.Label(
                                                            "Validation", className="field-label"
                                                        ),
                                                        dcc.Checklist(
                                                            id="run-sumo-check",
                                                            options=[
                                                                {
                                                                    "label": "Run short SUMO validation after generation",
                                                                    "value": "enabled",
                                                                }
                                                            ],
                                                            value=["enabled"],
                                                            className="check-row",
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        )
                                    ],
                                ),
                            ],
                        ),
                        html.Div(
                            className="action-bar",
                            children=[
                                html.Div(
                                    className="action-row",
                                    children=[
                                        html.Button(
                                            "Generate Demand and Scenario Files",
                                            id="generate-demand-button",
                                            className="action-button primary",
                                        ),
                                        html.Button(
                                            "Download demand.xml",
                                            id="download-demand-button",
                                            className="action-button",
                                            disabled=True,
                                        ),
                                        html.Button(
                                            "Download Settings Confirmation (.txt)",
                                            id="download-settings-button",
                                            className="action-button",
                                            disabled=True,
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        html.Div(
                            className="status-stack",
                            children=[
                                html.Div(id="generation-status"),
                                html.Div(id="output-path-summary"),
                                html.Div(id="confirmation-screen"),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)


@callback(
    Output("network-store", "data"),
    Output("network-status", "children"),
    Input("connect-network-button", "n_clicks"),
    State("network-file-dropdown", "value"),
    State("network-custom-path", "value"),
)
def connect_network(n_clicks: int | None, dropdown_value: str | None, custom_path: str | None):
    # Initial render should still auto-connect the default file.
    _ = n_clicks

    if not dropdown_value and not custom_path:
        return None, html.Div(
            "Select a network file and click Connect Network", className="status-inline"
        )

    try:
        network_path = resolve_network_path(dropdown_value, custom_path)
        summary = load_network_summary(network_path)
    except Exception as exc:
        return None, html.Div(f"Network connection failed: {exc}", className="status-inline error")

    store = {
        "network_key": summary["key"],
        "network_relpath": summary["network_relpath"],
        "edge_count": summary["edge_count"],
        "node_count": len(summary["node_ids"]),
        "tls_count": summary["tls_count"],
        "mapped_location_count": len(summary["location_mappings"]),
    }

    status = html.Div(
        [
            html.Div("Network connected", className="status-inline success"),
            html.Div(f"{summary['network_relpath']}", className="status-inline mono"),
        ]
    )

    return store, status


@callback(
    Output("location-selector", "options"),
    Output("location-selector", "value"),
    Input("network-store", "data"),
    Input("max-match-distance", "value"),
    State("location-selector", "value"),
)
def update_location_selector(
    network_store: dict[str, Any] | None,
    max_match_distance: float | int | None,
    current_values: list[str] | None,
):
    if not network_store:
        return [], []

    summary = NETWORK_CACHE.get(str(network_store.get("network_key")))
    if summary is None:
        return [], []

    limit = clamp_float(max_match_distance, 180.0, 1.0, 5000.0)

    filtered = [
        item for item in summary["location_mappings"] if float(item["distance_m"]) <= limit
    ]

    options = [
        {
            "label": f"{item['location_name']} | {item['junction_id']} | {item['distance_m']:.1f} m",
            "value": item["location_name"],
        }
        for item in filtered
    ]

    valid_values = {option["value"] for option in options}
    current_values = current_values or []
    selected = [value for value in current_values if value in valid_values]

    if not selected and options:
        selected = [option["value"] for option in options[: min(12, len(options))]]

    return options, selected


@callback(
    Output("network-meta", "children"),
    Input("network-store", "data"),
)
def update_network_meta(network_store: dict[str, Any] | None):
    return build_network_meta(network_store)


@callback(
    Output("network-graph", "figure"),
    Input("network-store", "data"),
    Input("show-signalized-only", "value"),
    Input("junction-marker-size", "value"),
    Input("location-selector", "value"),
    Input("max-match-distance", "value"),
)
def update_network_graph(
    network_store: dict[str, Any] | None,
    show_signalized_only_values: list[str] | None,
    marker_size_value: int | float | None,
    selected_locations: list[str] | None,
    max_match_distance: float | int | None,
):
    if not network_store:
        return empty_network_figure()

    network_key = str(network_store.get("network_key"))
    show_signalized_only = bool(
        show_signalized_only_values and "signalized_only" in show_signalized_only_values
    )
    marker_size = clamp_int(marker_size_value, 6, 2, 30)
    max_distance = clamp_float(max_match_distance, 180.0, 1.0, 5000.0)

    return build_network_figure(
        network_key=network_key,
        show_signalized_only=show_signalized_only,
        marker_size=marker_size,
        selected_locations=selected_locations or [],
        max_match_distance=max_distance,
    )


@callback(
    Output("junction-hover-details", "children"),
    Input("network-graph", "hoverData"),
)
def update_hover_panel(hover_data: dict[str, Any] | None):
    if not hover_data or "points" not in hover_data or not hover_data["points"]:
        return html.Div(
            "Hover any junction to inspect ID, signalization state, and degree",
            className="status-inline",
        )

    point = hover_data["points"][0]
    custom = point.get("customdata")
    if not custom:
        return html.Div(
            "Hover any junction to inspect ID, signalization state, and degree",
            className="status-inline",
        )

    return html.Div(
        [
            html.Div(f"Junction ID: {custom[0]}", className="status-inline mono"),
            html.Div(f"Signalized: {custom[1]}", className="status-inline"),
            html.Div(
                f"Incoming edges: {custom[2]} | Outgoing edges: {custom[3]}",
                className="status-inline",
            ),
            html.Div(f"Node type: {custom[4]}", className="status-inline"),
        ]
    )


@callback(
    Output("selected-date-wrapper", "style"),
    Output("date-range-wrapper", "style"),
    Input("date-policy", "value"),
)
def toggle_date_controls(date_policy: str):
    if date_policy == "selected_date":
        return {"display": "block"}, {"display": "none"}
    if date_policy == "date_range":
        return {"display": "none"}, {"display": "grid"}
    return {"display": "none"}, {"display": "none"}


@callback(
    Output("fixed-time-group", "style"),
    Output("max-pressure-group", "style"),
    Output("mappo-group", "style"),
    Input("controller-mode", "value"),
)
def toggle_controller_groups(controller_mode: str):
    show_fixed = (
        {"display": "block"} if controller_mode == "baseline_fixed_time" else {"display": "none"}
    )
    show_max_pressure = (
        {"display": "block"} if controller_mode == "baseline_max_pressure" else {"display": "none"}
    )
    show_mappo = {"display": "block"} if controller_mode == "mappo_custom" else {"display": "none"}
    return show_fixed, show_max_pressure, show_mappo


@callback(
    Output("generation-status", "children"),
    Output("output-path-summary", "children"),
    Output("generated-artifacts-store", "data"),
    Output("demand-download-state", "data"),
    Input("generate-demand-button", "n_clicks"),
    State("network-store", "data"),
    State("location-selector", "value"),
    State("max-match-distance", "value"),
    State("date-policy", "value"),
    State("selected-date", "value"),
    State("date-range-start", "value"),
    State("date-range-end", "value"),
    State("time-window-start", "value"),
    State("time-window-duration", "value"),
    State("sim-begin", "value"),
    State("sim-end", "value"),
    State("strict-route-check", "value"),
    State("min-count-threshold", "value"),
    State("global-demand-scale", "value"),
    State("mode-checklist", "value"),
    State("car-scale", "value"),
    State("truck-scale", "value"),
    State("bus-scale", "value"),
    State("streetcar-scale", "value"),
    State("pedestrian-scale", "value"),
    State("streetcar-share", "value"),
    State("controller-mode", "value"),
    State("include-cluster-signals", "value"),
    State("fixed-ew-green", "value"),
    State("fixed-ns-green", "value"),
    State("fixed-yellow", "value"),
    State("fixed-all-red", "value"),
    State("fixed-offset", "value"),
    State("mp-min-green", "value"),
    State("mp-max-green", "value"),
    State("mp-yellow", "value"),
    State("mp-all-red", "value"),
    State("mp-pressure-exp", "value"),
    State("mp-queue-exp", "value"),
    State("mappo-lr-actor", "value"),
    State("mappo-lr-critic", "value"),
    State("mappo-gamma", "value"),
    State("mappo-gae-lambda", "value"),
    State("mappo-clip-ratio", "value"),
    State("mappo-entropy-coef", "value"),
    State("mappo-value-coef", "value"),
    State("mappo-max-grad-norm", "value"),
    State("mappo-rollout-steps", "value"),
    State("mappo-ppo-epochs", "value"),
    State("mappo-minibatch-size", "value"),
    State("mappo-num-envs", "value"),
    State("mappo-hidden-dim", "value"),
    State("mappo-hidden-layers", "value"),
    State("mappo-target-kl", "value"),
    State("mappo-seed", "value"),
    State("mappo-shared-critic", "value"),
    State("mappo-normalize-advantages", "value"),
    State("heading-n-lo", "value"),
    State("heading-n-hi", "value"),
    State("heading-s-lo", "value"),
    State("heading-s-hi", "value"),
    State("heading-e-lo", "value"),
    State("heading-e-hi", "value"),
    State("heading-w-lo", "value"),
    State("heading-w-hi", "value"),
    State("car-accel", "value"),
    State("car-decel", "value"),
    State("car-sigma", "value"),
    State("car-length", "value"),
    State("car-maxspeed", "value"),
    State("truck-accel", "value"),
    State("truck-decel", "value"),
    State("truck-sigma", "value"),
    State("truck-length", "value"),
    State("truck-maxspeed", "value"),
    State("bus-accel", "value"),
    State("bus-decel", "value"),
    State("bus-sigma", "value"),
    State("bus-length", "value"),
    State("bus-maxspeed", "value"),
    State("streetcar-accel", "value"),
    State("streetcar-decel", "value"),
    State("streetcar-sigma", "value"),
    State("streetcar-length", "value"),
    State("streetcar-maxspeed", "value"),
    State("run-sumo-check", "value"),
)
def generate_demand_callback(
    n_clicks,
    network_store,
    location_values,
    max_match_distance,
    date_policy,
    selected_date,
    date_range_start,
    date_range_end,
    time_window_start,
    time_window_duration,
    sim_begin,
    sim_end,
    strict_route_check,
    min_count_threshold,
    global_demand_scale,
    mode_checklist,
    car_scale,
    truck_scale,
    bus_scale,
    streetcar_scale,
    pedestrian_scale,
    streetcar_share,
    controller_mode,
    include_cluster_signals,
    fixed_ew_green,
    fixed_ns_green,
    fixed_yellow,
    fixed_all_red,
    fixed_offset,
    mp_min_green,
    mp_max_green,
    mp_yellow,
    mp_all_red,
    mp_pressure_exp,
    mp_queue_exp,
    mappo_lr_actor,
    mappo_lr_critic,
    mappo_gamma,
    mappo_gae_lambda,
    mappo_clip_ratio,
    mappo_entropy_coef,
    mappo_value_coef,
    mappo_max_grad_norm,
    mappo_rollout_steps,
    mappo_ppo_epochs,
    mappo_minibatch_size,
    mappo_num_envs,
    mappo_hidden_dim,
    mappo_hidden_layers,
    mappo_target_kl,
    mappo_seed,
    mappo_shared_critic,
    mappo_normalize_advantages,
    heading_n_lo,
    heading_n_hi,
    heading_s_lo,
    heading_s_hi,
    heading_e_lo,
    heading_e_hi,
    heading_w_lo,
    heading_w_hi,
    car_accel,
    car_decel,
    car_sigma,
    car_length,
    car_maxspeed,
    truck_accel,
    truck_decel,
    truck_sigma,
    truck_length,
    truck_maxspeed,
    bus_accel,
    bus_decel,
    bus_sigma,
    bus_length,
    bus_maxspeed,
    streetcar_accel,
    streetcar_decel,
    streetcar_sigma,
    streetcar_length,
    streetcar_maxspeed,
    run_sumo_check,
):
    if not n_clicks:
        raise PreventUpdate

    included_modes = set(mode_checklist or [])

    mode_scales = {
        "cars": clamp_float(car_scale, 1.0, 0.0, 10.0),
        "trucks": clamp_float(truck_scale, 1.0, 0.0, 10.0),
        "buses": clamp_float(bus_scale, 1.0, 0.0, 10.0),
        "streetcars": clamp_float(streetcar_scale, 1.0, 0.0, 10.0),
        "pedestrians": clamp_float(pedestrian_scale, 1.0, 0.0, 10.0),
    }

    fixed_signal = {
        "ew_green": clamp_int(fixed_ew_green, DEFAULT_FIXED_SIGNAL["ew_green"], 1, 240),
        "ns_green": clamp_int(fixed_ns_green, DEFAULT_FIXED_SIGNAL["ns_green"], 1, 240),
        "yellow": clamp_int(fixed_yellow, DEFAULT_FIXED_SIGNAL["yellow"], 1, 20),
        "all_red": clamp_int(fixed_all_red, DEFAULT_FIXED_SIGNAL["all_red"], 0, 20),
        "offset": clamp_int(fixed_offset, DEFAULT_FIXED_SIGNAL["offset"], 0, 3600),
    }

    max_pressure_signal = {
        "min_green": clamp_int(mp_min_green, DEFAULT_MAX_PRESSURE_SIGNAL["min_green"], 1, 180),
        "max_green": clamp_int(mp_max_green, DEFAULT_MAX_PRESSURE_SIGNAL["max_green"], 2, 360),
        "yellow": clamp_int(mp_yellow, DEFAULT_MAX_PRESSURE_SIGNAL["yellow"], 1, 20),
        "all_red": clamp_int(mp_all_red, DEFAULT_MAX_PRESSURE_SIGNAL["all_red"], 0, 20),
        "pressure_exponent": clamp_float(
            mp_pressure_exp, DEFAULT_MAX_PRESSURE_SIGNAL["pressure_exponent"], 0.0, 5.0
        ),
        "queue_exponent": clamp_float(
            mp_queue_exp, DEFAULT_MAX_PRESSURE_SIGNAL["queue_exponent"], 0.0, 5.0
        ),
    }

    mappo_hyperparams = {
        "learning_rate_actor": clamp_float(
            mappo_lr_actor, DEFAULT_MAPPO["learning_rate_actor"], 1e-6, 1.0
        ),
        "learning_rate_critic": clamp_float(
            mappo_lr_critic, DEFAULT_MAPPO["learning_rate_critic"], 1e-6, 1.0
        ),
        "gamma": clamp_float(mappo_gamma, DEFAULT_MAPPO["gamma"], 0.0, 1.0),
        "gae_lambda": clamp_float(mappo_gae_lambda, DEFAULT_MAPPO["gae_lambda"], 0.0, 1.0),
        "clip_ratio": clamp_float(mappo_clip_ratio, DEFAULT_MAPPO["clip_ratio"], 0.01, 1.0),
        "entropy_coef": clamp_float(mappo_entropy_coef, DEFAULT_MAPPO["entropy_coef"], 0.0, 1.0),
        "value_loss_coef": clamp_float(
            mappo_value_coef, DEFAULT_MAPPO["value_loss_coef"], 0.0, 10.0
        ),
        "max_grad_norm": clamp_float(
            mappo_max_grad_norm, DEFAULT_MAPPO["max_grad_norm"], 0.0, 10.0
        ),
        "rollout_steps": clamp_int(
            mappo_rollout_steps, DEFAULT_MAPPO["rollout_steps"], 16, 1000000
        ),
        "ppo_epochs": clamp_int(mappo_ppo_epochs, DEFAULT_MAPPO["ppo_epochs"], 1, 500),
        "minibatch_size": clamp_int(
            mappo_minibatch_size, DEFAULT_MAPPO["minibatch_size"], 1, 1000000
        ),
        "num_envs": clamp_int(mappo_num_envs, DEFAULT_MAPPO["num_envs"], 1, 10000),
        "hidden_dim": clamp_int(mappo_hidden_dim, DEFAULT_MAPPO["hidden_dim"], 4, 4096),
        "hidden_layers": clamp_int(mappo_hidden_layers, DEFAULT_MAPPO["hidden_layers"], 1, 16),
        "target_kl": clamp_float(mappo_target_kl, DEFAULT_MAPPO["target_kl"], 0.0, 1.0),
        "seed": clamp_int(mappo_seed, DEFAULT_MAPPO["seed"], 0, 1000000),
        "shared_critic": bool(mappo_shared_critic and "enabled" in mappo_shared_critic),
        "normalize_advantages": bool(
            mappo_normalize_advantages and "enabled" in mappo_normalize_advantages
        ),
    }

    heading_ranges = {
        "n": (
            clamp_float(heading_n_lo, APPROACH_HEADING_RANGES_DEFAULT["n"][0], 0.0, 360.0),
            clamp_float(heading_n_hi, APPROACH_HEADING_RANGES_DEFAULT["n"][1], 0.0, 360.0),
        ),
        "s": (
            clamp_float(heading_s_lo, APPROACH_HEADING_RANGES_DEFAULT["s"][0], 0.0, 360.0),
            clamp_float(heading_s_hi, APPROACH_HEADING_RANGES_DEFAULT["s"][1], 0.0, 360.0),
        ),
        "e": (
            clamp_float(heading_e_lo, APPROACH_HEADING_RANGES_DEFAULT["e"][0], 0.0, 360.0),
            clamp_float(heading_e_hi, APPROACH_HEADING_RANGES_DEFAULT["e"][1], 0.0, 360.0),
        ),
        "w": (
            clamp_float(heading_w_lo, APPROACH_HEADING_RANGES_DEFAULT["w"][0], 0.0, 360.0),
            clamp_float(heading_w_hi, APPROACH_HEADING_RANGES_DEFAULT["w"][1], 0.0, 360.0),
        ),
    }

    vtype_settings = {
        "car": {
            "accel": clamp_float(car_accel, DEFAULT_VTYPE["car"]["accel"], 0.1, 10.0),
            "decel": clamp_float(car_decel, DEFAULT_VTYPE["car"]["decel"], 0.1, 10.0),
            "sigma": clamp_float(car_sigma, DEFAULT_VTYPE["car"]["sigma"], 0.0, 1.0),
            "length": clamp_float(car_length, DEFAULT_VTYPE["car"]["length"], 0.5, 100.0),
            "maxSpeed": clamp_float(car_maxspeed, DEFAULT_VTYPE["car"]["maxSpeed"], 1.0, 60.0),
        },
        "truck": {
            "accel": clamp_float(truck_accel, DEFAULT_VTYPE["truck"]["accel"], 0.1, 10.0),
            "decel": clamp_float(truck_decel, DEFAULT_VTYPE["truck"]["decel"], 0.1, 10.0),
            "sigma": clamp_float(truck_sigma, DEFAULT_VTYPE["truck"]["sigma"], 0.0, 1.0),
            "length": clamp_float(truck_length, DEFAULT_VTYPE["truck"]["length"], 0.5, 100.0),
            "maxSpeed": clamp_float(truck_maxspeed, DEFAULT_VTYPE["truck"]["maxSpeed"], 1.0, 60.0),
        },
        "bus": {
            "accel": clamp_float(bus_accel, DEFAULT_VTYPE["bus"]["accel"], 0.1, 10.0),
            "decel": clamp_float(bus_decel, DEFAULT_VTYPE["bus"]["decel"], 0.1, 10.0),
            "sigma": clamp_float(bus_sigma, DEFAULT_VTYPE["bus"]["sigma"], 0.0, 1.0),
            "length": clamp_float(bus_length, DEFAULT_VTYPE["bus"]["length"], 0.5, 100.0),
            "maxSpeed": clamp_float(bus_maxspeed, DEFAULT_VTYPE["bus"]["maxSpeed"], 1.0, 60.0),
        },
        "streetcar": {
            "accel": clamp_float(streetcar_accel, DEFAULT_VTYPE["streetcar"]["accel"], 0.1, 10.0),
            "decel": clamp_float(streetcar_decel, DEFAULT_VTYPE["streetcar"]["decel"], 0.1, 10.0),
            "sigma": clamp_float(streetcar_sigma, DEFAULT_VTYPE["streetcar"]["sigma"], 0.0, 1.0),
            "length": clamp_float(
                streetcar_length, DEFAULT_VTYPE["streetcar"]["length"], 0.5, 100.0
            ),
            "maxSpeed": clamp_float(
                streetcar_maxspeed, DEFAULT_VTYPE["streetcar"]["maxSpeed"], 1.0, 60.0
            ),
        },
    }

    result = generate_scenario(
        network_store=network_store,
        selected_locations=location_values or [],
        max_match_distance=clamp_float(max_match_distance, 180.0, 1.0, 5000.0),
        date_policy=str(date_policy or "latest_per_location"),
        selected_date=selected_date,
        date_range_start=date_range_start,
        date_range_end=date_range_end,
        time_window_start=clamp_int(time_window_start, 8 * 60, 0, 24 * 60 - 1),
        time_window_duration=clamp_int(time_window_duration, 60, 15, 24 * 60),
        simulation_begin=clamp_int(sim_begin, 0, 0, 100000000),
        simulation_end=clamp_int(sim_end, 3600, 1, 100000000),
        strict_route_check=bool(strict_route_check and "strict" in strict_route_check),
        min_count_threshold=clamp_int(min_count_threshold, 1, 0, 100000),
        global_demand_scale=clamp_float(global_demand_scale, 1.0, 0.0, 10.0),
        included_modes=included_modes,
        mode_scales=mode_scales,
        streetcar_share_from_bus=clamp_float(streetcar_share, 0.35, 0.0, 1.0),
        controller_mode=str(controller_mode or "mappo_custom"),
        include_cluster_signals=bool(
            include_cluster_signals and "include" in include_cluster_signals
        ),
        fixed_signal=fixed_signal,
        max_pressure_signal=max_pressure_signal,
        mappo_hyperparams=mappo_hyperparams,
        heading_ranges=heading_ranges,
        vtype_settings=vtype_settings,
        run_validation=bool(run_sumo_check and "enabled" in run_sumo_check),
    )

    artifacts = build_generation_artifacts(result)
    return (
        build_generation_status(result),
        build_output_paths(result),
        artifacts,
        {"downloaded": False},
    )


@callback(
    Output("download-demand-button", "disabled"),
    Output("download-settings-button", "disabled"),
    Input("generated-artifacts-store", "data"),
    Input("demand-download-state", "data"),
)
def update_download_button_states(
    artifacts: dict[str, Any] | None,
    download_state: dict[str, Any] | None,
):
    has_artifacts = bool(artifacts and artifacts.get("demand_abs"))
    downloaded = bool((download_state or {}).get("downloaded"))
    return (not has_artifacts, not (has_artifacts and downloaded))


@callback(
    Output("demand-file-download", "data"),
    Output("demand-download-state", "data"),
    Input("download-demand-button", "n_clicks"),
    State("generated-artifacts-store", "data"),
    prevent_initial_call=True,
)
def download_demand_file(
    n_clicks: int | None,
    artifacts: dict[str, Any] | None,
):
    if not n_clicks or not artifacts:
        raise PreventUpdate

    demand_path_raw = artifacts.get("demand_abs")
    if not demand_path_raw:
        raise PreventUpdate

    demand_path = Path(str(demand_path_raw))
    if not demand_path.exists():
        raise PreventUpdate

    return dcc.send_file(str(demand_path), filename="demand.xml"), {"downloaded": True}


@callback(
    Output("settings-txt-download", "data"),
    Input("download-settings-button", "n_clicks"),
    State("generated-artifacts-store", "data"),
    State("demand-download-state", "data"),
    prevent_initial_call=True,
)
def download_confirmation_text(
    n_clicks: int | None,
    artifacts: dict[str, Any] | None,
    download_state: dict[str, Any] | None,
):
    if not n_clicks or not artifacts:
        raise PreventUpdate

    downloaded = bool((download_state or {}).get("downloaded"))
    if not downloaded:
        raise PreventUpdate

    confirmation_text = artifacts.get("confirmation_text")
    if not confirmation_text:
        raise PreventUpdate

    filename = artifacts.get("confirmation_filename") or "demand_settings_confirmation.txt"
    return {
        "content": str(confirmation_text),
        "filename": str(filename),
        "type": "text/plain",
    }


@callback(
    Output("confirmation-screen", "children"),
    Input("generated-artifacts-store", "data"),
    Input("demand-download-state", "data"),
)
def render_confirmation_screen(
    artifacts: dict[str, Any] | None,
    download_state: dict[str, Any] | None,
):
    downloaded = bool((download_state or {}).get("downloaded"))
    return build_confirmation_screen(artifacts, downloaded)


if __name__ == "__main__":
    default_port = clamp_int(os.environ.get("TANGO_DEMAND_PORT", "8050"), 8050, 1024, 65535)
    app.run(host="127.0.0.1", port=default_port, debug=False)
