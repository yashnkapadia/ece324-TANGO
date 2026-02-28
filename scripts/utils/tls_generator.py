"""
Purpose
    Generate static, two-direction (EW/NS) TLS programs for all signalized intersections in the study network.
    The resulting programs are written to an additional file for use with SUMO.

Enforcing
    - Fixed 60 s cycle: EW green/yellow/all-red, then NS green/yellow/all-red
    - Yellow time: 3 s
    - All-red time: 2 s
    - Greens: EW=30 s, NS=20 s
    - Phase state lengths match the network's expected TLS signal index count

Baseline Plan (6 phases; two greens with intergreens)
    1) EW green, NS red
    2) EW yellow, NS red
    3) all-red
    4) NS green, EW red
    5) NS yellow, EW red
    6) all-red

Inputs
    - sumo/network/osm.net.xml.gz

Outputs
    - sumo/network/tls_overrides.add.xml.gz

Running: 
    python scripts\utils\tls_generator.py
    sumo-gui -n "sumo/network/osm.net.xml.gz" --additional-files "sumo/network/tls_overrides.add.xml.gz"
"""

import os
import math
import gzip
import sumolib
from lxml import etree

EW_GREEN = 30
NS_GREEN = 20
YELLOW = 3
ALL_RED = 2


def _edge_heading_degrees(edge) -> float | None:
    shape = edge.getShape()
    if len(shape) < 2:
        return None
    dx = shape[-1][0] - shape[-2][0]
    dy = shape[-1][1] - shape[-2][1]
    return math.degrees(math.atan2(dy, dx)) % 360


def _expected_state_size(tls_obj) -> int | None:
    programs = tls_obj.getPrograms() or {}
    if programs:
        prog = next(iter(programs.values()))
        phases = getattr(prog, "getPhases", lambda: [])()
        lengths = [len(p.state) for p in phases if hasattr(p, "state") and p.state is not None]
        if lengths:
            return max(lengths)

    links = tls_obj.getLinks() or {}
    if links:
        keys = list(links.keys())
        if keys:
            return max(keys) + 1

    conns = tls_obj.getConnections() or []
    if conns:
        idxs = [linkIdx for (_, _, linkIdx) in conns]
        return (max(idxs) + 1) if idxs else None

    return None


def _classify_signal_indices(tls_obj) -> tuple[set[int], set[int], set[int]]:
    ew, ns, excluded = set(), set(), set()
    links = tls_obj.getLinks() or {}

    for sig_idx, conn_list in links.items():
        if not conn_list:
            excluded.add(sig_idx)
            continue

        in_lane = conn_list[0][0]
        edge = in_lane.getEdge()

        func = edge.getFunction()
        if func in ("crossing", "walkingarea"):
            excluded.add(sig_idx)
            continue

        ang = _edge_heading_degrees(edge)
        if ang is None:
            ew.add(sig_idx)
            continue

        if (ang < 45) or (ang > 315) or (135 < ang < 225):
            ew.add(sig_idx)
        else:
            ns.add(sig_idx)

    return ew, ns, excluded


def _state_string(n_links: int, active: set[int], ch: str) -> str:
    s = ["r"] * n_links
    for i in active:
        if 0 <= i < n_links:
            s[i] = ch
    return "".join(s)


def main() -> None:
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    net_file = os.path.join(base_dir, "sumo", "network", "osm.net.xml.gz")

    net = sumolib.net.readNet(
        net_file,
        withInternal=True,
        withPrograms=True,
        withLatestPrograms=True,
        withPedestrianConnections=True,
    )

    root = etree.Element("additional")

    for tls_obj in net.getTrafficLights():
        tls_id = tls_obj.getID()
        n_links = _expected_state_size(tls_obj)
        if not n_links or n_links <= 0:
            continue

        ew, ns, excluded = _classify_signal_indices(tls_obj)

        all_indices = set(range(n_links))
        candidate = sorted(list(all_indices - excluded))

        if not ew or not ns:
            if len(candidate) >= 2:
                half = max(1, len(candidate) // 2)
                ew = set(candidate[:half])
                ns = set(candidate[half:]) if candidate[half:] else set(candidate[:half])
            else:
                continue

        all_red = "r" * n_links
        phases = [
            (EW_GREEN, _state_string(n_links, ew, "G")),
            (YELLOW,   _state_string(n_links, ew, "y")),
            (ALL_RED,  all_red),
            (NS_GREEN, _state_string(n_links, ns, "G")),
            (YELLOW,   _state_string(n_links, ns, "y")),
            (ALL_RED,  all_red),
        ]

        tl_elem = etree.SubElement(
            root,
            "tlLogic",
            id=tls_id,
            type="static",
            programID="baseline_fixed",
            offset="0",
        )

        for dur, st in phases:
            if len(st) != n_links:
                raise RuntimeError(f"{tls_id}: generated state length {len(st)} != expected {n_links}")
            etree.SubElement(tl_elem, "phase", duration=str(dur), state=st)

    out_path = os.path.join(base_dir, "sumo", "network", "tls_overrides.add.xml.gz")
    print(f"Writing TLS overrides to {out_path}...")
    xml_bytes = etree.tostring(root, pretty_print=True, xml_declaration=True, encoding="UTF-8")
    with gzip.open(out_path, "wb") as f:
        f.write(xml_bytes)


if __name__ == "__main__":
    main()