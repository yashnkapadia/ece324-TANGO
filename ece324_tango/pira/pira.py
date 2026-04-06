"""
PIRA - Planning Infrastructure Response Analyzer
Second layer of the TANGO traffic optimization system.

PIRA is a graph neural network surrogate that predicts traffic impact metrics
and timing recommendations from scenario-conditioned intersection graphs,
enabling fast "what-if" analysis without re-running SUMO simulations.

Inputs:
  - Per-intersection traffic state (from ASCE rollout logs in Parquet)
  - Scenario descriptors (disruption type, affected edges, demand changes)
  - Network topology (from SUMO osm.net.xml)

Outputs:
  - Impact metrics per intersection: delay, throughput, queue_total
  - Timing recommendations per intersection: NS and EW green durations

Success criteria (from TANGO proposal):
  - MAPE <= 10%
  - R^2  >= 0.85
  - Inference < 5 seconds per scenario
"""

import xml.etree.ElementTree as ET
import time
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# All 10 per-intersection features from the ASCE rollout schema (Table 1)
NODE_FEATURE_COLUMNS = [
    "queue_ns",
    "queue_ew",
    "arrivals_ns",
    "arrivals_ew",
    "avg_speed_ns",
    "avg_speed_ew",
    "current_phase",
    "time_of_day",
    "action_phase",
    "action_green_dur",
]

TARGET_COLUMNS = ["delay", "throughput", "queue_total"]

# Timing head outputs: recommended green durations for NS and EW phases
NUM_TIMING_OUTPUTS = 2

# Scenario disruption categories
DISRUPTION_TYPES = [
    "none",
    "construction",
    "lane_closure",
    "transit_line",
    "major_event",
]

# Scenario feature vector length: one-hot disruption + capacity + demand
SCENARIO_FEAT_DIM = len(DISRUPTION_TYPES) + 2  # 7


# ===========================================================================
# 1. SUMO Network Parsing
# ===========================================================================
def parse_sumo_network(xml_file_path):
    """
    Parse a SUMO ``osm.net.xml`` file and return:

    * edge_index  – ``[2, E]`` tensor of directed edges
    * node_id_to_idx – ``{junction_id: int}`` mapping
    * edge_meta   – list of dicts carrying per-edge attributes
                    (from_id, to_id, src_idx, dst_idx, num_lanes,
                     speed_limit, length)
    """
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    node_id_to_idx = {}
    idx = 0
    for junction in root.findall("junction"):
        j_id = junction.get("id")
        if j_id and j_id not in node_id_to_idx:
            node_id_to_idx[j_id] = idx
            idx += 1

    src_list, dst_list = [], []
    edge_meta = []

    for edge_el in root.findall("edge"):
        from_id = edge_el.get("from")
        to_id = edge_el.get("to")
        if from_id not in node_id_to_idx or to_id not in node_id_to_idx:
            continue

        src = node_id_to_idx[from_id]
        dst = node_id_to_idx[to_id]
        src_list.append(src)
        dst_list.append(dst)

        lanes = edge_el.findall("lane")
        num_lanes = len(lanes) if lanes else 1
        speed = float(lanes[0].get("speed", "13.89")) if lanes else 13.89
        length = float(lanes[0].get("length", "100.0")) if lanes else 100.0

        edge_meta.append(
            {
                "from_id": from_id,
                "to_id": to_id,
                "src_idx": src,
                "dst_idx": dst,
                "num_lanes": num_lanes,
                "speed_limit": speed,
                "length": length,
            }
        )

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    return edge_index, node_id_to_idx, edge_meta


# ===========================================================================
# 2. Scenario Descriptors
# ===========================================================================
class ScenarioDescriptor:
    """Encodes a single infrastructure scenario into tensors.

    Parameters
    ----------
    scenario_id : str
        Unique identifier (e.g. ``"construction_01"``).
    disruption_type : str
        One of :data:`DISRUPTION_TYPES`.
    affected_edge_indices : list[int]
        Indices into the base ``edge_index`` that are affected.
    capacity_reduction : float
        0.0 = full closure, 1.0 = no change.
    demand_multiplier : float
        Multiplicative demand shift (1.0 = baseline).
    """

    def __init__(
        self,
        scenario_id,
        disruption_type="none",
        affected_edge_indices=None,
        capacity_reduction=1.0,
        demand_multiplier=1.0,
    ):
        self.scenario_id = scenario_id
        self.disruption_type = disruption_type
        self.affected_edge_indices = affected_edge_indices or []
        self.capacity_reduction = capacity_reduction
        self.demand_multiplier = demand_multiplier

    def to_feature_vector(self):
        """Return a ``[SCENARIO_FEAT_DIM]`` tensor."""
        type_oh = [0.0] * len(DISRUPTION_TYPES)
        if self.disruption_type in DISRUPTION_TYPES:
            type_oh[DISRUPTION_TYPES.index(self.disruption_type)] = 1.0
        return torch.tensor(
            type_oh + [self.capacity_reduction, self.demand_multiplier],
            dtype=torch.float,
        )

    def apply_to_graph(self, edge_index):
        """Return (modified_edge_index, keep_mask).

        Full closures (``capacity_reduction == 0``) drop affected edges.
        Partial closures keep all edges (capacity encoded in features).
        """
        if not self.affected_edge_indices:
            return edge_index, torch.ones(edge_index.shape[1], dtype=torch.bool)

        keep = torch.ones(edge_index.shape[1], dtype=torch.bool)
        if self.capacity_reduction == 0.0:
            for i in self.affected_edge_indices:
                if i < edge_index.shape[1]:
                    keep[i] = False
        return edge_index[:, keep], keep


# ===========================================================================
# 3. Dataset Construction
# ===========================================================================
def infer_scenario_metadata(parquet_path):
    """Build default :class:`ScenarioDescriptor` objects from scenario IDs."""
    df = pd.read_parquet(parquet_path, columns=["scenario_id"])
    metadata = {}
    for sid in df["scenario_id"].unique():
        s = str(sid).lower()
        if "construction" in s:
            dtype = "construction"
        elif "closure" in s or "lane" in s:
            dtype = "lane_closure"
        elif "transit" in s:
            dtype = "transit_line"
        elif "event" in s:
            dtype = "major_event"
        else:
            dtype = "none"
        metadata[sid] = ScenarioDescriptor(scenario_id=sid, disruption_type=dtype)
    return metadata


def create_graph_dataset(parquet_path, edge_index, node_id_to_idx, scenario_metadata=None):
    """Build a list of :class:`torch_geometric.data.Data` from Parquet logs.

    Each sample corresponds to one ``(scenario_id, time_step)`` snapshot.
    Scenario features are concatenated to every node's feature vector, and
    affected edges are removed from the graph when the scenario calls for a
    full road closure.
    """
    df = pd.read_parquet(parquet_path)
    num_nodes = len(node_id_to_idx)
    dataset = []

    for (sid, ts), grp in df.groupby(["scenario_id", "time_step"]):
        grp = grp.sort_values("intersection_id")

        # Build per-node feature matrix (zero-padded for missing nodes)
        x_np = np.zeros((num_nodes, len(NODE_FEATURE_COLUMNS)), dtype=np.float32)
        y_np = np.zeros((num_nodes, len(TARGET_COLUMNS)), dtype=np.float32)
        for _, row in grp.iterrows():
            iid = row["intersection_id"]
            if iid not in node_id_to_idx:
                continue
            ni = node_id_to_idx[iid]
            x_np[ni] = [row.get(c, 0.0) for c in NODE_FEATURE_COLUMNS]
            y_np[ni] = [row.get(c, 0.0) for c in TARGET_COLUMNS]

        x = torch.tensor(x_np)
        y = torch.tensor(y_np)

        # Scenario conditioning
        scenario = scenario_metadata.get(sid) if scenario_metadata else None
        if scenario:
            sf = scenario.to_feature_vector().unsqueeze(0).expand(num_nodes, -1)
            sc_edge_index, _ = scenario.apply_to_graph(edge_index)
        else:
            sf = torch.zeros(num_nodes, SCENARIO_FEAT_DIM)
            sc_edge_index = edge_index

        x = torch.cat([x, sf], dim=1)

        dataset.append(
            Data(
                x=x,
                edge_index=sc_edge_index,
                y=y,
                scenario_id=sid,
            )
        )

    return dataset


def split_by_scenario(dataset, test_ratio=0.2, seed=42):
    """Split so entire scenarios stay in train **or** test (no leakage)."""
    ids = list({d.scenario_id for d in dataset})
    rng = np.random.RandomState(seed)
    rng.shuffle(ids)
    cut = max(1, int(len(ids) * (1 - test_ratio)))
    train_ids = set(ids[:cut])
    test_ids = set(ids[cut:])
    train = [d for d in dataset if d.scenario_id in train_ids]
    test = [d for d in dataset if d.scenario_id in test_ids]
    return train, test


# ===========================================================================
# 4. Edge Lookup and Inference API
# ===========================================================================
def build_edge_lookup(edge_meta, intersection_map_csv=None):
    """Return a helper object for finding edge indices by junction or name.

    Parameters
    ----------
    edge_meta : list[dict]
        Output of :func:`parse_sumo_network`.
    intersection_map_csv : str, optional
        Path to the intersection-map CSV produced by the data pipeline
        (columns: ``intersection_name``, ``junction_id``).  When provided,
        edges can also be looked up by human-readable intersection name.

    Returns
    -------
    EdgeLookup
        Call ``lookup.find(from_junction, to_junction)`` or
        ``lookup.find_by_name(from_name, to_name)`` to get edge indices.
    """
    return EdgeLookup(edge_meta, intersection_map_csv)


class EdgeLookup:
    """Helper for mapping real-world intersection names to edge indices.

    Example
    -------
    >>> lookup = build_edge_lookup(edge_meta, 'intersection_map.csv')
    >>> # Find the edge going westbound from Spadina/Dundas
    >>> indices = lookup.find_by_name('Spadina Ave / Dundas St W',
    ...                               'Bathurst St / Dundas St W')
    >>> print(indices)  # e.g. [42]
    """

    def __init__(self, edge_meta, intersection_map_csv=None):
        self._meta = edge_meta
        self._name_to_junction = {}

        if intersection_map_csv:
            df = pd.read_csv(intersection_map_csv)
            # Expected columns: intersection_name, junction_id
            for _, row in df.iterrows():
                self._name_to_junction[row["intersection_name"]] = row["junction_id"]

    def find(self, from_junction_id, to_junction_id):
        """Return all edge indices connecting two SUMO junction IDs.

        Parameters
        ----------
        from_junction_id : str
            SUMO junction ID of the origin intersection.
        to_junction_id : str
            SUMO junction ID of the destination intersection.

        Returns
        -------
        list[int]
            Indices into ``edge_meta`` (and therefore into ``edge_index``).
        """
        return [
            i
            for i, e in enumerate(self._meta)
            if e["from_id"] == from_junction_id and e["to_id"] == to_junction_id
        ]

    def find_by_name(self, from_name, to_name):
        """Same as :meth:`find` but accepts human-readable intersection names.

        Requires ``intersection_map_csv`` to have been provided at construction.
        """
        if not self._name_to_junction:
            raise RuntimeError(
                "No intersection map loaded. Pass intersection_map_csv to " "build_edge_lookup()."
            )
        fj = self._name_to_junction.get(from_name)
        tj = self._name_to_junction.get(to_name)
        if fj is None:
            raise KeyError(f"Intersection not found in map: '{from_name}'")
        if tj is None:
            raise KeyError(f"Intersection not found in map: '{to_name}'")
        return self.find(fj, tj)

    def list_all(self):
        """Print all edges for manual inspection."""
        print(f"{'idx':>5}  {'from_junction':<35} {'to_junction':<35}")
        print("-" * 78)
        for i, e in enumerate(self._meta):
            print(f"{i:>5}  {e['from_id']:<35} {e['to_id']:<35}")

    def list_named(self):
        """Print all known intersection names (requires intersection_map_csv)."""
        if not self._name_to_junction:
            print("No intersection map loaded.")
            return
        for name, jid in sorted(self._name_to_junction.items()):
            print(f"  {name}  ->  {jid}")


def predict_scenario(
    model, scenario, baseline_traffic_state, edge_index, node_id_to_idx, device="cpu"
):
    """Run PIRA inference for a single scenario.

    This is the main entry point for interactive planning.  You do not need
    a pre-built dataset -- just a baseline traffic snapshot and a scenario
    descriptor.

    Parameters
    ----------
    model : PIRAModel
        A trained PIRA model.
    scenario : ScenarioDescriptor
        The infrastructure scenario to evaluate.
    baseline_traffic_state : dict or pd.DataFrame
        Current traffic state, keyed by ``intersection_id``.  Each entry
        must have the fields in :data:`NODE_FEATURE_COLUMNS`.
        Can be a ``dict[intersection_id -> dict[feature -> value]]`` or a
        single-time-step slice of the Parquet log.
    edge_index : torch.Tensor
        Base ``[2, E]`` edge index from :func:`parse_sumo_network`.
    node_id_to_idx : dict
        Junction-ID-to-integer mapping from :func:`parse_sumo_network`.
    device : str

    Returns
    -------
    dict with keys:
        ``impact``  – ``DataFrame`` of predicted delay/throughput/queue per
                      intersection.
        ``timing``  – ``DataFrame`` of recommended green durations per
                      intersection.
        ``elapsed_ms`` – inference time in milliseconds.
    """
    model = model.to(device)
    model.eval()

    num_nodes = len(node_id_to_idx)

    # Build node feature matrix from baseline state
    x_np = np.zeros((num_nodes, len(NODE_FEATURE_COLUMNS)), dtype=np.float32)

    if isinstance(baseline_traffic_state, pd.DataFrame):
        for _, row in baseline_traffic_state.iterrows():
            iid = row.get("intersection_id")
            if iid in node_id_to_idx:
                ni = node_id_to_idx[iid]
                x_np[ni] = [row.get(c, 0.0) for c in NODE_FEATURE_COLUMNS]
    else:
        # dict[intersection_id -> dict[feature -> value]]
        for iid, feats in baseline_traffic_state.items():
            if iid in node_id_to_idx:
                ni = node_id_to_idx[iid]
                x_np[ni] = [feats.get(c, 0.0) for c in NODE_FEATURE_COLUMNS]

    x = torch.tensor(x_np)

    # Append scenario features
    sf = scenario.to_feature_vector().unsqueeze(0).expand(num_nodes, -1)
    x = torch.cat([x, sf], dim=1)

    # Apply scenario to graph topology
    sc_edge_index, _ = scenario.apply_to_graph(edge_index)

    data = Data(x=x.to(device), edge_index=sc_edge_index.to(device))

    t0 = time.perf_counter()
    with torch.no_grad():
        impact, timing = model(data)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Build reverse mapping: idx -> junction_id
    idx_to_junction = {v: k for k, v in node_id_to_idx.items()}

    impact_np = impact.cpu().numpy()
    timing_np = timing.cpu().numpy()

    impact_df = pd.DataFrame(
        impact_np,
        columns=TARGET_COLUMNS,
        index=[idx_to_junction.get(i, f"node_{i}") for i in range(num_nodes)],
    )
    impact_df.index.name = "junction_id"

    timing_df = pd.DataFrame(
        timing_np,
        columns=["green_ns_s", "green_ew_s"],
        index=impact_df.index,
    )
    timing_df.index.name = "junction_id"

    print(
        f"Inference: {elapsed_ms:.1f} ms  "
        f"({'PASS' if elapsed_ms < 5000 else 'FAIL'} < 5 s target)"
    )
    return {"impact": impact_df, "timing": timing_df, "elapsed_ms": elapsed_ms}


# ===========================================================================
# 5. Model
# ===========================================================================
class PIRAModel(nn.Module):
    """Planning Infrastructure Response Analyzer.

    Architecture
    ------------
    * **Node encoder** – 2-layer MLP projecting ``(node_features + scenario_features)``
      into ``hidden_dim``.
    * **3 x GATConv** layers with multi-head attention, residual connections,
      layer normalisation and dropout.
    * **Impact head** – predicts ``[delay, throughput, queue_total]`` per node.
    * **Timing head** – recommends ``[green_ns, green_ew]`` durations per node
      (``Softplus`` ensures positive outputs).
    """

    def __init__(
        self,
        num_node_features,
        num_scenario_features=SCENARIO_FEAT_DIM,
        hidden_dim=128,
        num_heads=4,
        num_impact_targets=3,
        num_timing_outputs=NUM_TIMING_OUTPUTS,
        dropout=0.1,
    ):
        super().__init__()
        in_dim = num_node_features + num_scenario_features

        self.node_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        head_dim = hidden_dim // num_heads
        self.conv1 = GATConv(hidden_dim, head_dim, heads=num_heads, concat=True, dropout=dropout)
        self.conv2 = GATConv(hidden_dim, head_dim, heads=num_heads, concat=True, dropout=dropout)
        self.conv3 = GATConv(hidden_dim, head_dim, heads=num_heads, concat=True, dropout=dropout)

        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

        self.impact_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_impact_targets),
        )

        self.timing_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_timing_outputs),
            nn.Softplus(),  # durations must be positive
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.node_encoder(x)

        # Layer 1
        h = self.conv1(x, edge_index)
        x = self.ln1(h + x)
        x = F.relu(x)
        x = self.drop(x)

        # Layer 2
        h = self.conv2(x, edge_index)
        x = self.ln2(h + x)
        x = F.relu(x)
        x = self.drop(x)

        # Layer 3
        h = self.conv3(x, edge_index)
        x = self.ln3(h + x)
        x = F.relu(x)

        impact = self.impact_head(x)
        timing = self.timing_head(x)
        return impact, timing


# ===========================================================================
# 5. Training
# ===========================================================================
def curriculum_sort(dataset, baseline_num_edges):
    """Sort samples by scenario complexity (edges removed + demand shift)."""

    def _key(d):
        removed = baseline_num_edges - d.edge_index.shape[1]
        # scenario features are the last SCENARIO_FEAT_DIM cols of x
        demand = d.x[0, -1].item()  # demand_multiplier
        return removed + abs(demand - 1.0)

    return sorted(dataset, key=_key)


def train_pira(
    model,
    train_data,
    val_data,
    optimizer,
    epochs=100,
    batch_size=32,
    curriculum=True,
    device="cpu",
    patience=15,
):
    """Train with optional curriculum learning and early stopping."""
    model = model.to(device)
    impact_crit = nn.MSELoss()
    best_val = float("inf")
    wait = 0
    best_state = None

    for epoch in range(epochs):
        model.train()

        # Curriculum: ramp up data complexity over the first half
        if curriculum and epoch < epochs // 2:
            frac = min(1.0, 0.3 + 0.7 * epoch / (epochs // 2))
            n = max(1, int(len(train_data) * frac))
            epoch_data = train_data[:n]
        else:
            epoch_data = train_data

        loader = DataLoader(epoch_data, batch_size=batch_size, shuffle=True)
        total_loss = 0.0

        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            impact_pred, timing_pred = model(batch)
            loss_impact = impact_crit(impact_pred, batch.y)

            # Timing regularisation: encourage durations in [7, 60] seconds
            loss_timing = torch.mean(F.relu(7.0 - timing_pred)) + torch.mean(
                F.relu(timing_pred - 60.0)
            )

            loss = loss_impact + 0.1 * loss_timing
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_train = total_loss / len(loader)

        # Validation
        val_loss = _validate(model, val_data, batch_size, device)

        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:>3}/{epochs} | "
                f"train {avg_train:.4f} | val {val_loss:.4f} | "
                f"samples {len(epoch_data)}/{len(train_data)}"
            )

        if wait >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _validate(model, data, batch_size, device):
    model.eval()
    loader = DataLoader(data, batch_size=batch_size, shuffle=False)
    crit = nn.MSELoss()
    total = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred, _ = model(batch)
            total += crit(pred, batch.y).item()
    return total / max(len(loader), 1)


# ===========================================================================
# 6. Evaluation
# ===========================================================================
def evaluate_pira(model, test_data, batch_size=32, device="cpu"):
    """Evaluate against TANGO success criteria (MAPE, R^2, latency)."""
    model = model.to(device)
    model.eval()
    loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    preds, targets, timings = [], [], []
    wall = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            t0 = time.perf_counter()
            ip, tp = model(batch)
            wall += time.perf_counter() - t0
            n_batches += 1
            preds.append(ip.cpu())
            targets.append(batch.y.cpu())
            timings.append(tp.cpu())

    preds = torch.cat(preds)
    targets = torch.cat(targets)

    eps = 1e-8
    mape = torch.mean(torch.abs((targets - preds) / (torch.abs(targets) + eps))) * 100

    ss_res = torch.sum((targets - preds) ** 2, dim=0)
    ss_tot = torch.sum((targets - targets.mean(dim=0)) ** 2, dim=0)
    r2_each = 1 - ss_res / (ss_tot + eps)
    r2_all = 1 - ss_res.sum() / (ss_tot.sum() + eps)

    avg_ms = wall / max(n_batches, 1) * 1000

    print()
    print("=" * 55)
    print("  PIRA Evaluation Results")
    print("=" * 55)
    for i, name in enumerate(TARGET_COLUMNS):
        print(f"  {name:>15s}   R^2 = {r2_each[i].item():.4f}")
    print(f"\n  {'MAPE':>15s} = {mape.item():.2f}%    (target: <= 10%)")
    print(f"  {'R^2 overall':>15s} = {r2_all.item():.4f}   (target: >= 0.85)")
    print(f"  {'Inference':>15s} = {avg_ms:.1f} ms/batch (target: < 5000 ms)")
    print()
    ok_mape = mape.item() <= 10.0
    ok_r2 = r2_all.item() >= 0.85
    ok_time = avg_ms < 5000
    print(f"  MAPE  <= 10%  :  {'PASS' if ok_mape else 'FAIL'}")
    print(f"  R^2   >= 0.85 :  {'PASS' if ok_r2 else 'FAIL'}")
    print(f"  Latency < 5 s :  {'PASS' if ok_time else 'FAIL'}")
    print("=" * 55)

    return {
        "mape": mape.item(),
        "r2_overall": r2_all.item(),
        "r2_per_target": {
            TARGET_COLUMNS[i]: r2_each[i].item() for i in range(len(TARGET_COLUMNS))
        },
        "avg_inference_ms": avg_ms,
    }


# ===========================================================================
# 7. Synthetic Data (for testing without SUMO data)
# ===========================================================================
def generate_synthetic_dataset(num_nodes=8, num_scenarios=20, steps_per_scenario=50, seed=42):
    """Create a synthetic dataset to verify the pipeline end to end.

    Returns ``(dataset, base_edge_index)`` where *dataset* is a list of
    :class:`Data` objects and *base_edge_index* is the unmodified corridor
    graph.
    """
    rng = np.random.RandomState(seed)

    # Bidirectional corridor: 0-1-2-...-N
    fwd = list(range(num_nodes - 1))
    bwd = list(range(1, num_nodes))
    src = fwd + bwd
    dst = bwd + fwd
    base_ei = torch.tensor([src, dst], dtype=torch.long)

    dataset = []

    for s in range(num_scenarios):
        dtype = rng.choice(DISRUPTION_TYPES)
        demand = rng.uniform(0.5, 2.0)
        cap = rng.uniform(0.0, 1.0) if dtype != "none" else 1.0

        sc = ScenarioDescriptor(
            scenario_id=f"scenario_{s:03d}",
            disruption_type=dtype,
            affected_edge_indices=([rng.randint(0, base_ei.shape[1])] if dtype != "none" else []),
            capacity_reduction=cap,
            demand_multiplier=demand,
        )
        sc_ei, _ = sc.apply_to_graph(base_ei)
        sf = sc.to_feature_vector().unsqueeze(0).expand(num_nodes, -1)

        for t in range(steps_per_scenario):
            queue = rng.poisson(5 * demand, (num_nodes, 2)).astype(np.float32)
            arrivals = rng.poisson(8 * demand, (num_nodes, 2)).astype(np.float32)
            speed = (rng.uniform(5, 15, (num_nodes, 2)) / demand).astype(np.float32)
            phase = rng.randint(0, 4, (num_nodes, 1)).astype(np.float32)
            tod = np.full((num_nodes, 1), t / steps_per_scenario, dtype=np.float32)
            act_phase = rng.randint(0, 4, (num_nodes, 1)).astype(np.float32)
            act_green = rng.uniform(7, 45, (num_nodes, 1)).astype(np.float32)

            x_np = np.hstack([queue, arrivals, speed, phase, tod, act_phase, act_green])
            x = torch.cat([torch.tensor(x_np), sf], dim=1)

            # Targets with causal structure so the model can learn
            cap_safe = max(cap, 0.1)
            delay = queue.sum(1) * demand / cap_safe
            throughput = arrivals.sum(1) * min(cap, 1.0)
            q_total = queue.sum(1)
            y = torch.tensor(np.stack([delay, throughput, q_total], 1))

            dataset.append(
                Data(
                    x=x,
                    edge_index=sc_ei,
                    y=y,
                    scenario_id=sc.scenario_id,
                )
            )

    return dataset, base_ei


# ===========================================================================
# 8. CLI entry point
# ===========================================================================
def main():
    import argparse

    p = argparse.ArgumentParser(description="PIRA - Planning Infrastructure Response Analyzer")
    p.add_argument("--network", type=str, default=None, help="Path to SUMO osm.net.xml")
    p.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to Parquet dataset (data/final/dataset.parquet)",
    )
    p.add_argument("--synthetic", action="store_true", help="Use synthetic data for testing")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--no-curriculum", action="store_true")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--save", type=str, default="pira_model.pt")
    args = p.parse_args()

    device = (
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    print(f"Device: {device}")
    curriculum = not args.no_curriculum

    # ── Load data ──
    if args.synthetic:
        print("Generating synthetic dataset ...")
        dataset, base_ei = generate_synthetic_dataset()
    elif args.network and args.data:
        base_ei, node_map, edge_meta = parse_sumo_network(args.network)
        print(f"Network: {len(node_map)} nodes, {base_ei.shape[1]} edges")
        sc_meta = infer_scenario_metadata(args.data)
        dataset = create_graph_dataset(args.data, base_ei, node_map, sc_meta)
    else:
        print("Provide --network + --data, or use --synthetic")
        return

    print(f"Dataset: {len(dataset)} samples")

    train_set, test_set = split_by_scenario(dataset)
    if curriculum:
        train_set = curriculum_sort(train_set, base_ei.shape[1])
    print(f"Train: {len(train_set)} | Test: {len(test_set)}")

    # ── Model ──
    model = PIRAModel(
        num_node_features=len(NODE_FEATURE_COLUMNS),
        num_scenario_features=SCENARIO_FEAT_DIM,
        hidden_dim=args.hidden_dim,
    )
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = Adam(model.parameters(), lr=args.lr)

    # ── Train ──
    model = train_pira(
        model,
        train_set,
        test_set,
        optimizer,
        epochs=args.epochs,
        batch_size=args.batch_size,
        curriculum=curriculum,
        device=device,
    )

    # ── Evaluate ──
    results = evaluate_pira(model, test_set, batch_size=args.batch_size, device=device)

    # ── Save ──
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "num_node_features": len(NODE_FEATURE_COLUMNS),
                "num_scenario_features": SCENARIO_FEAT_DIM,
                "hidden_dim": args.hidden_dim,
            },
            "results": results,
        },
        args.save,
    )
    print(f"Saved to {args.save}")


if __name__ == "__main__":
    main()
