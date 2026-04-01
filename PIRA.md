# PIRA — Planning Infrastructure Response Analyzer

Second layer of the **TANGO** traffic optimization system.

PIRA is a graph neural network surrogate that answers "what-if" infrastructure
questions — *if we close this lane, what happens to delay and queue across the
corridor?* — without re-running a full SUMO simulation. Given a description of
a disruption and the current traffic state, it predicts per-intersection impact
metrics and recommends updated signal timing, in milliseconds instead of minutes.

---

## Files

| File | Purpose |
|---|---|
| `pira.py` | Model definition, training, evaluation, dataset construction |
| `planner.py` | User-facing interface for running scenarios on a trained model |

---

## Inputs Required

| File | What it is | Where it comes from |
|---|---|---|
| `osm.net.xml` | SUMO road network for the corridor | SUMO OSM Web Wizard |
| `data/final/dataset.parquet` | ASCE rollout logs (one row per intersection per timestep per scenario) | Output of the ASCE training pipeline |
| `intersection_map.csv` | Maps readable names → SUMO junction IDs *(optional)* | Data pipeline / manual |
| `pira_model.pt` | Trained PIRA checkpoint | Output of `pira.py` training |

### Parquet schema
Each row in the dataset must have these columns:

**Node features (inputs):** `queue_ns`, `queue_ew`, `arrivals_ns`, `arrivals_ew`,
`avg_speed_ns`, `avg_speed_ew`, `current_phase`, `time_of_day`,
`action_phase`, `action_green_dur`

**Targets:** `delay`, `throughput`, `queue_total`

**Keys:** `scenario_id`, `time_step`, `intersection_id`

### Intersection map schema (optional)
```
intersection_name, junction_id
Spadina Ave / Dundas St W, cluster_123456_789012
Bathurst St / Dundas St W, cluster_234567_890123
...
```

---

## Architecture

Each training sample is one `(scenario, time_step)` snapshot of the corridor,
represented as a directed graph where **nodes are intersections** and **edges
are road segments**.

**Node features** are a 17-dimensional vector:
- 10 traffic state features from the ASCE rollout logs (queues, arrivals,
  speeds, phase, time of day, action)
- 7 scenario features appended to every node: a one-hot encoding of the
  disruption type (`none / construction / lane_closure / transit_line /
  major_event`), remaining capacity (0–1), and demand multiplier

**Graph topology is dynamic per scenario.** For full road closures
(`capacity = 0`), the affected edges are physically removed from the graph
before the forward pass, so the GNN cannot route information through a
closed road.

**The model has four stages:**

1. **Node encoder** — 2-layer MLP (input → 128 → 128) that projects the
   concatenated traffic + scenario features into a shared hidden space.

2. **3 × GATConv message-passing layers** — each intersection aggregates
   information from its neighbours using multi-head attention (4 heads,
   hidden size 128). Residual connections and LayerNorm after each layer
   stabilise training. Dropout (p=0.1) is applied between layers.

3. **Impact head** — 2-layer MLP (128 → 64 → 3) predicting `delay`,
   `throughput`, and `queue_total` per intersection.

4. **Timing head** — 2-layer MLP (128 → 64 → 2) with a Softplus output,
   recommending `green_ns_s` and `green_ew_s` phase durations (always
   positive). Regularisation penalises outputs outside [7, 60] seconds.

**Training** uses MSE loss on the impact targets, Adam optimiser (lr=1e-3),
and gradient clipping (max norm 1.0). **Curriculum learning** is on by
default: for the first half of training, the model sees only the simplest
scenarios (fewest edges removed, demand closest to 1.0), with the full
dataset phased in linearly. **Early stopping** halts training when validation
loss has not improved for 15 consecutive epochs, restoring the best checkpoint.

**Dataset split** is done by scenario ID, not by timestep, so no scenario
leaks between train and test.

**Success criteria** (as per our proposal):
- MAPE ≤ 10% across 10 seeds
- R² ≥ 0.85
- Inference < 5 seconds per scenario

---

## Usage

### 1. Train a model

With real data:
```bash
python pira.py --network osm.net.xml \
               --data data/final/dataset.parquet \
               --epochs 100 \
               --hidden-dim 128 \
               --save pira_model.pt
```

With synthetic data (to verify the pipeline works before SUMO data is ready):
```bash
python pira.py --synthetic --epochs 50
```

Curriculum learning is on by default. Disable with `--no-curriculum`.

---

### 2. Run a scenario — interactive terminal

```bash
python planner.py
```

Prompts you for file paths, shows available intersections, then loops on
scenario definitions until you quit.

---

### 3. Run a scenario — in a script or notebook

```python
from planner import PIRAPlanner

planner = PIRAPlanner(
    model_path       = 'pira_model.pt',
    network_path     = 'osm.net.xml',
    data_path        = 'data/final/dataset.parquet',
    intersection_map = 'intersection_map.csv',  # optional but recommended
)

# See what intersection names to use
planner.list_intersections()

# Run a scenario
result = planner.run(
    disruption_type   = 'lane_closure',
    from_intersection = 'Spadina Ave / Dundas St W',  # road origin (eastern end)
    to_intersection   = 'Bathurst St / Dundas St W',  # road destination (western end)
    capacity          = 0.5,   # 0 = fully closed, 0.5 = one lane blocked, 1 = open
    demand            = 1.0,   # 1.0 = normal, 1.3 = 30% more rerouted traffic
)

result['impact']      # DataFrame: delay / throughput / queue per intersection
result['timing']      # DataFrame: recommended green_ns_s / green_ew_s per intersection
result['elapsed_ms']  # inference time
```

`result['impact']` is sorted worst-first by delay so the most affected
intersections appear at the top.

**`disruption_type` options:** `none`, `construction`, `lane_closure`,
`transit_line`, `major_event`

**Direction matters.** "Spadina → Bathurst" is westbound on Dundas.
"Bathurst → Spadina" is eastbound. Run `planner.list_intersections()` to see
exact names, and think of the segment as *traffic flowing from → to*.

---

## Dependencies

```
torch
torch_geometric
pandas
numpy
```

Install:
```bash
pip install torch torch_geometric pandas numpy
```
