# ece324-TANGO

Traffic Adaptive Network Guidance &amp; Optimization &mdash; real-time adaptive signal control and scenario planning module that evaluates how nearby projects (construction, lane closures, new public transit lines) alter demand/capacity to recommend signal timing/phasing updates 🕺 💃

The full write-up is in [`reports/final/final_report-TANGO.pdf`](reports/final/final_report-TANGO.pdf). Headline result: the action-gate residual MAPPO controller (ASCE) beats Max-Pressure on person-time-loss across all four evaluated scenarios (AM Peak −10.1%, PM Peak −6.3%, Demand Surge −11.0%, Midday Multimodal −13.8%, 5 seeds each). PIRA (the scenario-planning surrogate) is deferred &mdash; see [`PIRA.md`](./PIRA.md) and final report Section 4.6.

## Documentation

- Simulating data for MAPPO training and the Demand Studio interface: [`DATA.md`](./DATA.md)
- Training, evaluating, and reproducing ASCE results: [`ASCE.md`](./ASCE.md)
- PIRA architecture, dataset, and deferred-work notes: [`PIRA.md`](./PIRA.md)

Each document carries its own setup and usage instructions. The full repository layout and reproduction commands also appear in [`ASCE.md`](./ASCE.md).

## Reproducing the headline numbers

```bash
pixi install                          # one environment for everything
pixi run eval-matrix                  # full 5-seed × 4-scenario × 4-controller sweep
pixi run generate-report-figures      # rebuild the figures used in the report
```

See [`ASCE.md`](./ASCE.md) for the training pipeline (warm-start + curriculum) and per-scenario eval recipes, and [`models/README.md`](models/README.md) for the canonical checkpoint layout.

## Presentation

[Project Presentation Link](https://yashnkapadia.github.io/ece324-TANGO/TANGO-presentation.html) (highlights below)

![alt text](pictures/image.png)
![alt text](pictures/image-1.png)
![alt text](pictures/pira_demo.gif)

## Repository layout

```
ece324-TANGO/
├── README.md             ← you are here
├── ASCE.md               ← Adaptive Signal Control Engine: training, eval, results
├── DATA.md               ← Data pipeline, TMC parsing, Demand Studio
├── PIRA.md               ← Planning Infrastructure Response Analyzer (deferred)
├── pixi.toml / pixi.lock ← single source of truth for the environment
├── environment.yml       ← legacy conda fallback (kept for older instructions)
├── requirements.txt      ← legacy pip fallback
├── pyproject.toml / setup.cfg / Makefile
│
├── ece324_tango/         ← project Python package
│   ├── asce/             ← MAPPO trainer, baselines, env, KPI, schema
│   ├── pira/             ← PIRA GNN model + planner (deferred)
│   ├── modeling/         ← train.py / predict.py CLI entry points
│   ├── sumo_rl/          ← vendored sumo-rl with native TLS-program patches
│   ├── plots.py / dataset.py / features.py / config.py
│
├── apps/demand_studio/   ← Dash web app for TMC → SUMO scenario generation
├── scripts/              ← data pipeline, training driver, eval matrix, figures
├── notebooks/            ← network/TMC inspection + PIRA exploration
│
├── sumo/                 ← network, TLS overrides, demand files (curriculum/)
├── data/                 ← raw TMC, processed CSVs, PIRA parquets
├── models/               ← canonical checkpoints (see models/README.md)
│
├── reports/
│   ├── final/            ← NeurIPS-format final report .tex, .bib, figures, PDF
│   ├── interim/          ← interim report
│   ├── proposal/         ← original proposal
│   ├── results/          ← training/eval CSVs
│   └── results/eval_matrix/  ← 5-seed × 4-scenario × 4-controller eval CSVs
│
├── pictures/             ← README/presentation imagery
└── tests/                ← unit and regression tests
```

Each top-level doc carries its own focused file tree: see [`DATA.md`](./DATA.md#data-pipeline-files), [`ASCE.md`](./ASCE.md#asce-files), and [`PIRA.md`](./PIRA.md#pira-files).

## Team

Aryan Shrivastava &middot; Kotaro Murakami &middot; Yash Kapadia

