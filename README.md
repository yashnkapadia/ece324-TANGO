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

## Team

Aryan Shrivastava &middot; Kotaro Murakami &middot; Yash Kapadia

