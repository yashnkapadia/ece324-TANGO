# TANGO — Traffic Adaptive Network Guidance & Optimization

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Real-time adaptive signal control (ASCE) and scenario planning (PIRA) for a Toronto corridor, trained with Multi-Agent PPO in SUMO.

> **Status:** ASCE is complete and beats Max-Pressure on every evaluated scenario. PIRA is deferred &mdash; see [`PIRA.md`](./PIRA.md) and final report Section 4.6. The full write-up lives at [`reports/final/final_report-TANGO.pdf`](reports/final/final_report-TANGO.pdf).

## Overview

This document covers the **Adaptive Signal Control Engine (ASCE)** pipeline:

1. **Train** a shared-parameter action-gate residual MAPPO policy across all signalized intersections.
2. **Evaluate** MAPPO against Max-Pressure, SUMO's Actuated Default, and a naive Fixed-Time baseline.
3. **Generate** reproducible report figures from evaluation artifacts.

The benchmark uses a Toronto OSM (OpenStreetMap) corridor with Turning Movement Counts (TMC) calibrated demand and four distinct demand scenarios generated through the [Demand Studio](./DATA.md) interface (AM Peak, PM Peak, Demand Surge, Midday Multimodal). Demand includes cars, trucks, buses, streetcars, and pedestrians.

## Current Toronto Setup

| Property | Value |
|---|---|
| Network | Toronto OSM corridor, 12 signalized intersections (8 named on Dundas) |
| Demand | 4 TMC-calibrated scenarios in `sumo/demand/curriculum/` (`am_peak`, `pm_peak`, `demand_surge`, `midday_multimodal`) |
| Modes | Cars, trucks, buses, streetcars, pedestrians |
| Simulation | 900–1200 s episodes, delta-time = 5 s |
| Training | 200 ep warm-start (AM Peak only) + 600 ep curriculum (4 scenarios, 8 parallel libsumo workers); `person_objective` reward; obs normalisation; action-gate residual on top of Max-Pressure |
| Evaluation | 5 random seeds per controller per scenario (MAPPO, Max-Pressure, Actuated Default, Fixed-Time) |

## Quick Start (requires pixi)

```bash
pixi install

# Phase 1 — warm-start MAPPO on AM Peak (200 episodes, single worker)
pixi run warm-start-asce

# Phase 2 — curriculum across all four scenarios (600 episodes, 8 workers)
pixi run train-asce-curriculum

# Single-scenario eval against the headline checkpoint (override --route-file
# to switch scenarios; see pixi.toml for the exact command)
pixi run eval-asce-curriculum-best

# Full 5-seed × 4-scenario × 4-controller eval matrix
pixi run eval-matrix

# Rebuild the figures used in the final report (training curve, gate fraction,
# bar chart) from the eval CSVs
pixi run generate-report-figures
```

The headline checkpoint &mdash; reported in Table 1 of the final report &mdash; is `models/asce_mappo_curriculum_best.pt` (curriculum episode 432 / total episode 632, selected by minimum worst-case MAPPO/MP ratio across the four scenarios). See [`models/README.md`](models/README.md) for the rest of the canonical checkpoint layout and what is archived under `models/_archive/`.

## Current Findings

The figures below come from `scripts/generate_report_figures.py` and the 5-seed eval matrix in `reports/results/eval_matrix/`:

![ASCE bar chart](reports/final/asce_bar_chart.pdf)

**Headline numbers (5 seeds, person-time-loss vs. Max-Pressure):**

| Scenario | MAPPO/MP | Δ vs. MP | Meets ≥10% target? |
|---|---|---|---|
| AM Peak | 0.899 | −10.1% | ✅ |
| PM Peak | 0.937 | −6.3% | ❌ |
| Demand Surge | 0.890 | −11.0% | ✅ |
| Midday Multimodal | 0.862 | −13.8% | ✅ |

**Key results:**

- **MAPPO beats every baseline on every scenario.** Across the four scenarios MAPPO has the lowest person-time-loss against Max-Pressure, SUMO's Actuated Default, *and* Fixed-Time. The relative ranking of the *baselines* shifts across scenarios (Max-Pressure is strongest on AM Peak / Midday; Actuated Default is strongest on PM / Surge), but MAPPO is consistently first.
- **Three of four scenarios meet the proposal's ≥10% reduction target.** PM Peak &mdash; the highest-volume scenario with a tidal reversal &mdash; falls short at 6.3%; we attribute this to insufficient curriculum allocation to the highest-volume scenario.
- **Fairness improves ~30% relative to Max-Pressure** (Jain index 0.43–0.52 for MAPPO vs. 0.34–0.40 for MP), though no controller reaches the proposal's aspirational target of 0.8.
- **Best-checkpointing is essential.** The final episode-600 model is uniformly worse than the episode-432 best (regression after curriculum gradient interference).
- **The action gate learns selective override**, rising from ~2% (initialization bias) to ~30% by end of training and peaking during demand transitions &mdash; consistent with the hypothesis that Max-Pressure is near-optimal under stationary demand but suboptimal during transients.

See final report Sections 4.4 (Results) and 4.5 (Analysis) for the full discussion, and Appendix F for the complete metrics table.

## Why Action-Gate MAPPO Beats Max-Pressure

Max-Pressure grants green to the phase with the highest queue differential, which is provably throughput-optimal under stationary single-commodity demand. The action-gate residual architecture starts the policy *at* Max-Pressure (gate bias initialised so the agent follows MP ~98% of the time at episode 0) and learns to override it only where MP's local greedy heuristic is suboptimal &mdash; specifically:

- **Person occupancy.** MP cannot see that a single bus carries 30× the people of a car; the `person_objective` reward and the learned override do.
- **Demand transitions.** MP reacts to current queues but cannot anticipate a surge that will arrive in 30 seconds; the override rate spikes during the demand-surge onset and tidal reversal.
- **Multimodal corridors.** Pedestrian-heavy and transit-heavy scenarios (Spadina at midday) are exactly where MP's vehicle-count-based heuristic underweights person-delay.

## Limitations and Future Work

These are the items called out in the final report's Limitations and Future Actions sections:

1. **PM Peak still misses the 10% target.** More principled curriculum strategies (prioritised replay across scenarios, multi-task loss weighting) may help.
2. **Round-robin curriculum is unstable.** Reward oscillates as scenario-specific gradients conflict; best-checkpointing mitigates but does not solve this.
3. **Fixed-Time baseline is unrealistic.** A 30 s uniform cycle should be replaced with the City of Toronto's actual fixed-time plans.
4. **Safety constraints are not enforced** on MAPPO's phase selections (minimum green, maximum green, pedestrian recall). Required for any real-world deployment.
5. **PIRA scenario-planning surrogate is incomplete.** Architecture and training loop exist; real ASCE-rollout dataset integration is deferred. See [`PIRA.md`](./PIRA.md).

## ASCE files

Files relevant to the ASCE training, evaluation, and reporting pipeline:

```
ece324-TANGO/
├── ASCE.md                                           ← this document
├── pixi.toml                                         ← env + task definitions
│
├── ece324_tango/
│   ├── asce/
│   │   ├── env.py                ← SUMO env factory, obs flatten/pad helpers
│   │   ├── mappo.py              ← Actor / GatedActor / Critic, PPO update
│   │   ├── baselines.py          ← FixedTimeController, MaxPressureController
│   │   ├── kpi.py                ← Per-episode KPI tracker
│   │   ├── traffic_metrics.py    ← Person-weighted delay/throughput, Jain
│   │   ├── obs_norm.py           ← Welford running normaliser
│   │   ├── runtime.py            ← Step-output normalisation, phase helpers
│   │   ├── schema.py             ← ASCE_DATASET_COLUMNS contract
│   │   └── trainers/
│   │       ├── base.py                ← TrainConfig / EvalConfig dataclasses
│   │       ├── factory.py             ← Backend selection
│   │       ├── local_mappo_backend.py ← Episode loop, parallel eval, ckpt I/O
│   │       ├── noise_control.py       ← Quiet-output context manager
│   │       └── training_tui.py        ← rich.Live training dashboard
│   ├── modeling/
│   │   ├── train.py              ← `python -m ece324_tango.modeling.train`
│   │   └── predict.py            ← `python -m ece324_tango.modeling.predict`
│   ├── sumo_rl/                  ← Vendored sumo-rl with native-TLS patches
│   ├── plots.py                  ← Interim figure generation
│   └── config.py                 ← PROJ_ROOT, MODELS_DIR, RESULTS_DIR
│
├── scripts/
│   ├── generate_curriculum.py    ← Build the four curriculum scenarios
│   ├── eval_matrix.sh            ← Full 5-seed × 4-scenario × 4-controller sweep
│   ├── eval_nema.py              ← Standalone Actuated Default eval driver
│   └── generate_report_figures.py ← Bar chart, training curve, gate fraction
│
├── sumo/
│   ├── network/
│   │   ├── osm.net.xml.gz        ← Toronto Dundas corridor SUMO network
│   │   └── tls_overrides.add.xml.gz ← Native TLS programs preserved by sumo-rl
│   └── demand/curriculum/        ← am_peak / pm_peak / demand_surge / midday_multimodal .rou.xml
│
├── models/
│   ├── README.md                          ← canonical-checkpoint layout
│   ├── asce_mappo_curriculum_best.pt      ← HEADLINE checkpoint (ep 632)
│   ├── asce_mappo_curriculum.pt           ← Final ep-600 (regression comparison)
│   ├── asce_mappo_curriculum_best_{am,pm,demand_surge,midday_multimodal}.pt
│   ├── asce_mappo_curriculum_train_state.json
│   └── _archive/20260406_pre_final_report/  ← stale experimental ckpts
│
├── reports/
│   ├── final/
│   │   ├── final_report-TANGO.tex / .pdf
│   │   ├── references.bib / neurips_2024.sty
│   │   ├── asce_architecture.tex          ← TikZ architecture diagram
│   │   ├── asce_bar_chart.pdf             ← Figure: bar chart by scenario
│   │   ├── asce_training_curve.pdf        ← Figure: reward per episode
│   │   └── asce_gate_fraction.pdf         ← Figure: override fraction
│   └── results/eval_matrix/
│       ├── asce_mappo_curriculum_best__{scenario}.csv  ← 5-seed headline matrix
│       ├── asce_mappo_curriculum__{scenario}.csv       ← Final-ep regression
│       ├── nema__{scenario}.csv                        ← Actuated Default seeds 1000-1004
│       └── nema_v2__{scenario}.csv                     ← Actuated Default verification
│
└── tests/                                ← Unit and regression tests
```
