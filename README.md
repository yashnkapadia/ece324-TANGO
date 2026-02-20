# ece324-TANGO

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Traffic Adaptive Network Guidance & Optimization - real-time adaptive signal control and cenario planning module that evaluates how nearby projects (construction, lane closures, new public transit lines) alter demand/capacity to recommend signal timing/phasing updates 🕺 💃

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pixi.toml          <- Pixi configuration file for environment and dependency management
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         ece324_tango and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   ├── figures        <- Plots for reports
│   ├── proposal       <- Project proposal
│   ├── results        <- Testing/evaluation results
│   └── final          <- Final project deliverables
│
├── setup.cfg          <- Configuration file for flake8
│
├── tests              <- Unit tests for the project
│   └── test_data.py   <- Tests for data processing
│
└── ece324_tango         <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes ece324_tango a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Dataset schema validation tooling
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── asce                    <- ASCE env adapters, baselines, MAPPO, schema
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

## Quickstart (Pixi)

```bash
pixi install
pixi run train-asce
pixi run eval-asce
pixi run validate-asce-schema
pixi run benchmark-backends
```
