# ece324-TANGO

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Traffic Adaptive Network Guidance & Optimization - real-time adaptive signal control and scenario planning module that evaluates how nearby projects (construction, lane closures, new public transit lines) alter demand/capacity to recommend signal timing/phasing updates 🕺 💃

## Corridor for Data Simulation
This project will focus on signalized intersections along Dundas Street West in Toronto, Ontario, Canada, starting from the intersection at University Avenue to the intersection at Bathurst St (see picture below). Data will be simulated for around 12 intersections along this corridor. This corridor is chosen because TMC data is available for these intersections, and it runs along a streetcar route which will be useful for transit-focused scenarios for PIRA [see [proposal](reports/proposal/TANGO-proposal.pdf)].

![alt text](image.png)
This figure shows the initial chosen area on SUMO web wizard. The initial scenario generation, for this selected area, is done only for cars and pedestrians. 
## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── environment.yml    <- Conda environment file for reproducing the environment
├── requirements.txt   <- Pip requirements file for reproducing the environment
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
└── ece324_tango-model   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes ece324_tango a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

## Environment Setup

This project environment is reproducible using either Conda (recommended) or pip. 

### Option 1: Conda

Requires [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

```bash
conda env create -f environment.yml
conda activate tango
```

> **Note:** The environment includes PyTorch with CUDA 12.1 support. If you do not have a compatible GPU, remove the `pytorch-cuda` line from `environment.yml` before creating the environment and install the CPU-only PyTorch build instead.

### Option 2: Pip

Requires Python 3.11+ and a virtual environment.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

> **Note:** `requirements.txt` does not pin a specific PyTorch variant. Visit [pytorch.org](https://pytorch.org/get-started/locally/) to install the correct build for your platform before running the above command.
--------

