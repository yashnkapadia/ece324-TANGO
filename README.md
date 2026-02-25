# ece324-TANGO

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Traffic Adaptive Network Guidance & Optimization - real-time adaptive signal control and scenario planning module that evaluates how nearby projects (construction, lane closures, new public transit lines) alter demand/capacity to recommend signal timing/phasing updates 🕺 💃

## Corridor for Data Simulation
This project will focus on signalized intersections along Dundas Street West in Toronto, Ontario, Canada, starting from the intersection at University Avenue to the intersection at Bathurst St (see picture below). Data will be simulated for 12 intersections along this corridor. This corridor is chosen because TMC data is available for these intersections, and it runs along a streetcar route which will be useful for transit-focused scenarios for PIRA [see [proposal](reports/proposal/TANGO-proposal.pdf)].

![alt text](docs/assets/images/sumo_web_wizard.png)
The figure above shows the initial chosen area on SUMO web wizard. The initial scenario generation, for this selected area, is done only for cars and pedestrians. The figure below shows the generated scenario for the selected area (the selected area is inspected using ```sumolib``` and annotated using ```matplotlib``` to highlight signalized intersections, in ```notebooks\01_inspect_network.ipynb```). The red markers indicate the locations of signalized intersections along the corridor. 
![alt text](docs/network_map.png)


## Project Organization

```
├── LICENSE                <- Open-source license if one is chosen
├── Makefile               <- Makefile with convenience commands like `make data` or `make train`
├── README.md              <- The top-level README for developers using this project
├── environment.yml        <- Conda environment file for reproducing the environment
├── requirements.txt       <- Pip requirements file for reproducing the environment
├── pixi.toml              <- Pixi configuration file for environment and dependency management
├── pyproject.toml         <- Project configuration file with package metadata for
│                             ece324_tango and configuration for tools like black
├── setup.cfg              <- Configuration file for flake8
│
├── data
│   ├── final              <- Final outputs for delivery or publication
│   ├── processed          <- The final, canonical data sets for modeling
│   │   └── intersection_map.csv   <- Mapping of intersection names to SUMO junction IDs
│   └── raw
│       └── tmc            <- Raw TMC (Traffic Monitoring Count) data
│
├── docs
│   ├── network_map.png            <- Annotated SUMO network map with signalized intersections
│   └── assets
│       └── images
│           └── sumo_web_wizard.png <- Screenshot of the SUMO web wizard area selection
│
├── models                 <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks              <- Jupyter notebooks for exploration and visualization
│   └── 01_inspect_network.ipynb   <- Loads the SUMO network, lists traffic-light junctions
│                                     and edges, and visualizes the network with signalized
│                                     intersections annotated on a matplotlib plot
│
├── references             <- Data dictionaries, manuals, and all other explanatory materials
│
├── reports                <- Generated analysis as HTML, PDF, LaTeX, etc.
│   ├── figures            <- Plots for reports
│   ├── proposal
│   │   └── TANGO-proposal.pdf    <- Project proposal document
│   ├── results            <- Testing/evaluation results
│   └── final              <- Final project deliverables
│
├── scripts                <- Standalone data-processing and utility scripts
│   ├── 03_map_intersections.py   <- Maps Toronto intersection names to SUMO junction IDs
│   │                                by matching WGS84 coordinates to the nearest junctions
│   │                                in the SUMO network; outputs intersection_map.csv
│   └── utils              <- Shared helper utilities for scripts
│
├── sumo                   <- SUMO simulation files
│   ├── config             <- SUMO configuration files
│   ├── demand             <- Traffic demand / route files
│   ├── network
│   │   ├── build.bat              <- Batch script to build/rebuild the SUMO network
│   │   ├── osm.net.xml.gz         <- Compressed SUMO network generated from OSM
│   │   ├── osm.poly.xml.gz        <- Compressed polygon (building/land-use) data
│   │   └── osm_bbox.osm.xml.gz    <- Raw OSM extract for the bounding box
│   ├── output
│   │   ├── baseline       <- Simulation output for the baseline scenario
│   │   └── scenarios      <- Simulation output for alternative scenarios
│   └── scenarios          <- Scenario definition files
│
├── tests                  <- Unit tests for the project
│   └── test_data.py       <- Tests for data processing
│
└── ece324_tango-model     <- Source code for use in this project
    ├── __init__.py                <- Makes ece324_tango a Python module
    ├── config.py                  <- Store useful variables and configuration
    ├── dataset.py                 <- Scripts to download or generate data
    ├── features.py                <- Code to create features for modeling
    ├── plots.py                   <- Code to create visualizations
    └── modeling
        ├── __init__.py
        ├── predict.py             <- Code to run model inference with trained models
        └── train.py               <- Code to train models
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

