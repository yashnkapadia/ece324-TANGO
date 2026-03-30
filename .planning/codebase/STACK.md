# Technology Stack

**Analysis Date:** 2026-03-29

## Languages

**Primary:**
- Python 3.12.12 - All source code, ML training, simulation, data processing

## Runtime

**Environment:**
- Linux 64-bit (conda-forge packages)
- CUDA 12.8 support built-in

**Package Manager:**
- Pixi (conda-based) - Defined in `pixi.toml`
- Lockfile: `pixi.lock` (present, fully locked)
- Python package manager: pip (included via pixi)

## Frameworks

**Core ML/RL:**
- PyTorch 2.7.1 (CUDA 12.6) - Neural network training and inference
  - Location: `ece324_tango/asce/mappo.py` - Actor/Critic networks
  - LibTorch 2.7.1 (C++ bindings included)

**Simulation & Control:**
- SUMO (Simulation of Urban Mobility) v1.26.0 - Traffic simulator via TraCI interface
  - Package: `sumo-rl>=1.4.5,<2` (PyPI)
  - Consumed in: `ece324_tango/asce/env.py` - Environment creation
  - Network files: `sumo/network/osm.net.xml` (OSM-derived, 29K lines)
  - Demand files: `sumo/demand/demand.rou.xml` (TMC-calibrated)

**CLI Framework:**
- Typer - Command-line interface building
  - Used in: `ece324_tango/modeling/train.py`, `ece324_tango/modeling/predict.py`

**Data & Analysis:**
- NumPy - Array computations, observation processing
- Pandas - CSV/data handling, metrics aggregation
  - Used in: `ece324_tango/plots.py` (evaluation result aggregation)
- Matplotlib - Figure generation
  - Used in: `ece324_tango/plots.py` (interim report figures)
- scikit-learn - Metrics and utilities

**Testing:**
- pytest - Test runner
  - Config: `pyproject.toml` with markers for slow and integration tests
  - Test dir: `tests/`

**Development:**
- black 26.1.0 - Code formatter (line-length: 99)
- isort 7.0.0 - Import sorting (profile: black)
- flake8 7.3.0 - Linter (config in `setup.cfg`)
- ruff >=0.15.4,<0.16 - Fast linter/formatter
- Jupyter/IPython - Interactive development

**Build/Install:**
- flit_core >=3.2,<4 - Build backend
- python-dotenv - Environment variable loading

**Parallel Computing:**
- OpenMPI >=5.0.8,<6 - Multi-agent distributed training (optional)

**Logging:**
- loguru - Structured logging with tqdm integration
  - Used in: `ece324_tango/config.py` (log output redirected to tqdm)

**Progress/Output:**
- tqdm - Progress bars
- IPython/Jupyter - Interactive notebooks

## Key Dependencies

**Critical:**
- `sumo-rl 1.4.5` - Multi-agent RL interface to SUMO
  - Provides `SumoEnvironment` with Gym-like API
  - Handles TraCI connection and action/observation spaces
- `pytorch-gpu 2.7.1` - GPU-accelerated tensor computation
  - Device detection: auto-resolves to CUDA if available
  - Fallback to CPU in `ece324_tango/asce/trainers/local_mappo_backend.py:38`
- `numpy` - Core numerical operations for observation padding, queue calculations
- `pandas` - Training/evaluation metrics CSV I/O

**Infrastructure:**
- `loguru` - Exception-safe logging via `ece324_tango/error_reporting.py`
- `typer` - Type-safe CLI argument parsing with help text
- `python-dotenv` - Environment configuration loading
- `sumolib 1.26.0` - SUMO utilities (imported transitively via sumo-rl)

## Configuration

**Environment:**
- Loaded via `python-dotenv` in `ece324_tango/config.py`
- No `.env` file present in repo (secrets not committed)
- Paths configured in `ece324_tango/config.py`:
  ```python
  DATA_DIR = PROJ_ROOT / "data"
  MODELS_DIR = PROJ_ROOT / "models"
  REPORTS_DIR = PROJ_ROOT / "reports"
  ```

**Build:**
- `pyproject.toml` - Flit-based package config, Black/isort/pytest settings
- `setup.cfg` - Flake8 configuration (E731, E266, E501, C901, W503 ignored)
- `pixi.toml` - Workspace definition, dependencies, and Pixi tasks

**Pixi Tasks Defined:**
- `train-asce` - Basic training on sample grid
- `train-asce-toronto-demand` - Full training pipeline (30 episodes, 300 s, objective reward)
- `eval-asce-toronto-demand` - Evaluation on Toronto network
- `validate-asce-schema` - Data validation via `ece324_tango.dataset`
- `plot-interim-figures` - Report figure generation via `ece324_tango.plots`

## Platform Requirements

**Development:**
- Linux 64-bit OS (WSL2 compatible, tested on Linux 6.6.87.2-microsoft-standard-WSL2)
- NVIDIA GPU with CUDA 12.x support (recommended for training)
- ~6GB disk for Pixi environment (.pixi/)
- ~3.5MB for SUMO network files (osm.net.xml.gz)

**Production/Evaluation:**
- Same as development (GPU optional for inference, slower on CPU)
- Docker containerization not currently in use

**GPU Support:**
- CUDA 12.8 declaratively specified
- CuDNN 9.10.2.21 included in lock file
- PyTorch auto-detects CUDA availability at runtime

## Dependencies by Module

**ece324_tango/asce/env.py:**
- sumo-rl, numpy

**ece324_tango/asce/mappo.py:**
- torch, numpy

**ece324_tango/modeling/train.py:**
- typer, loguru, torch (via LocalMappoBackend)

**ece324_tango/asce/trainers/local_mappo_backend.py:**
- numpy, pandas, torch, loguru, traci.exceptions (from sumo-rl)

**ece324_tango/plots.py:**
- matplotlib, pandas, numpy

**ece324_tango/config.py:**
- python-dotenv, loguru, pathlib, tqdm

---

*Stack analysis: 2026-03-29*
