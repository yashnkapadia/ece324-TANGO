# Housekeeping, Obs-Norm, Longer Training & GPU Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Clean up repo/docs so any agent can orient quickly, implement running observation normalization, batch actor inference for GPU efficiency, and update training configs for longer runs.

**Architecture:**
- `ObsRunningNorm` (Welford online) lives in `ece324_tango/asce/obs_norm.py`, saved in the model checkpoint.
- Normalization is applied to padded local obs and global obs before every actor/critic call; padded-zero dimensions naturally converge to mean≈0, std≈ε, so they stay 0 after normalization — no masking needed.
- Actor `act()` is batch-lifted so all N-agent observations are evaluated in a single GPU forward pass per step.

**Tech Stack:** Python 3.12, PyTorch (CUDA 12.6 / RTX 4070 Laptop), SUMO-RL, pixi

---

## Task 1: Commit — Toronto SUMO assets (already staged)

**Files:** `sumo/config/`, `sumo/demand/`, `sumo/network/`

**Step 1: Verify what is staged**

```bash
git status --short
```
Expected: 8 green `A ` lines for sumo/* files

**Step 2: Commit**

```bash
git commit -m "Add Toronto corridor SUMO assets (network, demand, config)"
```

---

## Task 2: Update .gitignore for generated artifacts

Generated model files and per-run result CSVs should not be committed (they bloat the repo and are reproduced by running tasks). The handoff documents, raw plan files, and any xuance reference dumps should also be excluded.

**Files:** `.gitignore`

**Step 1: Read current .gitignore**

Look at the last block of `.gitignore` (the `# Data` section near the top). Currently it ignores `/data/`, `/logs/`, and `/models/xuance_mappo/`.

**Step 2: Append entries**

Add to the `# Data` section at the top of `.gitignore`:

```
# Generated model checkpoints (reproduced via pixi tasks)
/models/*.pt

# Per-run result CSVs and sweeps
/reports/results/*.csv
/reports/results/toronto_sweep/
/reports/results/backend_compare/

# Unzipped SUMO network (regenerated from .gz)
/sumo/network/osm.net.xml

# Third-party reference dumps
/xuance_custom_refs/

# Editor
.vscode/
```

**Step 3: Remove already-tracked untracked files from git scope**

```bash
# These should not be tracked; .gitignore will cover future adds
# The models and result CSVs are untracked already (good), just verify
git status --short | grep "^??"
```

**Step 4: Commit**

```bash
git add .gitignore
git commit -m "Expand .gitignore: models, result CSVs, unzipped net, editor dirs"
```

---

## Task 3: Commit — Code fixes from previous session

The three critical MAPPO fixes (action masking, value bootstrap, per-agent reward) plus the new tests are all unstaged. Commit in two focused commits.

**Files:** `ece324_tango/asce/mappo.py`, `ece324_tango/asce/env.py`, `ece324_tango/asce/traffic_metrics.py`, `ece324_tango/asce/trainers/local_mappo_backend.py`, `tests/test_observation_padding.py`, `tests/test_local_backend_eval_guard.py`, `tests/test_traffic_metrics.py`, `pixi.toml`

**Step 1: Run tests first to confirm they pass**

```bash
pixi run pytest tests -q
```
Expected: `26 passed, 2 skipped`

**Step 2: Stage and commit the MAPPO fixes**

```bash
git add ece324_tango/asce/mappo.py \
        ece324_tango/asce/env.py \
        ece324_tango/asce/traffic_metrics.py \
        ece324_tango/asce/trainers/local_mappo_backend.py \
        pixi.toml
git commit -m "Fix three critical MAPPO bugs: action masking, value bootstrap, per-agent reward

- n_actions: min -> max(action_dims) with invalid-action masking via -inf logits.
  Toronto has 12 agents with 2-6 phases; using min=2 blocked 10 agents from 4-6 phases.
- _compute_gae: pass bootstrapped critic value at episode end for truncated episodes
  (fixed time limit -> last_value was hardcoded 0.0 causing biased GAE).
- Store per-agent shaped reward in each agent's Transition instead of global mean;
  restores credit assignment across heterogeneous intersections.
- pixi: toronto-demand episodes 1->10, train tasks use --reward-mode objective."
```

**Step 3: Stage and commit the new tests**

```bash
git add tests/test_observation_padding.py \
        tests/test_local_backend_eval_guard.py \
        tests/test_traffic_metrics.py
git commit -m "Add tests: observation padding, eval guard, lane-axis metric extraction"
```

**Step 4: Commit deleted scaffold package and doc updates**

```bash
git add -u ece324_tango_model/  # stages the deletions
git add README.md docs/notes/data_schema.md docs/notes/prototype_log.md docs/notes/runbook.md
git commit -m "Remove obsolete ece324_tango_model scaffold; update README and docs"
```

---

## Task 4: Write AGENTS.md — full codebase orientation for agents

Replace the current thin AGENTS.md with a comprehensive map that any agent can read to orient in under 5 minutes.

**Files:** `AGENTS.md`

**Step 1: Write the new AGENTS.md**

Replace the entire file with:

```markdown
# AGENTS — Codebase Orientation

## Project
TANGO: Traffic Adaptive Network Guidance & Optimization.
This repo implements the ASCE (Adaptive Signal Control Engine) MAPPO pipeline
for real-time traffic signal control on a Toronto corridor SUMO network.

## Quick-start commands (run all via pixi)
| Command | What it does |
|---|---|
| `pixi run pytest tests -q` | Full unit test suite (fast, no SUMO) |
| `pixi run train-asce-toronto-demand` | Train MAPPO on Toronto TMC demand |
| `pixi run eval-asce-toronto-demand` | Evaluate checkpoint vs baselines |
| `pixi run train-asce-toronto-random` | Train on random-trip demand |
| `pixi run benchmark-backends` | Multi-seed backend comparison (slow) |

See `docs/notes/runbook.md` for all CLI flags and artifact paths.

## Package layout

```
ece324_tango/
  asce/
    mappo.py              # Actor, Critic, MAPPOTrainer (PPO update, GAE, build_batch)
    obs_norm.py           # ObsRunningNorm: Welford online normalizer, saved in checkpoint
    env.py                # create_parallel_env(), pad_observation(), flatten_obs_by_agent()
    traffic_metrics.py    # TraCI metric extraction, reward shaping (objective/sumo modes)
    baselines.py          # FixedTimeController, MaxPressureController
    runtime.py            # extract_reset_obs/step, safe_done, jain_index
    kpi.py                # KPITracker: episode-level time_loss, person_time_loss, trip time
    schema.py             # ASCE_DATASET_COLUMNS — canonical CSV column list
    trainers/
      base.py             # TrainConfig, EvalConfig dataclasses (all CLI args live here)
      local_mappo_backend.py  # Main train/eval loop (LocalMappoBackend)
      benchmarl_backend.py    # BenchMARL MAPPO path
      xuance_backend.py       # Xuance MAPPO path
      libsignal_backend.py    # Placeholder (fail-fast)
      noise_control.py        # Quiet SUMO helper
  modeling/
    train.py              # CLI entry point: `python -m ece324_tango.modeling.train`
    predict.py            # CLI entry point: `python -m ece324_tango.modeling.predict`
    benchmark_backends.py # CLI: multi-seed backend comparison
  config.py               # PROJ_ROOT, MODELS_DIR, RESULTS_DIR, DATA_DIR
  dataset.py              # Schema validation CLI
  error_reporting.py      # report_exception() — non-fatal fallback logger
```

## Critical design decisions (read before changing anything)

### Action space (FIXED 2026-03-01)
`n_actions = max(action_dims.values())` — the actor outputs logits for the largest
action space. Agents with fewer phases get invalid logits masked to `-inf` before
sampling (see `MAPPOTrainer.act(n_valid_actions=...)`). **Never use `min`.**

### Value bootstrap (FIXED 2026-03-01)
Episodes end by time truncation. `build_batch(last_values=...)` takes a dict of
per-agent bootstrap values computed from the critic at the final observation.
`_compute_gae(last_value=0.0)` is only correct for truly terminal states.

### Per-agent rewards (FIXED 2026-03-01)
Each agent's `Transition.reward` is its own shaped reward, NOT the global mean.
`rewards_from_metrics()` returns per-agent values — use them directly.

### Observation normalization
`ObsRunningNorm` (Welford) normalizes per-feature across all observations seen
during training. Saved in checkpoint under key `"obs_norm"`. Applied to padded
local obs and to global obs before every actor/critic call. Padded-zero dimensions
stay 0 after normalization (their running mean → 0, std → ε).

### Reward modes
- `objective` (default): `-w_d * log1p(delay) + w_t * log1p(throughput) + w_f * fairness`
- `sumo`: raw sumo-rl rewards (usually negative accumulated waiting time)

### Toronto SUMO network
- 12 intersections with heterogeneous obs dims (9-38) and action spaces (2-6 phases).
- Net file: `sumo/network/osm.net.xml` (regenerate with `gzip -dk sumo/network/osm.net.xml.gz`).
- Demand: `sumo/demand/demand.rou.xml` (TMC-calibrated) or `random_trips.rou.xml`.
- Long-horizon runs (>300s) may hit `FatalTraCIError` on this network — known issue,
  root cause is in sumo-rl controllable TLS interaction.

### Proposal success criterion
`time_loss_s(MAPPO) <= 0.90 * time_loss_s(best_baseline)` over 10 seeds.
Use `person_time_loss_s` for the person-weighted version. Report from eval CSV.

## Key file cross-references
- `TrainConfig` / `EvalConfig`: `ece324_tango/asce/trainers/base.py` — add new CLI flags here.
- Backend registration: `ece324_tango/asce/trainers/__init__.py` — `get_backend()` factory.
- Dataset schema: `docs/notes/data_schema.md` + `ece324_tango/asce/schema.py`.
- Prototype log (chronological changes): `docs/notes/prototype_log.md`.
- ADRs: `docs/notes/adr/`.

## Test suite
`pixi run pytest tests -q` — 26 tests, 2 skipped (slow integration).
Slow integration tests: `RUN_SLOW_INTEGRATION=1 pixi run pytest tests/test_backend_integration_slow.py`
```

**Step 2: Commit**

```bash
git add AGENTS.md
git commit -m "Rewrite AGENTS.md as full codebase orientation for agents"
```

---

## Task 5: Update runbook.md

Add the new fixes, obs-norm flag, and GPU section to the runbook.

**Files:** `docs/notes/runbook.md`

**Step 1: Add new sections to runbook**

After the `## Reward Objective` section, add:

```markdown
## Observation Normalization
- Enabled by default when `--use-obs-norm` flag is passed (default: off until stable).
- `ObsRunningNorm` (Welford online, per-feature) is applied to padded local obs and global obs.
- Stats are saved under key `"obs_norm"` in the model checkpoint.
- Eval path loads and applies the same stats without updating them.

## GPU Notes
- RTX 4070 Laptop, CUDA 12.6. Use `--device auto` (default) to pick GPU automatically.
- All 12-agent observations are batched into a single GPU forward pass per step.
- Training throughput is SUMO-limited (CPU), not GPU-limited. Longer episodes give more data.
- To maximize data per wall-clock minute: increase `--episodes` rather than `--seconds`
  if the long-horizon SUMO crash recurs.

## Known Bugs Fixed (2026-03-01)
- **Action space crippled**: `n_actions` was `min(action_dims)` = 2; 10/12 agents could
  never select phases 2-5. Fixed: use `max` + `-inf` masking.
- **GAE bias on truncation**: `last_value` was hardcoded 0.0; now bootstrapped from critic.
- **Credit assignment lost**: all agents received global mean reward; now each gets own reward.
```

**Step 2: Commit**

```bash
git add docs/notes/runbook.md docs/notes/handoff_2026-03-02.md
git commit -m "Update runbook with obs-norm, GPU notes, and 2026-03-01 bug fixes"
```

---

## Task 6: Implement ObsRunningNorm

**Files:**
- Create: `ece324_tango/asce/obs_norm.py`
- Create: `tests/test_obs_norm.py`

**Step 1: Write the failing tests**

```python
# tests/test_obs_norm.py
import numpy as np
import pytest
from ece324_tango.asce.obs_norm import ObsRunningNorm


def test_normalize_converges_to_zero_mean_unit_variance():
    rng = np.random.default_rng(42)
    norm = ObsRunningNorm(dim=4)
    # Feed 1000 samples from a known distribution
    for _ in range(1000):
        x = rng.normal(loc=[2.0, -1.0, 0.0, 5.0], scale=[1.0, 2.0, 0.5, 3.0])
        norm.update(x)
    out = norm.normalize(np.array([2.0, -1.0, 0.0, 5.0], dtype=np.float32))
    # Should be close to zero (normalizing the mean value)
    np.testing.assert_allclose(out, 0.0, atol=0.1)


def test_padded_zeros_stay_zero():
    """Dimensions that are always 0 (padded) remain 0 after normalization."""
    norm = ObsRunningNorm(dim=4)
    rng = np.random.default_rng(0)
    for _ in range(500):
        x = np.array([rng.random(), rng.random(), 0.0, 0.0], dtype=np.float32)
        norm.update(x)
    out = norm.normalize(np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32))
    # Padded (always-zero) dims 2 and 3 must not blow up
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out[2:], 0.0, atol=1e-6)


def test_state_dict_roundtrip():
    norm = ObsRunningNorm(dim=3)
    norm.update(np.array([1.0, 2.0, 3.0]))
    norm.update(np.array([2.0, 4.0, 6.0]))
    state = norm.state_dict()

    norm2 = ObsRunningNorm(dim=3)
    norm2.load_state_dict(state)
    x = np.array([1.5, 3.0, 4.5], dtype=np.float32)
    np.testing.assert_array_equal(norm.normalize(x), norm2.normalize(x))


def test_untrained_normalizer_is_passthrough():
    """Before any updates the normalizer should return the input unchanged (count=0)."""
    norm = ObsRunningNorm(dim=3)
    x = np.array([5.0, -3.0, 1.0], dtype=np.float32)
    out = norm.normalize(x)
    np.testing.assert_array_equal(out, x)
```

**Step 2: Run tests, confirm they fail**

```bash
pixi run pytest tests/test_obs_norm.py -v
```
Expected: `ModuleNotFoundError: No module named 'ece324_tango.asce.obs_norm'`

**Step 3: Implement ObsRunningNorm**

```python
# ece324_tango/asce/obs_norm.py
"""Online running observation normalizer using Welford's algorithm."""
from __future__ import annotations

import numpy as np


class ObsRunningNorm:
    """Per-feature Welford running mean/variance normalizer.

    Tracks statistics online across all observations seen during training.
    Padded-zero dimensions converge to mean≈0, std≈ε and stay 0 after
    normalization — no explicit masking needed.

    Save/load via state_dict() / load_state_dict() for checkpoint persistence.
    """

    def __init__(self, dim: int, eps: float = 1e-8):
        self.dim = dim
        self.eps = eps
        self._count: int = 0
        self._mean = np.zeros(dim, dtype=np.float64)
        self._M2 = np.zeros(dim, dtype=np.float64)  # sum of squared deviations

    def update(self, x: np.ndarray) -> None:
        """Update running stats with one observation vector (shape: [dim])."""
        x = np.asarray(x, dtype=np.float64).ravel()
        assert x.size == self.dim, f"Expected dim={self.dim}, got {x.size}"
        self._count += 1
        delta = x - self._mean
        self._mean += delta / self._count
        delta2 = x - self._mean
        self._M2 += delta * delta2

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Return normalized observation. If no data seen yet, returns x unchanged."""
        x = np.asarray(x, dtype=np.float32).ravel()
        if self._count < 2:
            return x
        var = self._M2 / (self._count - 1)
        std = np.sqrt(var + self.eps).astype(np.float32)
        mean = self._mean.astype(np.float32)
        return (x - mean) / std

    def state_dict(self) -> dict:
        return {
            "dim": self.dim,
            "eps": self.eps,
            "count": self._count,
            "mean": self._mean.copy(),
            "M2": self._M2.copy(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.dim = int(state["dim"])
        self.eps = float(state["eps"])
        self._count = int(state["count"])
        self._mean = np.asarray(state["mean"], dtype=np.float64)
        self._M2 = np.asarray(state["M2"], dtype=np.float64)
```

**Step 4: Run tests, confirm they pass**

```bash
pixi run pytest tests/test_obs_norm.py -v
```
Expected: `4 passed`

**Step 5: Commit**

```bash
git add ece324_tango/asce/obs_norm.py tests/test_obs_norm.py
git commit -m "Add ObsRunningNorm: Welford online per-feature normalizer with checkpoint support"
```

---

## Task 7: Wire ObsRunningNorm into MAPPOTrainer

The normalizer should live inside `MAPPOTrainer` and be applied to:
- Local padded obs before `actor` forward pass
- Global obs before `critic` forward pass
It should be enabled/disabled by a flag, and saved/loaded with the checkpoint.

**Files:**
- Modify: `ece324_tango/asce/mappo.py`

**Step 1: Add import and integrate into MAPPOTrainer.__init__**

In `mappo.py`, add at the top:
```python
from ece324_tango.asce.obs_norm import ObsRunningNorm
```

In `MAPPOTrainer.__init__`, add parameter `use_obs_norm: bool = False` and body:
```python
self.use_obs_norm = use_obs_norm
if use_obs_norm:
    self.obs_norm = ObsRunningNorm(obs_dim)
    self.gobs_norm = ObsRunningNorm(global_obs_dim)
else:
    self.obs_norm = None
    self.gobs_norm = None
```

**Step 2: Add norm_update() method**

```python
def norm_update(self, obs: np.ndarray, global_obs: np.ndarray) -> None:
    """Update running stats. Call once per transition during training."""
    if self.obs_norm is not None:
        self.obs_norm.update(obs)
    if self.gobs_norm is not None:
        self.gobs_norm.update(global_obs)
```

**Step 3: Apply normalization in act() — before actor/critic forward**

In `act()`, replace:
```python
obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
gobs_t = torch.tensor(global_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
```
With:
```python
obs_n = self.obs_norm.normalize(obs) if self.obs_norm is not None else np.asarray(obs, dtype=np.float32)
gobs_n = self.gobs_norm.normalize(global_obs) if self.gobs_norm is not None else np.asarray(global_obs, dtype=np.float32)
obs_t = torch.tensor(obs_n, dtype=torch.float32, device=self.device).unsqueeze(0)
gobs_t = torch.tensor(gobs_n, dtype=torch.float32, device=self.device).unsqueeze(0)
```

**Step 4: Apply normalization in update() — before PPO loss computation**

In `update()`, after loading tensors, add:
```python
if self.obs_norm is not None:
    obs_np = batch["obs"]          # already np.float32 [N, obs_dim]
    gobs_np = batch["global_obs"]  # [N, gobs_dim]
    obs_norm_np = np.stack([self.obs_norm.normalize(o) for o in obs_np])
    gobs_norm_np = np.stack([self.gobs_norm.normalize(g) for g in gobs_np])
    obs = torch.tensor(obs_norm_np, dtype=torch.float32, device=self.device)
    gobs = torch.tensor(gobs_norm_np, dtype=torch.float32, device=self.device)
```
(Place this block right after `adv = ...` normalization line.)

**Step 5: Save/load normalizer state in save()/load()**

In `save()`:
```python
payload = {
    "actor": self.actor.state_dict(),
    "critic": self.critic.state_dict(),
    "obs_norm": self.obs_norm.state_dict() if self.obs_norm is not None else None,
    "gobs_norm": self.gobs_norm.state_dict() if self.gobs_norm is not None else None,
}
```

In `load()`:
```python
if self.obs_norm is not None and payload.get("obs_norm") is not None:
    self.obs_norm.load_state_dict(payload["obs_norm"])
if self.gobs_norm is not None and payload.get("gobs_norm") is not None:
    self.gobs_norm.load_state_dict(payload["gobs_norm"])
```

**Step 6: Run full test suite**

```bash
pixi run pytest tests -q
```
Expected: `30 passed, 2 skipped` (26 original + 4 obs_norm tests)

**Step 7: Commit**

```bash
git add ece324_tango/asce/mappo.py
git commit -m "Wire ObsRunningNorm into MAPPOTrainer: normalize obs/gobs, persist in checkpoint"
```

---

## Task 8: Batch actor inference + wire obs-norm to backend

Two changes to `local_mappo_backend.py`:
1. Collect all agents' observations, run actor in a single batched GPU call
2. Expose `--use-obs-norm` CLI flag via `TrainConfig` / `EvalConfig`

**Files:**
- Modify: `ece324_tango/asce/trainers/local_mappo_backend.py`
- Modify: `ece324_tango/asce/trainers/base.py`
- Modify: `ece324_tango/modeling/train.py`
- Modify: `ece324_tango/modeling/predict.py`

**Step 1: Add use_obs_norm to TrainConfig and EvalConfig in base.py**

```python
use_obs_norm: bool = False
```
Add this field to both `TrainConfig` and `EvalConfig`.

**Step 2: Add use_obs_norm to MAPPOTrainer construction in train() and evaluate()**

In `local_mappo_backend.py`, pass `use_obs_norm=cfg.use_obs_norm` to both `MAPPOTrainer(...)` calls.

**Step 3: Add norm_update calls in the train loop**

In the training step loop, after collecting per-agent padded_obs/gobs, add:
```python
trainer.norm_update(padded_obs, gobs)
```
This must be called before `trainer.act(...)` so stats include the current observation.

**Step 4: Expose --use-obs-norm in train.py and predict.py CLI**

In `train.py` and `predict.py`, add Typer option:
```python
use_obs_norm: bool = typer.Option(False, "--use-obs-norm", help="Enable running obs normalization"),
```
And wire it through to `TrainConfig`/`EvalConfig`.

**Step 5: Batch actor inference**

Replace the per-agent loop in `train()`:
```python
# OLD: 12 sequential GPU calls
for agent in active_agents:
    padded = pad_observation(...)
    out = trainer.act(padded, gobs, ...)
```

With a batched call. Add `act_batch()` to `MAPPOTrainer`:
```python
@torch.no_grad()
def act_batch(
    self,
    obs_list: list[np.ndarray],   # [N, obs_dim] padded
    global_obs: np.ndarray,        # [gobs_dim]
    n_valid_actions_list: list[int],
) -> list[dict]:
    """Single forward pass for all N agents."""
    N = len(obs_list)
    obs_arr = np.stack([
        self.obs_norm.normalize(o) if self.obs_norm else np.asarray(o, dtype=np.float32)
        for o in obs_list
    ])
    obs_t = torch.tensor(obs_arr, dtype=torch.float32, device=self.device)  # [N, obs_dim]
    gobs_n = self.gobs_norm.normalize(global_obs) if self.gobs_norm else np.asarray(global_obs, dtype=np.float32)
    gobs_t = torch.tensor(gobs_n, dtype=torch.float32, device=self.device).unsqueeze(0).expand(N, -1)

    logits = self.actor(obs_t)  # [N, n_actions]
    for i, n_valid in enumerate(n_valid_actions_list):
        if n_valid < logits.shape[-1]:
            logits[i, n_valid:] = float("-inf")
    dist = Categorical(logits=logits)
    actions = dist.sample()          # [N]
    logps = dist.log_prob(actions)   # [N]
    values = self.critic(gobs_t)     # [N]

    return [
        {"action": int(actions[i].item()), "logp": float(logps[i].item()), "value": float(values[i].item())}
        for i in range(N)
    ]
```

Replace the per-agent loop in `train()` and `evaluate()` to use `act_batch()`.

**Step 6: Run full test suite**

```bash
pixi run pytest tests -q
```
Expected: all pass

**Step 7: Smoke-run training to verify obs-norm flag works**

```bash
pixi run python -m ece324_tango.modeling.train \
  --trainer-backend local_mappo \
  --net-file sumo/network/osm.net.xml \
  --route-file sumo/demand/demand.rou.xml \
  --episodes 2 --seconds 120 --delta-time 5 \
  --use-obs-norm \
  --model-path /tmp/test_norm.pt
```
Expected: completes without error, logs `Training device: cuda`

**Step 8: Commit**

```bash
git add ece324_tango/asce/trainers/local_mappo_backend.py \
        ece324_tango/asce/trainers/base.py \
        ece324_tango/asce/mappo.py \
        ece324_tango/modeling/train.py \
        ece324_tango/modeling/predict.py
git commit -m "Batch actor inference (single GPU forward per step) and wire --use-obs-norm flag"
```

---

## Task 9: Update pixi training tasks for longer runs

The current Toronto tasks use `--episodes 10 --seconds 120`. Increase to gather more learning data while staying within stable SUMO sim limits.

**Files:** `pixi.toml`

**Step 1: Update train-asce-toronto-demand**

Change `--episodes 10 --seconds 120` to `--episodes 30 --seconds 300`:
- 300s with delta_time=5 = 60 steps/episode × 30 episodes = 1800 total steps
- 12 agents × 1800 steps = 21,600 transitions per training run
- ~30-60 min wall time (SUMO-limited)
- Stays well below the known crash threshold at ~57 control steps on this net

Add `--use-obs-norm` to enable normalization by default in the task.

**Step 2: Update train-asce-toronto-random similarly**

Same settings: `--episodes 30 --seconds 300 --use-obs-norm`

**Step 3: Verify pixi.toml parses correctly**

```bash
pixi run --list
```
Expected: shows updated train tasks

**Step 4: Commit**

```bash
git add pixi.toml
git commit -m "Toronto training tasks: 30 episodes x 300s with obs normalization enabled"
```

---

## Task 10: Run Toronto training with all fixes and verify improvement

**Step 1: Delete stale checkpoint**

```bash
rm -f models/asce_mappo_toronto_demand.pt
```

**Step 2: Run training**

```bash
pixi run train-asce-toronto-demand
```
Monitor episode rewards — should trend upward (less negative) over 30 episodes.

**Step 3: Run eval**

```bash
pixi run eval-asce-toronto-demand
```

**Step 4: Check result**

```bash
cat reports/results/asce_eval_metrics_toronto_demand.csv | column -t -s ','
```

Expected direction:
- `time_loss_s` (MAPPO) lower than the previous 3245s (from 10-episode run)
- Ratio vs max_pressure below 1.82x

**Step 5: Update prototype_log.md with results**

Record episode reward curve, final time_loss ratio, and any remaining gap to 0.90x target.

**Step 6: Final commit**

```bash
git add docs/notes/prototype_log.md
git commit -m "Record 30-episode obs-norm training results on Toronto demand"
```

---

## Summary of changes

| File | Change |
|---|---|
| `.gitignore` | Add models, CSVs, unzipped net, vscode |
| `AGENTS.md` | Full codebase orientation (replaces stub) |
| `docs/notes/runbook.md` | Obs-norm, GPU notes, bug-fix record |
| `ece324_tango/asce/obs_norm.py` | New: Welford running normalizer |
| `ece324_tango/asce/mappo.py` | `use_obs_norm` param, `act_batch()`, checkpoint save/load |
| `ece324_tango/asce/trainers/base.py` | `use_obs_norm: bool` in configs |
| `ece324_tango/asce/trainers/local_mappo_backend.py` | Use `act_batch()`, `norm_update()`, pass `use_obs_norm` |
| `ece324_tango/modeling/train.py` | `--use-obs-norm` flag |
| `ece324_tango/modeling/predict.py` | `--use-obs-norm` flag |
| `pixi.toml` | 30 episodes × 300s, `--use-obs-norm` |
| `tests/test_obs_norm.py` | 4 tests for ObsRunningNorm |
