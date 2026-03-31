---
phase: quick
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - ece324_tango/asce/trainers/base.py
  - ece324_tango/modeling/train.py
  - ece324_tango/asce/trainers/local_mappo_backend.py
autonomous: true
must_haves:
  truths:
    - "Training with --num-workers 1 produces identical behavior to current code (no regression)"
    - "Training with --num-workers 4 runs 4 SUMO episodes in parallel per PPO update"
    - "PPO update receives merged batch from all N episodes with correct GAE computation"
    - "libsumo is enabled in worker subprocesses via LIBSUMO_AS_TRACI=1"
  artifacts:
    - path: "ece324_tango/asce/trainers/base.py"
      provides: "num_workers field on TrainConfig"
      contains: "num_workers"
    - path: "ece324_tango/modeling/train.py"
      provides: "--num-workers CLI flag"
      contains: "num_workers"
    - path: "ece324_tango/asce/trainers/local_mappo_backend.py"
      provides: "Parallel episode collection with multiprocessing.Pool"
      contains: "_run_episode_worker"
  key_links:
    - from: "ece324_tango/modeling/train.py"
      to: "ece324_tango/asce/trainers/base.py"
      via: "TrainConfig(num_workers=...)"
      pattern: "num_workers"
    - from: "ece324_tango/asce/trainers/local_mappo_backend.py"
      to: "multiprocessing.Pool"
      via: "pool.map(_run_episode_worker, ...)"
      pattern: "multiprocessing"
---

<objective>
Add parallel SUMO episode collection via `--num-workers N` flag. When N > 1, use Python multiprocessing to run N SUMO episodes simultaneously (each with its own libsumo instance), merge trajectories, and perform one PPO update on the combined batch. This gives ~Nx speedup on the simulation bottleneck (~98.7% of training time).

Purpose: Cut wall-clock training time from ~5.5 hours to ~1.5 hours for 200 episodes on RTX 4070 laptop.
Output: Modified training pipeline with backward-compatible parallel collection.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@ece324_tango/asce/trainers/base.py
@ece324_tango/modeling/train.py
@ece324_tango/asce/trainers/local_mappo_backend.py
@ece324_tango/asce/mappo.py

<interfaces>
<!-- Key types the executor needs -->

From ece324_tango/asce/mappo.py:
```python
@dataclass
class Transition:
    obs: np.ndarray
    global_obs: np.ndarray
    action: int
    logp: float
    reward: float
    done: bool
    value: float
    n_valid_actions: int
    mp_action: int = 0
    gate: int = 0

class MAPPOTrainer:
    def build_batch(self, trajectories: Dict[str, List[Transition]], last_values: Dict[str, float] | None = None) -> dict:
        # Returns: {"obs", "global_obs", "actions", "logp", "returns", "advantages", "n_valid_actions"}
    def update(self, batch, ppo_epochs, minibatch_size) -> dict:
        # Returns: {"actor_loss", "critic_loss", "entropy"}
    def save(self, out_path: str): ...
    def load(self, in_path: str): ...

class ResidualMAPPOTrainer(MAPPOTrainer):
    def act_batch_residual(self, obs_list, global_obs, n_valid_list, mp_actions_list) -> list: ...
    def build_batch(...) -> dict:
        # Also includes "gate_decisions" and "mp_action" arrays
```

From ece324_tango/asce/env.py:
```python
def create_parallel_env(net_file, route_file, seed, use_gui, seconds, delta_time, quiet_sumo) -> SumoEnvironment:
def flatten_obs_by_agent(obs: dict, ordered_agents: list) -> np.ndarray:
def pad_observation(obs: np.ndarray, target_dim: int) -> np.ndarray:
```

From ece324_tango/asce/trainers/base.py:
```python
@dataclass
class TrainConfig:
    # ... all current fields ...
    resume: bool = False
```
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Add num_workers to TrainConfig and CLI</name>
  <files>ece324_tango/asce/trainers/base.py, ece324_tango/modeling/train.py</files>
  <action>
1. In `base.py`, add `num_workers: int = 1` to `TrainConfig` dataclass (after `resume` field).

2. In `train.py`:
   - Add CLI parameter: `num_workers: int = typer.Option(1, "--num-workers", help="Number of parallel SUMO workers for episode collection (1=sequential)")`.
   - Pass `num_workers=num_workers` to the `TrainConfig(...)` constructor.
   - Add validation: if `num_workers < 1`, raise `typer.BadParameter`.

No other files change in this task.
  </action>
  <verify>
    <automated>cd /home/as04/ece324-TANGO && pixi run python -c "from ece324_tango.asce.trainers.base import TrainConfig; print('num_workers' in TrainConfig.__dataclass_fields__)"</automated>
  </verify>
  <done>TrainConfig has num_workers field, CLI accepts --num-workers flag, default is 1</done>
</task>

<task type="auto">
  <name>Task 2: Implement parallel episode collection in LocalMappoBackend</name>
  <files>ece324_tango/asce/trainers/local_mappo_backend.py</files>
  <action>
This is the core change. Modify `LocalMappoBackend.train()` to support parallel episode collection.

**A. Create a module-level worker function `_run_episode_worker(args):`**

This function runs in a subprocess. It must:
1. Set `os.environ["LIBSUMO_AS_TRACI"] = "1"` at the top.
2. Add `/usr/share/sumo/tools` to `sys.path` if not already present (for libsumo discovery).
3. Accept a single dict argument (pickle-serializable) containing:
   - `net_file`, `route_file`, `seed`, `seconds`, `delta_time`, `scenario_id`
   - `model_state_dict` (actor + critic + obs_norm + gobs_norm state dicts)
   - `obs_dim`, `global_obs_dim`, `n_actions`, `action_dims_dict`, `ordered_agents`
   - `reward_mode`, `reward_weights_dict` (delay/throughput/fairness/residual as floats)
   - `residual_mode`, `use_obs_norm`
   - `episode_num` (for logging)
4. Create its own `SumoEnvironment` via `create_parallel_env(quiet_sumo=True)`.
5. Create its own `MAPPOTrainer` or `ResidualMAPPOTrainer` (matching `residual_mode`), load the state dict, set to eval mode (no gradient tracking needed for collection).
6. Create its own `MaxPressureController`.
7. Run one full episode loop (same logic as current inner loop in `train()`), collecting:
   - `trajectories: Dict[str, List[Transition]]`
   - `last_values: Dict[str, float]`
   - `all_rows: List[dict]` (rollout CSV rows)
   - Episode metrics: `ep_reward`, `ep_steps`, `gate_fraction`
8. Close the env.
9. Return a dict with all collected data. Transitions use numpy arrays, so they're pickle-safe.

**IMPORTANT worker details:**
- Do NOT update obs_norm in workers. Pass the current norm state for inference only. The main process will update obs_norm from raw observations collected by workers.
- Actually, for simplicity in this first version: DO update obs_norm in workers (each worker updates its local copy), then after all workers finish, the main process averages the norm states OR just uses the single-worker path's norm. Simpler: skip obs_norm update in workers entirely. The main process obs_norm from before the parallel batch is good enough for inference; it will get updated on the next sequential inline-eval or when num_workers=1.
- Actually, the simplest correct approach: workers use the current obs_norm state for inference (normalize obs before acting) but do NOT call `obs_norm.update()`. Collect raw obs arrays in the worker return value. Main process calls `obs_norm.update()` on all raw obs after workers return. Add a list `raw_obs_for_norm: List[np.ndarray]` and `raw_gobs_for_norm: List[np.ndarray]` to the worker return.

**B. Modify the episode loop in `train()` method:**

When `cfg.num_workers > 1`:
1. Before the episode loop, pre-compute everything the worker needs that's static: `obs_dim`, `global_obs_dim`, `n_actions`, `action_dims`, `ordered_agents`, `effective_reward_mode`, `reward_weights`.
2. Change the loop to step by `cfg.num_workers` episodes at a time: `for ep_batch_start in range(start_episode, cfg.episodes, cfg.num_workers)`.
3. For each batch:
   - Serialize current model state: `trainer.actor.state_dict()`, `trainer.critic.state_dict()`, obs_norm/gobs_norm state dicts.
   - Build worker args list (one per worker, with different seeds: `cfg.seed + ep_batch_start + i`).
   - Use `multiprocessing.Pool(cfg.num_workers)` (or `get_context("spawn").Pool`) to run `_run_episode_worker` on each.
   - IMPORTANT: Use `spawn` context, not `fork` — fork + CUDA = deadlock.
   - Collect results from all workers.
   - Update obs_norm with raw obs from all workers.
   - For each worker result: call `trainer.build_batch(traj, last_values=lv)` to get a batch with correct per-episode GAE.
   - Merge batches by concatenating all arrays: `{k: np.concatenate([b[k] for b in batches]) for k in batches[0]}`.
   - Call `trainer.update(merged_batch, ...)` once.
   - Append each worker's ep_metrics and all_rows to the main lists.
   - Run checkpoint/eval/interrupt logic per batch (not per episode within batch). For episode numbering in metrics, use `ep_batch_start + i` for worker `i`.
4. Handle edge case: if remaining episodes < num_workers, only spawn that many workers.

When `cfg.num_workers == 1`:
- Keep EXACTLY the current code path. Do not call `_run_episode_worker`. This ensures zero regression risk. Wrap the existing single-episode logic in an `if cfg.num_workers == 1: ... else: ...` block.

**C. Pool lifecycle:**
- Create pool once before the episode loop, reuse for all batches, close after loop.
- Use `pool.map()` (blocking, simpler than `apply_async`).
- Set `maxtasksperchild=1` to avoid libsumo global state leaks between episodes.

**D. Logging:**
- Workers should NOT use loguru (subprocess logging is messy). Instead, include timing info in their return dict.
- Main process logs per-worker results after collecting: `"Workers completed: ep {start}-{end}, avg {time:.1f}s/episode"`.

**E. What NOT to change:**
- `_run_inline_eval` stays sequential in main process.
- `evaluate()` method unchanged.
- Checkpoint save/resume logic unchanged (just saves after merged update).
- Graceful interrupt: check `_interrupt_requested` after each batch, not each episode.
  </action>
  <verify>
    <automated>cd /home/as04/ece324-TANGO && pixi run python -c "
from ece324_tango.asce.trainers.local_mappo_backend import LocalMappoBackend, _run_episode_worker
print('Worker function importable:', callable(_run_episode_worker))
print('Backend importable:', LocalMappoBackend.name)
"</automated>
  </verify>
  <done>
- `_run_episode_worker` is a module-level function that creates its own SUMO env + model, runs one episode, returns trajectories + metrics.
- `train()` with num_workers=1 uses the original sequential code path (no multiprocessing import).
- `train()` with num_workers>1 uses spawn-context Pool to run N episodes in parallel, merges batches (per-episode GAE then concatenate), does one PPO update.
- libsumo enabled in workers via env var + sys.path.
- obs_norm updated in main process from worker-collected raw observations.
  </done>
</task>

<task type="checkpoint:human-verify" gate="non-blocking">
  <what-built>Parallel SUMO training with --num-workers flag</what-built>
  <how-to-verify>
1. Smoke test (2 episodes, 1 worker — regression check):
   ```
   pixi run python -m ece324_tango.modeling.train --episodes 2 --seconds 60 --num-workers 1 --no-resume
   ```
   Confirm: completes without error, episode metrics CSV written.

2. Parallel test (4 episodes, 2 workers):
   ```
   pixi run python -m ece324_tango.modeling.train --episodes 4 --seconds 60 --num-workers 2 --no-resume
   ```
   Confirm: runs 2 batches of 2 episodes each, logs show worker completion, episode metrics has 4 rows.

3. Full parallel test (4 episodes, 4 workers):
   ```
   pixi run python -m ece324_tango.modeling.train --episodes 4 --seconds 60 --num-workers 4 --no-resume
   ```
   Confirm: runs 1 batch of 4 episodes, wall time is significantly less than 4x single-episode time.
  </how-to-verify>
  <resume-signal>Type "approved" or describe issues</resume-signal>
</task>

</tasks>

<verification>
- `pixi run python -c "from ece324_tango.asce.trainers.base import TrainConfig; tc = TrainConfig.__dataclass_fields__; assert 'num_workers' in tc"` passes
- `pixi run python -m ece324_tango.modeling.train --help` shows `--num-workers` option
- Module imports cleanly: `from ece324_tango.asce.trainers.local_mappo_backend import _run_episode_worker`
- Training with `--num-workers 1` produces same output structure as before
</verification>

<success_criteria>
- --num-workers 1 is a no-op (identical to current behavior)
- --num-workers N > 1 runs N episodes in parallel via multiprocessing spawn pool
- Each worker uses libsumo (LIBSUMO_AS_TRACI=1) for faster per-step simulation
- GAE computed per-episode (not across merged trajectories) then batches concatenated
- obs_norm updated in main process from worker-collected raw observations
- No CUDA-in-subprocess issues (spawn context, model on CPU in workers)
</success_criteria>

<output>
After completion, create `.planning/quick/260330-ods-parallel-sumo-training-with-multiprocess/260330-ods-SUMMARY.md`
</output>
