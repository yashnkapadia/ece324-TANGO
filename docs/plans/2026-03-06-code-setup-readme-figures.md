# Code-Setup README And Figures Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Upgrade the `code-setup` branch README into a polished interim-project overview and add a reproducible, report-quality composite figure built from the current Toronto objective-run artifacts.

**Architecture:** Keep the branch narrative tightly scoped to the ASCE/MAPPO code pipeline. Generate one primary composite figure directly from existing CSV artifacts so the README and interim report both reference the same reproducible asset. Use the objective retest run as the main evidence source and defer the time-loss run to later analysis.

**Tech Stack:** Python, pandas, matplotlib, pixi, Markdown

---

### Task 1: Add a figure-generation script for interim report assets

**Files:**
- Create: `ece324_tango/plots.py`
- Modify: `pixi.toml`
- Test: `reports/results/asce_train_episode_metrics_toronto_demand.csv`
- Test: `reports/results/asce_eval_metrics_toronto_demand_objective_retest_e10.csv`

**Step 1: Write the script skeleton**

Create a plotting entry point that:
- reads the objective training CSV,
- reads the objective retest eval CSV,
- creates `reports/figures/` if missing,
- writes a composite PNG and optionally a PDF.

Include functions for:
- loading and validating required columns,
- computing aggregate controller summaries,
- rendering the four-panel figure,
- saving outputs to stable filenames.

**Step 2: Run the script once to verify it fails only on missing implementation details**

Run: `pixi run python -m ece324_tango.plots`

Expected:
- initial failure if the file/function wiring is incomplete,
- no ambiguity about which inputs are expected.

**Step 3: Implement the figure**

Implement a `2 x 2` composite figure with:
- Panel A: objective training progress from `asce_train_episode_metrics_toronto_demand.csv`
- Panel B: per-controller `time_loss_s` distribution from `asce_eval_metrics_toronto_demand_objective_retest_e10.csv`
- Panel C: aggregated `objective_mean_reward` and `person_time_loss_s`
- Panel D: environment/takeaway context card

Use stable output names such as:
- `reports/figures/interim_objective_results.png`
- `reports/figures/interim_objective_results.pdf`

Do not include the `time_loss_e10` run in the main figure.

**Step 4: Add a pixi task for regenerating figures**

Add a task such as:
- `plot-interim-figures = "python -m ece324_tango.plots"`

Keep the task narrow and branch-specific.

**Step 5: Run the figure command and verify output**

Run: `pixi run plot-interim-figures`

Expected:
- figure files written into `reports/figures/`
- no stack traces
- the figure uses only the intended objective-run artifacts

**Step 6: Commit**

```bash
git add ece324_tango/plots.py pixi.toml reports/figures
git commit -m "Add interim report figure generation"
```

### Task 2: Rewrite the code-setup README around the interim-report story

**Files:**
- Modify: `README.md`
- Test: `docs/notes/runbook.md`
- Test: `docs/notes/prototype_log.md`
- Test: `sumo/demand/demand.rou.xml`

**Step 1: Draft the new README structure**

Replace the scaffold-style content with sections for:
- overview,
- branch scope,
- current Toronto setup,
- quick start,
- current findings,
- why max-pressure is strong here,
- next steps.

Include one short sentence pointing to the `data-setup` branch for demand-generation/calibration details.

**Step 2: Insert verified environment facts**

Use facts already confirmed in repo artifacts:
- current demand file contains `70` flows,
- all flows are `car`,
- all active flows span `0` to `3600` seconds,
- the current eval story should center on `objective_retest_e10`.

Avoid vague claims that are not evidenced by the current files.

**Step 3: Insert the current findings section**

Summarize the objective retest evidence clearly:
- MAPPO underperforms max-pressure on `time_loss_s`
- max-pressure also leads on `objective_mean_reward`
- this is expected in the current nominal regime

Do not foreground the `time_loss_e10` run here beyond a brief note that further analysis is pending.

**Step 4: Embed the generated figure**

Reference the generated composite figure from `reports/figures/`.

Ensure the surrounding text explains what the reader should take from the figure.

**Step 5: Proofread for branch correctness**

Verify the README does not accidentally:
- duplicate the full `data-setup` workflow,
- imply transit/pedestrian demand is already active in the current benchmark,
- overclaim MAPPO failure beyond the present environment.

**Step 6: Commit**

```bash
git add README.md
git commit -m "Rewrite code-setup README for interim report"
```

### Task 3: Verify the final presentation assets

**Files:**
- Test: `README.md`
- Test: `reports/figures/interim_objective_results.png`
- Test: `reports/figures/interim_objective_results.pdf`

**Step 1: Regenerate the figure from scratch**

Run: `pixi run plot-interim-figures`

Expected:
- successful regeneration,
- no manual editing required.

**Step 2: Open the README and verify references**

Check that:
- the figure path is correct,
- the README narrative matches the generated figure,
- the branch pointer to `data-setup` is present and brief.

**Step 3: Run the fast test suite**

Run: `pixi run pytest tests -q`

Expected:
- existing tests still pass,
- README/plotting changes did not disturb the package.

**Step 4: Summarize outputs**

Record the final deliverables:
- updated `README.md`
- reproducible plot task
- generated composite figure(s)

**Step 5: Commit**

```bash
git add README.md reports/figures pixi.toml ece324_tango/plots.py
git commit -m "Prepare code-setup branch for interim report"
```
