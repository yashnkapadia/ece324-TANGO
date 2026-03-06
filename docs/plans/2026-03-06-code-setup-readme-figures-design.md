# Code-Setup README And Interim Figures Design

## Goal

Prepare the `code-setup` branch for the interim report by:

1. Replacing the scaffold-style README with a polished, branch-specific overview of the ASCE/MAPPO codebase.
2. Generating a concise, report-quality composite figure that explains the current training/evaluation evidence.

This design is intentionally scoped to `code-setup`. It should not absorb the full demand-generation and calibration workflow from `data-setup`; it should only reference that branch briefly as the source of those details.

## Branch Positioning

The README for `code-setup` should describe what this branch actually owns:

- the local MAPPO training/evaluation pipeline,
- Toronto SUMO asset consumption,
- baseline controllers,
- KPI logging,
- current empirical findings,
- immediate research next steps.

It should include one short sentence directing readers to the `data-setup` branch for demand-generation and calibration details. It should not duplicate the data-setup README section-by-section.

## README Structure

The new README should read like an interim-project landing page, not a generic repo scaffold.

Planned sections:

1. **Project Overview**
   - TANGO and ASCE in one paragraph.
   - Toronto corridor scope and current modeling objective.

2. **What This Branch Implements**
   - Local MAPPO backend.
   - Fixed-time and max-pressure baselines.
   - KPI outputs (`time_loss_s`, `person_time_loss_s`, `avg_trip_time_s`, `arrived_vehicles`).
   - Observation normalization and heterogeneous intersection support, described briefly.

3. **Current Toronto Setup**
   - Current demand file is the integrated Toronto corridor demand.
   - Current active calibrated demand is narrow:
     - `70` flows,
     - all `car`,
     - one hour (`begin="0"`, `end="3600"`),
     - regular morning traffic regime.
   - The network contains pedestrian infrastructure, but the present demand/eval setup is still effectively car-only.
   - Short note pointing to `data-setup` for demand-generation/calibration details.

4. **Quick Start**
   - `pixi install`
   - `pixi run pytest tests -q`
   - `pixi run train-asce-toronto-demand`
   - `pixi run eval-asce-toronto-demand`
   - `pixi run train-asce-toronto-random`

5. **Current Findings**
   - Plain statement: naive MAPPO currently underperforms max-pressure on the nominal Toronto demand setup.
   - Use the objective retest run as the main evidence source:
     - `reports/results/asce_train_episode_metrics_toronto_demand.csv`
     - `reports/results/asce_eval_metrics_toronto_demand_objective_retest_e10.csv`
   - Mention the key aggregate comparison:
     - MAPPO `time_loss_s = 5405.39`
     - max-pressure `time_loss_s = 4353.33`
     - fixed-time `time_loss_s = 6184.94`
     - MAPPO / best-baseline ratio `= 1.2417`
   - Mention that max-pressure also leads on `objective_mean_reward` in the same retest.

6. **Why This Result Is Plausible**
   - Brief explanation of max-pressure:
     - it is a queue-pressure controller,
     - it directly favors phases that relieve the strongest local queue imbalances,
     - it is especially strong in regular, vehicle-dominated traffic where queue pressure is a good approximation of control quality.
   - Brief explanation of the present environment:
     - narrow one-hour regime,
     - car-only flows,
     - no TTC/streetcar demand yet,
     - no pedestrian actuation yet,
     - no incident/special-event/pathological demand patterns yet.
   - Conclusion:
     - in this setup, naive MAPPO should not be expected to beat max-pressure by default.

7. **Next Steps**
   - richer daily traffic patterns,
   - recurring commuter variation over a full day,
   - special-event surges,
   - incident-induced rerouting,
   - TTC/transit demand and priority,
   - pedestrian flows/signals,
   - max-pressure-informed MAPPO initialization or residual policy.

## Figure Design

### Primary Figure

Create one publication-style composite figure for the interim report and README. It should be dense but readable, with a layout suitable for direct insertion into the report.

Recommended layout: `2 x 2` grid in a single image.

#### Panel A: Objective-Run Training Progress

Source:
- `reports/results/asce_train_episode_metrics_toronto_demand.csv`

Purpose:
- show that the objective-trained Toronto demand run stabilizes over 30 episodes,
- provide evidence of training behavior without overclaiming convergence.

Plotted content:
- episode vs `mean_global_reward`,
- episode vs `critic_loss`,
- optionally episode vs `entropy` if it improves interpretability without clutter.

Presentation:
- use smoothed trend plus raw values, or raw values with a light line and markers.
- avoid cramming too many metrics on a single axis unless dual-axis readability remains high.

#### Panel B: Controller Comparison On Proposal KPI

Source:
- `reports/results/asce_eval_metrics_toronto_demand_objective_retest_e10.csv`

Purpose:
- show the per-episode distribution of the proposal KPI.

Plotted content:
- per-controller distribution of `time_loss_s` across 10 eval episodes,
- controllers:
  - `mappo`
  - `fixed_time`
  - `max_pressure`

Presentation:
- boxplot + stripplot, or violin + stripplot if the result remains legible.
- annotate controller means directly.
- this panel is the core “MAPPO underperforms max-pressure” evidence.

#### Panel C: Secondary KPI Summary

Source:
- `reports/results/asce_eval_metrics_toronto_demand_objective_retest_e10.csv`

Purpose:
- show that the same conclusion is visible in a second summary view, not only one metric plot.

Plotted content:
- aggregated controller means for:
  - `objective_mean_reward`
  - `person_time_loss_s`

Rationale:
- `objective_mean_reward` shows that max-pressure also leads on the current shaped objective.
- `person_time_loss_s` ties the figure back to the report’s person-weighted framing.

Presentation:
- grouped bars or two aligned horizontal bar charts.
- annotate exact means or rounded values.

#### Panel D: Environment And Takeaway Context Card

Purpose:
- compress the environment assumptions and interpretation into one compact visual panel.

Content:
- `12` intersections,
- `70` car-only flows,
- one-hour demand window,
- current regime approximates regular morning traffic,
- no TTC or pedestrian demand in active evaluation,
- takeaway:
  - max-pressure is strong because queue-pressure logic matches the current environment,
  - next step is richer demand plus max-pressure-informed RL.

Presentation:
- a styled text panel inside the figure, not a separate paragraph outside it.
- this keeps the figure self-contained for report use.

### Deferred Figure / Separate Discussion

The `time_loss` run should not be mixed into the primary figure.

Files:
- `reports/results/asce_train_episode_metrics_toronto_demand_time_loss.csv`
- `reports/results/asce_eval_metrics_toronto_demand_time_loss_e10.csv`

Reason:
- the current story for the interim report should remain focused and clean.
- the time-loss-only objective underperformance is still not well explained.
- mixing it into the main figure would dilute the narrative and invite a discussion we are not ready to defend yet.

Planned handling:
- keep the main README and main figure focused on the objective retest evidence.
- mention the time-loss run, if at all, as a deferred investigation or future appendix/support item.

## Implementation Shape

Expected repo changes:

- rewrite `README.md`,
- add a small plotting script that reads the existing CSV artifacts and writes reproducible figure assets into `reports/figures/`,
- optionally add a pixi task or documented command for regenerating the figure,
- avoid changing experiment logic or rerunning training/evaluation as part of this task.

## Acceptance Criteria

This design is complete when:

- `README.md` is clearly branch-specific and polished,
- the README includes the plain-language takeaway and explanation,
- the README includes a short pointer to `data-setup`,
- the figure is reproducible from checked-in result CSVs,
- the primary composite figure uses the objective retest run as the central evidence source,
- the time-loss run is explicitly excluded from the main figure and noted as future analysis.
