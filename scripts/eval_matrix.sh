  #!/bin/bash
  # Full 5-seed × 4-scenario × 4-controller eval matrix.
  #
  # Stage 1: sweep MAPPO checkpoints × scenarios via predict.py. Each
  #   predict.py invocation internally evaluates {mappo, fixed_time,
  #   max_pressure} for the requested checkpoint, so one CSV per
  #   (checkpoint, scenario) contains 3 controllers × 5 seeds (1000–1004).
  # Stage 2: run NEMA (Actuated Default) over the same scenarios via
  #   eval_nema.py with seeds 1000–1004, producing nema__<scenario>.csv.
  # Together stages 1+2 cover all four controllers documented in the
  # final report (MAPPO / Max-Pressure / Actuated Default / Fixed-Time).
  set -euo pipefail

  NET="sumo/network/osm.net.xml.gz"
  SCENARIOS=(
    "sumo/demand/curriculum/am_peak.rou.xml"
    "sumo/demand/curriculum/pm_peak.rou.xml"
    "sumo/demand/curriculum/demand_surge.rou.xml"
    "sumo/demand/curriculum/midday_multimodal.rou.xml"
  )
  # Canonical eval set — only the checkpoints documented in models/README.md.
  # Older experimental checkpoints have been moved to models/_archive/.
  MODELS=(
    "models/asce_mappo_curriculum_best.pt"
    "models/asce_mappo_curriculum.pt"
    "models/asce_mappo_curriculum_best_am_peak.pt"
    "models/asce_mappo_curriculum_best_pm_peak.pt"
    "models/asce_mappo_curriculum_best_demand_surge.pt"
    "models/asce_mappo_curriculum_best_midday_multimodal.pt"
  )

  OUTDIR="reports/results/eval_matrix"
  mkdir -p "$OUTDIR"

  for model in "${MODELS[@]}"; do
    mname=$(basename "$model" .pt | tr ' ' '_')
    for scenario in "${SCENARIOS[@]}"; do
      sname=$(basename "$scenario" .rou.xml)
      outcsv="$OUTDIR/${mname}__${sname}.csv"
      echo "PYTHONUNBUFFERED=1 pixi run python -m ece324_tango.modeling.predict \
        --model-path \"$model\" \
        --net-file $NET \
        --route-file $scenario \
        --episodes 5 --seconds 900 --delta-time 5 --seed 1000 \
        --reward-mode person_objective \
        --residual-mode action_gate --use-obs-norm \
        --out-csv $outcsv"
    done
  done | xargs -P 8 -I {} bash -c '{}'

  echo "Stage 2: NEMA (Actuated Default) sweep, seeds 1000-1004"
  PYTHONUNBUFFERED=1 pixi run python scripts/eval_nema.py \
    --seed-base 1000 --num-seeds 5 --out-suffix ""

  echo "Done. Results in $OUTDIR/"
