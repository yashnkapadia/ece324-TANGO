  #!/bin/bash
  # Full 5-seed curriculum eval matrix — 8 parallel SUMO instances
  set -euo pipefail

  NET="sumo/network/osm.net.xml"
  SCENARIOS=(
    "sumo/demand/curriculum/am_peak.rou.xml"
    "sumo/demand/curriculum/pm_peak.rou.xml"
    "sumo/demand/curriculum/demand_surge.rou.xml"
    "sumo/demand/curriculum/midday_multimodal.rou.xml"
  )
  MODELS=(
    "models/asce_mappo_curriculum_best.pt"
    "models/asce_mappo_curriculum.pt"
    "models/asce_mappo_curriculum_best_am_peak.pt"
    "models/asce_mappo_curriculum_best_pm_peak.pt"
    "models/asce_mappo_curriculum_best_demand_surge.pt"
    "models/asce_mappo_curriculum_best_midday_multimodal.pt"
    "models/asce_mappo_curriculum - Copy.pt"
    "models/asce_mappo_curriculum_best - Copy.pt"
    "models/asce_mappo_person_obj.pt"
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

  echo "Done. Results in $OUTDIR/"
