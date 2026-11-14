#!/bin/bash
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
if [[ -n "${PACE_MODULES:-}" ]]; then
  for pace_module in $PACE_MODULES; do
    module load "$pace_module"
  done
fi
if [[ -n "${PACE_ENV:-}" ]]; then
  source "$PACE_ENV/bin/activate"
fi
make hyperparam MODEL="${HPO_MODEL:-yolo}" HPO_CONFIG="${HPO_CONFIG:-}" \
  COCO_JSON="${COCO_JSON:-}" SPLIT_JSON="${SPLIT_JSON:-}" \
  IMAGES_DIR="${IMAGES_DIR:-}"
