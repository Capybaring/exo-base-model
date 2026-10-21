#!/usr/bin/env bash
# Save in the repository's scripts/ directory, then run:
# bash scripts/run_pjm_patch_base_lift.sh
# Same protocol as the current NP/BE experiments; no data preparation is performed.
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd -- "$REPO_ROOT"
PYTHON_BIN=${PYTHON_BIN:-python}
CONFIG_PATH="$REPO_ROOT/configs/table3_irregular/PJM.json"
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Missing configs/table3_irregular/PJM.json. Place this file in the repository's scripts/ directory." >&2
  exit 1
fi

# Prefer the existing prepared irregular cache. Accept the alternate flat layout.
PJM_DATA_PATH=${PJM_DATA_PATH:-}
if [[ -z "$PJM_DATA_PATH" ]]; then
  for CANDIDATE in "$REPO_ROOT/data/processed/table3_irregular/PJM.npz" "$REPO_ROOT/data/processed/PJM.npz"; do
    if [[ -f "$CANDIDATE" ]]; then
      PJM_DATA_PATH="$CANDIDATE"
      break
    fi
  done
fi
if [[ -z "$PJM_DATA_PATH" || ! -f "$PJM_DATA_PATH" ]]; then
  echo "Prepared PJM.npz not found. Set PJM_DATA_PATH=/absolute/path/to/PJM.npz and rerun." >&2
  exit 1
fi

RUN_ROOT="$REPO_ROOT/results/rolling_12h_patch12h_pjm/run_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p -- "$RUN_ROOT"

COMMON_ARGS=(
  --config "$CONFIG_PATH"
  --data-path "$PJM_DATA_PATH"
  --patch-encoder ttcn
  --ttcn-execution chunked
  --task-protocol dataset-short
  --history 604800 --forecast 86400
  --patch-seconds 43200 --patch-stride 43200
  --sample-stride 43200
  --lag-estimator fft --refiner-mode global
  --split chronological --normalization train
  --batch-size 64 --epochs 100 --patience 10
  --lr 0.001 --d-model 64 --n-heads 4 --n-layers 1 --state-num 8
  --device "${DEVICE:-cuda:0}" --threads 2 --seed 1
)

echo "PJM data: $PJM_DATA_PATH"
echo "PJM results: $RUN_ROOT"
for VARIANT in patch-base patch-lift; do
  SAVE_PATH="$RUN_ROOT/PJM/$VARIANT"
  mkdir -p -- "$SAVE_PATH"
  echo "Running PJM $VARIANT: 168h -> 24h, patch 12h, sample stride 12h"
  "$PYTHON_BIN" -u -m exobase.train "${COMMON_ARGS[@]}" \
    --variant "$VARIANT" --save-path "$SAVE_PATH"
done

echo "Completed. Compare these two files:"
echo "$RUN_ROOT/PJM/patch-base/metrics.json"
echo "$RUN_ROOT/PJM/patch-lift/metrics.json"
