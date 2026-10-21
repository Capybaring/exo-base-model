#!/usr/bin/env bash
# Save this file in the repository's scripts/ directory.
# Run: bash scripts/run_np_be_patch_base.sh
# Match the patch-lift experiment; change only variant and output subdirectory.
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd -- "$REPO_ROOT"
PYTHON_BIN=${PYTHON_BIN:-python}

for DATASET in NP BE; do
  if [[ ! -f "$REPO_ROOT/configs/$DATASET.json" ]]; then
    echo "Missing configs/$DATASET.json. Place this script in the repository's scripts/ directory." >&2
    exit 1
  fi
done

RUN_ROOT="$REPO_ROOT/results/rolling_12h_patch12h_base/run_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p -- "$RUN_ROOT"

COMMON_ARGS=(
  --variant patch-base
  --patch-encoder ttcn
  --task-protocol dataset-short
  --history 604800 --forecast 86400
  --patch-seconds 43200 --patch-stride 43200
  --sample-stride 43200
  --lag-estimator fft --refiner-mode global
  --split chronological --normalization train
  --batch-size 64 --epochs 100 --patience 10
  --device "${DEVICE:-cuda:0}" --threads 2 --seed 1
)

echo "NP and BE patch-base results: $RUN_ROOT"
for DATASET in NP BE; do
  SAVE_PATH="$RUN_ROOT/$DATASET/patch-base"
  mkdir -p -- "$SAVE_PATH"
  echo "Running $DATASET patch-base: 168h -> 24h, patch 12h, sample stride 12h"
  "$PYTHON_BIN" -u -m exobase.train "${COMMON_ARGS[@]}" \
    --config "$REPO_ROOT/configs/$DATASET.json" --save-path "$SAVE_PATH"
done
