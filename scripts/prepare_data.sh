#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
"${PYTHON:-python}" -m exobase.prepare \
  --dist "${DIST_PATH:-data/raw/DIST_irregular_9x9.csv}" \
  --dataset-dir "${DATASET_DIR:-data/raw/dataset}" \
  --drop-rate "${DROP_RATE:-0.3}" --seed "${SEED:-2021}" "$@"
