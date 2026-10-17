#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
dataset="${1:-BE}"
if (($#)); then shift; fi
for variant in tpatch-y tpatch exo delay; do
  "${PYTHON:-python}" -u -m exobase.train --config "configs/$dataset.json" \
    --variant "$variant" --save-path "results/$dataset/$variant" "$@"
done
