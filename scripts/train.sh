#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
dataset="${1:-DIST}"
if (($#)); then shift; fi
"${PYTHON:-python}" -u -m exobase.train --config "configs/$dataset.json" "$@"
