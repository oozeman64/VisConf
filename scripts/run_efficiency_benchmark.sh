#!/usr/bin/env bash
set -euo pipefail

output_root=${1:?usage: run_efficiency_benchmark.sh OUTPUT_ROOT [OPTIONS]}
shift

python scripts/run_full_output_benchmark.py \
  --output-root "$output_root" \
  "$@"
