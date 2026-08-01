#!/usr/bin/env bash
set -euo pipefail

group_id=${1:?usage: benchmark_hardware.sh GROUP_ID RUN_ID OUTPUT_ROOT [REPORT]}
run_id=${2:?usage: benchmark_hardware.sh GROUP_ID RUN_ID OUTPUT_ROOT [REPORT]}
output_root=${3:?usage: benchmark_hardware.sh GROUP_ID RUN_ID OUTPUT_ROOT [REPORT]}
report=${4:-"$output_root/$group_id/benchmarks/$run_id.json"}

python -m visconf benchmark \
  --group "$group_id" \
  --run "$run_id" \
  --output-root "$output_root" \
  --output "$report"
