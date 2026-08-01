#!/usr/bin/env bash
set -euo pipefail

group_id=${1:?usage: run_single_run.sh GROUP_ID RUN_ID OUTPUT_ROOT}
run_id=${2:?usage: run_single_run.sh GROUP_ID RUN_ID OUTPUT_ROOT}
output_root=${3:?usage: run_single_run.sh GROUP_ID RUN_ID OUTPUT_ROOT}

python -m visconf run \
  --group "$group_id" \
  --run "$run_id" \
  --output-root "$output_root"
