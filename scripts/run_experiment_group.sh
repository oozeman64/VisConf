#!/usr/bin/env bash
set -euo pipefail

config=${1:?usage: run_experiment_group.sh CONFIG GROUP_ID OUTPUT_ROOT}
group_id=${2:?usage: run_experiment_group.sh CONFIG GROUP_ID OUTPUT_ROOT}
output_root=${3:?usage: run_experiment_group.sh CONFIG GROUP_ID OUTPUT_ROOT}

python -m visconf plan \
  --config "$config" \
  --group-id "$group_id" \
  --output-root "$output_root"
python -m visconf group \
  --group "$group_id" \
  --output-root "$output_root"
