#!/usr/bin/env bash
set -euo pipefail

output_root=${1:?usage: verify_persistent_storage.sh OUTPUT_ROOT [--initialize]}
mode=${2:-}

if [[ "$mode" == "--initialize" ]]; then
  python -m visconf verify-storage \
    --output-root "$output_root" \
    --initialize
else
  python -m visconf verify-storage \
    --output-root "$output_root"
fi
