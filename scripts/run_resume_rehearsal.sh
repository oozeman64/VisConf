#!/usr/bin/env bash
set -euo pipefail

pytest -q \
  tests/test_phase3.py \
  tests/test_phase7.py \
  tests/test_phase9.py
