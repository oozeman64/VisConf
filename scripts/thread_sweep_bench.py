"""Benchmark one fixed (prompt_count, rollout_microbatch) shape, for comparing
OMP_NUM_THREADS/MKL_NUM_THREADS settings. Bypasses the CLI's --microbatch
flag, which currently expects (prompt, rollout) pairs but only accepts a
flat list of ints.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from visconf.benchmark import benchmark_run
from visconf.planning import load_experiment_plan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-count", type=int, required=True)
    parser.add_argument("--rollout-count", type=int, required=True)
    parser.add_argument(
        "--total-prompt-count",
        type=int,
        help="number of prompts to benchmark; defaults to --prompt-count",
    )
    parser.add_argument("--max-new-tokens", type=int)
    args = parser.parse_args()

    group_dir = args.output_root.resolve() / args.group
    plan = load_experiment_plan(group_dir)
    run = next(item for item in plan.runs if item.run_id == args.run)

    report = benchmark_run(
        run,
        args.output,
        candidates=[(args.prompt_count, args.rollout_count)],
        prompt_count=args.total_prompt_count or args.prompt_count,
        max_new_tokens=args.max_new_tokens,
    )
    print(json.dumps(asdict(report), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
