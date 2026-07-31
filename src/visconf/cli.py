"""Command-line interface for VisConf."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from visconf.config import ConfigError, load_experiment_group_config
from visconf.planning import (
    PlanningError,
    load_experiment_plan,
    plan_experiment_group,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="visconf")
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="plan the six-run experiment group")
    plan.add_argument("--config", type=Path, required=True)
    plan.add_argument("--group-id")
    plan.add_argument("--output-root", type=Path)

    validate = commands.add_parser(
        "validate", help="validate a planned experiment group"
    )
    validate.add_argument("--group", required=True)
    validate.add_argument("--output-root", type=Path, default=Path("outputs"))

    run = commands.add_parser("run", help="execute one resolved run")
    run.add_argument("--group", required=True)
    run.add_argument("--run", required=True)

    group = commands.add_parser("group", help="execute incomplete group runs")
    group.add_argument("--group", required=True)

    score = commands.add_parser("score", help="score committed generations")
    score.add_argument("--group", required=True)
    score.add_argument("--run")

    return parser


def _plan(args: argparse.Namespace) -> int:
    config = load_experiment_group_config(args.config)
    plan = plan_experiment_group(
        config,
        experiment_group_id=args.group_id,
        output_root=args.output_root,
    )
    summary = {
        "experiment_group_id": plan.manifest.experiment_group_id,
        "group_dir": str(plan.group_dir),
        "runs": [
            {
                "run_id": run.run_id,
                "label": run.run_label,
                "dataset": run.dataset.name,
                "strategy": run.sampling.name,
            }
            for run in plan.runs
        ],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _validate(args: argparse.Namespace) -> int:
    plan = load_experiment_plan(
        args.output_root.resolve() / args.group
    )
    print(
        json.dumps(
            {
                "experiment_group_id": plan.manifest.experiment_group_id,
                "run_count": len(plan.runs),
                "status": plan.manifest.status,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "plan":
            return _plan(args)
        if args.command == "validate":
            return _validate(args)
        print(
            f"visconf {args.command} is introduced in a later implementation phase",
            file=sys.stderr,
        )
        return 2
    except (ConfigError, PlanningError) as exc:
        print(f"visconf: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
