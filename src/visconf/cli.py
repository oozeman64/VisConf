"""Command-line interface for VisConf."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from visconf.config import ConfigError, load_experiment_group_config
from visconf.planning import (
    PlanningError,
    load_experiment_plan,
    plan_experiment_group,
)
from visconf.storage.persistence import (
    initialize_persistence_marker,
    verify_persistence_marker,
)
from visconf.storage.manifest import current_git_state
from visconf.storage.resume import (
    build_resume_index,
    discover_orphan_parts,
)
from visconf.utils.logging import configure_logging


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
    validate.add_argument("--production", action="store_true")
    validate.add_argument("--benchmark-report", type=Path)

    run = commands.add_parser("run", help="execute one resolved run")
    run.add_argument("--group", required=True)
    run.add_argument("--run", required=True)
    run.add_argument("--output-root", type=Path, default=Path("outputs"))

    group = commands.add_parser("group", help="execute incomplete group runs")
    group.add_argument("--group", required=True)
    group.add_argument("--output-root", type=Path, default=Path("outputs"))

    score = commands.add_parser("score", help="score committed generations")
    score.add_argument("--group", required=True)
    score.add_argument("--run")
    score.add_argument("--output-root", type=Path, default=Path("outputs"))

    benchmark = commands.add_parser(
        "benchmark",
        help="benchmark rollout microbatches on the resolved target GPU",
    )
    benchmark.add_argument("--group", required=True)
    benchmark.add_argument("--run", required=True)
    benchmark.add_argument("--output-root", type=Path, default=Path("outputs"))
    benchmark.add_argument("--output", type=Path)
    benchmark.add_argument("--microbatch", type=int, nargs="+")
    benchmark.add_argument("--rollouts", type=int)
    benchmark.add_argument("--max-new-tokens", type=int)

    storage = commands.add_parser(
        "verify-storage",
        help="initialize or verify a persistent output root",
    )
    storage.add_argument("--output-root", type=Path, required=True)
    storage.add_argument("--initialize", action="store_true")

    return parser


def _group_dir(args: argparse.Namespace) -> Path:
    return args.output_root.resolve() / args.group


def _resolved_run(plan, run_id: str):
    try:
        return next(item for item in plan.runs if item.run_id == run_id)
    except StopIteration as exc:
        raise PlanningError(f"unknown run_id {run_id!r}") from exc


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
    plan = load_experiment_plan(_group_dir(args))
    repository_root = Path(__file__).resolve().parents[2]
    commit, dirty = current_git_state(repository_root)
    if (
        commit != plan.manifest.git_commit
        or dirty != plan.manifest.git_dirty
    ):
        raise PlanningError(
            "current Git state differs from the experiment plan"
        )
    if args.production:
        from visconf.production import validate_production_readiness

        if args.benchmark_report is None:
            raise ValueError(
                "--benchmark-report is required with --production"
            )
        readiness = validate_production_readiness(
            plan,
            repository_root,
            args.benchmark_report,
        )
        print(json.dumps(asdict(readiness), indent=2, sort_keys=True))
        return 0

    runs = []
    for run in plan.runs:
        resume = build_resume_index(run)
        runs.append(
            {
                "run_id": run.run_id,
                "committed_rollouts": len(resume.completed_rollouts),
                "core_shards": len(resume.core_shards),
                "score_shards": len(resume.score_shards),
                "orphan_parts": len(discover_orphan_parts(run)),
            }
        )
    print(
        json.dumps(
            {
                "experiment_group_id": plan.manifest.experiment_group_id,
                "run_count": len(plan.runs),
                "status": plan.manifest.status,
                "runs": runs,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _run(args: argparse.Namespace) -> int:
    from visconf.runner import execute_run

    plan = load_experiment_plan(_group_dir(args))
    run = _resolved_run(plan, args.run)
    print(json.dumps(asdict(execute_run(run)), indent=2, sort_keys=True))
    return 0


def _group(args: argparse.Namespace) -> int:
    from visconf.runner import execute_experiment_group

    summary = execute_experiment_group(_group_dir(args))
    print(json.dumps(asdict(summary), indent=2, sort_keys=True))
    return 0 if all(run.status.value == "complete" for run in summary.runs) else 2


def _score(args: argparse.Namespace) -> int:
    from visconf.scoring.engine import score_experiment_group

    summaries = score_experiment_group(
        _group_dir(args),
        run_id=args.run,
    )
    print(
        json.dumps(
            [asdict(summary) for summary in summaries],
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _benchmark(args: argparse.Namespace) -> int:
    from visconf.benchmark import benchmark_run

    plan = load_experiment_plan(_group_dir(args))
    run = _resolved_run(plan, args.run)
    output = args.output or (
        plan.group_dir / "benchmarks" / f"{run.run_id}.json"
    )
    report = benchmark_run(
        run,
        output,
        candidates=args.microbatch,
        rollout_count=args.rollouts,
        max_new_tokens=args.max_new_tokens,
    )
    print(json.dumps(asdict(report), indent=2, sort_keys=True))
    return 0


def _verify_storage(args: argparse.Namespace) -> int:
    marker = (
        initialize_persistence_marker(args.output_root)
        if args.initialize
        else verify_persistence_marker(args.output_root)
    )
    print(json.dumps(marker, indent=2, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    args = _parser().parse_args(argv)
    try:
        if args.command == "plan":
            return _plan(args)
        if args.command == "validate":
            return _validate(args)
        if args.command == "run":
            return _run(args)
        if args.command == "group":
            return _group(args)
        if args.command == "score":
            return _score(args)
        if args.command == "benchmark":
            return _benchmark(args)
        if args.command == "verify-storage":
            return _verify_storage(args)
        return 2
    except (ConfigError, PlanningError, ValueError, RuntimeError) as exc:
        print(f"visconf: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
