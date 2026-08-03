"""Benchmark MathVerse prompt-batch shapes on a real CUDA model.

The benchmark runs the same prompts, rollout count, and generation limit for
each requested (prompt_batch_size, rollout_microbatch_size) shape. It reports
synchronized prefill/decode timings, peak allocated/reserved VRAM, and both
decode-only and end-to-end generated-token throughput.
"""

from __future__ import annotations

import argparse
import uuid
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

from visconf.benchmark import BenchmarkReport, benchmark_run
from visconf.config import (
    GenerationSettings,
    HardwareSettings,
    LoadedExperimentGroup,
    load_experiment_group_config,
    sha256_json,
)
from visconf.planning import plan_experiment_group
from visconf.storage.manifest import atomic_write_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHAPES = ((1,32), (2,32), (4, 32), (8, 32), (16, 32), (32, 32))


def _shape(value: str) -> tuple[int, int]:
    try:
        prompt_text, rollout_text = value.lower().split("x", 1)
        prompt_size = int(prompt_text)
        rollout_size = int(rollout_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid shape {value!r}; expected PROMPTSxROLLOUTS, e.g. 8x32"
        ) from exc
    if prompt_size <= 0 or rollout_size <= 0:
        raise argparse.ArgumentTypeError("benchmark shape values must be positive")
    return prompt_size, rollout_size


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark MathVerse prompt-batch shapes on CUDA."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "experiment_group_4090_full_mb32.yaml",
        help="experiment-group config providing model and MathVerse settings",
    )
    parser.add_argument(
        "--hardware-config",
        type=Path,
        default=ROOT / "configs" / "hardware" / "rtx_4090.yaml",
        help="hardware profile; use a100_80gb.yaml on an 80 GB A100",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs" / "benchmarks",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "benchmarks" / "mathverse_prompt_batch.json",
    )
    parser.add_argument("--group-id")
    parser.add_argument("--strategy", default="diverse")
    parser.add_argument("--prompts", type=int, default=32)
    parser.add_argument("--rollouts", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument(
        "--shape",
        dest="shapes",
        type=_shape,
        nargs="+",
        default=DEFAULT_SHAPES,
        help="one or more PROMPT_BATCHxROLLOUT_MICROBATCH values",
    )
    return parser


def _hardware_config(path: Path) -> HardwareSettings:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        return HardwareSettings.model_validate(payload)
    except (OSError, yaml.YAMLError, ValueError) as exc:
        raise ValueError(f"invalid hardware config {path}: {exc}") from exc


def _benchmark_config(
    loaded: LoadedExperimentGroup,
    hardware_base: HardwareSettings,
    shapes: tuple[tuple[int, int], ...],
    prompts: int,
    rollouts: int,
    max_new_tokens: int,
) -> LoadedExperimentGroup:
    if not shapes or len(set(shapes)) != len(shapes):
        raise ValueError("benchmark shapes must be non-empty and unique")
    if prompts <= 0 or rollouts <= 0 or max_new_tokens <= 0:
        raise ValueError("prompts, rollouts, and max_new_tokens must be positive")
    largest_prompt = max(prompt for prompt, _ in shapes)
    largest_rollout = max(rollout for _, rollout in shapes)
    if largest_prompt > prompts:
        raise ValueError("a prompt batch size cannot exceed the prompt count")
    if largest_rollout > rollouts:
        raise ValueError("a rollout microbatch cannot exceed the rollout count")
    if largest_rollout > hardware_base.max_rollout_microbatch_size:
        raise ValueError(
            f"hardware profile permits rollout microbatches up to "
            f"{hardware_base.max_rollout_microbatch_size}, but the benchmark "
            f"requests {largest_rollout}"
        )
    largest_active = max(prompt * rollout for prompt, rollout in shapes)
    if largest_active > hardware_base.max_active_decode_rows:
        raise ValueError(
            f"hardware profile permits {hardware_base.max_active_decode_rows} "
            f"active decode rows, but the benchmark requests {largest_active}"
        )

    generation = GenerationSettings(
        rollouts_per_example=rollouts,
        prompt_batch_size=largest_prompt,
        rollout_microbatch_size=largest_rollout,
        prompt_batching_strategy=loaded.generation.prompt_batching_strategy,
        prompt_bucket_window_size=loaded.generation.prompt_bucket_window_size,
        prompt_scheduler_algorithm_version=(
            loaded.generation.prompt_scheduler_algorithm_version
        ),
        max_new_tokens=max_new_tokens,
    )
    hardware = hardware_base.model_copy(
        update={"benchmark_batch_shapes": shapes}
    )
    group_config_hash = sha256_json(
        {
            "experiment_group": loaded.group.model_dump(mode="json"),
            "model": loaded.model.model_dump(mode="json"),
            "hardware": hardware.model_dump(mode="json"),
            "datasets": [
                item.model_dump(mode="json") for item in loaded.datasets
            ],
            "strategies": [
                item.model_dump(mode="json") for item in loaded.strategies
            ],
            "generation": generation.model_dump(mode="json"),
            "scoring": loaded.scoring.model_dump(mode="json"),
            "storage": loaded.storage.model_dump(mode="json"),
            "schemas": loaded.schemas.model_dump(mode="json"),
        }
    )
    return loaded.model_copy(
        update={
            "hardware": hardware,
            "generation": generation,
            "group_config_hash": group_config_hash,
        }
    )


def _summary(report: BenchmarkReport) -> list[dict[str, object]]:
    rows = []
    for measurement in report.measurements:
        decode_seconds = measurement.decode_seconds
        total_seconds = measurement.total_seconds
        rows.append(
            {
                "prompt_batch_size": measurement.requested_prompt_batch_size,
                "rollout_microbatch_size": (
                    measurement.requested_rollout_microbatch_size
                ),
                "status": measurement.status,
                "oom_fallbacks": measurement.oom_fallbacks,
                "peak_allocated_gib": (
                    measurement.peak_allocated_bytes / 1024**3
                ),
                "peak_reserved_gib": (
                    measurement.peak_reserved_bytes / 1024**3
                ),
                "prompt_seconds": measurement.prompt_seconds,
                "decode_seconds": decode_seconds,
                "total_seconds": total_seconds,
                "retained_tokens": measurement.retained_tokens,
                "decode_tokens_per_second": (
                    measurement.retained_tokens / decode_seconds
                    if decode_seconds > 0
                    else None
                ),
                "end_to_end_tokens_per_second": (
                    measurement.retained_tokens / total_seconds
                    if total_seconds > 0
                    else None
                ),
            }
        )
    return rows


def _print_summary(
    report: BenchmarkReport,
    summary: list[dict[str, object]],
) -> None:
    print(f"GPU: {report.environment.get('gpu_model')}")
    rollout_count = report.measurements[0].rollout_count if report.measurements else 0
    print(f"Prompts: {report.prompt_count}; rollouts/prompt: {rollout_count}")
    print(
        "shape       status     peak_alloc  peak_reserved  decode_s  "
        "total_s  decode_tok/s  e2e_tok/s"
    )
    for row in summary:
        shape = f"{row['prompt_batch_size']}x{row['rollout_microbatch_size']}"
        print(
            f"{shape:<11} {str(row['status']):<10} "
            f"{float(row['peak_allocated_gib']):>10.2f} "
            f"{float(row['peak_reserved_gib']):>14.2f} "
            f"{float(row['decode_seconds']):>9.2f} "
            f"{float(row['total_seconds']):>8.2f} "
            f"{float(row['decode_tokens_per_second'] or 0):>13.2f} "
            f"{float(row['end_to_end_tokens_per_second'] or 0):>10.2f}"
        )


def main() -> int:
    args = _parser().parse_args()
    loaded = load_experiment_group_config(args.config)
    if args.strategy not in {item.name for item in loaded.strategies}:
        raise ValueError(f"strategy {args.strategy!r} is not present in {args.config}")
    if "mathverse" not in {item.name for item in loaded.datasets}:
        raise ValueError(f"{args.config} does not include the mathverse dataset")
    loaded = _benchmark_config(
        loaded,
        _hardware_config(args.hardware_config),
        tuple(args.shapes),
        args.prompts,
        args.rollouts,
        args.max_new_tokens,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    group_id = args.group_id or f"mathverse-prompt-batch-{uuid.uuid4().hex[:8]}"
    plan = plan_experiment_group(
        loaded,
        experiment_group_id=group_id,
        output_root=args.output_root.resolve(),
    )
    run = next(
        item
        for item in plan.runs
        if item.dataset.name == "mathverse"
        and item.sampling.name == args.strategy
    )
    benchmark_metadata = {
        "group_id": group_id,
        "shapes": [list(shape) for shape in args.shapes],
        "throughput_definition": {
            "decode_tokens_per_second": "retained_tokens / decode_seconds",
            "end_to_end_tokens_per_second": "retained_tokens / total_seconds",
        },
    }
    progress_measurements = []

    def on_measurement(measurement) -> None:
        progress_measurements.append(measurement)
        decode_rate = (
            measurement.retained_tokens / measurement.decode_seconds
            if measurement.decode_seconds > 0
            else None
        )
        print(
            f"Completed {len(progress_measurements)}/{len(args.shapes)} "
            f"shape={measurement.requested_prompt_batch_size}x"
            f"{measurement.requested_rollout_microbatch_size} "
            f"status={measurement.status} "
            f"decode_s={measurement.decode_seconds:.2f} "
            f"decode_tok_s={decode_rate or 0:.2f} "
            f"peak_allocated_gib="
            f"{measurement.peak_allocated_bytes / 1024**3:.2f}",
            flush=True,
        )
        atomic_write_json(
            args.output,
            {
                "status": "running",
                "dataset": "mathverse",
                "prompt_count": args.prompts,
                "rollouts_per_prompt": args.rollouts,
                "max_new_tokens": args.max_new_tokens,
                "completed_shapes": len(progress_measurements),
                "total_shapes": len(args.shapes),
                "benchmark": benchmark_metadata,
                "measurements": [
                    asdict(item) for item in progress_measurements
                ],
            },
        )

    report = benchmark_run(
        run,
        args.output,
        candidates=tuple(args.shapes),
        rollout_count=args.rollouts,
        max_new_tokens=args.max_new_tokens,
        prompt_count=args.prompts,
        on_measurement=on_measurement,
    )
    summary = _summary(report)
    payload = asdict(report)
    payload["summary"] = summary
    payload["benchmark"] = benchmark_metadata
    atomic_write_json(args.output, payload)
    _print_summary(report, summary)
    print(f"Report: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
