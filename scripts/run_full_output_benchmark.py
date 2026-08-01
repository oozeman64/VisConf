"""Run a bounded real-model benchmark through the full output transaction path."""

from __future__ import annotations

import argparse
import gc
import json
import time
import uuid
from itertools import islice
from pathlib import Path

import pyarrow.parquet as pq
import torch

from visconf.config import (
    GenerationSettings,
    HardwareSettings,
    load_experiment_group_config,
    sha256_json,
)
from visconf.datasets import create_dataset_adapter
from visconf.planning import plan_experiment_group
from visconf.runner import execute_run
from visconf.storage.persistence import (
    MARKER_NAME,
    initialize_persistence_marker,
    verify_persistence_marker,
)
from visconf.storage.resume import build_resume_index, discover_orphan_parts
from visconf.types import RunStatus
from visconf.utils.logging import configure_logging


ROOT = Path(__file__).resolve().parents[1]
CORE_TABLES = (
    "generations",
    "tokens",
    "token_probability_metrics",
    "token_attention_metrics",
    "token_hidden_state_metrics",
)


class _LimitedAdapter:
    def __init__(self, delegate, prompt_count: int) -> None:
        self.delegate = delegate
        self.prompt_count = prompt_count
        self.name = delegate.name

    def load_examples(self, config):
        return tuple(
            islice(
                self.delegate.load_examples(config),
                self.prompt_count,
            )
        )

    def __getattr__(self, name: str):
        return getattr(self.delegate, name)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "experiment_group_4090.yaml",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--group-id")
    parser.add_argument("--dataset", default="mathverse")
    parser.add_argument("--strategy", default="diverse")
    parser.add_argument("--prompts", type=int, default=1)
    parser.add_argument("--rollouts", type=int, default=16)
    parser.add_argument("--microbatch", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    return parser


def _group_hash(loaded, hardware, generation) -> str:
    return sha256_json(
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


def _resolved_config(args):
    loaded = load_experiment_group_config(args.config)
    generation = GenerationSettings(
        rollouts_per_example=args.rollouts,
        rollout_microbatch_size=args.microbatch,
        max_new_tokens=args.max_new_tokens,
    )
    candidates = tuple(
        sorted(
            set(
                loaded.hardware.benchmark_microbatch_sizes
                + (args.microbatch,)
            )
        )
    )
    hardware = HardwareSettings.model_validate(
        {
            **loaded.hardware.model_dump(mode="python"),
            "default_rollout_microbatch_size": args.microbatch,
            "benchmark_microbatch_sizes": candidates,
            "max_rollout_microbatch_size": candidates[-1],
        }
    )
    return loaded.model_copy(
        update={
            "hardware": hardware,
            "generation": generation,
            "group_config_hash": _group_hash(
                loaded,
                hardware,
                generation,
            ),
        }
    )


def _table_rows(run) -> dict[str, int]:
    rows = {
        "examples": pq.ParquetFile(
            run.output_dir / "examples.parquet"
        ).metadata.num_rows
    }
    for table in CORE_TABLES:
        rows[table] = sum(
            pq.ParquetFile(path).metadata.num_rows
            for path in sorted((run.output_dir / table).glob("part-*.parquet"))
        )
    return rows


def _ensure_persistent_output_root(output_root: Path) -> dict[str, object]:
    """Initialize a fresh root, or verify the existing persistence marker."""

    root = output_root.resolve()
    if (root / MARKER_NAME).exists():
        return verify_persistence_marker(root)
    return initialize_persistence_marker(root)


def main() -> int:
    args = _parser().parse_args()
    if min(
        args.prompts,
        args.rollouts,
        args.microbatch,
        args.max_new_tokens,
    ) <= 0:
        raise ValueError("benchmark sizes must be positive")
    configure_logging()
    _ensure_persistent_output_root(args.output_root)
    loaded = _resolved_config(args)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    gpu_name = torch.cuda.get_device_name(loaded.model.device)
    if loaded.hardware.accelerator not in gpu_name.casefold():
        raise RuntimeError(
            f"active GPU {gpu_name!r} does not match "
            f"{loaded.hardware.accelerator!r}"
        )

    group_id = args.group_id or (
        f"exp-full-output-{uuid.uuid4().hex[:8]}"
    )
    plan = plan_experiment_group(
        loaded,
        experiment_group_id=group_id,
        output_root=args.output_root.resolve(),
    )
    run = next(
        value
        for value in plan.runs
        if value.dataset.name == args.dataset
        and value.sampling.name == args.strategy
    )
    adapter = _LimitedAdapter(
        create_dataset_adapter(run.dataset.adapter),
        args.prompts,
    )

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(run.model.device)
    summary = execute_run(
        run,
        adapter=adapter,
        attempt_id="full-output-benchmark",
    )
    elapsed = time.perf_counter() - started
    peak_allocated_gib = (
        torch.cuda.max_memory_allocated(run.model.device) / 1024**3
    )
    resume = build_resume_index(run)
    rows = _table_rows(run)
    expected_rollouts = args.prompts * args.rollouts

    if summary.status is not RunStatus.COMPLETE:
        raise RuntimeError(f"benchmark run ended with {summary.status}")
    if len(resume.completed_rollouts) != expected_rollouts:
        raise RuntimeError("resume index has the wrong rollout count")
    if len(resume.core_shards) != args.prompts:
        raise RuntimeError("checkpoint count differs from prompt count")
    if discover_orphan_parts(run):
        raise RuntimeError("benchmark output contains orphan parts")
    if rows["examples"] != args.prompts:
        raise RuntimeError("examples row count differs from prompt count")
    if rows["generations"] != expected_rollouts:
        raise RuntimeError("generation row count differs from rollout count")
    token_rows = rows["tokens"]
    if token_rows <= 0 or any(
        rows[table] != token_rows
        for table in CORE_TABLES
        if table not in {"generations", "tokens"}
    ):
        raise RuntimeError("token and metric-family row counts differ")

    print(
        json.dumps(
            {
                "committed_rollouts": len(resume.completed_rollouts),
                "committed_shards": len(resume.core_shards),
                "dataset": run.dataset.name,
                "elapsed_seconds": elapsed,
                "gpu": gpu_name,
                "group_id": group_id,
                "max_new_tokens": args.max_new_tokens,
                "microbatch": args.microbatch,
                "orphan_parts": 0,
                "output_dir": str(run.output_dir),
                "peak_allocated_gib": peak_allocated_gib,
                "prompts": args.prompts,
                "rollouts_per_prompt": args.rollouts,
                "run_id": run.run_id,
                "status": summary.status,
                "strategy": run.sampling.name,
                "table_rows": rows,
            },
            indent=2,
            sort_keys=True,
        )
    )
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
