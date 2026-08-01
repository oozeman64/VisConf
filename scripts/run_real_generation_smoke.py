"""Run the two-rollout real-model generation, storage, and resume smoke test."""

from __future__ import annotations

import argparse
import gc
import json
import tempfile
import time
import uuid
from itertools import islice
from pathlib import Path

import torch

from visconf.config import GenerationSettings, load_experiment_group_config
from visconf.datasets import create_dataset_adapter
from visconf.generation.engine import GenerationEngine
from visconf.models.instrumentation import QwenInstrumentation
from visconf.models.qwen25vl import QwenModelFacade
from visconf.planning import plan_experiment_group
from visconf.storage.parquet_writer import write_examples
from visconf.storage.resume import build_resume_index
from visconf.storage.transaction import CoreShardTransaction
from visconf.types import (
    ExampleRecord,
    ImageRecord,
    PromptConfig,
    RolloutKey,
)
from visconf.utils.logging import configure_logging


ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "experiment_group_4090.yaml",
    )
    parser.add_argument("--dataset", default="mathverse")
    parser.add_argument("--strategy", default="diverse")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--output-root", type=Path)
    return parser


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _retained_ids(results) -> dict[int, tuple[int, ...]]:
    return {
        bundle.generation.key.rollout_index:
        bundle.generation.generated_token_ids
        for bundle in results
    }


def main() -> int:
    args = _parser().parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    configure_logging()
    loaded = load_experiment_group_config(args.config).model_copy(
        update={
            "generation": GenerationSettings(
                rollouts_per_example=2,
                rollout_microbatch_size=2,
                max_new_tokens=args.max_new_tokens,
            )
        }
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    gpu_name = torch.cuda.get_device_name(loaded.model.device)
    if loaded.hardware.accelerator not in gpu_name.casefold():
        raise RuntimeError(
            f"active GPU {gpu_name!r} does not match "
            f"{loaded.hardware.accelerator!r}"
        )

    temporary = (
        tempfile.TemporaryDirectory(prefix="visconf-real-smoke-")
        if args.output_root is None
        else None
    )
    output_root = (
        Path(temporary.name)
        if temporary is not None
        else args.output_root.resolve()
    )
    facade = None
    results_one = results_two = ()
    started = time.perf_counter()
    try:
        plan = plan_experiment_group(
            loaded,
            experiment_group_id=f"exp-real-smoke-{uuid.uuid4().hex[:8]}",
            output_root=output_root,
        )
        run = next(
            value
            for value in plan.runs
            if value.dataset.name == args.dataset
            and value.sampling.name == args.strategy
        )
        adapter = create_dataset_adapter(run.dataset.adapter)
        example = next(iter(islice(adapter.load_examples(run.dataset), 1)))
        messages = adapter.build_messages(
            example,
            PromptConfig(run.dataset.prompt_template),
        )
        keys = tuple(
            RolloutKey(
                run_id=run.run_id,
                dataset=run.dataset.name,
                split=run.dataset.split,
                sample_id=example.sample_id,
                strategy=run.sampling.name,
                rollout_index=index,
            )
            for index in range(2)
        )

        facade = QwenModelFacade.load(run.model)
        torch.cuda.reset_peak_memory_stats(facade.device)
        with QwenInstrumentation(facade.model) as instrumentation:
            for microbatch_size in (1, 2):
                engine = GenerationEngine(
                    facade,
                    instrumentation,
                    base_seed=run.base_seed,
                    max_new_tokens=args.max_new_tokens,
                    rollout_microbatch_size=microbatch_size,
                    seed_derivation_version=(
                        run.schemas.seed_derivation_version
                    ),
                )
                results = tuple(
                    engine.generate_example(
                        example,
                        facade.prepare_example(messages),
                        keys,
                        run.sampling.as_domain(),
                    )
                )
                if microbatch_size == 1:
                    results_one = results
                else:
                    results_two = results

        if _retained_ids(results_one) != _retained_ids(results_two):
            raise RuntimeError("rollout IDs changed with microbatch size")

        prompt = facade.prepare_example(messages).prompt_record
        write_examples(
            run.output_dir / "examples.parquet",
            (
                ExampleRecord(
                    run_id=run.run_id,
                    dataset=run.dataset.name,
                    split=run.dataset.split,
                    sample_id=example.sample_id,
                    source_row_index=example.source_row_index,
                    question=example.question,
                    rendered_prompt=prompt.rendered_prompt,
                    prompt_token_ids=prompt.prompt_token_ids,
                    ground_truth_json=_canonical_json(
                        adapter.ground_truth(example)
                    ),
                    answer_type=example.answer_type,
                    images=tuple(
                        ImageRecord(
                            image.source_ref,
                            image.sha256,
                            image.width,
                            image.height,
                            image.mode,
                        )
                        for image in example.images
                    ),
                    metadata_json=_canonical_json(example.metadata),
                ),
            ),
            run.storage,
        )
        CoreShardTransaction(run).commit(
            "real-smoke",
            "real-smoke-attempt",
            results_two,
        )
        resume = build_resume_index(run)
        if len(resume.completed_rollouts) != 2:
            raise RuntimeError("resume did not reconstruct both rollouts")

        print(
            json.dumps(
                {
                    "config": str(args.config.resolve()),
                    "gpu": gpu_name,
                    "dataset": run.dataset.name,
                    "strategy": run.sampling.name,
                    "max_new_tokens": args.max_new_tokens,
                    "retained_ids": {
                        str(key): list(value)
                        for key, value in _retained_ids(results_two).items()
                    },
                    "committed_rollouts": len(resume.completed_rollouts),
                    "committed_shards": len(resume.core_shards),
                    "elapsed_seconds": time.perf_counter() - started,
                    "peak_allocated_gib": (
                        torch.cuda.max_memory_allocated(facade.device)
                        / 1024**3
                    ),
                    "output_dir": str(run.output_dir),
                    "output_is_temporary": temporary is not None,
                },
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    finally:
        del results_one
        del results_two
        if facade is not None:
            del facade
        gc.collect()
        torch.cuda.empty_cache()
        if temporary is not None:
            temporary.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
