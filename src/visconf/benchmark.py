"""Target-GPU rollout microbatch benchmarking."""

from __future__ import annotations

import gc
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

from visconf.config import ResolvedRunConfig
from visconf.datasets import create_dataset_adapter
from visconf.generation.engine import GenerationEngine
from visconf.generation.scheduler import (
    prompt_batch_telemetry,
    schedule_prompt_batches,
)
from visconf.models.instrumentation import QwenInstrumentation
from visconf.models.qwen25vl import QwenModelFacade
from visconf.storage.manifest import (
    atomic_write_json,
    capture_environment,
    utc_now,
)
from visconf.types import (
    PromptBatchWorkItem,
    PromptConfig,
    RolloutKey,
)


class BenchmarkError(RuntimeError):
    """Raised when the active GPU does not match the hardware profile."""


@dataclass(frozen=True, slots=True)
class BenchmarkMeasurement:
    requested_prompt_batch_size: int
    effective_prompt_batch_size: int
    requested_rollout_microbatch_size: int
    effective_rollout_microbatch_size: int
    maximum_active_decode_rows: int
    rollout_count: int
    prompt_count: int
    completed_prompt_count: int
    prompt_batching_strategy: str
    min_prompt_length: int
    max_prompt_length: int
    min_image_token_count: int
    max_image_token_count: int
    total_unpadded_prompt_tokens: int
    total_padded_prompt_tokens: int
    padding_fraction: float
    status: str
    prompt_seconds: float
    decode_seconds: float
    total_seconds: float
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    retained_tokens: int
    tokens_per_second: float | None
    oom_fallbacks: int
    retained_ids_sha256: str | None



@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    created_at_utc: str
    run_id: str
    run_config_hash: str
    dataset: str
    sample_id: str
    sample_ids: tuple[str, ...]
    prompt_count: int
    hardware_profile: dict[str, Any]
    environment: dict[str, Any]
    max_new_tokens: int
    measurements: tuple[BenchmarkMeasurement, ...]


class _TimingFacade:
    def __init__(self, facade: QwenModelFacade) -> None:
        self.facade = facade
        self.prompt_seconds = 0.0
        self.decode_seconds = 0.0
        self.oom_fallbacks = 0

    def __getattr__(self, name: str):
        return getattr(self.facade, name)

    def _timed(self, field: str, operation, *args, **kwargs):
        torch.cuda.synchronize(self.facade.device)
        started = time.perf_counter()
        try:
            return operation(*args, **kwargs)
        except torch.cuda.OutOfMemoryError:
            self.oom_fallbacks += 1
            raise
        finally:
            torch.cuda.synchronize(self.facade.device)
            setattr(
                self,
                field,
                getattr(self, field) + time.perf_counter() - started,
            )

    def prefill(self, prepared, instrumentation):
        return self._timed(
            "prompt_seconds", self.facade.prefill, prepared, instrumentation
        )

    def prefill_batch(self, prepared, instrumentation):
        return self._timed(
            "prompt_seconds",
            self.facade.prefill_batch,
            prepared,
            instrumentation,
        )

    def repeat_cache(self, cache, batch_size):
        try:
            return self.facade.repeat_cache(cache, batch_size)
        except torch.cuda.OutOfMemoryError:
            self.oom_fallbacks += 1
            raise

    def select_cache_sources(self, cache, indices):
        try:
            return self.facade.select_cache_sources(cache, indices)
        except torch.cuda.OutOfMemoryError:
            self.oom_fallbacks += 1
            raise

    def decode_step(self, *args, **kwargs):
        return self._timed(
            "decode_seconds",
            self.facade.decode_step,
            *args,
            **kwargs,
        )


def _verify_hardware(run: ResolvedRunConfig) -> None:
    name = torch.cuda.get_device_name(torch.device(run.model.device)).lower()
    if run.hardware.accelerator not in name:
        raise BenchmarkError(
            f"active GPU {name!r} does not match "
            f"{run.hardware.accelerator!r}"
        )
    memory_gb = (
        torch.cuda.get_device_properties(run.model.device).total_memory
        / 1024**3
    )
    if memory_gb + 2 < run.hardware.memory_gb:
        raise BenchmarkError(
            f"active GPU has {memory_gb:.1f} GiB, "
            f"profile requires {run.hardware.memory_gb} GiB"
        )


def benchmark_candidates(
    run: ResolvedRunConfig,
) -> tuple[tuple[int, int], ...]:
    return run.hardware.benchmark_batch_shapes


def _keys(
    run: ResolvedRunConfig,
    sample_id: str,
    count: int,
) -> tuple[RolloutKey, ...]:
    return tuple(
        RolloutKey(
            run_id=run.run_id,
            dataset=run.dataset.name,
            split=run.dataset.split,
            sample_id=sample_id,
            strategy=run.sampling.name,
            rollout_index=index,
        )
        for index in range(count)
    )


def _update_ids_hash(digest, result) -> int:
    """Add one rollout to a deterministic streaming benchmark digest."""

    payload = {
        "sample_id": result.generation.key.sample_id,
        "rollout_index": result.generation.key.rollout_index,
        "generated_token_ids": result.generation.generated_token_ids,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)
    return len(result.generation.generated_token_ids)


def benchmark_run(
    run: ResolvedRunConfig,
    output_path: str | Path,
    *,
    candidates: Sequence[tuple[int, int]] | None = None,
    rollout_count: int | None = None,
    max_new_tokens: int | None = None,
    prompt_count: int = 1,
    sample_ids: Sequence[str] | None = None,
    on_measurement: Callable[[BenchmarkMeasurement], None] | None = None,
) -> BenchmarkReport:
    """Benchmark real visual examples on the resolved target GPU."""

    _verify_hardware(run)
    sizes = tuple(candidates or benchmark_candidates(run))
    if (
        not sizes
        or any(prompt <= 0 or rollout <= 0 for prompt, rollout in sizes)
        or len(set(sizes)) != len(sizes)
    ):
        raise BenchmarkError("benchmark batch shapes must be unique and positive")
    largest_rollout = max(rollout for _, rollout in sizes)
    count = largest_rollout if rollout_count is None else rollout_count
    if count < largest_rollout:
        raise BenchmarkError(
            "rollout_count must exercise the largest rollout axis"
        )

    token_limit = (
        run.generation.max_new_tokens
        if max_new_tokens is None
        else max_new_tokens
    )
    if token_limit <= 0:
        raise BenchmarkError("max_new_tokens must be positive")
    requested_sample_ids = (
        tuple(str(sample_id).strip() for sample_id in sample_ids)
        if sample_ids is not None
        else None
    )
    if requested_sample_ids is not None and (
        not requested_sample_ids
        or any(not sample_id for sample_id in requested_sample_ids)
        or len(set(requested_sample_ids)) != len(requested_sample_ids)
    ):
        raise BenchmarkError("sample_ids must be non-empty and unique")
    if requested_sample_ids is not None:
        prompt_count = len(requested_sample_ids)
    if prompt_count <= 0:
        raise BenchmarkError("prompt_count must be positive")
    if prompt_count < max(prompt for prompt, _ in sizes):
        raise BenchmarkError("prompt_count must exercise the largest prompt axis")

    adapter = create_dataset_adapter(run.dataset.adapter)
    if requested_sample_ids is None:
        examples = tuple(
            islice(adapter.load_examples(run.dataset), prompt_count)
        )
    else:
        requested_set = set(requested_sample_ids)
        examples_by_id = {}
        for example in adapter.load_examples(run.dataset):
            if example.sample_id in requested_set:
                examples_by_id[example.sample_id] = example
                if len(examples_by_id) == len(requested_set):
                    break
        missing = tuple(
            sample_id
            for sample_id in requested_sample_ids
            if sample_id not in examples_by_id
        )
        if missing:
            raise BenchmarkError(
                "dataset did not contain requested sample_ids: "
                + ", ".join(missing)
            )
        examples = tuple(
            examples_by_id[sample_id] for sample_id in requested_sample_ids
        )
    if len(examples) != prompt_count:
        raise BenchmarkError(
            f"dataset provided {len(examples)} prompts, "
            f"but {prompt_count} were requested"
        )
    facade = QwenModelFacade.load(run.model)
    measurements = []
    instrumentation = None
    timed = None
    engine = None
    prepared = None
    results = ()
    try:
        with QwenInstrumentation(facade.model) as instrumentation:
            for prompt_size, rollout_size in sizes:
                timed = _TimingFacade(facade)
                engine = GenerationEngine(
                    timed,
                    instrumentation,
                    base_seed=run.base_seed,
                    max_new_tokens=token_limit,
                    rollout_microbatch_size=rollout_size,
                    seed_derivation_version=run.schemas.seed_derivation_version,
                )
                torch.cuda.reset_peak_memory_stats(facade.device)
                torch.cuda.synchronize(facade.device)
                started = time.perf_counter()
                retained = 0
                completed_prompts = 0
                ids_digest = hashlib.sha256()
                prepared_items = []
                try:
                    for ordinal, example in enumerate(examples):
                        messages = adapter.build_messages(
                            example,
                            PromptConfig(run.dataset.prompt_template),
                        )
                        prepared = facade.prepare_example(messages)
                        prepared_items.append(
                            PromptBatchWorkItem(
                                example=example,
                                sample_id=example.sample_id,
                                canonical_source_ordinal=ordinal,
                                prompt_record=prepared.prompt_record,
                                token_groups=prepared.token_groups,
                                image_token_count=int(
                                    prepared.token_groups.image_positions.numel()
                                ),
                                pending_rollout_keys=_keys(
                                    run, example.sample_id, count
                                ),
                                prepared=prepared,
                                messages=tuple(messages),
                            )
                        )
                    units = tuple(schedule_prompt_batches(
                        prepared_items,
                        prompt_batch_size=prompt_size,
                        strategy=run.generation.prompt_batching_strategy,
                        bucket_window_size=(
                            run.generation.prompt_bucket_window_size
                        ),
                    ))
                    all_telemetry = [
                        prompt_batch_telemetry(
                            unit, rollout_size, rollout_size
                        )
                        for unit in units
                    ]
                    for unit in units:
                        for result in engine.generate_prompt_batch(
                            unit, run.sampling.as_domain()
                        ):
                            retained += _update_ids_hash(ids_digest, result)
                        completed_prompts += len(unit)
                    torch.cuda.synchronize(facade.device)
                    total = time.perf_counter() - started
                    lengths = [
                        item.prompt_record.prompt_token_count
                        for item in prepared_items
                    ]
                    images = [item.image_token_count for item in prepared_items]
                    unpadded = sum(
                        item.total_unpadded_prompt_tokens
                        for item in all_telemetry
                    )
                    padded = sum(
                        item.total_padded_prompt_tokens
                        for item in all_telemetry
                    )
                    fallback = timed.oom_fallbacks
                    measurement = BenchmarkMeasurement(
                        requested_prompt_batch_size=prompt_size,
                        effective_prompt_batch_size=(
                            prompt_size if fallback == 0 else 0
                        ),
                        requested_rollout_microbatch_size=rollout_size,
                        effective_rollout_microbatch_size=(
                            rollout_size if fallback == 0 else 0
                        ),
                        maximum_active_decode_rows=prompt_size * rollout_size,
                        rollout_count=count,
                        prompt_count=prompt_count,
                        completed_prompt_count=completed_prompts,
                        prompt_batching_strategy=run.generation.prompt_batching_strategy,
                        min_prompt_length=min(lengths),
                        max_prompt_length=max(lengths),
                        min_image_token_count=min(images),
                        max_image_token_count=max(images),
                        total_unpadded_prompt_tokens=unpadded,
                        total_padded_prompt_tokens=padded,
                        padding_fraction=(padded - unpadded) / padded,
                        status="complete" if fallback == 0 else "fallback",
                        prompt_seconds=timed.prompt_seconds,
                        decode_seconds=timed.decode_seconds,
                        total_seconds=total,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(facade.device),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(facade.device),
                        retained_tokens=retained,
                        tokens_per_second=retained / total if total > 0 else None,
                        oom_fallbacks=fallback,
                        retained_ids_sha256=ids_digest.hexdigest(),
                    )
                    measurements.append(measurement)
                    if on_measurement is not None:
                        on_measurement(measurement)
                except torch.cuda.OutOfMemoryError:
                    instrumentation.cancel_step()
                    measurement = BenchmarkMeasurement(
                        requested_prompt_batch_size=prompt_size,
                        effective_prompt_batch_size=0,
                        requested_rollout_microbatch_size=rollout_size,
                        effective_rollout_microbatch_size=0,
                        maximum_active_decode_rows=prompt_size * rollout_size,
                        rollout_count=count,
                        prompt_count=prompt_count,
                        completed_prompt_count=completed_prompts,
                        prompt_batching_strategy=run.generation.prompt_batching_strategy,
                        min_prompt_length=0,
                        max_prompt_length=0,
                        min_image_token_count=0,
                        max_image_token_count=0,
                        total_unpadded_prompt_tokens=0,
                        total_padded_prompt_tokens=0,
                        padding_fraction=0.0,
                        status="oom",
                        prompt_seconds=timed.prompt_seconds,
                        decode_seconds=timed.decode_seconds,
                        total_seconds=time.perf_counter() - started,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(facade.device),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(facade.device),
                        retained_tokens=retained,
                        tokens_per_second=None,
                        oom_fallbacks=timed.oom_fallbacks,
                        retained_ids_sha256=None,
                    )
                    measurements.append(measurement)
                    if on_measurement is not None:
                        on_measurement(measurement)
                    torch.cuda.empty_cache()
    finally:
        del results
        del prepared
        del engine
        del timed
        del instrumentation
        del facade
        gc.collect()
        torch.cuda.empty_cache()

    report = BenchmarkReport(
        created_at_utc=utc_now().isoformat(),
        run_id=run.run_id,
        run_config_hash=run.config_hash,
        dataset=run.dataset.name,
        sample_id=examples[0].sample_id,
        sample_ids=tuple(example.sample_id for example in examples),
        prompt_count=prompt_count,
        hardware_profile=run.hardware.model_dump(mode="json"),
        environment=capture_environment(run.model.device),
        max_new_tokens=token_limit,
        measurements=tuple(measurements),
    )
    atomic_write_json(Path(output_path), asdict(report))
    return report
