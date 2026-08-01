"""Target-GPU rollout microbatch benchmarking."""

from __future__ import annotations

import gc
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Sequence

import torch

from visconf.config import ResolvedRunConfig
from visconf.datasets import create_dataset_adapter
from visconf.generation.engine import GenerationEngine
from visconf.models.instrumentation import QwenInstrumentation
from visconf.models.qwen25vl import QwenModelFacade
from visconf.storage.manifest import (
    atomic_write_json,
    capture_environment,
    utc_now,
)
from visconf.types import PromptConfig, RolloutKey


class BenchmarkError(RuntimeError):
    """Raised when the active GPU does not match the hardware profile."""


@dataclass(frozen=True, slots=True)
class BenchmarkMeasurement:
    requested_microbatch_size: int
    rollout_count: int
    status: str
    prompt_seconds: float
    decode_seconds: float
    total_seconds: float
    peak_allocated_bytes: int
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

    def _timed(self, field: str, operation, *args):
        torch.cuda.synchronize(self.facade.device)
        started = time.perf_counter()
        try:
            return operation(*args)
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
            "prompt_seconds",
            self.facade.prefill,
            prepared,
            instrumentation,
        )

    def repeat_cache(self, cache, batch_size):
        try:
            return self.facade.repeat_cache(cache, batch_size)
        except torch.cuda.OutOfMemoryError:
            self.oom_fallbacks += 1
            raise

    def decode_step(self, *args):
        return self._timed(
            "decode_seconds",
            self.facade.decode_step,
            *args,
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


def benchmark_candidates(run: ResolvedRunConfig) -> tuple[int, ...]:
    return run.hardware.benchmark_microbatch_sizes


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


def _ids_hash(results) -> str:
    payload = [
        {
            "rollout_index": result.generation.key.rollout_index,
            "generated_token_ids": result.generation.generated_token_ids,
        }
        for result in results
    ]
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def benchmark_run(
    run: ResolvedRunConfig,
    output_path: str | Path,
    *,
    candidates: Sequence[int] | None = None,
    rollout_count: int | None = None,
    max_new_tokens: int | None = None,
) -> BenchmarkReport:
    """Benchmark one real visual example on the resolved target GPU."""

    _verify_hardware(run)
    sizes = tuple(candidates or benchmark_candidates(run))
    if not sizes or any(size <= 0 for size in sizes) or len(set(sizes)) != len(sizes):
        raise BenchmarkError("benchmark candidates must be unique and positive")
    count = max(sizes) if rollout_count is None else rollout_count
    if count < max(sizes):
        raise BenchmarkError(
            "rollout_count must exercise the largest microbatch"
        )

    token_limit = (
        run.generation.max_new_tokens
        if max_new_tokens is None
        else max_new_tokens
    )
    if token_limit <= 0:
        raise BenchmarkError("max_new_tokens must be positive")

    adapter = create_dataset_adapter(run.dataset.adapter)
    example = next(iter(islice(adapter.load_examples(run.dataset), 1)))
    messages = adapter.build_messages(
        example,
        PromptConfig(run.dataset.prompt_template),
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
            for size in sizes:
                timed = _TimingFacade(facade)
                engine = GenerationEngine(
                    timed,
                    instrumentation,
                    base_seed=run.base_seed,
                    max_new_tokens=token_limit,
                    rollout_microbatch_size=size,
                    seed_derivation_version=(
                        run.schemas.seed_derivation_version
                    ),
                )
                prepared = facade.prepare_example(messages)
                torch.cuda.reset_peak_memory_stats(facade.device)
                torch.cuda.synchronize(facade.device)
                started = time.perf_counter()
                try:
                    results = tuple(
                        engine.generate_example(
                            example,
                            prepared,
                            _keys(run, example.sample_id, count),
                            run.sampling.as_domain(),
                        )
                    )
                    torch.cuda.synchronize(facade.device)
                    total = time.perf_counter() - started
                    retained = sum(
                        len(result.generation.generated_token_ids)
                        for result in results
                    )
                    measurements.append(
                        BenchmarkMeasurement(
                            requested_microbatch_size=size,
                            rollout_count=count,
                            status=(
                                "complete"
                                if timed.oom_fallbacks == 0
                                else "fallback"
                            ),
                            prompt_seconds=timed.prompt_seconds,
                            decode_seconds=timed.decode_seconds,
                            total_seconds=total,
                            peak_allocated_bytes=(
                                torch.cuda.max_memory_allocated(
                                    facade.device
                                )
                            ),
                            retained_tokens=retained,
                            tokens_per_second=(
                                retained / total if total > 0 else None
                            ),
                            oom_fallbacks=timed.oom_fallbacks,
                            retained_ids_sha256=_ids_hash(results),
                        )
                    )
                except torch.cuda.OutOfMemoryError:
                    measurements.append(
                        BenchmarkMeasurement(
                            requested_microbatch_size=size,
                            rollout_count=count,
                            status="oom",
                            prompt_seconds=timed.prompt_seconds,
                            decode_seconds=timed.decode_seconds,
                            total_seconds=time.perf_counter() - started,
                            peak_allocated_bytes=(
                                torch.cuda.max_memory_allocated(
                                    facade.device
                                )
                            ),
                            retained_tokens=0,
                            tokens_per_second=None,
                            oom_fallbacks=timed.oom_fallbacks,
                            retained_ids_sha256=None,
                        )
                    )
                    instrumentation.cancel_step()
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
        sample_id=example.sample_id,
        hardware_profile=run.hardware.model_dump(mode="json"),
        environment=capture_environment(run.model.device),
        max_new_tokens=token_limit,
        measurements=tuple(measurements),
    )
    atomic_write_json(Path(output_path), asdict(report))
    return report
