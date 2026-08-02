"""Predictor-aligned cached rollout generation."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence

from visconf.generation.rollout_state import RolloutState
from visconf.generation.scheduler import prompt_batch_telemetry
from visconf.generation.sampling import sample_next_tokens_prepared
from visconf.generation.stopping import StopReason, decide_stop
from visconf.metrics.accumulator import AttentionScenarioAggregate
from visconf.metrics.attention import (
    ATTENTION_METRICS,
    AttentionScenarioMetrics,
    StepTokenGroups,
    compute_attention_metrics,
    compute_attention_metrics_batch,
)
from visconf.metrics.probability import (
    ProbabilityMetrics,
    compute_probability_metrics,
    compute_probability_metrics_from_workspace,
    prepare_probability_batch,
)
from visconf.metrics.validation import MetricInputError
from visconf.models.instrumentation import (
    QwenInstrumentation,
    RowStepObservation,
)
from visconf.models.token_positions import (
    predictor_position,
)
from visconf.types import (
    AttentionMetricRecord,
    CompletedRollout,
    DecodeRowMapping,
    Example,
    GenerationRecord,
    HiddenStateMetricRecord,
    ProbabilityMetricRecord,
    PromptBatchWorkItem,
    RolloutKey,
    SamplingConfig,
    TokenKey,
    TokenRecord,
)
from visconf.utils.seeds import make_rollout_generator
from visconf.utils.logging import log_event


logger = logging.getLogger(__name__)


class GenerationError(ValueError):
    """Raised when rollout inputs do not describe one example and strategy."""


def _invalid_attention() -> AttentionScenarioMetrics:
    return AttentionScenarioMetrics(
        valid=False,
        **{name: None for name in ATTENTION_METRICS},
    )


def _scenario_metrics(
    aggregate: AttentionScenarioAggregate,
    groups,
) -> AttentionScenarioMetrics:
    if not aggregate.valid or aggregate.vector is None:
        return _invalid_attention()
    return compute_attention_metrics(aggregate.vector, groups)


def _scenario_metrics_batch(
    observations: list[RowStepObservation],
    row_indices: list[int],
    groups_by_row: dict[int, StepTokenGroups],
) -> dict[int, dict[str, AttentionScenarioMetrics]]:
    """Pad ragged logical vectors with excluded zeros and transfer once."""
    results = {row_index: {} for row_index in row_indices}
    for scenario in (
        "all_layers_all_heads",
        "early_visual_integration",
        "visual_reasoning",
    ):
        compatible_rows = []
        for row_index in row_indices:
            aggregate = observations[row_index].attention[scenario]
            if not aggregate.valid or aggregate.vector is None:
                results[row_index][scenario] = _invalid_attention()
                continue
            compatible_rows.append(row_index)
        if compatible_rows:
            row_vectors = [
                observations[row].attention[scenario].vector
                for row in compatible_rows
            ]
            if any(vector is None for vector in row_vectors):
                raise GenerationError("valid attention row has no vector")
            vectors_present = [
                vector for vector in row_vectors if vector is not None
            ]
            widths = {int(vector.numel()) for vector in vectors_present}
            vectors = torch.stack(
                vectors_present
            ) if len(widths) == 1 else pad_sequence(
                vectors_present,
                batch_first=True,
                padding_value=0.0,
            )
            metrics = compute_attention_metrics_batch(
                vectors,
                [groups_by_row[row] for row in compatible_rows],
            )
            for row, metric in zip(compatible_rows, metrics, strict=True):
                results[row][scenario] = metric
    return results



class GenerationEngine:
    def __init__(
        self,
        facade: Any,
        instrumentation: QwenInstrumentation,
        *,
        base_seed: int,
        max_new_tokens: int,
        rollout_microbatch_size: int,
        seed_derivation_version: int = 1,
    ) -> None:
        self.facade = facade
        self.instrumentation = instrumentation
        self.base_seed = base_seed
        self.max_new_tokens = max_new_tokens
        self.rollout_microbatch_size = rollout_microbatch_size
        self.seed_derivation_version = seed_derivation_version

    def generate_example(
        self,
        example: Example,
        prepared: Any,
        rollout_keys: Sequence[RolloutKey],
        sampling: SamplingConfig,
    ) -> Iterable[CompletedRollout]:
        self._validate_keys(example, rollout_keys, sampling)
        prefill = self.facade.prefill(prepared, self.instrumentation)

        offset = 0
        effective_batch_size = self.rollout_microbatch_size
        while offset < len(rollout_keys):
            batch_size = min(
                effective_batch_size,
                len(rollout_keys) - offset,
            )
            keys = rollout_keys[offset : offset + batch_size]
            first = keys[0]
            log_event(
                logger,
                "microbatch_started",
                run_id=first.run_id,
                dataset=first.dataset,
                strategy=first.strategy,
                sample_id=first.sample_id,
                offset=offset,
                microbatch_size=batch_size,
            )
            try:
                completed = self._generate_microbatch(
                    keys,
                    prepared,
                    prefill,
                    sampling,
                )
            except torch.cuda.OutOfMemoryError:
                self.instrumentation.cancel_step()
                if batch_size == 1:
                    raise
                effective_batch_size = max(1, batch_size // 2)
                log_event(
                    logger,
                    "oom_fallback",
                    run_id=first.run_id,
                    sample_id=first.sample_id,
                    requested_microbatch_size=batch_size,
                    effective_microbatch_size=effective_batch_size,
                )
                torch.cuda.empty_cache()
                continue
            log_event(
                logger,
                "microbatch_completed",
                run_id=first.run_id,
                sample_id=first.sample_id,
                offset=offset,
                microbatch_size=batch_size,
                rollout_count=len(completed),
            )
            yield from completed
            offset += batch_size

    def generate_prompt_batch(
        self,
        work_items: Sequence[PromptBatchWorkItem],
        sampling: SamplingConfig,
    ) -> tuple[CompletedRollout, ...]:
        """Prefill independent prompts once and decode their rollout rows together."""
        items = tuple(work_items)
        if not items:
            return ()
        for item in items:
            self._validate_keys(item.example, item.pending_rollout_keys, sampling)
        if (
            len(items) == 1
            and hasattr(self.facade, "prefill")
            and hasattr(self.facade, "repeat_cache")
        ):
            item = items[0]
            return tuple(
                self.generate_example(
                    item.example,
                    item.prepared,
                    item.pending_rollout_keys,
                    sampling,
                )
            )
        try:
            prepared_batch = self.facade.prepare_batch(
                tuple(item.prepared for item in items)
            )
            prefill = self.facade.prefill_batch(
                prepared_batch,
                self.instrumentation,
            )
        except torch.cuda.OutOfMemoryError:
            self.instrumentation.cancel_step()
            if len(items) == 1:
                raise
            midpoint = max(1, len(items) // 2)
            torch.cuda.empty_cache()
            return (
                *self.generate_prompt_batch(items[:midpoint], sampling),
                *self.generate_prompt_batch(items[midpoint:], sampling),
            )

        offsets = [0] * len(items)
        completed: list[CompletedRollout] = []
        effective_rollouts = self.rollout_microbatch_size
        while any(
            offsets[row] < len(item.pending_rollout_keys)
            for row, item in enumerate(items)
        ):
            mappings: list[DecodeRowMapping] = []
            for prompt_row, item in enumerate(items):
                keys = item.pending_rollout_keys[
                    offsets[prompt_row] :
                    offsets[prompt_row] + effective_rollouts
                ]
                for key in keys:
                    state = self._new_state(key)
                    mappings.append(
                        DecodeRowMapping(
                            model_row=len(mappings),
                            prompt_row=prompt_row,
                            sample_id=item.sample_id,
                            rollout_key=key,
                            rollout_state=state,
                        )
                    )
            telemetry = prompt_batch_telemetry(
                items,
                self.rollout_microbatch_size,
                max(
                    (
                        sum(
                            mapping.prompt_row == row
                            for mapping in mappings
                        )
                        for row in range(len(items))
                    ),
                    default=0,
                ),
            )
            log_event(
                logger,
                "prompt_batch_started",
                **asdict(telemetry),
                requested_prompt_count=len(items),
                total_active_rows=len(mappings),
            )
            try:
                wave = self._generate_prompt_rows(
                    items,
                    mappings,
                    prefill,
                    sampling,
                )
            except torch.cuda.OutOfMemoryError:
                self.instrumentation.cancel_step()
                if effective_rollouts == 1:
                    raise
                effective_rollouts = max(1, effective_rollouts // 2)
                torch.cuda.empty_cache()
                log_event(
                    logger,
                    "oom_fallback",
                    requested_prompt_count=len(items),
                    effective_prompt_count=len(items),
                    requested_rollout_rows_per_prompt=self.rollout_microbatch_size,
                    effective_rollout_rows_per_prompt=effective_rollouts,
                    total_active_rows=len(mappings),
                )
                continue
            completed.extend(wave)
            counts = [0] * len(items)
            for mapping in mappings:
                counts[mapping.prompt_row] += 1
            offsets = [
                offset + count
                for offset, count in zip(offsets, counts, strict=True)
            ]
        completed.sort(
            key=lambda bundle: (
                next(
                    item.canonical_source_ordinal
                    for item in items
                    if item.sample_id == bundle.generation.key.sample_id
                ),
                bundle.generation.key.rollout_index,
            )
        )
        return tuple(completed)

    def _generate_prompt_rows(
        self,
        items: tuple[PromptBatchWorkItem, ...],
        mappings: list[DecodeRowMapping],
        prefill: Any,
        sampling: SamplingConfig,
    ) -> tuple[CompletedRollout, ...]:
        started = time.perf_counter()
        active = list(mappings)
        source_indices = torch.tensor(
            [row.prompt_row for row in active],
            dtype=torch.long,
            device=self.facade.device,
        )
        cache = self.facade.select_cache_sources(prefill.cache, source_indices)
        attention_mask = prefill.attention_mask.index_select(0, source_indices)
        raw_logits = prefill.raw_logits.index_select(0, source_indices)
        observations = [
            prefill.observations.rows[row.prompt_row] for row in active
        ]
        rope_deltas = (
            prefill.rope_deltas.index_select(0, source_indices)
            if prefill.rope_deltas.ndim > 0
            and prefill.rope_deltas.shape[0] == len(items)
            else prefill.rope_deltas
        )
        logical_lengths = prefill.logical_context_lengths.index_select(
            0, source_indices
        )
        stop_token_ids = self.facade.stop_token_ids()
        prompt_groups = tuple(item.token_groups for item in items)
        groups_cache: dict[tuple[int, int], StepTokenGroups] = {}

        def groups_for(prompt_row: int, retained_before: int) -> StepTokenGroups:
            key = (prompt_row, retained_before)
            if key not in groups_cache:
                token_groups = items[prompt_row].token_groups
                groups_cache[key] = StepTokenGroups(
                    image_positions=tuple(token_groups.image_positions.tolist()),
                    prompt_text_positions=tuple(
                        token_groups.prompt_text_positions.tolist()
                    ),
                    generated_text_positions=tuple(
                        range(
                            token_groups.prompt_token_count,
                            token_groups.prompt_token_count + retained_before,
                        )
                    ),
                )
            return groups_cache[key]

        while active:
            workspace = prepare_probability_batch(raw_logits)
            states = [row.rollout_state for row in active]
            sampled_ids = sample_next_tokens_prepared(
                workspace.logits,
                workspace.sorted_indices,
                sampling,
                tuple(state.generator for state in states),
            )
            decisions = [
                decide_stop(
                    selected,
                    len(state.generated_token_ids),
                    self.max_new_tokens,
                    stop_token_ids,
                )
                for state, selected in zip(states, sampled_ids, strict=True)
            ]
            retained_indices = [
                index for index, decision in enumerate(decisions)
                if decision.retain_token
            ]
            probability_by_row: dict[
                int, tuple[ProbabilityMetrics | None, str | None]
            ] = {}
            if retained_indices:
                retained_tensor = torch.tensor(
                    retained_indices, dtype=torch.long, device=self.facade.device
                )
                selected_tensor = torch.tensor(
                    [sampled_ids[index] for index in retained_indices],
                    dtype=torch.long,
                    device=self.facade.device,
                )
                try:
                    values = compute_probability_metrics_from_workspace(
                        workspace, selected_tensor, retained_tensor
                    )
                    probability_by_row.update({
                        row: (metric, None)
                        for row, metric in zip(retained_indices, values, strict=True)
                    })
                except MetricInputError:
                    for row in retained_indices:
                        try:
                            probability_by_row[row] = (
                                compute_probability_metrics(
                                    raw_logits[row], sampled_ids[row]
                                ),
                                None,
                            )
                        except MetricInputError:
                            probability_by_row[row] = (
                                None, "invalid_probability_logits"
                            )
            groups_by_row = {
                row: groups_for(
                    active[row].prompt_row,
                    len(active[row].rollout_state.generated_token_ids),
                )
                for row in retained_indices
            }
            attention_by_row = _scenario_metrics_batch(
                observations, retained_indices, groups_by_row
            )
            selected_ids: list[int] = []
            survivors: list[DecodeRowMapping] = []
            survivor_indices: list[int] = []
            for row_index, (mapping, selected, decision) in enumerate(
                zip(active, sampled_ids, decisions, strict=True)
            ):
                state = mapping.rollout_state
                if decision.retain_token:
                    probability, invalid_reason = probability_by_row[row_index]
                    self._append_token(
                        state,
                        selected,
                        probability,
                        invalid_reason,
                        observations[row_index],
                        items[mapping.prompt_row].prepared,
                        groups_by_row[row_index],
                        attention_by_row[row_index],
                    )
                if decision.stop:
                    state.finish(decision.reason, decision.terminating_token_id)
                else:
                    mapping.model_row = len(survivors)
                    selected_ids.append(selected)
                    survivors.append(mapping)
                    survivor_indices.append(row_index)
            if not survivors:
                break
            if len(survivors) != len(active):
                indices = torch.tensor(
                    survivor_indices, dtype=torch.long, device=self.facade.device
                )
                cache = self.facade.select_cache_rows(cache, indices)
                attention_mask = attention_mask.index_select(0, indices)
                rope_deltas = (
                    rope_deltas.index_select(0, indices)
                    if rope_deltas.ndim > 0
                    and rope_deltas.shape[0] == len(active)
                    else rope_deltas
                )
                logical_lengths = logical_lengths.index_select(0, indices)
            source_mapping = tuple(row.prompt_row for row in survivors)
            decoded = self.facade.decode_step(
                torch.tensor(selected_ids, dtype=torch.long, device=self.facade.device),
                cache,
                attention_mask,
                rope_deltas,
                prompt_groups,
                self.instrumentation,
                source_prompt_indices=source_mapping,
                logical_context_lengths=logical_lengths,
            )
            active = survivors
            cache = decoded.cache
            attention_mask = decoded.attention_mask
            raw_logits = decoded.raw_logits
            observations = list(decoded.observations.rows)
            logical_lengths = logical_lengths + 1

        elapsed = time.perf_counter() - started
        attributed = elapsed / len(mappings)
        completed_at = datetime.now(timezone.utc)
        return tuple(
            self._completed_rollout(
                mapping.rollout_state,
                sampling,
                items[mapping.prompt_row].prepared,
                attributed,
                completed_at,
            )
            for mapping in mappings
        )

    def _validate_keys(
        self,
        example: Example,
        rollout_keys: Sequence[RolloutKey],
        sampling: SamplingConfig,
    ) -> None:
        if not rollout_keys:
            raise GenerationError("at least one rollout key is required")
        first = rollout_keys[0]
        expected = (
            first.run_id,
            example.dataset,
            example.split,
            example.sample_id,
            sampling.name,
        )
        seen_indices = set()
        for key in rollout_keys:
            actual = (
                key.run_id,
                key.dataset,
                key.split,
                key.sample_id,
                key.strategy,
            )
            if actual != expected:
                raise GenerationError(
                    "rollout keys must share one run, example, and strategy"
                )
            if key.rollout_index in seen_indices:
                raise GenerationError("rollout indices must be unique")
            seen_indices.add(key.rollout_index)

    def _new_state(self, key: RolloutKey) -> RolloutState:
        seed, generator = make_rollout_generator(
            base_seed=self.base_seed,
            dataset=key.dataset,
            sample_id=key.sample_id,
            strategy=key.strategy,
            rollout_index=key.rollout_index,
            device=self.facade.device,
            seed_derivation_version=self.seed_derivation_version,
        )
        return RolloutState(
            key=key,
            rollout_seed=seed,
            generator=generator,
        )

    def _generate_microbatch(
        self,
        keys: Sequence[RolloutKey],
        prepared: Any,
        prefill: Any,
        sampling: SamplingConfig,
    ) -> tuple[CompletedRollout, ...]:
        started = time.perf_counter()
        states = [self._new_state(key) for key in keys]
        active = list(states)
        cache = self.facade.repeat_cache(prefill.cache, len(active))
        attention_mask = prefill.attention_mask.repeat(len(active), 1)
        raw_logits = prefill.raw_logits.unsqueeze(0).expand(
            len(active), -1
        )
        observations = [
            prefill.observations.rows[0] for _ in active
        ]
        stop_token_ids = self.facade.stop_token_ids()
        image_positions = tuple(
            prepared.token_groups.image_positions.tolist()
        )
        prompt_text_positions = tuple(
            prepared.token_groups.prompt_text_positions.tolist()
        )
        groups_cache: dict[int, StepTokenGroups] = {}

        def groups_for(retained_before: int) -> StepTokenGroups:
            if retained_before not in groups_cache:
                groups_cache[retained_before] = StepTokenGroups(
                    image_positions=image_positions,
                    prompt_text_positions=prompt_text_positions,
                    generated_text_positions=tuple(
                        range(
                            prepared.token_groups.prompt_token_count,
                            prepared.token_groups.prompt_token_count
                            + retained_before,
                        )
                    ),
                )
            return groups_cache[retained_before]

        while active:
            probability_workspace = prepare_probability_batch(raw_logits)
            sampled_ids = sample_next_tokens_prepared(
                probability_workspace.logits,
                probability_workspace.sorted_indices,
                sampling,
                tuple(state.generator for state in active),
            )
            decisions = [
                decide_stop(
                    selected,
                    len(state.generated_token_ids),
                    self.max_new_tokens,
                    stop_token_ids,
                )
                for state, selected in zip(
                    active,
                    sampled_ids,
                    strict=True,
                )
            ]
            retained_indices = [
                index
                for index, decision in enumerate(decisions)
                if decision.retain_token
            ]
            probability_by_row: dict[
                int,
                tuple[ProbabilityMetrics | None, str | None],
            ] = {}
            if retained_indices:
                retained_tensor = torch.tensor(
                    retained_indices,
                    dtype=torch.long,
                    device=self.facade.device,
                )
                selected_tensor = torch.tensor(
                    [sampled_ids[index] for index in retained_indices],
                    dtype=torch.long,
                    device=self.facade.device,
                )
                try:
                    batch_metrics = compute_probability_metrics_from_workspace(
                        probability_workspace,
                        selected_tensor,
                        retained_tensor,
                    )
                    probability_by_row.update(
                        {
                            row_index: (metric, None)
                            for row_index, metric in zip(
                                retained_indices,
                                batch_metrics,
                                strict=True,
                            )
                        }
                    )
                except MetricInputError:
                    for row_index in retained_indices:
                        try:
                            metric = compute_probability_metrics(
                                raw_logits[row_index],
                                sampled_ids[row_index],
                            )
                            probability_by_row[row_index] = (metric, None)
                        except MetricInputError:
                            probability_by_row[row_index] = (
                                None,
                                "invalid_probability_logits",
                            )
            groups_by_row = {
                row_index: groups_for(
                    len(active[row_index].generated_token_ids)
                )
                for row_index in retained_indices
            }
            attention_by_row = _scenario_metrics_batch(
                observations,
                retained_indices,
                groups_by_row,
            )

            selected_ids: list[int] = []
            survivors: list[RolloutState] = []
            survivor_indices: list[int] = []
            for row_index, (state, selected, decision) in enumerate(
                zip(active, sampled_ids, decisions, strict=True)
            ):
                if decision.retain_token:
                    probability, invalid_reason = probability_by_row[row_index]
                    self._append_token(
                        state,
                        selected,
                        probability,
                        invalid_reason,
                        observations[row_index],
                        prepared,
                        groups_by_row[row_index],
                        attention_by_row[row_index],
                    )
                if decision.stop:
                    state.finish(
                        decision.reason,
                        decision.terminating_token_id,
                    )
                else:
                    selected_ids.append(selected)
                    survivors.append(state)
                    survivor_indices.append(row_index)

            if not survivors:
                break
            if len(survivors) != len(active):
                indices = torch.tensor(
                    survivor_indices,
                    dtype=torch.long,
                    device=self.facade.device,
                )
                cache = self.facade.select_cache_rows(cache, indices)
                attention_mask = attention_mask.index_select(0, indices)
            decoded = self.facade.decode_step(
                torch.tensor(
                    selected_ids,
                    dtype=torch.long,
                    device=self.facade.device,
                ),
                cache,
                attention_mask,
                prefill.rope_deltas,
                prepared.token_groups,
                self.instrumentation,
            )
            active = survivors
            cache = decoded.cache
            attention_mask = decoded.attention_mask
            raw_logits = decoded.raw_logits
            observations = list(decoded.observations.rows)

        elapsed = time.perf_counter() - started
        attributed = elapsed / len(states)
        completed_at = datetime.now(timezone.utc)
        return tuple(
            self._completed_rollout(
                state,
                sampling,
                prepared,
                attributed,
                completed_at,
            )
            for state in states
        )

    def _append_token(
        self,
        state: RolloutState,
        selected_token_id: int,
        probability: ProbabilityMetrics | None,
        probability_invalid_reason: str | None,
        observation: RowStepObservation,
        prepared: Any,
        groups: StepTokenGroups,
        attention_results: dict[str, AttentionScenarioMetrics],
    ) -> None:
        retained_before = len(state.generated_token_ids)
        step = retained_before + 1
        token_key = TokenKey(state.key, step)
        if probability is not None:
            probability_record = ProbabilityMetricRecord(
                key=token_key,
                metrics_valid=True,
                invalid_reason=None,
                metrics=probability,
            )
        else:
            probability_record = ProbabilityMetricRecord(
                key=token_key,
                metrics_valid=False,
                invalid_reason=probability_invalid_reason,
                metrics=None,
            )

        state.generated_token_ids.append(selected_token_id)
        token_strings = getattr(self.facade, "token_strings", None)
        if token_strings is None:
            token_piece = self.facade.token_piece(selected_token_id)
            token_text = self.facade.token_text(selected_token_id)
        else:
            token_piece, token_text = token_strings(selected_token_id)
        state.tokens.append(
            TokenRecord(
                key=token_key,
                token_id=selected_token_id,
                token_piece=token_piece,
                token_text=token_text,
                predictor_position=predictor_position(
                    prepared.token_groups,
                    retained_before,
                ),
                context_length=(
                    prepared.prompt_record.prompt_token_count
                    + retained_before
                ),
            )
        )
        state.probability.append(probability_record)
        state.attention.append(
            AttentionMetricRecord(
                key=token_key,
                n_image_tokens=len(groups.image_positions),
                n_prompt_text_tokens=len(groups.prompt_text_positions),
                n_generated_text_tokens=len(
                    groups.generated_text_positions
                ),
                all_layers_all_heads=attention_results[
                    "all_layers_all_heads"
                ],
                early_visual_integration=attention_results[
                    "early_visual_integration"
                ],
                visual_reasoning=attention_results["visual_reasoning"],
            )
        )
        state.hidden_state.append(
            HiddenStateMetricRecord(
                key=token_key,
                metrics=observation.hidden_state,
            )
        )

    def _completed_rollout(
        self,
        state: RolloutState,
        sampling: SamplingConfig,
        prepared: Any,
        wall_time_seconds: float,
        completed_at: datetime,
    ) -> CompletedRollout:
        if state.active or state.stop_reason is None:
            raise GenerationError("rollout did not reach a stop condition")
        token_ids = tuple(state.generated_token_ids)
        generation = GenerationRecord(
            key=state.key,
            rollout_seed=state.rollout_seed,
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            top_k=sampling.top_k,
            repetition_penalty=sampling.repetition_penalty,
            generated_token_ids=token_ids,
            generated_text=self.facade.decode_retained(token_ids),
            stop_reason=state.stop_reason.value,
            terminating_token_id=state.terminating_token_id,
            hit_max_new_tokens=(
                state.stop_reason is StopReason.MAX_NEW_TOKENS
            ),
            prompt_token_count=prepared.prompt_record.prompt_token_count,
            wall_time_seconds=wall_time_seconds,
            tokens_per_second=(
                len(token_ids) / wall_time_seconds
                if wall_time_seconds > 0
                else None
            ),
            completed_at_utc=completed_at,
        )
        return CompletedRollout(
            generation=generation,
            tokens=tuple(state.tokens),
            probability=tuple(state.probability),
            attention=tuple(state.attention),
            hidden_state=tuple(state.hidden_state),
        )
