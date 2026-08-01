"""Predictor-aligned cached rollout generation."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from typing import Any

import torch

from visconf.generation.rollout_state import RolloutState
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
    Example,
    GenerationRecord,
    HiddenStateMetricRecord,
    ProbabilityMetricRecord,
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
    results = {row_index: {} for row_index in row_indices}
    for scenario in (
        "all_layers_all_heads",
        "early_visual_integration",
        "visual_reasoning",
    ):
        valid_rows = [
            row_index
            for row_index in row_indices
            if observations[row_index].attention[scenario].valid
            and observations[row_index].attention[scenario].vector is not None
        ]
        invalid_rows = set(row_indices) - set(valid_rows)
        for row_index in invalid_rows:
            results[row_index][scenario] = _invalid_attention()
        if not valid_rows:
            continue
        vectors = torch.stack(
            [
                observations[row_index].attention[scenario].vector
                for row_index in valid_rows
            ]
        )
        metrics = compute_attention_metrics_batch(
            vectors,
            [groups_by_row[row_index] for row_index in valid_rows],
        )
        for row_index, metric in zip(valid_rows, metrics, strict=True):
            results[row_index][scenario] = metric
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
