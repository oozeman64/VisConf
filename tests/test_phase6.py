"""Phase 6 rollout-engine acceptance tests."""

from types import SimpleNamespace

import torch

from visconf.generation.engine import GenerationEngine
from visconf.metrics.accumulator import AttentionScenarioAggregate
from visconf.metrics.attention import ATTENTION_SCENARIOS
from visconf.metrics.hidden_state import HiddenStateMetrics
from visconf.models.instrumentation import RowStepObservation, StepObservations
from visconf.types import (
    Example,
    PromptRecord,
    RolloutKey,
    SamplingConfig,
    TokenGroups,
)


class FakeInstrumentation:
    def __init__(self) -> None:
        self.cancelled = 0

    def cancel_step(self) -> None:
        self.cancelled += 1


def observation(marker: float, width: int) -> RowStepObservation:
    aggregate = AttentionScenarioAggregate(
        valid=True,
        vector=torch.full((width,), marker),
    )
    hidden = HiddenStateMetrics(
        valid_layers_all=36,
        valid_layers_early_visual_integration=21,
        valid_layers_visual_reasoning=13,
        last_layer_valid=True,
        cosine_gen_imgproto_hidden_avg_all_layers=marker,
        cosine_gen_imgproto_hidden_last_layer=marker,
        cosine_gen_imgproto_hidden_early_visual_integration=marker,
        cosine_gen_imgproto_hidden_visual_reasoning=marker,
    )
    return RowStepObservation(
        attention={name: aggregate for name in ATTENTION_SCENARIOS},
        hidden_state=hidden,
    )


class FakeFacade:
    device = torch.device("cpu")

    def __init__(
        self,
        *,
        oom_above: int | None = None,
        forced_prefill_token: int | None = None,
    ) -> None:
        self.oom_above = oom_above
        self.forced_prefill_token = forced_prefill_token
        self.select_calls = 0

    def prefill(self, prepared, instrumentation):
        logits = torch.zeros(6)
        if self.forced_prefill_token is not None:
            logits.fill_(-100)
            logits[self.forced_prefill_token] = 100
        return SimpleNamespace(
            raw_logits=logits,
            cache=[[]],
            observations=StepObservations((observation(0.1, 2),)),
            attention_mask=torch.ones((1, 2), dtype=torch.long),
            rope_deltas=torch.zeros((1, 1), dtype=torch.long),
        )

    def repeat_cache(self, cache, batch_size: int):
        if self.oom_above is not None and batch_size > self.oom_above:
            raise torch.cuda.OutOfMemoryError("synthetic OOM")
        return [[] for _ in range(batch_size)]

    def select_cache_rows(self, cache, indices):
        self.select_calls += 1
        return [cache[index] for index in indices.tolist()]

    def decode_step(
        self,
        selected_token_ids,
        cache,
        attention_mask,
        rope_deltas,
        prompt_groups,
        instrumentation,
    ):
        histories = [
            [*history, int(token_id)]
            for history, token_id in zip(
                cache,
                selected_token_ids.tolist(),
                strict=True,
            )
        ]
        logits = []
        rows = []
        for history in histories:
            row = torch.full((6,), -100.0)
            if len(history) >= 2 or history[0] % 2 == 0:
                row[5] = 100
            else:
                row[1] = 100
            logits.append(row)
            rows.append(observation(0.1 + len(history) / 10, 2 + len(history)))
        return SimpleNamespace(
            raw_logits=torch.stack(logits),
            cache=histories,
            observations=StepObservations(tuple(rows)),
            attention_mask=torch.cat(
                (
                    attention_mask,
                    torch.ones(
                        (len(histories), 1),
                        dtype=attention_mask.dtype,
                    ),
                ),
                dim=1,
            ),
        )

    def stop_token_ids(self):
        return frozenset({5})

    def token_piece(self, token_id: int):
        return f"<{token_id}>"

    def token_text(self, token_id: int):
        return str(token_id)

    def decode_retained(self, token_ids):
        return " ".join(str(token_id) for token_id in token_ids)


def prepared_input():
    return SimpleNamespace(
        prompt_record=PromptRecord(
            rendered_prompt="prompt",
            prompt_token_ids=(10, 11),
            prompt_token_count=2,
        ),
        token_groups=TokenGroups(
            image_positions=torch.tensor([0]),
            prompt_text_positions=torch.tensor([1]),
            prompt_last_position=1,
            prompt_token_count=2,
        ),
    )


def rollout_keys(count: int):
    return tuple(
        RolloutKey(
            run_id="run",
            dataset="synthetic",
            split="test",
            sample_id="sample",
            strategy="sample_t1",
            rollout_index=index,
        )
        for index in range(count)
    )


SAMPLING = SamplingConfig(
    name="sample_t1",
    temperature=1.0,
    top_p=1.0,
    top_k=None,
    repetition_penalty=1.0,
)
EXAMPLE = Example(
    dataset="synthetic",
    split="test",
    sample_id="sample",
    source_row_index=0,
    question="question",
    images=(),
    ground_truth={},
    answer_type=None,
    metadata={},
)


def run_with_batch(batch_size: int, facade=None):
    instrumentation = FakeInstrumentation()
    engine = GenerationEngine(
        facade or FakeFacade(),
        instrumentation,
        base_seed=1234,
        max_new_tokens=3,
        rollout_microbatch_size=batch_size,
    )
    return (
        tuple(
            engine.generate_example(
                EXAMPLE,
                prepared_input(),
                rollout_keys(32),
                SAMPLING,
            )
        ),
        instrumentation,
    )


def generated_ids(results):
    return {
        row.generation.key.rollout_index: row.generation.generated_token_ids
        for row in results
    }


def test_rollouts_are_batch_invariant_and_predictor_aligned():
    single, _ = run_with_batch(1)
    batched, _ = run_with_batch(4)

    assert generated_ids(single) == generated_ids(batched)
    assert {len(row.tokens) for row in batched} == {0, 1, 2}

    for rollout in batched:
        count = len(rollout.tokens)
        assert len(rollout.probability) == count
        assert len(rollout.attention) == count
        assert len(rollout.hidden_state) == count
        for index, token in enumerate(rollout.tokens):
            retained_before = index
            assert token.key.step == index + 1
            assert token.predictor_position == (1 if index == 0 else 1 + index)
            assert token.context_length == 2 + retained_before
            assert rollout.attention[index].n_generated_text_tokens == retained_before
            marker = 0.1 + retained_before / 10
            assert (
                rollout.hidden_state[index]
                .metrics.cosine_gen_imgproto_hidden_last_layer
                == marker
            )


def test_oom_retries_same_rollouts_and_maximum_is_recorded():
    expected, _ = run_with_batch(2)
    recovered, instrumentation = run_with_batch(
        8,
        FakeFacade(oom_above=2),
    )
    assert generated_ids(recovered) == generated_ids(expected)
    assert instrumentation.cancelled == 2

    engine = GenerationEngine(
        FakeFacade(forced_prefill_token=0),
        FakeInstrumentation(),
        base_seed=1234,
        max_new_tokens=1,
        rollout_microbatch_size=1,
    )
    result = tuple(
        engine.generate_example(
            EXAMPLE,
            prepared_input(),
            rollout_keys(1),
            SAMPLING,
        )
    )[0]
    assert result.generation.generated_token_ids == (0,)
    assert result.generation.stop_reason == "max_new_tokens"
    assert result.generation.hit_max_new_tokens


def test_identity_cache_compaction_is_skipped():
    facade = FakeFacade(forced_prefill_token=1)
    engine = GenerationEngine(
        facade,
        FakeInstrumentation(),
        base_seed=1234,
        max_new_tokens=3,
        rollout_microbatch_size=4,
    )
    results = tuple(
        engine.generate_example(
            EXAMPLE,
            prepared_input(),
            rollout_keys(4),
            SAMPLING,
        )
    )
    assert len(results) == 4
    assert facade.select_calls == 0
