from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import visconf.models.instrumentation as instrumentation_module
from visconf.config import GenerationSettings, HardwareSettings
from visconf.generation.engine import GenerationEngine
from visconf.generation.scheduler import (
    prompt_batch_telemetry,
    schedule_prompt_batches,
)
from visconf.metrics.accumulator import AttentionScenarioAggregate
from visconf.metrics.attention import ATTENTION_SCENARIOS
from visconf.metrics.hidden_state import HiddenStateMetrics
from visconf.models.instrumentation import (
    QwenInstrumentation,
    RowStepObservation,
    StepObservations,
)
from visconf.types import (
    Example,
    PromptBatchWorkItem,
    PromptRecord,
    RolloutKey,
    SamplingConfig,
    TokenGroups,
)


def _example(sample: str, ordinal: int) -> Example:
    return Example(
        dataset="synthetic",
        split="test",
        sample_id=sample,
        source_row_index=ordinal,
        question=sample,
        images=(),
        ground_truth={},
        answer_type=None,
        metadata={},
    )


def _key(sample: str, index: int) -> RolloutKey:
    return RolloutKey("run", "synthetic", "test", sample, "sample", index)


def _work(sample: str, ordinal: int, length: int, images: int = 0) -> PromptBatchWorkItem:
    groups = TokenGroups(
        image_positions=torch.arange(images, dtype=torch.long),
        prompt_text_positions=torch.arange(images, length, dtype=torch.long),
        prompt_last_position=length - 1,
        prompt_token_count=length,
    )
    record = PromptRecord(sample, tuple(range(length)), length)
    prepared = SimpleNamespace(prompt_record=record, token_groups=groups)
    return PromptBatchWorkItem(
        example=_example(sample, ordinal),
        sample_id=sample,
        canonical_source_ordinal=ordinal,
        prompt_record=record,
        token_groups=groups,
        image_token_count=images,
        pending_rollout_keys=(_key(sample, 0), _key(sample, 2)),
        prepared=prepared,
    )


def test_configuration_and_bounded_scheduler_contracts():
    assert GenerationSettings().prompt_batch_size == 1
    with pytest.raises(ValueError, match="requires"):
        GenerationSettings(prompt_batching_strategy="token_count_bucketed")
    with pytest.raises(ValueError, match="at least"):
        GenerationSettings(
            prompt_batch_size=3,
            prompt_batching_strategy="token_count_bucketed",
            prompt_bucket_window_size=2,
        )
    hardware = HardwareSettings(
        name="test",
        accelerator="h100",
        memory_gb=80,
        default_rollout_microbatch_size=2,
        benchmark_microbatch_sizes=(1, 2),
        max_rollout_microbatch_size=2,
        max_active_decode_rows=4,
        benchmark_batch_shapes=((1, 2), (2, 2)),
    )
    assert hardware.benchmark_batch_shapes == ((1, 2), (2, 2))
    with pytest.raises(ValueError, match="batch shapes"):
        HardwareSettings(
            name="test",
            accelerator="h100",
            memory_gb=80,
            default_rollout_microbatch_size=2,
            benchmark_microbatch_sizes=(1, 2),
            max_rollout_microbatch_size=2,
            max_active_decode_rows=3,
            benchmark_batch_shapes=((2, 2),),
        )

    items = (
        _work("a", 0, 9, 2),
        _work("b", 1, 3, 0),
        _work("c", 2, 3, 0),
        _work("d", 3, 8, 1),
        _work("e", 4, 2, 0),
    )
    contiguous = tuple(
        schedule_prompt_batches(
            items, prompt_batch_size=2, strategy="contiguous"
        )
    )
    assert [[x.sample_id for x in unit] for unit in contiguous] == [
        ["a", "b"], ["c", "d"], ["e"]
    ]
    bucketed = tuple(
        schedule_prompt_batches(
            items,
            prompt_batch_size=2,
            strategy="token_count_bucketed",
            bucket_window_size=4,
        )
    )
    # The final item cannot move into or reorder the first bounded window.
    assert [[x.sample_id for x in unit] for unit in bucketed] == [
        ["b", "c"], ["d", "a"], ["e"]
    ]
    assert sorted(x.sample_id for unit in bucketed for x in unit) == [
        "a", "b", "c", "d", "e"
    ]
    one = tuple(
        schedule_prompt_batches(
            items,
            prompt_batch_size=1,
            strategy="token_count_bucketed",
            bucket_window_size=3,
        )
    )
    assert [unit[0].sample_id for unit in one] == ["a", "b", "c", "d", "e"]
    assert sum(
        prompt_batch_telemetry(unit, 2, 2).total_padded_prompt_tokens
        for unit in bucketed
    ) <= sum(
        prompt_batch_telemetry(unit, 2, 2).total_padded_prompt_tokens
        for unit in contiguous
    )


class _Layer(nn.Module):
    def __init__(self, number: int):
        super().__init__()
        self.number = number

    def forward(self, hidden_states, output_attentions=False, **kwargs):
        hidden = hidden_states + self.number
        batch, width, _ = hidden.shape
        attention = torch.zeros(batch, 1, width, width)
        for row in range(batch):
            attention[row].fill_(self.number + row)
        return (hidden, attention) if output_attentions else (hidden,)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SimpleNamespace(
            language_model=SimpleNamespace(
                layers=nn.ModuleList(_Layer(i) for i in range(1, 37))
            )
        )


def _forward(model, hidden):
    for layer in model.model.language_model.layers:
        hidden = layer(hidden)[0]


def test_row_specific_instrumentation_padding_images_and_queries():
    model = _Model()
    groups = (
        TokenGroups(
            image_positions=torch.tensor([], dtype=torch.long),
            prompt_text_positions=torch.tensor([0, 1, 2]),
            prompt_last_position=2,
            prompt_token_count=3,
        ),
        TokenGroups(
            image_positions=torch.tensor([1, 2]),
            prompt_text_positions=torch.tensor([0, 3, 4]),
            prompt_last_position=4,
            prompt_token_count=5,
        ),
    )
    mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
    hidden = torch.tensor([
        [[99.0, 99.0], [99.0, 99.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
        [[0.0, 1.0], [0.0, 2.0], [0.0, 4.0], [0.0, 8.0], [0.0, 16.0]],
    ])
    with QwenInstrumentation(model) as instrumentation:
        instrumentation.begin_prefill(groups, mask)
        _forward(model, hidden)
        rows = instrumentation.finish_step().rows
        assert rows[0].attention["all_layers_all_heads"].vector.numel() == 3
        assert rows[1].attention["all_layers_all_heads"].vector.numel() == 5
        assert rows[0].hidden_state.valid_layers_all == 0
        assert rows[1].hidden_state.valid_layers_all == 36
        prototypes = instrumentation.image_prototypes_by_prompt
        assert all(layer is None for layer in prototypes[0])
        assert all(layer is not None for layer in prototypes[1])


def test_instrumentation_vectorizes_rollouts_by_source_prompt(monkeypatch):
    model = _Model()
    groups = (
        TokenGroups(
            image_positions=torch.tensor([1]),
            prompt_text_positions=torch.tensor([0, 2]),
            prompt_last_position=2,
            prompt_token_count=3,
        ),
        TokenGroups(
            image_positions=torch.tensor([1, 2]),
            prompt_text_positions=torch.tensor([0, 3, 4]),
            prompt_last_position=4,
            prompt_token_count=5,
        ),
    )
    prompt_mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
    prompt_hidden = torch.ones((2, 5, 2))
    calls = []
    original = instrumentation_module.compute_layer_cosines_tensor

    def tracked(hidden, prototype):
        calls.append(hidden.shape[0])
        return original(hidden, prototype)

    monkeypatch.setattr(
        instrumentation_module,
        "compute_layer_cosines_tensor",
        tracked,
    )
    with QwenInstrumentation(model) as instrumentation:
        instrumentation.begin_prefill(groups, prompt_mask)
        _forward(model, prompt_hidden)
        instrumentation.finish_step()
        calls.clear()
        rollout_mask = prompt_mask.index_select(
            0,
            torch.tensor([0, 0, 1, 1]),
        )
        instrumentation.begin_decode(
            groups,
            4,
            rollout_mask,
            source_prompt_indices=(0, 0, 1, 1),
        )
        _forward(model, torch.ones((4, 5, 2)))
        observations = instrumentation.finish_step()

    assert len(observations.rows) == 4
    assert calls == [2, 2] * 36


def _observation(marker: float, width: int) -> RowStepObservation:
    aggregate = AttentionScenarioAggregate(True, torch.full((width,), marker))
    hidden = HiddenStateMetrics(
        36, 21, 13, True, marker, marker, marker, marker
    )
    return RowStepObservation(
        {name: aggregate for name in ATTENTION_SCENARIOS}, hidden
    )


class _BatchFacade:
    device = torch.device("cpu")

    def prepare_batch(self, prompts):
        return prompts

    def prefill_batch(self, prompts, instrumentation):
        logits = torch.full((len(prompts), 6), -100.0)
        for row, prompt in enumerate(prompts):
            logits[row, 1 if prompt.prompt_record.rendered_prompt == "a" else 2] = 100.0
        lengths = torch.tensor(
            [p.prompt_record.prompt_token_count for p in prompts]
        )
        width = int(lengths.max())
        masks = torch.stack([
            torch.cat((torch.zeros(width - int(n)), torch.ones(int(n))))
            for n in lengths
        ]).long()
        return SimpleNamespace(
            raw_logits=logits,
            cache=[[] for _ in prompts],
            observations=StepObservations(tuple(
                _observation(row + 0.1, int(lengths[row]))
                for row in range(len(prompts))
            )),
            attention_mask=masks,
            rope_deltas=torch.zeros((len(prompts), 1), dtype=torch.long),
            logical_context_lengths=lengths,
        )

    def select_cache_sources(self, cache, indices):
        return [list(cache[index]) for index in indices.tolist()]

    def select_cache_rows(self, cache, indices):
        return [cache[index] for index in indices.tolist()]

    def decode_step(
        self, selected_token_ids, cache, attention_mask, rope_deltas,
        prompt_groups, instrumentation, source_prompt_indices=None,
        logical_context_lengths=None,
    ):
        histories = [
            [*history, int(token)]
            for history, token in zip(cache, selected_token_ids, strict=True)
        ]
        logits = torch.full((len(histories), 6), -100.0)
        logits[:, 5] = 100.0
        masks = torch.cat((
            attention_mask,
            torch.ones((len(histories), 1), dtype=attention_mask.dtype),
        ), dim=1)
        return SimpleNamespace(
            raw_logits=logits,
            cache=histories,
            observations=StepObservations(tuple(
                _observation(source + 0.2, int(masks[row].sum()))
                for row, source in enumerate(source_prompt_indices)
            )),
            attention_mask=masks,
        )

    def stop_token_ids(self):
        return frozenset({5})

    def token_strings(self, token):
        return f"<{token}>", str(token)

    def decode_retained(self, tokens):
        return " ".join(map(str, tokens))


class _Instrumentation:
    def __init__(self):
        self.cancelled = 0

    def cancel_step(self):
        self.cancelled += 1


def test_fake_prompt_batched_generation_matches_size_one_and_demultiplexes():
    items = (_work("a", 0, 3), _work("b", 1, 5))
    sampling = SamplingConfig("sample", 1.0, 1.0, None, 1.0)
    engine = GenerationEngine(
        _BatchFacade(),
        _Instrumentation(),
        base_seed=7,
        max_new_tokens=3,
        rollout_microbatch_size=2,
    )
    batched = engine.generate_prompt_batch(items, sampling)
    baseline = tuple(
        result
        for item in items
        for result in engine.generate_prompt_batch((item,), sampling)
    )
    ids = {
        (x.generation.key.sample_id, x.generation.key.rollout_index):
        x.generation.generated_token_ids
        for x in batched
    }
    assert ids == {
        (x.generation.key.sample_id, x.generation.key.rollout_index):
        x.generation.generated_token_ids
        for x in baseline
    }
    assert {key[0] for key in ids} == {"a", "b"}
    assert {value for key, value in ids.items() if key[0] == "a"} == {(1,)}
    assert {value for key, value in ids.items() if key[0] == "b"} == {(2,)}
    for rollout in batched:
        assert rollout.tokens[0].context_length == (
            3 if rollout.generation.key.sample_id == "a" else 5
        )
        assert rollout.attention[0].n_generated_text_tokens == 0
