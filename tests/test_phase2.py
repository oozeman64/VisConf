import math
from dataclasses import fields

import pytest
import torch

from visconf.generation.sampling import (
    apply_sampling_transforms,
    apply_sampling_transforms_batch,
    sample_next_token,
    sample_next_tokens,
)
from visconf.generation.stopping import (
    StopReason,
    decide_stop,
    resolve_stop_token_ids,
)
from visconf.metrics.accumulator import (
    AttentionAccumulator,
    HiddenStateAccumulator,
)
from visconf.metrics.attention import (
    ATTENTION_METRICS,
    ATTENTION_SCENARIOS,
    AttentionScenarioMetrics,
    StepTokenGroups,
    compute_attention_metrics,
    compute_attention_metrics_batch,
)
from visconf.metrics.hidden_state import (
    HIDDEN_STATE_METRICS,
    aggregate_hidden_metrics,
    compute_layer_cosine,
    compute_layer_cosines,
)
from visconf.metrics.probability import (
    PROBABILITY_METRICS,
    ProbabilityMetrics,
    compute_probability_metrics,
    compute_probability_metrics_batch,
)
from visconf.types import SamplingConfig
from visconf.utils.seeds import (
    derive_rollout_seed,
    make_rollout_generator,
)


def test_probability_metrics_follow_the_full_support_definitions() -> None:
    logits = torch.log(torch.tensor([0.5, 0.25, 0.25]))
    metrics = compute_probability_metrics(logits, selected_token_id=1)

    assert len(PROBABILITY_METRICS) == 31
    assert {field.name for field in fields(ProbabilityMetrics)} == set(
        PROBABILITY_METRICS
    )
    assert metrics.logp == pytest.approx(math.log(0.25))
    assert metrics.gini == pytest.approx(0.375)
    assert metrics.entropy == pytest.approx(
        0.5 * math.log(0.5) + 0.5 * math.log(0.25)
    )
    assert metrics.kl_p_u == pytest.approx(
        math.log(3) + metrics.entropy
    )
    assert metrics.max_prob == pytest.approx(0.5)
    assert metrics.margin_top2 == pytest.approx(0.25)
    assert metrics.log_ratio_margin_top2 == pytest.approx(math.log(2))
    assert metrics.selected_dominance == pytest.approx(-math.log(2))
    assert metrics.selected_rank == -2
    assert metrics.selected_logrank == pytest.approx(-math.log(2))
    assert metrics.topk_mass_2 == pytest.approx(0.75)
    assert metrics.tail_mass_2 == pytest.approx(-0.25)
    assert metrics.nucleus_size_0p9 == -3
    assert metrics.renyi_entropy_1 == pytest.approx(metrics.entropy)
    assert metrics.renyi_entropy_2 == pytest.approx(math.log(metrics.gini))
    assert metrics.renyi_entropy_inf == pytest.approx(
        math.log(metrics.max_prob)
    )

    point_mass = compute_probability_metrics(
        torch.tensor([0.0, -torch.inf, -torch.inf]),
        selected_token_id=0,
    )
    assert math.isinf(point_mass.kl_u_p)
    assert math.isinf(point_mass.log_ratio_margin_top2)
    assert math.isinf(point_mass.selected_dominance)


def test_attention_and_hidden_metrics_cover_fixed_aggregates() -> None:
    vector = torch.tensor([0.4, 0.3, 0.2, 0.1, 0.0])
    groups = StepTokenGroups(
        image_positions=(0, 4),
        prompt_text_positions=(1, 2),
        generated_text_positions=(3,),
    )
    metrics = compute_attention_metrics(vector, groups)
    assert compute_attention_metrics_batch(
        torch.stack((vector, vector)),
        (groups, groups),
    ) == (metrics, metrics)

    assert len(ATTENTION_METRICS) == 41
    assert len(ATTENTION_SCENARIOS) * len(ATTENTION_METRICS) == 123
    assert metrics.valid
    assert metrics.img_attn_total == pytest.approx(0.4)
    assert metrics.img_attn_avg == pytest.approx(0.2)
    assert math.isinf(metrics.img_attn_kl_u_p)
    assert metrics.all_attn_total == pytest.approx(1.0)
    assert (
        metrics.attn_ratio_img_to_all
        + metrics.attn_ratio_prompt_generated_text_to_all
    ) == pytest.approx(1.0)

    invalid_vector = vector.clone()
    invalid_vector[0] = -1e-7
    invalid = compute_attention_metrics(invalid_vector, groups)
    assert not invalid.valid
    assert all(
        getattr(invalid, field.name) is None
        for field in fields(AttentionScenarioMetrics)
        if field.name != "valid"
    )

    attention_accumulator = AttentionAccumulator()
    for layer_number in range(1, 37):
        attention_accumulator.add_layer(layer_number, vector.unsqueeze(0))
    aggregates = attention_accumulator.finalize()
    assert set(aggregates) == set(ATTENTION_SCENARIOS)
    assert all(aggregate.valid for aggregate in aggregates.values())
    assert all(
        torch.allclose(aggregate.vector, vector)
        for aggregate in aggregates.values()
    )

    assert len(HIDDEN_STATE_METRICS) == 4
    assert compute_layer_cosine(torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0])) == pytest.approx(1.0)
    assert compute_layer_cosine(torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0])) == pytest.approx(0.0)
    assert compute_layer_cosine(torch.zeros(2), torch.ones(2)) is None
    cosines = compute_layer_cosines(
        torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 0.0]]),
        torch.tensor([1.0, 0.0]),
    )
    assert cosines[:2] == pytest.approx((1.0, 0.0))
    assert cosines[2] is None

    layer_values: list[float | None] = [0.5] * 36
    layer_values[0] = None
    layer_values[35] = 1.0
    hidden = aggregate_hidden_metrics(layer_values)
    assert hidden.valid_layers_all == 35
    assert hidden.valid_layers_early_visual_integration == 20
    assert hidden.valid_layers_visual_reasoning == 13
    assert hidden.last_layer_valid
    assert hidden.cosine_gen_imgproto_hidden_avg_all_layers == pytest.approx(
        18 / 35
    )
    assert hidden.cosine_gen_imgproto_hidden_last_layer == pytest.approx(1.0)

    hidden_accumulator = HiddenStateAccumulator()
    for layer_number, value in enumerate(layer_values, start=1):
        hidden_accumulator.add_layer(layer_number, value)
    assert hidden_accumulator.finalize() == hidden


def test_sampling_seeds_generators_and_stopping_form_one_decoding_contract() -> None:
    config = SamplingConfig(
        name="test",
        temperature=2.0,
        top_p=0.7,
        top_k=3,
        repetition_penalty=1.0,
    )
    raw_logits = torch.tensor([0.0, 1.0, 2.0, 3.0])
    untouched = raw_logits.clone()
    transformed = apply_sampling_transforms(raw_logits, config)

    assert torch.equal(raw_logits, untouched)
    assert set(torch.nonzero(torch.isfinite(transformed)).flatten().tolist()) == {
        2,
        3,
    }

    seed = derive_rollout_seed(
        base_seed=42,
        dataset="mathverse",
        sample_id="sample-001",
        strategy="diverse",
        rollout_index=7,
    )
    assert seed == 6012316891626256825

    _, generator_a = make_rollout_generator(
        base_seed=42,
        dataset="mathverse",
        sample_id="sample-001",
        strategy="diverse",
        rollout_index=0,
        device="cpu",
    )
    _, generator_b = make_rollout_generator(
        base_seed=42,
        dataset="mathverse",
        sample_id="sample-001",
        strategy="diverse",
        rollout_index=1,
        device="cpu",
    )
    interleaved_a = []
    interleaved_b = []
    for _ in range(8):
        interleaved_a.append(sample_next_token(raw_logits, config, generator_a))
        interleaved_b.append(sample_next_token(raw_logits, config, generator_b))

    _, replay_a = make_rollout_generator(
        base_seed=42,
        dataset="mathverse",
        sample_id="sample-001",
        strategy="diverse",
        rollout_index=0,
        device="cpu",
    )
    _, replay_b = make_rollout_generator(
        base_seed=42,
        dataset="mathverse",
        sample_id="sample-001",
        strategy="diverse",
        rollout_index=1,
        device="cpu",
    )
    assert interleaved_a == [
        sample_next_token(raw_logits, config, replay_a) for _ in range(8)
    ]
    assert interleaved_b == [
        sample_next_token(raw_logits, config, replay_b) for _ in range(8)
    ]

    _, batch_a = make_rollout_generator(
        base_seed=42,
        dataset="mathverse",
        sample_id="sample-001",
        strategy="diverse",
        rollout_index=0,
        device="cpu",
    )
    _, batch_b = make_rollout_generator(
        base_seed=42,
        dataset="mathverse",
        sample_id="sample-001",
        strategy="diverse",
        rollout_index=1,
        device="cpu",
    )
    _, expected_a = make_rollout_generator(
        base_seed=42,
        dataset="mathverse",
        sample_id="sample-001",
        strategy="diverse",
        rollout_index=0,
        device="cpu",
    )
    _, expected_b = make_rollout_generator(
        base_seed=42,
        dataset="mathverse",
        sample_id="sample-001",
        strategy="diverse",
        rollout_index=1,
        device="cpu",
    )
    logits_batch = torch.stack((raw_logits, raw_logits))
    assert torch.equal(
        apply_sampling_transforms_batch(logits_batch, config)[0],
        transformed,
    )
    assert sample_next_tokens(
        logits_batch,
        config,
        (batch_a, batch_b),
    ) == (
        sample_next_token(raw_logits, config, expected_a),
        sample_next_token(raw_logits, config, expected_b),
    )

    selected = torch.tensor([1, 3])
    batched_metrics = compute_probability_metrics_batch(
        logits_batch,
        selected,
    )
    assert batched_metrics[0] == compute_probability_metrics(raw_logits, 1)
    assert batched_metrics[1] == compute_probability_metrics(raw_logits, 3)

    stop_ids = resolve_stop_token_ids(151645, 151645)
    immediate = decide_stop(151645, 0, 4, stop_ids)
    assert not immediate.retain_token
    assert immediate.reason is StopReason.STOP_TOKEN
    assert immediate.terminating_token_id == 151645

    final_retained = decide_stop(10, 3, 4, stop_ids)
    assert final_retained.retain_token
    assert final_retained.stop
    assert final_retained.reason is StopReason.MAX_NEW_TOKENS
    assert final_retained.terminating_token_id is None
