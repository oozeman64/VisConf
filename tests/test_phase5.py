from __future__ import annotations

from itertools import islice
from pathlib import Path

import pytest
import torch
from torch import nn

from visconf.config import load_experiment_group_config
from visconf.datasets.mathverse import MathVerseAdapter
from visconf.models.instrumentation import QwenInstrumentation
from visconf.models.qwen25vl import QwenModelFacade
from visconf.types import PromptConfig, TokenGroups


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiment_group.yaml"


class _FakeLayer(nn.Module):
    def __init__(self, layer_number: int) -> None:
        super().__init__()
        self.layer_number = layer_number

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
        **kwargs,
    ):
        hidden_output = hidden_states + float(self.layer_number)
        batch, query_length, _ = hidden_states.shape
        attention = torch.full(
            (batch, 1, query_length, query_length),
            float(self.layer_number),
        )
        output = (hidden_output,)
        if output_attentions:
            output += (attention,)
        return output


class _FakeLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            _FakeLayer(index) for index in range(1, 37)
        )


class _FakeOuterModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = _FakeLanguageModel()


class _FakeQwen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _FakeOuterModel()


def _fake_forward(
    model: _FakeQwen,
    hidden: torch.Tensor,
) -> torch.Tensor:
    for layer in model.model.language_model.layers:
        output = layer(hidden, output_attentions=False)
        assert len(output) == 1
        hidden = output[0]
    return hidden


def test_instrumentation_reduces_online_and_always_restores() -> None:
    model = _FakeQwen()
    layers = tuple(model.model.language_model.layers)
    originals = tuple(layer.forward for layer in layers)
    groups = TokenGroups(
        image_positions=torch.tensor([1]),
        prompt_text_positions=torch.tensor([0, 2]),
        prompt_last_position=3,
        prompt_token_count=4,
    )
    hidden = torch.tensor(
        [[[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]]
    )

    with QwenInstrumentation(model) as instrumentation:
        instrumentation.begin_prefill(groups)
        _fake_forward(model, hidden)
        result = instrumentation.finish_step()
        row = result.rows[0]
        assert row.attention["all_layers_all_heads"].vector.tolist() == pytest.approx(
            [18.5] * 4
        )
        assert row.attention["early_visual_integration"].vector.tolist() == pytest.approx(
            [11.0] * 4
        )
        assert row.attention["visual_reasoning"].vector.tolist() == pytest.approx(
            [28.0] * 4
        )
        assert row.hidden_state.valid_layers_all == 36
        assert row.hidden_state.last_layer_valid
        assert row.hidden_state.cosine_gen_imgproto_hidden_last_layer == pytest.approx(
            1.0
        )
        assert len(instrumentation.image_prototypes) == 36
        assert all(
            prototype is not None and prototype.dtype == torch.float32
            for prototype in instrumentation.image_prototypes
        )

    assert all(
        layer.forward == original
        for layer, original in zip(layers, originals, strict=True)
    )
    with pytest.raises(RuntimeError, match="injected"):
        with QwenInstrumentation(model):
            raise RuntimeError("injected")
    assert all(
        layer.forward == original
        for layer, original in zip(layers, originals, strict=True)
    )


def test_real_qwen_prefill_cache_copy_and_two_step_decode() -> None:
    config = load_experiment_group_config(CONFIG)
    dataset = next(item for item in config.datasets if item.name == "mathverse")
    example = next(
        iter(islice(MathVerseAdapter().load_examples(dataset), 1))
    )
    messages = MathVerseAdapter().build_messages(
        example,
        PromptConfig(dataset.prompt_template),
    )
    facade = QwenModelFacade.load(config.model)
    try:
        with QwenInstrumentation(facade.model) as instrumentation:
            prepared = facade.prepare_example(messages)
            prefill = facade.prefill(prepared, instrumentation)
            assert prefill.raw_logits.ndim == 1
            assert prefill.raw_logits.shape[0] == facade.model.config.text_config.vocab_size
            assert prefill.cache.get_seq_length() == prepared.prompt_record.prompt_token_count
            assert len(prefill.observations.rows) == 1
            assert len(instrumentation.image_prototypes) == 36
            assert all(
                prototype is not None and prototype.dtype == torch.float32
                for prototype in instrumentation.image_prototypes
            )
            assert "pixel_values" not in prepared.model_inputs

            selected = int(torch.argmax(prefill.raw_logits).item())
            copied_cache = facade.repeat_cache(prefill.cache, batch_size=2)
            assert prefill.cache.to_legacy_cache()[0][0].shape[0] == 1
            assert copied_cache.to_legacy_cache()[0][0].shape[0] == 2
            copied = facade.decode_step(
                torch.tensor([selected, selected]),
                copied_cache,
                prefill.attention_mask,
                prefill.rope_deltas,
                prepared.token_groups,
                instrumentation,
            )
            assert len(copied.observations.rows) == 2
            assert torch.allclose(
                copied.raw_logits[0],
                copied.raw_logits[1],
                atol=1e-4,
                rtol=1e-4,
            )

            independent_prepared = facade.prepare_example(messages)
            independent_prefill = facade.prefill(
                independent_prepared,
                instrumentation,
            )
            independent = facade.decode_step(
                torch.tensor([selected]),
                facade.repeat_cache(
                    independent_prefill.cache,
                    batch_size=1,
                ),
                independent_prefill.attention_mask,
                independent_prefill.rope_deltas,
                independent_prepared.token_groups,
                instrumentation,
            )
            assert torch.allclose(
                prefill.raw_logits,
                independent_prefill.raw_logits,
                atol=1e-4,
                rtol=1e-4,
            )
            assert int(torch.argmax(copied.raw_logits[0])) == int(
                torch.argmax(independent.raw_logits[0])
            )
    finally:
        del facade
        torch.cuda.empty_cache()
