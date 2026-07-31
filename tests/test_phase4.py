from __future__ import annotations

from itertools import islice
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

from visconf.config import load_experiment_group_config
from visconf.datasets import DatasetAdapter, MathVerseAdapter
from visconf.datasets.base import (
    prompt_record_hash,
    render_chat_prompt,
)
from visconf.models.token_positions import (
    TokenPositionError,
    build_prompt_token_groups,
    predictor_position,
    step_token_groups,
)
from visconf.types import PromptConfig


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiment_group.yaml"


@pytest.fixture(scope="module")
def experiment_and_tokenizer():
    config = load_experiment_group_config(CONFIG)
    tokenizer_path = config.model.tokenizer_path or config.model.model_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
        trust_remote_code=config.model.trust_remote_code,
    )
    return config, tokenizer


def test_mathverse_selection_images_and_prompt_are_reproducible(
    experiment_and_tokenizer,
) -> None:
    config, tokenizer = experiment_and_tokenizer
    dataset = next(item for item in config.datasets if item.name == "mathverse")
    adapter = MathVerseAdapter()
    assert isinstance(adapter, DatasetAdapter)

    first_pass = tuple(islice(adapter.load_examples(dataset), 2))
    repeated_first = next(iter(adapter.load_examples(dataset)))
    assert len(first_pass) == 2
    assert repeated_first.sample_id == first_pass[0].sample_id
    assert repeated_first.source_row_index == first_pass[0].source_row_index
    assert repeated_first.images[0].sha256 == first_pass[0].images[0].sha256

    example = first_pass[0]
    assert example.dataset == "mathverse"
    assert example.split == dataset.split
    assert example.metadata["problem_version"] == "Vision Intensive"
    assert len(example.images) == 1
    assert example.images[0].mode == "RGB"
    assert example.images[0].width > 0 and example.images[0].height > 0
    assert len(example.images[0].sha256) == 64
    assert adapter.ground_truth(example)["answer"] is not None

    prompt_config = PromptConfig(name=dataset.prompt_template)
    messages = adapter.build_messages(example, prompt_config)
    assert [item["type"] for item in messages[0]["content"][:-1]] == [
        "image"
    ]
    assert messages[0]["content"][-1]["text"].endswith(
        "then give your final answer inside \\boxed{}."
    )
    prompt_a = render_chat_prompt(tokenizer, messages)
    prompt_b = render_chat_prompt(
        tokenizer,
        adapter.build_messages(example, prompt_config),
    )
    assert prompt_a == prompt_b
    assert prompt_record_hash(prompt_a) == prompt_record_hash(prompt_b)
    assert prompt_a.prompt_token_count > 0


def test_actual_qwen_tokenizer_groups_logical_positions_and_dynamic_g(
    experiment_and_tokenizer,
) -> None:
    _, tokenizer = experiment_and_tokenizer

    def token_id(token: str) -> int:
        return int(tokenizer.convert_tokens_to_ids(token))

    vision_start = token_id("<|vision_start|>")
    vision_end = token_id("<|vision_end|>")
    image_pad = token_id("<|image_pad|>")
    im_start = token_id("<|im_start|>")
    im_end = token_id("<|im_end|>")
    text_a = tokenizer.encode(" alpha", add_special_tokens=False)
    text_b = tokenizer.encode(" beta", add_special_tokens=False)
    image_newline = tokenizer.encode("\n", add_special_tokens=False)

    span_one_start = 1 + len(text_a)
    span_one = [
        vision_start,
        image_pad,
        *image_newline,
        image_pad,
        vision_end,
    ]
    text_b_start = span_one_start + len(span_one)
    span_two_start = text_b_start + len(text_b)
    valid_ids = [
        im_start,
        *text_a,
        *span_one,
        *text_b,
        vision_start,
        image_pad,
        vision_end,
        im_end,
    ]
    expected_image = (
        span_one_start + 1,
        span_one_start + 2 + len(image_newline),
        span_two_start + 1,
    )
    expected_prompt = tuple(
        range(1, 1 + len(text_a))
    ) + tuple(range(text_b_start, text_b_start + len(text_b)))

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    left_ids = torch.tensor([pad_id, pad_id, *valid_ids])
    left_mask = torch.tensor([0, 0, *([1] * len(valid_ids))])
    right_ids = torch.tensor([*valid_ids, pad_id, pad_id])
    right_mask = torch.tensor([*([1] * len(valid_ids)), 0, 0])

    left = build_prompt_token_groups(left_ids, left_mask, tokenizer)
    right = build_prompt_token_groups(right_ids, right_mask, tokenizer)
    assert tuple(left.image_positions.tolist()) == expected_image
    assert tuple(left.prompt_text_positions.tolist()) == expected_prompt
    assert torch.equal(left.image_positions, right.image_positions)
    assert torch.equal(left.prompt_text_positions, right.prompt_text_positions)
    assert left.prompt_token_count == len(valid_ids)
    assert left.prompt_last_position == len(valid_ids) - 1

    first_step = step_token_groups(left, retained_generated_tokens=0)
    later_step = step_token_groups(left, retained_generated_tokens=3)
    assert first_step.generated_text_positions == ()
    assert later_step.generated_text_positions == tuple(
        range(len(valid_ids), len(valid_ids) + 3)
    )
    assert predictor_position(left, 0) == len(valid_ids) - 1
    assert predictor_position(left, 3) == len(valid_ids) + 2

    with pytest.raises(TokenPositionError, match="unterminated"):
        build_prompt_token_groups(
            torch.tensor([vision_start, image_pad]),
            torch.ones(2, dtype=torch.long),
            tokenizer,
        )

