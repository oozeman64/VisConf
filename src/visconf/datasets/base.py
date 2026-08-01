"""Shared dataset and prompt contracts."""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Iterable, Mapping
from typing import Any, Protocol, runtime_checkable

from visconf.config import DatasetSettings, ScoringSettings
from visconf.types import Example, JsonValue, PromptConfig, PromptRecord


class DatasetError(ValueError):
    """Raised when configured dataset content violates its adapter contract."""


class PromptError(ValueError):
    """Raised when a prompt template or tokenizer cannot render a prompt."""


PROMPT_SUFFIXES = {
    "reasoning_boxed_v1": (
        "\n\nThink step by step inside <think> </think> tags, "
        "then give your final answer inside \\boxed{}."
    ),
}


def prompt_template_hash(name: str) -> str:
    """Hash the configured prompt body and the renderer that applies it."""

    try:
        suffix = PROMPT_SUFFIXES[name]
    except KeyError as exc:
        raise PromptError(f"unknown prompt template {name!r}") from exc
    payload = {
        "name": name,
        "suffix": suffix,
        "renderer": inspect.getsource(render_question),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def render_question(question: str, config: PromptConfig) -> str:
    try:
        suffix = PROMPT_SUFFIXES[config.name]
    except KeyError as exc:
        raise PromptError(f"unknown prompt template {config.name!r}") from exc
    return question + suffix


def render_chat_prompt(
    tokenizer: Any,
    messages: list[dict[str, object]],
) -> PromptRecord:
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        token_ids = tokenizer.encode(
            rendered,
            add_special_tokens=False,
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise PromptError("tokenizer could not render the chat prompt") from exc
    return PromptRecord(
        rendered_prompt=rendered,
        prompt_token_ids=tuple(int(token_id) for token_id in token_ids),
        prompt_token_count=len(token_ids),
    )


def prompt_record_hash(record: PromptRecord) -> str:
    payload = json.dumps(
        {
            "rendered_prompt": record.rendered_prompt,
            "prompt_token_ids": record.prompt_token_ids,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@runtime_checkable
class DatasetAdapter(Protocol):
    name: str

    def load_examples(
        self,
        config: DatasetSettings,
    ) -> Iterable[Example]: ...

    def build_messages(
        self,
        example: Example,
        prompt_config: PromptConfig,
    ) -> list[dict[str, object]]: ...

    def ground_truth(
        self,
        example: Example,
    ) -> Mapping[str, JsonValue]: ...

    def score(
        self,
        example: Example,
        response: str,
        scorer_config: ScoringSettings,
    ) -> Mapping[str, JsonValue] | None: ...
