"""MathVista Parquet adapter."""

from __future__ import annotations

import hashlib
import io
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from PIL import Image

from visconf.config import DatasetSettings, ScoringSettings
from visconf.datasets.base import DatasetError, render_question
from visconf.scoring.answer_normalization import (
    normalize_choice_text,
    score_generation,
)
from visconf.types import (
    Example,
    ExampleImage,
    JsonValue,
    PromptConfig,
)


def _rgb(image: Image.Image) -> Image.Image:
    if image.mode in {"RGBA", "LA", "PA", "P"}:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    return image.convert("RGB")


def _image(value: object, source_directory: Path) -> ExampleImage:
    if not isinstance(value, dict):
        raise DatasetError("MathVista decoded_image is not an image record")
    raw = value.get("bytes")
    source_ref = value.get("path") or None
    if raw:
        data = bytes(raw)
    elif source_ref:
        path = Path(str(source_ref))
        if not path.is_absolute():
            path = source_directory / path
        data = path.read_bytes()
    else:
        raise DatasetError("MathVista image has neither bytes nor a path")
    try:
        with Image.open(io.BytesIO(data)) as opened:
            image = _rgb(opened)
            image.load()
    except (OSError, ValueError) as exc:
        raise DatasetError("cannot decode a MathVista image") from exc
    return ExampleImage(
        source_ref=str(source_ref) if source_ref else None,
        image=image,
        sha256=hashlib.sha256(data).hexdigest(),
        width=image.width,
        height=image.height,
        mode=image.mode,
    )


def _matches(row: Mapping[str, object], filters: Mapping[str, JsonValue]):
    return all(row.get(name) == expected for name, expected in filters.items())


def _format_question(question: str, choices: object) -> str:
    if not isinstance(choices, list) or not choices:
        return question
    options = "\n".join(
        f"{chr(65 + index)}. {choice}"
        for index, choice in enumerate(choices)
    )
    return f"{question}\n{options}"


def _answer(row: Mapping[str, object], choices: object, answer_type: str):
    answer = str(row.get("answer") or "")
    if answer_type != "multiple_choice" or not isinstance(choices, list):
        return answer
    if len(answer.strip()) == 1 and answer.strip().upper().isalpha():
        return answer.strip().upper()
    normalized = normalize_choice_text(answer)
    for index, choice in enumerate(choices):
        if normalize_choice_text(choice) == normalized:
            return chr(65 + index)
    return answer


def _metadata(row: Mapping[str, object]) -> dict[str, JsonValue]:
    nested = row.get("metadata")
    values = nested if isinstance(nested, dict) else {}
    return {
        "question_type": str(row.get("question_type") or ""),
        "source_answer_type": str(row.get("answer_type") or ""),
        "unit": str(row.get("unit") or ""),
        "precision": row.get("precision"),
        "category": str(values.get("category") or ""),
        "context": str(values.get("context") or ""),
        "grade": str(values.get("grade") or ""),
        "language": str(values.get("language") or ""),
        "skills": [str(value) for value in values.get("skills") or []],
        "source": str(values.get("source") or ""),
        "source_split": str(values.get("split") or ""),
        "task": str(values.get("task") or ""),
    }


class MathVistaAdapter:
    name = "mathvista"

    def load_examples(
        self,
        config: DatasetSettings,
    ) -> Iterable[Example]:
        if config.name != self.name or config.adapter != self.name:
            raise DatasetError("MathVista adapter received another dataset config")
        if not config.source.is_file():
            raise DatasetError(f"MathVista source does not exist: {config.source}")

        seen = set()
        row_index = 0
        parquet = pq.ParquetFile(config.source)
        for batch in parquet.iter_batches(batch_size=64):
            for row in batch.to_pylist():
                source_index = row_index
                row_index += 1
                if not _matches(row, config.filters):
                    continue
                sample_id = str(row.get("pid") or "").strip()
                if not sample_id:
                    raise DatasetError("MathVista row has no stable pid")
                if sample_id in seen:
                    raise DatasetError(
                        f"duplicate MathVista sample ID {sample_id!r}"
                    )
                seen.add(sample_id)
                choices = row.get("choices")
                question = _format_question(
                    str(row.get("question") or ""),
                    choices,
                )
                question_type = str(
                    row.get("question_type") or ""
                ).casefold()
                answer_type = (
                    "multiple_choice"
                    if "choice" in question_type
                    else "freeform"
                )
                answer = _answer(row, choices, answer_type)
                yield Example(
                    dataset=self.name,
                    split=config.split,
                    sample_id=sample_id,
                    source_row_index=source_index,
                    question=question,
                    images=(
                        _image(row.get("decoded_image"), config.source.parent),
                    ),
                    ground_truth={
                        "answer": answer,
                        "question_for_eval": question,
                    },
                    answer_type=answer_type,
                    metadata=_metadata(row),
                )

    def build_messages(
        self,
        example: Example,
        prompt_config: PromptConfig,
    ) -> list[dict[str, object]]:
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": example.images[0].image,
                    },
                    {
                        "type": "text",
                        "text": render_question(
                            example.question,
                            prompt_config,
                        ),
                    },
                ],
            }
        ]

    def ground_truth(
        self,
        example: Example,
    ) -> Mapping[str, JsonValue]:
        return example.ground_truth

    def score(
        self,
        example: Example,
        response: str,
        scorer_config: ScoringSettings,
    ) -> Mapping[str, JsonValue] | None:
        return score_generation(
            response,
            str(
                example.ground_truth.get("question_for_eval")
                or example.question
            ),
            str(example.ground_truth.get("answer") or ""),
            example.answer_type or "freeform",
        )
