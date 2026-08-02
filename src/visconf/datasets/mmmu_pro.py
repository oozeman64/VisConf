"""MMMU-Pro standard Parquet adapter with variable option counts."""

from __future__ import annotations

import ast
import hashlib
import io
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from PIL import Image

from visconf.config import DatasetSettings, ScoringSettings
from visconf.datasets.base import DatasetError, render_question
from visconf.scoring.answer_normalization import score_generation
from visconf.types import (
    Example,
    ExampleImage,
    JsonValue,
    PromptConfig,
)


_IMAGE_FIELDS = tuple(f"image_{index}" for index in range(1, 8))
_IMAGE_PLACEHOLDER = re.compile(r"<image\s+(\d+)>", re.IGNORECASE)


def _rgb(image: Image.Image) -> Image.Image:
    if image.mode in {"RGBA", "LA", "PA", "P"}:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    return image.convert("RGB")


def _image(
    value: object,
    source_directory: Path,
) -> ExampleImage:
    if not isinstance(value, dict):
        raise DatasetError("MMMU-Pro image is not an image record")
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
        raise DatasetError("MMMU-Pro image has neither bytes nor a path")
    try:
        with Image.open(io.BytesIO(data)) as opened:
            image = _rgb(opened)
            image.load()
    except (OSError, ValueError) as exc:
        raise DatasetError("cannot decode an MMMU-Pro image") from exc
    return ExampleImage(
        source_ref=str(source_ref) if source_ref else None,
        image=image,
        sha256=hashlib.sha256(data).hexdigest(),
        width=image.width,
        height=image.height,
        mode=image.mode,
    )


def _options(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(option) for option in value]
    try:
        parsed = ast.literal_eval(str(value))
    except (ValueError, SyntaxError) as exc:
        raise DatasetError("MMMU-Pro options are not a list") from exc
    if not isinstance(parsed, (list, tuple)):
        raise DatasetError("MMMU-Pro options are not a list")
    return [str(option) for option in parsed]


def _options_block(options: list[str]) -> str:
    return "\n".join(
        f"{chr(65 + index)}. {option}"
        for index, option in enumerate(options)
    )


def _matches(row: Mapping[str, object], filters: Mapping[str, JsonValue]):
    return all(row.get(name) == expected for name, expected in filters.items())


class MMMUProAdapter:
    name = "mmmu_pro"

    def load_examples(
        self,
        config: DatasetSettings,
    ) -> Iterable[Example]:
        if config.name != self.name or config.adapter != self.name:
            raise DatasetError("MMMU-Pro adapter received another dataset config")
        if not config.source.is_dir():
            raise DatasetError(
                f"MMMU-Pro source directory does not exist: {config.source}"
            )
        paths = tuple(sorted(config.source.glob("*.parquet")))
        if not paths:
            raise DatasetError("MMMU-Pro source has no Parquet shards")

        seen = set()
        source_index = 0
        for path in paths:
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=64):
                for row in batch.to_pylist():
                    row_index = source_index
                    source_index += 1
                    if not _matches(row, config.filters):
                        continue
                    sample_id = str(row.get("id") or "").strip()
                    if not sample_id:
                        raise DatasetError("MMMU-Pro row has no stable id")
                    if sample_id in seen:
                        raise DatasetError(
                            f"duplicate MMMU-Pro sample ID {sample_id!r}"
                        )
                    seen.add(sample_id)
                    options = _options(row.get("options"))
                    if not 2 <= len(options) <= 26:
                        raise DatasetError(
                            "MMMU-Pro options must contain between two and 26 "
                            "choices"
                        )
                    image_numbers = tuple(
                        index
                        for index, field in enumerate(
                            _IMAGE_FIELDS,
                            start=1,
                        )
                        if row.get(field) is not None
                    )
                    if not image_numbers:
                        raise DatasetError("MMMU-Pro row has no images")
                    images = tuple(
                        _image(
                            row[_IMAGE_FIELDS[number - 1]],
                            path.parent,
                        )
                        for number in image_numbers
                    )
                    question = str(row.get("question") or "")
                    question_for_eval = (
                        f"{question}\n{_options_block(options)}"
                    ).strip()
                    yield Example(
                        dataset=self.name,
                        split=config.split,
                        sample_id=sample_id,
                        source_row_index=row_index,
                        question=question,
                        images=images,
                        ground_truth={
                            "answer": str(row.get("answer") or ""),
                            "question_for_eval": question_for_eval,
                        },
                        answer_type="multiple_choice",
                        metadata={
                            "options": options,
                            "image_numbers": list(image_numbers),
                            "explanation": str(
                                row.get("explanation") or ""
                            ),
                            "img_type": str(row.get("img_type") or ""),
                            "topic_difficulty": str(
                                row.get("topic_difficulty") or ""
                            ),
                            "subject": str(row.get("subject") or ""),
                        },
                    )

    def build_messages(
        self,
        example: Example,
        prompt_config: PromptConfig,
    ) -> list[dict[str, object]]:
        numbers = [
            int(value)
            for value in example.metadata.get("image_numbers", [])
        ]
        by_number = {
            number: image
            for number, image in zip(
                numbers,
                example.images,
                strict=True,
            )
        }
        content: list[dict[str, object]] = []
        used = set()
        parts = _IMAGE_PLACEHOLDER.split(example.question)
        for index, part in enumerate(parts):
            if index % 2:
                number = int(part)
                image = by_number.get(number)
                if image is None or number in used:
                    content.append(
                        {
                            "type": "text",
                            "text": f"<image {number}>",
                        }
                    )
                else:
                    content.append(
                        {"type": "image", "image": image.image}
                    )
                    used.add(number)
            elif part:
                content.append({"type": "text", "text": part})

        unreferenced = [
            {"type": "image", "image": image.image}
            for number, image in by_number.items()
            if number not in used
        ]
        content.extend(unreferenced)
        options = [
            str(value)
            for value in example.metadata.get("options", [])
        ]
        tail = render_question(
            "\n\n" + _options_block(options),
            prompt_config,
        )
        content.append({"type": "text", "text": tail})
        return [{"role": "user", "content": content}]

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
            "multiple_choice",
        )
