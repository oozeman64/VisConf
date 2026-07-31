"""MathVerse Parquet adapter."""

from __future__ import annotations

import hashlib
import io
import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from PIL import Image

from visconf.config import DatasetSettings, ScoringSettings
from visconf.datasets.base import DatasetError, render_question
from visconf.types import (
    Example,
    ExampleImage,
    JsonValue,
    PromptConfig,
)


_MULTIPLE_CHOICE_MARKER = re.compile(r"(?:\([A-Z]\)|\b[A-Z][.)])")


def _to_rgb(image: Image.Image) -> Image.Image:
    if image.mode in {"RGBA", "LA", "PA", "P"}:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    return image.convert("RGB")


def _image_bytes(
    value: object,
    source_directory: Path,
) -> tuple[bytes, str | None]:
    if isinstance(value, dict):
        raw = value.get("bytes")
        source_ref = value.get("path") or None
        if raw:
            return bytes(raw), str(source_ref) if source_ref else None
        if source_ref:
            path = Path(str(source_ref))
            if not path.is_absolute():
                path = source_directory / path
            return path.read_bytes(), str(source_ref)
    raise DatasetError("MathVerse image field has neither bytes nor a path")


def _load_image(
    value: object,
    source_directory: Path,
) -> ExampleImage:
    data, source_ref = _image_bytes(value, source_directory)
    try:
        with Image.open(io.BytesIO(data)) as source:
            image = _to_rgb(source)
            image.load()
    except (OSError, ValueError) as exc:
        raise DatasetError("cannot decode a MathVerse image") from exc
    return ExampleImage(
        source_ref=source_ref,
        image=image,
        sha256=hashlib.sha256(data).hexdigest(),
        width=image.width,
        height=image.height,
        mode=image.mode,
    )


def _matches_filters(
    row: Mapping[str, object],
    filters: Mapping[str, JsonValue],
) -> bool:
    for name, expected in filters.items():
        actual = row.get(name)
        if isinstance(actual, str) and isinstance(expected, str):
            if actual.strip().casefold() != expected.strip().casefold():
                return False
        elif actual != expected:
            return False
    return True


def _answer_type(row: Mapping[str, object]) -> str:
    question_type = str(row.get("question_type") or "").casefold()
    if "choice" in question_type:
        return "multiple_choice"
    answer = str(row.get("answer") or "").strip().upper()
    question = str(row.get("question") or "")
    if (
        len(answer) == 1
        and "A" <= answer <= "Z"
        and _MULTIPLE_CHOICE_MARKER.search(question)
    ):
        return "multiple_choice"
    return "freeform"


def _stable_sample_id(
    row: Mapping[str, object],
    image_sha256: str,
) -> str:
    for name in ("sample_index", "problem_index"):
        value = str(row.get(name) or "").strip()
        if value:
            return value
    payload = json.dumps(
        {
            "question": str(row.get("question") or ""),
            "answer": str(row.get("answer") or ""),
            "image_sha256": image_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return "sha256-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _metadata(row: Mapping[str, object]) -> dict[str, JsonValue]:
    source_metadata = row.get("metadata")
    nested = source_metadata if isinstance(source_metadata, dict) else {}
    return {
        "problem_index": str(row.get("problem_index") or ""),
        "problem_version": str(row.get("problem_version") or ""),
        "question_type": str(row.get("question_type") or ""),
        "query_wo": str(row.get("query_wo") or ""),
        "query_cot": str(row.get("query_cot") or ""),
        "source": str(nested.get("source") or ""),
        "subject": str(nested.get("subject") or ""),
        "subfield": str(nested.get("subfield") or ""),
        "source_split": str(nested.get("split") or ""),
    }


class MathVerseAdapter:
    name = "mathverse"

    def load_examples(
        self,
        config: DatasetSettings,
    ) -> Iterable[Example]:
        if config.name != self.name or config.adapter != self.name:
            raise DatasetError("MathVerse adapter received another dataset config")
        if not config.source.is_file():
            raise DatasetError(f"MathVerse source does not exist: {config.source}")

        seen_sample_ids: set[str] = set()
        source_row_index = 0
        parquet = pq.ParquetFile(config.source)
        for batch in parquet.iter_batches(batch_size=64):
            for row in batch.to_pylist():
                row_index = source_row_index
                source_row_index += 1
                if not _matches_filters(row, config.filters):
                    continue

                image = _load_image(row.get("image"), config.source.parent)
                sample_id = _stable_sample_id(row, image.sha256)
                if sample_id in seen_sample_ids:
                    raise DatasetError(
                        f"duplicate MathVerse sample ID {sample_id!r}"
                    )
                seen_sample_ids.add(sample_id)
                question = str(row.get("question") or "")
                yield Example(
                    dataset=self.name,
                    split=config.split,
                    sample_id=sample_id,
                    source_row_index=row_index,
                    question=question,
                    images=(image,),
                    ground_truth={
                        "answer": str(row.get("answer") or ""),
                        "question_for_eval": str(
                            row.get("question_for_eval") or question
                        ),
                    },
                    answer_type=_answer_type(row),
                    metadata=_metadata(row),
                )

    def build_messages(
        self,
        example: Example,
        prompt_config: PromptConfig,
    ) -> list[dict[str, object]]:
        content: list[dict[str, object]] = [
            {"type": "image", "image": image.image}
            for image in example.images
        ]
        content.append(
            {
                "type": "text",
                "text": render_question(example.question, prompt_config),
            }
        )
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
        return None

