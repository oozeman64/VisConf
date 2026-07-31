"""Dataset adapter registry."""

from visconf.datasets.base import (
    DatasetAdapter,
    DatasetError,
    PromptError,
)
from visconf.datasets.mathverse import MathVerseAdapter


def create_dataset_adapter(name: str) -> DatasetAdapter:
    if name == MathVerseAdapter.name:
        return MathVerseAdapter()
    raise DatasetError(f"dataset adapter is not implemented: {name!r}")


__all__ = [
    "DatasetAdapter",
    "DatasetError",
    "MathVerseAdapter",
    "PromptError",
    "create_dataset_adapter",
]
