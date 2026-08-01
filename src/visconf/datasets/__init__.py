"""Dataset adapter registry."""

from visconf.datasets.base import (
    DatasetAdapter,
    DatasetError,
    PromptError,
)
from visconf.datasets.mathverse import MathVerseAdapter
from visconf.datasets.mathvista import MathVistaAdapter
from visconf.datasets.mmmu_pro import MMMUProAdapter


_ADAPTERS = {
    adapter.name: adapter
    for adapter in (
        MathVerseAdapter,
        MathVistaAdapter,
        MMMUProAdapter,
    )
}


def create_dataset_adapter(name: str) -> DatasetAdapter:
    try:
        return _ADAPTERS[name]()
    except KeyError as exc:
        raise DatasetError(
            f"dataset adapter is not implemented: {name!r}"
        ) from exc


__all__ = [
    "DatasetAdapter",
    "DatasetError",
    "MMMUProAdapter",
    "MathVerseAdapter",
    "MathVistaAdapter",
    "PromptError",
    "create_dataset_adapter",
]
