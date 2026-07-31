"""Typed configuration loading and resolution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from visconf.types import JsonValue, SamplingConfig, ScoringMode


INITIAL_DATASETS = frozenset({"mathverse", "mathvista", "mmmu_pro"})
INITIAL_STRATEGIES = frozenset({"diverse", "concentrated"})
NORMATIVE_DOCUMENTS = (
    "docs/PROBABILITY_CONFIDENCE_METRICS.md",
    "docs/ATTENTION_METRICS.MD",
    "docs/VISUAL_HIDDEN_STATE_METRICS.MD",
    "docs/OUTPUT_SCHEMA.md",
    "docs/REPO_SCHEMA.MD",
)


class ConfigError(ValueError):
    """Raised when experiment configuration is invalid."""


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExperimentGroupSettings(FrozenModel):
    id: str | None = None
    output_root: Path = Path("outputs")
    base_seed: int = Field(default=42, ge=0, le=(2**64 - 1))


class GenerationSettings(FrozenModel):
    rollouts_per_example: int = Field(default=32, gt=0)
    rollout_microbatch_size: int = Field(default=16, gt=0)
    max_new_tokens: int = Field(default=1024, gt=0)

    @model_validator(mode="after")
    def validate_microbatch(self) -> "GenerationSettings":
        if self.rollout_microbatch_size > self.rollouts_per_example:
            raise ValueError(
                "rollout_microbatch_size cannot exceed rollouts_per_example"
            )
        return self


class ScoringSettings(FrozenModel):
    mode: ScoringMode = ScoringMode.OFFLINE
    scorer_name: str = "dataset_default"
    scorer_version: str = "1"


class StorageSettings(FrozenModel):
    compression: Literal["zstd"] = "zstd"
    compression_level: int = Field(default=3, ge=1)
    target_row_group_size: int = Field(default=65_536, gt=0)
    use_dictionary: bool = True
    write_statistics: bool = True


class SchemaSettings(FrozenModel):
    output_schema_version: int = Field(default=1, gt=0)
    metric_schema_version: int = Field(default=1, gt=0)
    seed_derivation_version: int = Field(default=1, gt=0)


class ExperimentGroupConfig(FrozenModel):
    experiment_group: ExperimentGroupSettings
    model: str
    hardware: str
    datasets: tuple[str, ...]
    strategies: tuple[str, ...]
    generation: GenerationSettings = GenerationSettings()
    scoring: ScoringSettings = ScoringSettings()
    storage: StorageSettings = StorageSettings()
    schemas: SchemaSettings = SchemaSettings()

    @model_validator(mode="after")
    def validate_initial_matrix(self) -> "ExperimentGroupConfig":
        if len(self.datasets) != len(set(self.datasets)):
            raise ValueError("datasets must be unique")
        if len(self.strategies) != len(set(self.strategies)):
            raise ValueError("strategies must be unique")
        if set(self.datasets) != INITIAL_DATASETS:
            raise ValueError(
                "the initial experiment requires mathverse, mathvista, and mmmu_pro"
            )
        if set(self.strategies) != INITIAL_STRATEGIES:
            raise ValueError(
                "the initial experiment requires diverse and concentrated"
            )
        return self


class ModelSettings(FrozenModel):
    name: str
    model_path: Path
    tokenizer_path: Path | None = None
    revision: str | None = None
    dtype: Literal["bfloat16"] = "bfloat16"
    attention_implementation: Literal["eager"] = "eager"
    num_hidden_layers: Literal[36] = 36
    device: str = "cuda:0"
    cpu_offload: Literal[False] = False
    quantization: None = None
    trust_remote_code: bool = True


class HardwareSettings(FrozenModel):
    name: str
    accelerator: Literal["a100", "h100"]
    memory_gb: int = Field(gt=0)
    max_rollout_microbatch_size: int = Field(gt=0)


class DatasetSettings(FrozenModel):
    name: str
    adapter: str
    source: Path
    revision: str | None = None
    split: str
    filter_id: str
    filters: dict[str, JsonValue] = Field(default_factory=dict)
    prompt_template: str


class SamplingSettings(FrozenModel):
    name: str
    temperature: float = Field(gt=0)
    top_p: float = Field(gt=0, le=1)
    top_k: int | None = Field(default=None, gt=0)
    repetition_penalty: Literal[1.0] = 1.0

    def as_domain(self) -> SamplingConfig:
        return SamplingConfig(
            name=self.name,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
        )


class LoadedExperimentGroup(FrozenModel):
    repository_root: Path
    config_path: Path
    group: ExperimentGroupSettings
    model: ModelSettings
    hardware: HardwareSettings
    datasets: tuple[DatasetSettings, ...]
    strategies: tuple[SamplingSettings, ...]
    generation: GenerationSettings
    scoring: ScoringSettings
    storage: StorageSettings
    schemas: SchemaSettings
    document_hashes: dict[str, str]
    group_config_hash: str


class ResolvedRunConfig(FrozenModel):
    experiment_group_id: str
    run_id: str
    run_label: str
    output_dir: Path
    base_seed: int
    dataset: DatasetSettings
    sampling: SamplingSettings
    model: ModelSettings
    hardware: HardwareSettings
    generation: GenerationSettings
    scoring: ScoringSettings
    storage: StorageSettings
    schemas: SchemaSettings
    document_hashes: dict[str, str]
    config_hash: str


def canonical_json(value: Any) -> str:
    """Return the canonical JSON representation used for configuration hashes."""

    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot load YAML configuration {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError(f"configuration {path} must contain a YAML mapping")
    return value


def _load_model(model: type[FrozenModel], path: Path) -> FrozenModel:
    try:
        return model.model_validate(_load_yaml_mapping(path))
    except ValueError as exc:
        raise ConfigError(f"invalid configuration {path}: {exc}") from exc


def _resolve_path(path: Path, repository_root: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (repository_root / path).resolve()


def _hash_documents(repository_root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in NORMATIVE_DOCUMENTS:
        path = repository_root / relative
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise ConfigError(f"cannot hash normative document {path}: {exc}") from exc
        hashes[relative] = hashlib.sha256(data).hexdigest()
    return hashes


def _group_hash_payload(
    *,
    group: ExperimentGroupSettings,
    model: ModelSettings,
    hardware: HardwareSettings,
    datasets: tuple[DatasetSettings, ...],
    strategies: tuple[SamplingSettings, ...],
    generation: GenerationSettings,
    scoring: ScoringSettings,
    storage: StorageSettings,
    schemas: SchemaSettings,
) -> dict[str, Any]:
    return {
        "experiment_group": group.model_dump(mode="json"),
        "model": model.model_dump(mode="json"),
        "hardware": hardware.model_dump(mode="json"),
        "datasets": [item.model_dump(mode="json") for item in datasets],
        "strategies": [item.model_dump(mode="json") for item in strategies],
        "generation": generation.model_dump(mode="json"),
        "scoring": scoring.model_dump(mode="json"),
        "storage": storage.model_dump(mode="json"),
        "schemas": schemas.model_dump(mode="json"),
    }


def load_experiment_group_config(path: str | Path) -> LoadedExperimentGroup:
    """Load and fully resolve an experiment-group YAML configuration."""

    config_path = Path(path).resolve()
    config_dir = config_path.parent
    repository_root = config_dir.parent

    try:
        raw = ExperimentGroupConfig.model_validate(_load_yaml_mapping(config_path))
    except ValueError as exc:
        raise ConfigError(f"invalid configuration {config_path}: {exc}") from exc

    group = raw.experiment_group.model_copy(
        update={
            "output_root": _resolve_path(
                raw.experiment_group.output_root, repository_root
            )
        }
    )

    model = _load_model(
        ModelSettings, config_dir / "model" / f"{raw.model}.yaml"
    )
    assert isinstance(model, ModelSettings)
    model = model.model_copy(
        update={
            "model_path": _resolve_path(model.model_path, repository_root),
            "tokenizer_path": (
                _resolve_path(model.tokenizer_path, repository_root)
                if model.tokenizer_path is not None
                else None
            ),
        }
    )
    if model.name != raw.model:
        raise ConfigError("model fragment name does not match its reference")

    hardware = _load_model(
        HardwareSettings, config_dir / "hardware" / f"{raw.hardware}.yaml"
    )
    assert isinstance(hardware, HardwareSettings)
    if hardware.name != raw.hardware:
        raise ConfigError("hardware fragment name does not match its reference")
    if (
        raw.generation.rollout_microbatch_size
        > hardware.max_rollout_microbatch_size
    ):
        raise ConfigError(
            "rollout_microbatch_size exceeds the selected hardware limit"
        )

    datasets: list[DatasetSettings] = []
    for name in raw.datasets:
        dataset = _load_model(
            DatasetSettings, config_dir / "datasets" / f"{name}.yaml"
        )
        assert isinstance(dataset, DatasetSettings)
        if dataset.name != name:
            raise ConfigError(
                f"dataset fragment {name!r} declares name {dataset.name!r}"
            )
        datasets.append(
            dataset.model_copy(
                update={"source": _resolve_path(dataset.source, repository_root)}
            )
        )

    strategies: list[SamplingSettings] = []
    for name in raw.strategies:
        strategy = _load_model(
            SamplingSettings, config_dir / "sampling" / f"{name}.yaml"
        )
        assert isinstance(strategy, SamplingSettings)
        if strategy.name != name:
            raise ConfigError(
                f"sampling fragment {name!r} declares name {strategy.name!r}"
            )
        strategies.append(strategy)

    dataset_tuple = tuple(datasets)
    strategy_tuple = tuple(strategies)
    group_config_hash = sha256_json(
        _group_hash_payload(
            group=group,
            model=model,
            hardware=hardware,
            datasets=dataset_tuple,
            strategies=strategy_tuple,
            generation=raw.generation,
            scoring=raw.scoring,
            storage=raw.storage,
            schemas=raw.schemas,
        )
    )

    return LoadedExperimentGroup(
        repository_root=repository_root,
        config_path=config_path,
        group=group,
        model=model,
        hardware=hardware,
        datasets=dataset_tuple,
        strategies=strategy_tuple,
        generation=raw.generation,
        scoring=raw.scoring,
        storage=raw.storage,
        schemas=raw.schemas,
        document_hashes=_hash_documents(repository_root),
        group_config_hash=group_config_hash,
    )

