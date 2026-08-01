"""Checkpoint-based validation, resume indexes, and orphan quarantine."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from visconf.config import ResolvedRunConfig
from visconf.storage.schema import (
    CORE_TABLES,
    SCHEMAS,
    TOKEN_KEY_COLUMNS,
)


class ResumeError(ValueError):
    """Raised when committed storage cannot be trusted for resume."""


@dataclass(frozen=True, slots=True)
class RolloutResumeKey:
    dataset: str
    split: str
    sample_id: str
    strategy: str
    rollout_index: int


@dataclass(frozen=True, slots=True)
class ScoreResumeKey:
    rollout: RolloutResumeKey
    scorer_name: str
    scorer_version: str


@dataclass(frozen=True, slots=True)
class ResumeIndex:
    completed_rollouts: frozenset[RolloutResumeKey]
    completed_scores: frozenset[ScoreResumeKey]
    core_shards: tuple[str, ...]
    score_shards: tuple[str, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResumeError(f"cannot load checkpoint {path}") from exc
    if not isinstance(payload, dict):
        raise ResumeError(f"checkpoint is not an object: {path}")
    return payload


def _part_paths(
    run: ResolvedRunConfig,
    checkpoint: dict[str, Any],
) -> dict[str, Path]:
    parts = checkpoint.get("parts")
    if not isinstance(parts, list):
        raise ResumeError("checkpoint has no part inventory")
    resolved: dict[str, Path] = {}
    run_dir = run.output_dir.resolve()
    for item in parts:
        if not isinstance(item, dict):
            raise ResumeError("checkpoint part inventory entry is not an object")
        table_name = item.get("table_name")
        relative_path = item.get("relative_path")
        if table_name not in SCHEMAS or not isinstance(relative_path, str):
            raise ResumeError("checkpoint part inventory is malformed")
        path = (run_dir / relative_path).resolve()
        if not path.is_relative_to(run_dir):
            raise ResumeError("checkpoint part escapes its run directory")
        if table_name in resolved:
            raise ResumeError("checkpoint repeats a table part")
        if not path.is_file():
            raise ResumeError(f"checkpoint part is missing: {path}")
        if path.stat().st_size != item.get("byte_size"):
            raise ResumeError(f"part byte size differs: {path}")
        if _sha256(path) != item.get("sha256"):
            raise ResumeError(f"part hash differs: {path}")
        metadata = pq.read_metadata(path)
        if metadata.num_rows != item.get("row_count"):
            raise ResumeError(f"part row count differs: {path}")
        if not pq.read_schema(path).equals(SCHEMAS[table_name]):
            raise ResumeError(f"part schema differs: {path}")
        resolved[table_name] = path
    return resolved


def _tuple_key(row: dict[str, Any]) -> tuple[object, ...]:
    return tuple(row[name] for name in TOKEN_KEY_COLUMNS)


def _rollout_key(row: dict[str, Any]) -> tuple[object, ...]:
    return tuple(row[name] for name in TOKEN_KEY_COLUMNS[:-1])


def _score_key(row: dict[str, Any]) -> tuple[object, ...]:
    return (*_rollout_key(row), row["scorer_name"], row["scorer_version"])


def _declared_rollout_key(row: dict[str, Any]) -> tuple[object, ...]:
    key = _rollout_key(row)
    if any(not isinstance(value, str) or not value for value in key[:-1]):
        raise TypeError("rollout identity strings are invalid")
    rollout_index = key[-1]
    if (
        isinstance(rollout_index, bool)
        or not isinstance(rollout_index, int)
        or rollout_index < 0
    ):
        raise TypeError("rollout index is invalid")
    return key


def _declared_score_key(row: dict[str, Any]) -> tuple[object, ...]:
    key = _declared_rollout_key(row)
    scorer_name = row["scorer_name"]
    scorer_version = row["scorer_version"]
    if any(
        not isinstance(value, str) or not value
        for value in (scorer_name, scorer_version)
    ):
        raise TypeError("scorer identity is invalid")
    return (*key, scorer_name, scorer_version)


def _validate_core_relations(
    run: ResolvedRunConfig,
    checkpoint: dict[str, Any],
    paths: dict[str, Path],
) -> None:
    if set(paths) != set(CORE_TABLES):
        raise ResumeError("core checkpoint must inventory all five parts")

    tables = {
        name: pq.read_table(path).to_pylist()
        for name, path in paths.items()
    }
    shard_id = checkpoint["shard_id"]
    for table_name, rows in tables.items():
        shard_column = "shard_id"
        if any(row[shard_column] != shard_id for row in rows):
            raise ResumeError(
                f"{table_name} contains a row for another shard"
            )
    token_keys = [_tuple_key(row) for row in tables["tokens"]]
    if len(token_keys) != len(set(token_keys)):
        raise ResumeError("committed token keys are not unique")
    for family in CORE_TABLES[2:]:
        if [_tuple_key(row) for row in tables[family]] != token_keys:
            raise ResumeError(f"{family} keys differ from tokens")

    generation_by_rollout = {}
    for row in tables["generations"]:
        rollout = _rollout_key(row)
        if rollout in generation_by_rollout:
            raise ResumeError("committed rollout keys are not unique")
        generation_by_rollout[rollout] = row

    tokens_by_rollout: dict[tuple[object, ...], list[dict[str, Any]]] = {}
    for row in tables["tokens"]:
        rollout = _rollout_key(row)
        tokens_by_rollout.setdefault(rollout, []).append(row)

    for rollout, generation in generation_by_rollout.items():
        rows = tokens_by_rollout.get(rollout, [])
        rows.sort(key=lambda row: row["step"])
        if [row["step"] for row in rows] != list(
            range(1, generation["num_retained_tokens"] + 1)
        ):
            raise ResumeError("committed token steps are not contiguous")
        if [row["token_id"] for row in rows] != generation[
            "generated_token_ids"
        ]:
            raise ResumeError("committed token IDs differ from generation")
        if generation["num_retained_tokens"] != len(
            generation["generated_token_ids"]
        ):
            raise ResumeError("generation retained-token count is inconsistent")
        if generation["stop_reason"] == "max_new_tokens":
            stop_valid = (
                generation["terminating_token_id"] is None
                and generation["hit_max_new_tokens"] is True
            )
        elif generation["stop_reason"] == "stop_token":
            stop_valid = (
                generation["terminating_token_id"] is not None
                and generation["hit_max_new_tokens"] is False
            )
        else:
            stop_valid = False
        if not stop_valid:
            raise ResumeError("generation stop fields are inconsistent")

    if set(tokens_by_rollout) - set(generation_by_rollout):
        raise ResumeError("token rows exist without a generation")

    actual_rollouts = list(generation_by_rollout)
    declared = checkpoint.get("completed_rollouts")
    if not isinstance(declared, list):
        raise ResumeError("core checkpoint has no completed rollout inventory")
    try:
        declared_rollouts = [_declared_rollout_key(value) for value in declared]
    except (KeyError, TypeError) as exc:
        raise ResumeError("completed rollout inventory is malformed") from exc
    if (
        len(declared_rollouts) != len(set(declared_rollouts))
        or set(declared_rollouts) != set(actual_rollouts)
    ):
        raise ResumeError(
            "checkpoint completed rollouts differ from generation rows"
        )

    steps = [row["step"] for row in tables["tokens"]]
    token_key_count = checkpoint.get("token_key_count")
    if (
        isinstance(token_key_count, bool)
        or not isinstance(token_key_count, int)
        or token_key_count != len(token_keys)
    ):
        raise ResumeError("checkpoint token-key count differs from tokens")
    expected_min = min(steps) if steps else None
    expected_max = max(steps) if steps else None
    recorded_min = checkpoint.get("min_token_step")
    recorded_max = checkpoint.get("max_token_step")
    if any(
        value is not None
        and (isinstance(value, bool) or not isinstance(value, int))
        for value in (recorded_min, recorded_max)
    ) or recorded_min != expected_min or recorded_max != expected_max:
        raise ResumeError("checkpoint token-step bounds differ from tokens")
    for rows in tables.values():
        for row in rows:
            if (
                row["run_id"] != run.run_id
                or row["dataset"] != run.dataset.name
                or row["split"] != run.dataset.split
                or row["strategy"] != run.sampling.name
            ):
                raise ResumeError("committed row does not match its run")


def _validate_score_relations(
    run: ResolvedRunConfig,
    checkpoint: dict[str, Any],
    path: Path,
) -> None:
    rows = pq.read_table(path).to_pylist()
    shard_id = checkpoint["shard_id"]
    keys = [_score_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise ResumeError("committed score keys are not unique")
    if any(row["score_shard_id"] != shard_id for row in rows):
        raise ResumeError("scores contain a row for another score shard")
    for row in rows:
        if (
            row["run_id"] != run.run_id
            or row["dataset"] != run.dataset.name
            or row["split"] != run.dataset.split
            or row["strategy"] != run.sampling.name
        ):
            raise ResumeError("committed score does not match its run")
    declared = checkpoint.get("completed_scores")
    if not isinstance(declared, list):
        raise ResumeError("score checkpoint has no completed score inventory")
    try:
        declared_keys = [_declared_score_key(value) for value in declared]
    except (KeyError, TypeError) as exc:
        raise ResumeError("completed score inventory is malformed") from exc
    if len(declared_keys) != len(set(declared_keys)) or set(declared_keys) != set(keys):
        raise ResumeError("checkpoint completed scores differ from score rows")


def validate_checkpoint(
    run: ResolvedRunConfig,
    path: Path,
) -> dict[str, Any]:
    checkpoint = _load_json(path)
    shard_id = checkpoint.get("shard_id")
    if not isinstance(shard_id, str) or not shard_id:
        raise ResumeError("checkpoint shard ID is missing")
    expected_path = path.resolve().relative_to(run.output_dir.resolve()).as_posix()
    if checkpoint.get("checkpoint_path") != expected_path:
        raise ResumeError("checkpoint path inventory differs from its location")
    if not isinstance(checkpoint.get("attempt_id"), str) or not checkpoint["attempt_id"]:
        raise ResumeError("checkpoint attempt ID is missing")
    if checkpoint.get("run_id") != run.run_id:
        raise ResumeError("checkpoint belongs to another run")
    if checkpoint.get("output_schema_version") != (
        run.schemas.output_schema_version
    ):
        raise ResumeError("checkpoint output schema version differs")
    if checkpoint.get("metric_schema_version") != (
        run.schemas.metric_schema_version
    ):
        raise ResumeError("checkpoint metric schema version differs")

    paths = _part_paths(run, checkpoint)
    checkpoint_type = checkpoint.get("checkpoint_type")
    if checkpoint_type == "core":
        _validate_core_relations(run, checkpoint, paths)
    elif checkpoint_type == "score":
        if set(paths) != {"scores"}:
            raise ResumeError("score checkpoint must inventory one score part")
        _validate_score_relations(run, checkpoint, paths["scores"])
    else:
        raise ResumeError("unknown checkpoint type")
    return checkpoint


def _rollout_resume_key(value: dict[str, Any]) -> RolloutResumeKey:
    return RolloutResumeKey(
        dataset=value["dataset"],
        split=value["split"],
        sample_id=value["sample_id"],
        strategy=value["strategy"],
        rollout_index=value["rollout_index"],
    )


def build_resume_index(run: ResolvedRunConfig) -> ResumeIndex:
    checkpoint_dir = run.output_dir / "checkpoints"
    core_paths = (
        sorted(checkpoint_dir.glob("shard-*.json"))
        if checkpoint_dir.exists()
        else []
    )
    score_paths = (
        sorted(checkpoint_dir.glob("score-*.json"))
        if checkpoint_dir.exists()
        else []
    )
    rollouts: set[RolloutResumeKey] = set()
    scores: set[ScoreResumeKey] = set()
    core_shards: list[str] = []
    score_shards: list[str] = []

    for path in core_paths:
        checkpoint = validate_checkpoint(run, path)
        shard_id = checkpoint["shard_id"]
        if shard_id in core_shards:
            raise ResumeError("duplicate core shard ID")
        core_shards.append(shard_id)
        for value in checkpoint["completed_rollouts"]:
            key = _rollout_resume_key(value)
            if key in rollouts:
                raise ResumeError("rollout completed by more than one shard")
            rollouts.add(key)

    for path in score_paths:
        checkpoint = validate_checkpoint(run, path)
        shard_id = checkpoint["shard_id"]
        if shard_id in score_shards:
            raise ResumeError("duplicate score shard ID")
        score_shards.append(shard_id)
        for value in checkpoint["completed_scores"]:
            key = ScoreResumeKey(
                rollout=_rollout_resume_key(value),
                scorer_name=value["scorer_name"],
                scorer_version=value["scorer_version"],
            )
            if key in scores:
                raise ResumeError("score completed by more than one shard")
            scores.add(key)

    return ResumeIndex(
        completed_rollouts=frozenset(rollouts),
        completed_scores=frozenset(scores),
        core_shards=tuple(core_shards),
        score_shards=tuple(score_shards),
    )


def referenced_part_paths(run: ResolvedRunConfig) -> frozenset[Path]:
    checkpoint_dir = run.output_dir / "checkpoints"
    paths: set[Path] = set()
    if checkpoint_dir.exists():
        for checkpoint_path in sorted(checkpoint_dir.glob("*.json")):
            checkpoint = validate_checkpoint(run, checkpoint_path)
            paths.update(_part_paths(run, checkpoint).values())
    return frozenset(paths)


def discover_orphan_parts(run: ResolvedRunConfig) -> tuple[Path, ...]:
    referenced = referenced_part_paths(run)
    candidates: set[Path] = set()
    for table_name in (*CORE_TABLES, "scores"):
        directory = run.output_dir / table_name
        if directory.exists():
            candidates.update(
                path.resolve()
                for path in directory.iterdir()
                if path.is_file()
            )
    return tuple(sorted(candidates - referenced))


def quarantine_orphan_parts(
    run: ResolvedRunConfig,
) -> tuple[Path, ...]:
    moved: list[Path] = []
    for source in discover_orphan_parts(run):
        relative = source.relative_to(run.output_dir.resolve())
        destination = run.output_dir / ".orphaned" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if _sha256(destination) != _sha256(source):
                raise ResumeError(f"orphan quarantine collision: {relative}")
            source.unlink()
        else:
            os.rename(source, destination)
        moved.append(destination)
    return tuple(moved)
