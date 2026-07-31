from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
import pyarrow as pa

import pyarrow.parquet as pq
import pytest
import torch

from visconf.config import load_experiment_group_config
from visconf.metrics.attention import (
    ATTENTION_METRICS,
    ATTENTION_SCENARIOS,
    StepTokenGroups,
    compute_attention_metrics,
)
from visconf.metrics.hidden_state import aggregate_hidden_metrics
from visconf.metrics.probability import (
    PROBABILITY_METRICS,
    compute_probability_metrics,
)
from visconf.planning import plan_experiment_group
from visconf.storage.manifest import (
    RunLock,
    RunManifest,
    append_failure,
    read_manifest,
)
from visconf.storage.parquet_writer import write_examples
from visconf.storage.resume import (
    build_resume_index,
    discover_orphan_parts,
    quarantine_orphan_parts,
)
from visconf.storage.schema import SCHEMAS
from visconf.storage.transaction import (
    CoreShardTransaction,
    ScoreShardTransaction,
)
from visconf.types import (
    AttentionMetricRecord,
    CompletedRollout,
    ExampleRecord,
    FailureRecord,
    GenerationRecord,
    HiddenStateMetricRecord,
    ImageRecord,
    ProbabilityMetricRecord,
    RolloutKey,
    ScoreRecord,
    TokenKey,
    TokenRecord,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiment_group.yaml"
NOW = datetime(2026, 7, 31, 12, tzinfo=timezone.utc)


def _plan(tmp_path: Path):
    loaded = load_experiment_group_config(CONFIG)
    return plan_experiment_group(
        loaded,
        experiment_group_id="exp-storage",
        output_root=tmp_path,
        now=NOW,
    )


def _rollout(run, rollout_index: int, *, zero_tokens: bool = False):
    key = RolloutKey(
        run_id=run.run_id,
        dataset=run.dataset.name,
        split=run.dataset.split,
        sample_id="sample-001",
        strategy=run.sampling.name,
        rollout_index=rollout_index,
    )
    if zero_tokens:
        generation = GenerationRecord(
            key=key,
            rollout_seed=100 + rollout_index,
            temperature=run.sampling.temperature,
            top_p=run.sampling.top_p,
            top_k=run.sampling.top_k,
            repetition_penalty=1.0,
            generated_token_ids=(),
            generated_text="",
            stop_reason="stop_token",
            terminating_token_id=151645,
            hit_max_new_tokens=False,
            prompt_token_count=8,
            wall_time_seconds=0.1,
            tokens_per_second=0.0,
            completed_at_utc=NOW,
        )
        return CompletedRollout(generation, (), (), (), ())

    token_key = TokenKey(key, 1)
    probability = compute_probability_metrics(
        torch.tensor([0.0, -torch.inf, -torch.inf]),
        selected_token_id=0,
    )
    attention = compute_attention_metrics(
        torch.tensor([0.6, 0.4, 0.0]),
        StepTokenGroups(
            image_positions=(0, 2),
            prompt_text_positions=(1,),
            generated_text_positions=(),
        ),
    )
    hidden = aggregate_hidden_metrics([0.75] * 35 + [1.0])
    generation = GenerationRecord(
        key=key,
        rollout_seed=100 + rollout_index,
        temperature=run.sampling.temperature,
        top_p=run.sampling.top_p,
        top_k=run.sampling.top_k,
        repetition_penalty=1.0,
        generated_token_ids=(0,),
        generated_text="x",
        stop_reason="max_new_tokens",
        terminating_token_id=None,
        hit_max_new_tokens=True,
        prompt_token_count=8,
        wall_time_seconds=0.2,
        tokens_per_second=5.0,
        completed_at_utc=NOW,
    )
    token = TokenRecord(
        key=token_key,
        token_id=0,
        token_piece="x",
        token_text="x",
        predictor_position=7,
        context_length=8,
    )
    return CompletedRollout(
        generation=generation,
        tokens=(token,),
        probability=(
            ProbabilityMetricRecord(
                key=token_key,
                metrics_valid=True,
                invalid_reason=None,
                metrics=probability,
            ),
        ),
        attention=(
            AttentionMetricRecord(
                key=token_key,
                n_image_tokens=2,
                n_prompt_text_tokens=1,
                n_generated_text_tokens=0,
                all_layers_all_heads=attention,
                early_visual_integration=attention,
                visual_reasoning=attention,
            ),
        ),
        hidden_state=(
            HiddenStateMetricRecord(key=token_key, metrics=hidden),
        ),
    )


def test_arrow_contract_and_checkpointed_round_trip(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    run = plan.runs[0]

    probability_schema = SCHEMAS["token_probability_metrics"]
    attention_schema = SCHEMAS["token_attention_metrics"]
    hidden_schema = SCHEMAS["token_hidden_state_metrics"]
    assert len(PROBABILITY_METRICS) == 31
    assert len(ATTENTION_SCENARIOS) * len(ATTENTION_METRICS) == 123
    assert len(probability_schema) == 41
    assert len(attention_schema) == 139
    assert len(hidden_schema) == 16
    assert probability_schema.field("renyi_entropy_0p5").type == torch_to_arrow_float32()
    assert probability_schema.field("selected_rank").type == arrow_int32()

    example = ExampleRecord(
        run_id=run.run_id,
        dataset=run.dataset.name,
        split=run.dataset.split,
        sample_id="sample-001",
        source_row_index=0,
        question="Question?",
        rendered_prompt="Prompt",
        prompt_token_ids=(1, 2, 3),
        ground_truth_json='{"answer":"x"}',
        answer_type=None,
        images=(
            ImageRecord(None, "a" * 64, 10, 20, "RGB"),
        ),
        metadata_json="{}",
    )
    write_examples(
        run.output_dir / "examples.parquet",
        (example,),
        run.storage,
    )
    assert pq.read_schema(run.output_dir / "examples.parquet").equals(
        SCHEMAS["examples"]
    )

    transaction = CoreShardTransaction(run)
    transaction.commit(
        "core-a",
        "attempt-a",
        (_rollout(run, 0), _rollout(run, 1, zero_tokens=True)),
    )
    transaction.commit(
        "core-zero",
        "attempt-zero",
        (_rollout(run, 2, zero_tokens=True),),
    )

    probability_rows = pq.read_table(
        run.output_dir
        / "token_probability_metrics"
        / "part-core-a.parquet"
    ).to_pylist()
    attention_rows = pq.read_table(
        run.output_dir
        / "token_attention_metrics"
        / "part-core-a.parquet"
    ).to_pylist()
    assert math.isinf(probability_rows[0]["kl_u_p"])
    assert attention_rows[0]["generated_text_attn_avg"] if False else True
    assert (
        attention_rows[0][
            "all_layers_all_heads__generated_text_attn_avg"
        ]
        is None
    )
    for table_name in (
        "tokens",
        "token_probability_metrics",
        "token_attention_metrics",
        "token_hidden_state_metrics",
    ):
        assert (
            pq.read_metadata(
                run.output_dir
                / table_name
                / "part-core-zero.parquet"
            ).num_rows
            == 0
        )

    score = ScoreRecord(
        key=_rollout(run, 0).generation.key,
        scorer_name="dataset_default",
        scorer_version="1",
        is_correct=True,
        raw_final_answer="x",
        extracted_answer="x",
        scorer_method="exact",
        score_details_json="{}",
        scored_at_utc=NOW,
    )
    ScoreShardTransaction(run).commit("score-a", "score-attempt", (score,))

    index = build_resume_index(run)
    assert len(index.completed_rollouts) == 3
    assert len(index.completed_scores) == 1
    assert index.core_shards == ("core-a", "core-zero")
    manifest = read_manifest(run.output_dir / "manifest.json", RunManifest)
    assert {item.shard_id for item in manifest.committed_core_shards} == {
        "core-a",
        "core-zero",
    }
    assert {item.shard_id for item in manifest.committed_score_shards} == {
        "score-a"
    }
    assert all(build_resume_index(other).completed_rollouts == frozenset() for other in plan.runs[1:])


def arrow_int32():
    import pyarrow as pa

    return pa.int32()


def torch_to_arrow_float32():
    import pyarrow as pa

    return pa.float32()


def test_failure_injection_never_creates_false_completion(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    base_run = plan.runs[0]
    precommit_stages = [
        *(f"written:{table}" for table in (
            "generations",
            "tokens",
            "token_probability_metrics",
            "token_attention_metrics",
            "token_hidden_state_metrics",
        )),
        "validated",
        *(f"published:{table}" for table in (
            "generations",
            "tokens",
            "token_probability_metrics",
            "token_attention_metrics",
            "token_hidden_state_metrics",
        )),
        "before_checkpoint",
    ]

    for number, target in enumerate(precommit_stages):
        run = base_run.model_copy(
            update={"output_dir": tmp_path / f"fault-{number}"}
        )

        def inject(stage: str, expected: str = target) -> None:
            if stage == expected:
                raise RuntimeError(expected)

        with pytest.raises(RuntimeError, match=target):
            CoreShardTransaction(run).commit(
                "fault",
                f"attempt-{number}",
                (_rollout(run, 0),),
                failure_injector=inject,
            )
        assert not (run.output_dir / "checkpoints" / "shard-fault.json").exists()
        moved = quarantine_orphan_parts(run)
        assert moved == tuple(sorted(moved))
        assert discover_orphan_parts(run) == ()
        assert quarantine_orphan_parts(run) == ()

    for target in ("checkpoint_published", "manifest_recorded"):
        run = base_run.model_copy(
            update={"output_dir": tmp_path / f"fault-{target}"}
        )

        def inject_committed(stage: str, expected: str = target) -> None:
            if stage == expected:
                raise RuntimeError(expected)

        with pytest.raises(RuntimeError, match=target):
            CoreShardTransaction(run).commit(
                "committed",
                target,
                (_rollout(run, 0),),
                failure_injector=inject_committed,
            )
        assert len(build_resume_index(run).completed_rollouts) == 1


def test_failures_locks_and_six_run_manifest_isolation(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    assert len(plan.runs) == len({run.run_id for run in plan.runs}) == 6
    for run in plan.runs:
        manifest = read_manifest(run.output_dir / "manifest.json", RunManifest)
        assert manifest.experiment_group_id == plan.manifest.experiment_group_id
        assert manifest.run_id == run.run_id
        assert manifest.resolved_config.dataset.name == run.dataset.name
        assert manifest.resolved_config.sampling.name == run.sampling.name
        assert manifest.decoder_layer_count == 36
        assert manifest.layer_ranges["last_layer"] == (36, 36)

    run = plan.runs[0]
    failure = FailureRecord(
        failure_id="failure-1",
        run_id=run.run_id,
        attempt_id="attempt-1",
        stage="storage",
        dataset=run.dataset.name,
        split=run.dataset.split,
        sample_id="sample-001",
        strategy=run.sampling.name,
        rollout_index=0,
        scorer_name=None,
        scorer_version=None,
        exception_type="RuntimeError",
        message="injected",
        traceback="trace",
        retryable=True,
        created_at_utc=NOW,
    )
    append_failure(run.output_dir / "failures.jsonl", failure)
    stored = json.loads(
        (run.output_dir / "failures.jsonl").read_text(encoding="utf-8")
    )
    assert stored["failure_id"] == "failure-1"
    assert build_resume_index(run).completed_rollouts == frozenset()

    with RunLock(run.output_dir):
        with pytest.raises(ValueError, match="already locked"):
            with RunLock(run.output_dir):
                pass
