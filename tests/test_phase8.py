"""Phase 8 versioned-scoring acceptance tests."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

from visconf.config import (
    ScoringSettings,
    load_experiment_group_config,
)
from visconf.datasets.mathverse import MathVerseAdapter, _answer_type
from visconf.planning import plan_experiment_group
from visconf.scoring.answer_normalization import score_generation
from visconf.scoring.engine import (
    score_completed_rollouts,
    score_run,
)
from visconf.storage.manifest import RunManifest, read_manifest
from visconf.storage.parquet_writer import write_examples
from visconf.storage.resume import build_resume_index
from visconf.storage.transaction import CoreShardTransaction
from visconf.types import (
    CompletedRollout,
    Example,
    ExampleRecord,
    GenerationRecord,
    RolloutKey,
    ScoringMode,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiment_group.yaml"
NOW = datetime(2026, 7, 31, 12, tzinfo=timezone.utc)


class MathScorer:
    def score(self, example, response, config):
        return score_generation(
            response,
            example.question,
            str(example.ground_truth["answer"]),
            example.answer_type or "freeform",
        )


class AbstainingScorer:
    def score(self, example, response, config):
        return None


class FailingScorer:
    def score(self, example, response, config):
        raise RuntimeError("synthetic scoring failure")


def plan_run(tmp_path: Path, group_id: str):
    config = load_experiment_group_config(CONFIG)
    return plan_experiment_group(
        config,
        experiment_group_id=group_id,
        output_root=tmp_path,
        now=NOW,
    ).runs[0]


def example_for(run):
    return Example(
        dataset=run.dataset.name,
        split=run.dataset.split,
        sample_id="sample",
        source_row_index=0,
        question="Choose one\nA. first\nB. second",
        images=(),
        ground_truth={"answer": "B"},
        answer_type="multiple_choice",
        metadata={},
    )


def write_core(run):
    example = example_for(run)
    write_examples(
        run.output_dir / "examples.parquet",
        (
            ExampleRecord(
                run_id=run.run_id,
                dataset=run.dataset.name,
                split=run.dataset.split,
                sample_id=example.sample_id,
                source_row_index=0,
                question=example.question,
                rendered_prompt="prompt",
                prompt_token_ids=(1, 2),
                ground_truth_json='{"answer":"B"}',
                answer_type=example.answer_type,
                images=(),
                metadata_json="{}",
            ),
        ),
        run.storage,
    )
    key = RolloutKey(
        run_id=run.run_id,
        dataset=run.dataset.name,
        split=run.dataset.split,
        sample_id=example.sample_id,
        strategy=run.sampling.name,
        rollout_index=0,
    )
    bundle = CompletedRollout(
        generation=GenerationRecord(
            key=key,
            rollout_seed=1,
            temperature=run.sampling.temperature,
            top_p=run.sampling.top_p,
            top_k=run.sampling.top_k,
            repetition_penalty=1.0,
            generated_token_ids=(),
            generated_text=r"reasoning \boxed{B}",
            stop_reason="stop_token",
            terminating_token_id=5,
            hit_max_new_tokens=False,
            prompt_token_count=2,
            wall_time_seconds=0.1,
            tokens_per_second=0.0,
            completed_at_utc=NOW,
        ),
        tokens=(),
        probability=(),
        attention=(),
        hidden_state=(),
    )
    CoreShardTransaction(run).commit("core", "attempt", (bundle,))
    return example, bundle


def score_rows(run):
    rows = []
    for path in sorted((run.output_dir / "scores").glob("*.parquet")):
        rows.extend(pq.read_table(path).to_pylist())
    return rows


def test_answer_normalization_covers_numeric_choice_and_malformed_box():
    numeric = score_generation(
        r"work \boxed{1.15}",
        "question",
        "1.2",
        "freeform",
    )
    assert numeric["is_correct"]

    fraction = score_generation(
        r"work \boxed{\frac{3}{4}}",
        "question",
        "0.8",
        "freeform",
    )
    assert not fraction["is_correct"]

    malformed_choice = score_generation(
        "therefore <boxed>C",
        "A. x\nB. y\nC. z",
        "C",
        "multiple_choice",
    )
    assert malformed_choice["is_correct"]
    assert malformed_choice["scorer_method"] == "boxed_letter_lenient"

    malformed_freeform = score_generation(
        r"unfinished \boxed{42",
        "question",
        "42",
        "freeform",
    )
    assert not malformed_freeform["is_correct"]
    assert malformed_freeform["scorer_method"] == "none"


def test_answer_normalization_matches_draft_edge_cases():
    fraction_with_unit = score_generation(
        r"work \boxed{\frac{3}{4} cm}",
        "question",
        "0.75",
        "freeform",
    )
    assert fraction_with_unit["is_correct"]

    degree_answer = score_generation(
        "work \\boxed{45°}",
        "question",
        "45",
        "freeform",
    )
    assert degree_answer["is_correct"]

    for marker in ("(A)", "[A]", "A)", "A]"):
        option_text = score_generation(
            r"work \boxed{red}",
            f"Choose one\n{marker} red\nB. blue",
            "A",
            "multiple_choice",
        )
        assert option_text["is_correct"]
        assert option_text["scorer_method"] == "boxed_option_text"

    ideographic_stop = score_generation(
        "Answer is: A。 Answer is: B",
        "A. first\nB. second",
        "B",
        "multiple_choice",
    )
    assert ideographic_stop["is_correct"]
    assert ideographic_stop["scorer_method"] == "explicit_answer_letter"


def test_mathverse_adapter_scores_and_classifies_choice_rows():
    example = Example(
        dataset="mathverse",
        split="testmini",
        sample_id="mathverse-synthetic",
        source_row_index=0,
        question="A. first\nB. second",
        images=(),
        ground_truth={
            "answer": "B",
            "question_for_eval": "A. first\nB. second",
        },
        answer_type="multiple_choice",
        metadata={},
    )
    score = MathVerseAdapter().score(
        example,
        r"reasoning \boxed{B}",
        ScoringSettings(),
    )
    assert score is not None and score["is_correct"]

    for marker in ("(A)", "[A]", "A)", "A]"):
        assert _answer_type(
            {"question": f"Choose\n{marker} first\nB. second", "answer": "A"}
        ) == "multiple_choice"

    assert _answer_type(
        {"question": "A. first\nB. second", "answer": "C"}
    ) == "freeform"
    assert _answer_type(
        {"question": "The point A. has value 4.", "answer": "A"}
    ) == "freeform"


def test_online_and_offline_scoring_have_identical_semantics(tmp_path):
    offline_run = plan_run(tmp_path, "exp-score-offline")
    online_run = plan_run(tmp_path, "exp-score-online")
    offline_example, _ = write_core(offline_run)
    online_example, online_bundle = write_core(online_run)

    offline = score_run(
        offline_run,
        adapter=MathScorer(),
        attempt_id="offline",
        scored_at=NOW,
    )
    online = score_completed_rollouts(
        online_run,
        MathScorer(),
        {online_example.sample_id: online_example},
        (online_bundle,),
        core_shard_id="core",
        attempt_id="online",
        scored_at=NOW,
    )
    assert offline.committed_scores == online.committed_scores == 1

    left = score_rows(offline_run)[0]
    right = score_rows(online_run)[0]
    comparable = (
        "dataset",
        "split",
        "sample_id",
        "strategy",
        "rollout_index",
        "scorer_name",
        "scorer_version",
        "is_correct",
        "raw_final_answer",
        "extracted_answer",
        "scorer_method",
        "score_details_json",
        "scored_at_utc",
    )
    assert {key: left[key] for key in comparable} == {
        key: right[key] for key in comparable
    }


def test_versions_abstention_and_failures_are_independent(tmp_path):
    run = plan_run(tmp_path, "exp-score-versions")
    write_core(run)
    score_run(run, adapter=MathScorer(), scored_at=NOW)

    version_two = ScoringSettings(
        mode=ScoringMode.OFFLINE,
        scorer_name="dataset_default",
        scorer_version="2",
    )
    score_run(
        run,
        adapter=MathScorer(),
        scoring_config=version_two,
        scored_at=NOW,
    )

    abstention = ScoringSettings(
        mode=ScoringMode.OFFLINE,
        scorer_name="dataset_default",
        scorer_version="3",
    )
    score_run(
        run,
        adapter=AbstainingScorer(),
        scoring_config=abstention,
        scored_at=NOW,
    )
    rows = score_rows(run)
    assert {row["scorer_version"] for row in rows} == {"1", "2", "3"}
    assert next(
        row["is_correct"]
        for row in rows
        if row["scorer_version"] == "3"
    ) is None

    failing = ScoringSettings(
        mode=ScoringMode.OFFLINE,
        scorer_name="dataset_default",
        scorer_version="4",
    )
    result = score_run(
        run,
        adapter=FailingScorer(),
        scoring_config=failing,
        scored_at=NOW,
    )
    assert result.failed_scores == 1
    assert len(build_resume_index(run).completed_rollouts) == 1
    assert len(build_resume_index(run).completed_scores) == 3
    assert (run.output_dir / "failures.jsonl").is_file()

    manifest = read_manifest(
        run.output_dir / "manifest.json",
        RunManifest,
    )
    assert {
        identity["version"] for identity in manifest.scorer_versions
    } == {"1", "2", "3", "4"}
