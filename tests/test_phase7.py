"""Phase 7 runner and CLI integration tests."""

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq
import torch

from visconf.cli import main
from visconf.config import load_experiment_group_config
from visconf.planning import plan_experiment_group
from visconf.runner import (
    RunSummary,
    execute_experiment_group,
    execute_run,
)
from visconf.storage.manifest import (
    ExperimentManifest,
    RunManifest,
    read_manifest,
)
from visconf.storage.resume import build_resume_index
from visconf.types import (
    CompletedRollout,
    Example,
    GenerationRecord,
    PromptRecord,
    RunStatus,
    TokenGroups,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiment_group.yaml"
NOW = datetime(2026, 7, 31, 12, tzinfo=timezone.utc)


class FakeAdapter:
    name = "mathverse"

    def load_examples(self, config):
        return tuple(
            Example(
                dataset=config.name,
                split=config.split,
                sample_id=f"sample-{index}",
                source_row_index=index,
                question=f"question {index}",
                images=(),
                ground_truth={"answer": str(index)},
                answer_type="freeform",
                metadata={"index": index},
            )
            for index in range(2)
        )

    def build_messages(self, example, prompt_config):
        return [{"role": "user", "content": example.question}]

    def ground_truth(self, example):
        return example.ground_truth


class FakeFacade:
    def prepare_example(self, messages):
        return SimpleNamespace(
            prompt_record=PromptRecord(
                rendered_prompt=str(messages),
                prompt_token_ids=(1, 2),
                prompt_token_count=2,
            ),
            token_groups=TokenGroups(
                image_positions=torch.tensor([], dtype=torch.long),
                prompt_text_positions=torch.tensor([0, 1]),
                prompt_last_position=1,
                prompt_token_count=2,
            ),
        )


class FakeInstrumentation:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return None


class FakeEngine:
    def __init__(self, facade, instrumentation, **settings):
        self.facade = facade

    def generate_example(self, example, prepared, rollout_keys, sampling):
        for key in rollout_keys:
            yield CompletedRollout(
                generation=GenerationRecord(
                    key=key,
                    rollout_seed=key.rollout_index,
                    temperature=sampling.temperature,
                    top_p=sampling.top_p,
                    top_k=sampling.top_k,
                    repetition_penalty=sampling.repetition_penalty,
                    generated_token_ids=(),
                    generated_text="",
                    stop_reason="stop_token",
                    terminating_token_id=5,
                    hit_max_new_tokens=False,
                    prompt_token_count=2,
                    wall_time_seconds=0.0,
                    tokens_per_second=0.0,
                    completed_at_utc=NOW,
                ),
                tokens=(),
                probability=(),
                attention=(),
                hidden_state=(),
            )


def planned(tmp_path: Path, group_id: str):
    config = load_experiment_group_config(CONFIG)
    return plan_experiment_group(
        config,
        experiment_group_id=group_id,
        output_root=tmp_path,
        now=NOW,
    )


def test_one_run_commits_examples_and_resumes_without_duplicates(
    tmp_path, capsys
):
    plan = planned(tmp_path, "exp-runner")
    run = plan.runs[0]

    summary = execute_run(
        run,
        adapter=FakeAdapter(),
        facade=FakeFacade(),
        instrumentation_factory=lambda facade: FakeInstrumentation(),
        engine_factory=FakeEngine,
        attempt_id="attempt-test",
    )
    progress_output = capsys.readouterr().err
    assert "0/2 prompts complete | ETA --:--" in progress_output
    assert "2/2 prompts complete | ETA 00:00" in progress_output

    assert summary.status is RunStatus.COMPLETE
    assert summary.committed_rollouts == 64
    assert summary.committed_shards == 2
    assert pq.read_table(run.output_dir / "examples.parquet").num_rows == 2

    resume = build_resume_index(run)
    assert len(resume.completed_rollouts) == 64
    assert len(resume.core_shards) == 2

    resumed = execute_run(run)
    assert resumed.committed_rollouts == 64
    assert resumed.skipped_rollouts == 64
    assert len(build_resume_index(run).core_shards) == 2

    run_manifest = read_manifest(
        run.output_dir / "manifest.json",
        RunManifest,
    )
    group_manifest = read_manifest(
        plan.group_dir / "experiment_manifest.json",
        ExperimentManifest,
    )
    assert run_manifest.status is RunStatus.COMPLETE
    assert next(
        entry.status
        for entry in group_manifest.runs
        if entry.run_id == run.run_id
    ) is RunStatus.COMPLETE

    assert (
        main(
            [
                "validate",
                "--group",
                plan.manifest.experiment_group_id,
                "--output-root",
                str(tmp_path),
            ]
        )
        == 0
    )


def test_group_orchestration_visits_each_isolated_run(tmp_path, monkeypatch):
    plan = planned(tmp_path, "exp-group")
    visited = []

    def fake_execute(run):
        visited.append((run.dataset.name, run.sampling.name, run.run_id))
        return RunSummary(
            run_id=run.run_id,
            status=RunStatus.COMPLETE,
            committed_rollouts=0,
            committed_shards=0,
            skipped_rollouts=0,
            quarantined_parts=0,
        )

    monkeypatch.setattr("visconf.runner.execute_run", fake_execute)
    summary = execute_experiment_group(plan.group_dir)

    assert len(summary.runs) == 6
    assert len(set(visited)) == 6
    assert {
        (run.dataset.name, run.sampling.name)
        for run in plan.runs
    } == {(dataset, strategy) for dataset, strategy, _ in visited}
