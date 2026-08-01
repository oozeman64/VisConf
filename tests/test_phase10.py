"""Phase 10 MathVista and MMMU-Pro adapter acceptance."""

from itertools import islice
from pathlib import Path

from visconf.config import load_experiment_group_config
from visconf.datasets import (
    DatasetAdapter,
    MMMUProAdapter,
    MathVistaAdapter,
    create_dataset_adapter,
)
from visconf.planning import plan_experiment_group
from visconf.storage.parquet_writer import records_to_table
from visconf.types import ExampleRecord, ImageRecord, PromptConfig


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiment_group.yaml"


def dataset_config(loaded, name):
    return next(
        config for config in loaded.datasets if config.name == name
    )


def test_mathvista_adapter_maps_choice_text_and_scores():
    loaded = load_experiment_group_config(CONFIG)
    config = dataset_config(loaded, "mathvista").model_copy(
        update={"filters": {"pid": "3"}}
    )
    adapter = MathVistaAdapter()
    example = next(iter(islice(adapter.load_examples(config), 1)))

    assert isinstance(adapter, DatasetAdapter)
    assert example.sample_id == "3"
    assert example.answer_type == "multiple_choice"
    assert example.ground_truth["answer"] == "C"
    assert "A. 135" in example.question
    assert len(example.images) == 1
    assert example.images[0].mode == "RGB"

    messages = adapter.build_messages(
        example,
        PromptConfig(config.prompt_template),
    )
    assert [part["type"] for part in messages[0]["content"]] == [
        "image",
        "text",
    ]
    score = adapter.score(
        example,
        r"reasoning \boxed{C}",
        loaded.scoring,
    )
    assert score is not None and score["is_correct"]


def test_mmmu_pro_preserves_two_images_and_shared_storage_schema():
    loaded = load_experiment_group_config(CONFIG)
    config = dataset_config(loaded, "mmmu_pro").model_copy(
        update={"filters": {"id": "test_History_76"}}
    )
    adapter = MMMUProAdapter()
    example = next(iter(islice(adapter.load_examples(config), 1)))

    assert isinstance(adapter, DatasetAdapter)
    assert example.sample_id == "test_History_76"
    assert example.answer_type == "multiple_choice"
    assert len(example.images) == 2
    assert example.metadata["image_numbers"] == [1, 2]

    messages = adapter.build_messages(
        example,
        PromptConfig(config.prompt_template),
    )
    image_parts = [
        part
        for part in messages[0]["content"]
        if part["type"] == "image"
    ]
    assert len(image_parts) == 2

    answer = str(example.ground_truth["answer"])
    score = adapter.score(
        example,
        "reasoning " + r"\boxed{" + answer + "}",
        loaded.scoring,
    )
    assert score is not None and score["is_correct"]

    record = ExampleRecord(
        run_id="run",
        dataset=example.dataset,
        split=example.split,
        sample_id=example.sample_id,
        source_row_index=example.source_row_index,
        question=example.question,
        rendered_prompt="prompt",
        prompt_token_ids=(1,),
        ground_truth_json='{"answer":"' + answer + '"}',
        answer_type=example.answer_type,
        images=tuple(
            ImageRecord(
                image.source_ref,
                image.sha256,
                image.width,
                image.height,
                image.mode,
            )
            for image in example.images
        ),
        metadata_json="{}",
    )
    table = records_to_table("examples", (record,))
    assert len(table.to_pylist()[0]["images"]) == 2


def test_registry_and_planner_keep_six_dataset_strategy_run_ids(tmp_path):
    loaded = load_experiment_group_config(CONFIG)
    assert {
        create_dataset_adapter(name).name
        for name in ("mathverse", "mathvista", "mmmu_pro")
    } == {"mathverse", "mathvista", "mmmu_pro"}

    plan = plan_experiment_group(
        loaded,
        experiment_group_id="exp-all-adapters",
        output_root=tmp_path,
    )
    assert len(plan.runs) == 6
    for dataset in ("mathverse", "mathvista", "mmmu_pro"):
        runs = [run for run in plan.runs if run.dataset.name == dataset]
        assert len(runs) == 2
        assert len({run.run_id for run in runs}) == 2
        assert {run.sampling.name for run in runs} == {
            "diverse",
            "concentrated",
        }
