from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from visconf.config import ResolvedRunConfig, load_experiment_group_config
from visconf.planning import load_experiment_plan, plan_experiment_group


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiment_group.yaml"


def test_initial_config_matches_the_fixed_experiment() -> None:
    config = load_experiment_group_config(CONFIG)

    assert config.model.num_hidden_layers == 36
    assert config.model.attention_implementation == "eager"
    assert {item.name for item in config.datasets} == {
        "mathverse",
        "mathvista",
        "mmmu_pro",
    }
    selections = {
        item.name: (item.split, item.filter_id, item.filters)
        for item in config.datasets
    }
    assert selections["mathverse"] == (
        "testmini",
        "vision_intensive",
        {"problem_version": "Vision Intensive"},
    )
    assert selections["mathvista"] == ("testmini", "all", {})
    assert selections["mmmu_pro"] == ("test", "all", {})
    assert {item.name for item in config.strategies} == {
        "diverse",
        "concentrated",
    }


def test_plan_is_six_isolated_idempotent_runs(tmp_path: Path) -> None:
    config = load_experiment_group_config(CONFIG)
    plan = plan_experiment_group(
        config,
        experiment_group_id="exp-test",
        output_root=tmp_path,
        now=datetime(2026, 7, 31, tzinfo=timezone.utc),
    )

    cells = {
        (
            run.dataset.name,
            run.dataset.split,
            run.dataset.filter_id,
            run.sampling.name,
        )
        for run in plan.runs
    }
    assert len(plan.runs) == len(cells) == 6
    assert len({run.run_id for run in plan.runs}) == 6
    assert all((run.output_dir / "manifest.json").is_file() for run in plan.runs)

    reopened = plan_experiment_group(
        config,
        experiment_group_id="exp-test",
        output_root=tmp_path,
    )
    assert reopened.runs == plan.runs
    assert load_experiment_plan(plan.group_dir) == plan

    invalid = plan.runs[0].model_dump()
    invalid["datasets"] = [invalid["dataset"]]
    invalid["strategies"] = [invalid["sampling"]]
    with pytest.raises(ValidationError):
        ResolvedRunConfig.model_validate(invalid)
