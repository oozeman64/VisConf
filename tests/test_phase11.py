"""Phase 11 production-readiness acceptance."""

import json
import subprocess
from pathlib import Path

import yaml

from visconf.benchmark import benchmark_candidates
from visconf.config import (
    HardwareSettings,
    load_experiment_group_config,
)
from visconf.planning import plan_experiment_group
from visconf.production import validate_production_readiness
from visconf.storage.persistence import (
    initialize_persistence_marker,
    verify_persistence_marker,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiment_group.yaml"


def test_hardware_profiles_and_revisions_are_frozen(tmp_path):
    loaded = load_experiment_group_config(CONFIG)
    assert loaded.model.revision
    assert all(dataset.revision for dataset in loaded.datasets)

    expected = {
        "a100_40gb": (4, (4, 8)),
        "a100_80gb": (8, (8, 16)),
        "h100": (16, (16, 32)),
        "rtx_4090": (2, (1, 2, 4)),
    }
    for name, (default, candidates) in expected.items():
        raw = yaml.safe_load(
            (ROOT / "configs" / "hardware" / f"{name}.yaml")
            .read_text(encoding="utf-8")
        )
        profile = HardwareSettings.model_validate(raw)
        assert profile.default_rollout_microbatch_size == default
        assert profile.benchmark_microbatch_sizes == candidates

    plan = plan_experiment_group(
        loaded,
        experiment_group_id="exp-hardware",
        output_root=tmp_path,
    )
    assert benchmark_candidates(plan.runs[0]) == (16, 32)


def test_persistence_marker_survives_independent_verification(tmp_path):
    initialized = initialize_persistence_marker(tmp_path)
    verified = verify_persistence_marker(tmp_path)

    assert initialized["marker_id"] == verified["marker_id"]
    assert initialized["output_root"] == str(tmp_path.resolve())
    assert "verified_at_utc" in verified


def test_production_validator_and_launch_scripts(tmp_path):
    loaded = load_experiment_group_config(CONFIG)
    plan = plan_experiment_group(
        loaded,
        experiment_group_id="exp-production",
        output_root=tmp_path,
    )
    marker = initialize_persistence_marker(tmp_path)
    benchmarked_run = plan.runs[0]
    hardware = plan.runs[0].hardware
    report = {
        "run_id": benchmarked_run.run_id,
        "run_config_hash": benchmarked_run.config_hash,
        "hardware_profile": hardware.model_dump(mode="json"),
        "measurements": [
            {
                "requested_microbatch_size": size,
                "status": "complete",
                "retained_ids_sha256": "same-sequence",
            }
            for size in hardware.benchmark_microbatch_sizes
        ],
    }
    report_path = tmp_path / "benchmark.json"
    report_path.write_text(
        json.dumps(report),
        encoding="utf-8",
    )

    readiness = validate_production_readiness(
        plan,
        ROOT,
        report_path,
    )
    assert readiness.run_count == 6
    assert readiness.hardware_profile == "h100"
    assert readiness.persistent_marker_id == marker["marker_id"]

    scripts = sorted((ROOT / "scripts").glob("*.sh"))
    assert {path.name for path in scripts} == {
        "benchmark_hardware.sh",
        "run_experiment_group.sh",
        "run_resume_rehearsal.sh",
        "run_single_run.sh",
        "run_smoke_test.sh",
        "verify_persistent_storage.sh",
    }
    for script in scripts:
        subprocess.run(["bash", "-n", str(script)], check=True)
