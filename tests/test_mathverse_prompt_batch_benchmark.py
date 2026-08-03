import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "mathverse_prompt_batch_benchmark",
    ROOT / "scripts" / "mathverse_prompt_batch_benchmark.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

DEFAULT_SHAPES = MODULE.DEFAULT_SHAPES
_benchmark_config = MODULE._benchmark_config
_hardware_config = MODULE._hardware_config
_shape = MODULE._shape
from visconf.config import load_experiment_group_config


ROOT = Path(__file__).resolve().parents[1]


def test_mathverse_benchmark_shape_parser():
    assert _shape("16x32") == (16, 32)
    with pytest.raises(Exception):
        _shape("16-by-32")
    with pytest.raises(Exception):
        _shape("0x32")


def test_mathverse_benchmark_config_overrides_generation_and_shapes():
    loaded = load_experiment_group_config(
        ROOT / "configs" / "experiment_group_4090_full_mb32.yaml"
    )
    configured = _benchmark_config(
        loaded,
        _hardware_config(ROOT / "configs" / "hardware" / "rtx_4090.yaml"),
        DEFAULT_SHAPES,
        prompts=16,
        rollouts=32,
        max_new_tokens=100,
    )

    assert configured.generation.prompt_batch_size == 16
    assert configured.generation.rollout_microbatch_size == 32
    assert configured.generation.rollouts_per_example == 32
    assert configured.generation.max_new_tokens == 100
    assert configured.hardware.benchmark_batch_shapes == DEFAULT_SHAPES
    assert configured.hardware.max_active_decode_rows >= 16 * 32


def test_mathverse_benchmark_rejects_insufficient_hardware_profile():
    loaded = load_experiment_group_config(
        ROOT / "configs" / "experiment_group.yaml"
    )
    with pytest.raises(ValueError, match="rollout microbatches"):
        _benchmark_config(
            loaded,
            _hardware_config(ROOT / "configs" / "hardware" / "a100.yaml"),
            ((16, 32),),
            prompts=16,
            rollouts=32,
            max_new_tokens=100,
        )
