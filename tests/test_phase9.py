"""Phase 9 real-Qwen MathVerse acceptance."""

from __future__ import annotations

import gc
import json
import math
from itertools import islice
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import torch

from visconf.config import (
    GenerationSettings,
    load_experiment_group_config,
)
from visconf.datasets.mathverse import MathVerseAdapter
from visconf.generation.engine import GenerationEngine
from visconf.metrics.probability import compute_probability_metrics
from visconf.models.instrumentation import (
    QwenInstrumentation,
    discover_decoder_layers,
)
from visconf.models.qwen25vl import QwenModelFacade
from visconf.planning import plan_experiment_group
from visconf.storage.parquet_writer import write_examples
from visconf.storage.resume import (
    build_resume_index,
    discover_orphan_parts,
    quarantine_orphan_parts,
    validate_checkpoint,
)
from visconf.storage.schema import CORE_TABLES, SCHEMAS
from visconf.storage.transaction import CoreShardTransaction
from visconf.types import (
    ExampleRecord,
    ImageRecord,
    PromptConfig,
    RolloutKey,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiment_group.yaml"


class OOMOnceFacade:
    def __init__(self, facade):
        self.facade = facade
        self.raised = False

    def __getattr__(self, name):
        return getattr(self.facade, name)

    def repeat_cache(self, cache, batch_size):
        if batch_size == 2 and not self.raised:
            self.raised = True
            raise torch.cuda.OutOfMemoryError("injected acceptance OOM")
        return self.facade.repeat_cache(cache, batch_size)


def keys_for(run, sample_id):
    return tuple(
        RolloutKey(
            run_id=run.run_id,
            dataset=run.dataset.name,
            split=run.dataset.split,
            sample_id=sample_id,
            strategy=run.sampling.name,
            rollout_index=index,
        )
        for index in range(2)
    )


def retained_ids(results):
    return {
        result.generation.key.rollout_index:
        result.generation.generated_token_ids
        for result in results
    }


def test_legacy_probability_overlap_and_intentional_support_difference():
    logits = torch.tensor([0.0, -1.0, 2.0, 0.5])
    selected = 2
    current = compute_probability_metrics(logits, selected)
    log_probabilities = torch.log_softmax(logits.float(), dim=-1)
    probabilities = log_probabilities.exp()
    legacy_entropy = torch.sum(probabilities * log_probabilities)
    legacy_kl = -math.log(logits.numel()) - log_probabilities.mean()

    assert current.logp == pytest.approx(float(log_probabilities[selected]))
    assert current.kl_u_p == pytest.approx(float(legacy_kl))
    assert current.gini == pytest.approx(float(torch.sum(probabilities.square())))
    assert current.entropy == pytest.approx(float(legacy_entropy))
    assert current.dist_perplexity == pytest.approx(
        float(-torch.exp(-legacy_entropy))
    )

    full_support = compute_probability_metrics(
        torch.tensor([0.0, -torch.inf, 1.0]),
        2,
    )
    assert math.isinf(full_support.kl_u_p)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_real_two_rollout_storage_resume_and_cleanup(tmp_path):
    loaded = load_experiment_group_config(CONFIG)
    plan = plan_experiment_group(
        loaded,
        experiment_group_id="exp-real-acceptance",
        output_root=tmp_path,
    )
    run = plan.runs[0].model_copy(
        update={
            "generation": GenerationSettings(
                rollouts_per_example=2,
                rollout_microbatch_size=2,
                max_new_tokens=2,
            )
        }
    )
    adapter = MathVerseAdapter()
    example = next(iter(islice(adapter.load_examples(run.dataset), 1)))
    messages = adapter.build_messages(
        example,
        PromptConfig(run.dataset.prompt_template),
    )
    facade = QwenModelFacade.load(run.model)
    layers = discover_decoder_layers(facade.model)
    originals = tuple(layer.forward for layer in layers)
    allocated_with_model = torch.cuda.memory_allocated()
    results_one = ()
    results_two = ()
    results_recovered = ()
    instrumentation = None
    single = None
    batched = None
    recovered = None
    oom_facade = None

    try:
        with QwenInstrumentation(facade.model) as instrumentation:
            single = GenerationEngine(
                facade,
                instrumentation,
                base_seed=run.base_seed,
                max_new_tokens=2,
                rollout_microbatch_size=1,
            )
            results_one = tuple(
                single.generate_example(
                    example,
                    facade.prepare_example(messages),
                    keys_for(run, example.sample_id),
                    run.sampling.as_domain(),
                )
            )

            batched = GenerationEngine(
                facade,
                instrumentation,
                base_seed=run.base_seed,
                max_new_tokens=2,
                rollout_microbatch_size=2,
            )
            results_two = tuple(
                batched.generate_example(
                    example,
                    facade.prepare_example(messages),
                    keys_for(run, example.sample_id),
                    run.sampling.as_domain(),
                )
            )

            oom_facade = OOMOnceFacade(facade)
            recovered = GenerationEngine(
                oom_facade,
                instrumentation,
                base_seed=run.base_seed,
                max_new_tokens=2,
                rollout_microbatch_size=2,
            )
            results_recovered = tuple(
                recovered.generate_example(
                    example,
                    facade.prepare_example(messages),
                    keys_for(run, example.sample_id),
                    run.sampling.as_domain(),
                )
            )
            assert oom_facade.raised
            assert retained_ids(results_one) == retained_ids(results_two)
            assert retained_ids(results_one) == retained_ids(
                results_recovered
            )

        assert all(
            layer.forward == original
            for layer, original in zip(layers, originals, strict=True)
        )

        prompt = facade.prepare_example(messages).prompt_record
        write_examples(
            run.output_dir / "examples.parquet",
            (
                ExampleRecord(
                    run_id=run.run_id,
                    dataset=run.dataset.name,
                    split=run.dataset.split,
                    sample_id=example.sample_id,
                    source_row_index=example.source_row_index,
                    question=example.question,
                    rendered_prompt=prompt.rendered_prompt,
                    prompt_token_ids=prompt.prompt_token_ids,
                    ground_truth_json=json.dumps(
                        example.ground_truth,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                    ),
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
                    metadata_json=json.dumps(
                        example.metadata,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                    ),
                ),
            ),
            run.storage,
        )

        transaction = CoreShardTransaction(run)

        def interrupt(stage):
            if stage == "before_checkpoint":
                raise RuntimeError("injected interruption")

        with pytest.raises(RuntimeError, match="injected interruption"):
            transaction.commit(
                "acceptance",
                "attempt-interrupted",
                results_two,
                failure_injector=interrupt,
            )
        assert len(discover_orphan_parts(run)) == 5
        assert len(quarantine_orphan_parts(run)) == 5

        transaction.commit(
            "acceptance",
            "attempt-resumed",
            results_two,
        )
        resume = build_resume_index(run)
        assert len(resume.completed_rollouts) == 2
        assert resume.core_shards == ("acceptance",)

        checkpoint_path = (
            run.output_dir / "checkpoints" / "shard-acceptance.json"
        )
        checkpoint = validate_checkpoint(run, checkpoint_path)
        paths = {
            part["table_name"]:
            run.output_dir / part["relative_path"]
            for part in checkpoint["parts"]
        }
        assert set(paths) == set(CORE_TABLES)
        tables = {
            name: pq.read_table(path)
            for name, path in paths.items()
        }
        assert all(
            tables[name].schema.equals(SCHEMAS[name])
            for name in CORE_TABLES
        )

        token_keys = [
            tuple(row[name] for name in (
                "run_id",
                "dataset",
                "split",
                "sample_id",
                "strategy",
                "rollout_index",
                "step",
            ))
            for row in tables["tokens"].to_pylist()
        ]
        for family in CORE_TABLES[2:]:
            family_keys = [
                tuple(row[name] for name in (
                    "run_id",
                    "dataset",
                    "split",
                    "sample_id",
                    "strategy",
                    "rollout_index",
                    "step",
                ))
                for row in tables[family].to_pylist()
            ]
            assert family_keys == token_keys

        stop_ids = facade.stop_token_ids()
        for bundle in results_two:
            assert not stop_ids.intersection(
                bundle.generation.generated_token_ids
            )
            if bundle.generation.stop_reason == "stop_token":
                assert bundle.generation.terminating_token_id in stop_ids
            for index, token in enumerate(bundle.tokens):
                assert token.predictor_position == (
                    prompt.prompt_token_count - 1
                    if index == 0
                    else prompt.prompt_token_count + index - 1
                )
                assert token.context_length == (
                    prompt.prompt_token_count + index
                )
                attention = bundle.attention[index]
                assert attention.n_generated_text_tokens == index
                hidden = bundle.hidden_state[index].metrics
                assert hidden.valid_layers_all == 36
                assert (
                    hidden.valid_layers_early_visual_integration == 21
                )
                assert hidden.valid_layers_visual_reasoning == 13
                assert hidden.last_layer_valid

        for table_name in CORE_TABLES[2:]:
            for row in tables[table_name].to_pylist():
                for value in row.values():
                    if isinstance(value, float):
                        assert not math.isnan(value)
    finally:
        del results_one
        del results_two
        del results_recovered
        del single
        del batched
        del recovered
        del oom_facade
        del instrumentation
        del facade
        del layers
        del originals
        gc.collect()
        torch.cuda.empty_cache()
        assert torch.cuda.memory_allocated() < allocated_with_model / 4
