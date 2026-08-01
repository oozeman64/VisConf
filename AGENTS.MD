# VisConf Repository Plan

## Project goal

Build a modular experiment runner for Qwen 2.5 VL 3B that generates multimodal
rollouts and records every per-decoding-step metric defined in `docs/`.

The initial experiment scope is:

- Model: Qwen 2.5 VL 3B.
- Datasets: MathVerse, MathVista, and MMMU-Pro.
  - MathVerse defaults to the Vision Intensive subset of `testmini`.
  - MathVista defaults to every row in `testmini`.
  - MMMU-Pro defaults to every row in the standard four-option test source.
  - Dataset source, split, and filters remain configurable in YAML.
- Sampling strategies:
  - `diverse`: temperature `1.0`, top-p `0.99`, and no top-k filtering.
  - `concentrated`: temperature `0.7`, top-p `0.8`, and top-k `20`.
- Run matrix: one independent `run_id` for every dataset and sampling-strategy
  pair, producing six runs in the initial experiment group.
- Run boundary: a run fixes exactly one dataset, one resolved split/filter
  selection, and one sampling strategy. A run must never mix datasets or
  strategies.
- Rollouts: 32 rollouts per dataset example in each run, for 64 total rollouts
  per example across the two strategy runs for a dataset.
- Attention implementation: eager.
- Hardware: one RunPod H100 or A100 GPU.

The six run manifests share an `experiment_group_id`, while every run writes to
its own output directory and resumes independently. Keep maximum generated
tokens, dataset split and filter, prompt template, online versus offline scoring,
output location, and RunPod persistence behavior configurable rather than
hard-coded.

## Authoritative metric definitions

The files in `docs/` are the normative definitions:

- `docs/PROBABILITY_CONFIDENCE_METRICS.md`
- `docs/ATTENTION_METRICS.MD`
- `docs/VISUAL_HIDDEN_STATE_METRICS.MD`

Implementations must follow their predictor alignment, token grouping, numeric
precision, edge-case, and invalid-value rules exactly. If code and documentation
disagree, do not silently choose one: update the implementation or resolve the
definition explicitly.

`docs/OUTPUT_SCHEMA.md` is the normative storage contract. Its table keys,
column names, types, row-completeness rules, and commit semantics must remain
synchronized with the implementation.

`docs/REPO_SCHEMA.MD` is the normative code-architecture contract. Module
boundaries, public data types, dependency directions, orchestration flow, and
configuration resolution must remain synchronized with the implementation.

## Architectural principles

- Do not recreate the previous single-file runner.
- Keep generation, model instrumentation, metrics, dataset adapters, and storage
  independent.
- Metric functions must be pure and must not depend on model loading, datasets,
  CLI state, or the filesystem.
- Dataset adapters must not implement generation or metric calculations.
- Storage code must not know Qwen internals.
- Process attention and hidden states online and discard large tensors promptly.
- Use a custom cached decoding loop for precise predictor alignment and raw-logit
  handling. Reuse Qwen's `prepare_inputs_for_generation()` where needed for cache
  positions, position IDs, and multimodal compatibility.
- Preserve the old runner's proven layer-patching approach, but isolate it in a
  restorable instrumentation context manager.

## Intended repository structure

~~~text
VisConf/
|-- pyproject.toml
|-- README.md
|-- AGENTS.MD
|-- configs/
|   |-- experiment_group.yaml
|   |-- experiment_group_4090.yaml
|   |-- experiment_group_4090_full_mb32.yaml
|   |-- model/
|   |   `-- qwen2_5_vl_3b.yaml
|   |-- datasets/
|   |   |-- mathverse.yaml
|   |   |-- mathvista.yaml
|   |   `-- mmmu_pro.yaml
|   |-- sampling/
|   |   |-- diverse.yaml
|   |   `-- concentrated.yaml
|   `-- hardware/
|       |-- a100.yaml
|       |-- a100_40gb.yaml
|       |-- a100_80gb.yaml
|       |-- rtx_4090.yaml
|       |-- rtx_4090_mb32.yaml
|       `-- h100.yaml
|-- docs/
|   |-- ATTENTION_METRICS.MD
|   |-- LEGACY_COMPARISON.md
|   |-- PRODUCTION_RUNBOOK.md
|   |-- PROBABILITY_CONFIDENCE_METRICS.md
|   |-- VISUAL_HIDDEN_STATE_METRICS.MD
|   |-- REPO_SCHEMA.MD
|   `-- OUTPUT_SCHEMA.md
|-- environment/
|   `-- runpod-qwen3vl-metrics.lock
|-- src/visconf/
|   |-- benchmark.py
|   |-- cli.py
|   |-- config.py
|   |-- planning.py
|   |-- production.py
|   |-- runner.py
|   |-- types.py
|   |-- generation/
|   |   |-- engine.py
|   |   |-- sampling.py
|   |   |-- rollout_state.py
|   |   `-- stopping.py
|   |-- models/
|   |   |-- qwen25vl.py
|   |   |-- instrumentation.py
|   |   `-- token_positions.py
|   |-- metrics/
|   |   |-- probability.py
|   |   |-- attention.py
|   |   |-- hidden_state.py
|   |   |-- accumulator.py
|   |   `-- validation.py
|   |-- datasets/
|   |   |-- base.py
|   |   |-- mathverse.py
|   |   |-- mathvista.py
|   |   `-- mmmu_pro.py
|   |-- scoring/
|   |   |-- engine.py
|   |   |-- answer_normalization.py
|   |   `-- identity.py
|   |-- storage/
|   |   |-- schema.py
|   |   |-- parquet_writer.py
|   |   |-- transaction.py
|   |   |-- manifest.py
|   |   |-- resume.py
|   |   `-- persistence.py
|   `-- utils/
|       |-- seeds.py
|       `-- logging.py
|-- tests/
|   |-- test_phase1.py
|   |-- test_phase2.py
|   |-- test_phase3.py
|   |-- test_phase4.py
|   |-- test_phase5.py
|   |-- test_phase6.py
|   |-- test_phase7.py
|   |-- test_phase8.py
|   |-- test_phase9.py
|   |-- test_phase10.py
|   `-- test_phase11.py
`-- scripts/
    |-- benchmark_hardware.sh
    |-- run_efficiency_benchmark.sh
    |-- run_experiment_group.sh
    |-- run_full_output_benchmark.py
    |-- run_real_generation_smoke.py
    |-- run_resume_rehearsal.sh
    |-- run_single_run.sh
    |-- run_smoke_test.sh
    `-- verify_persistent_storage.sh
~~~

The tree records the implemented production-ready organization. Add new modules
only when they preserve the dependency rules below.

## Module responsibilities

### Runner and configuration

- `planning.py` expands the configured dataset-by-strategy matrix into immutable
  run specifications. The initial group resolves to six run specifications.
- Every resolved run specification contains exactly one dataset/split/filter and
  one sampling strategy, and receives its own `run_id`.
- `runner.py` executes one resolved run specification. It coordinates examples,
  rollout microbatches, resumability, failure handling, and output writing, but
  must not switch datasets or strategies within a run.
- `config.py` owns typed and validated group configuration and resolved run
  configuration. Avoid scattered accesses to raw dictionaries.
- `benchmark.py` measures target-GPU microbatch candidates without changing
  rollout identities or sampling semantics.
- `production.py` validates frozen revisions, schemas, checkpoints,
  benchmark evidence, environment identity, and persistent-storage evidence.
- `storage/persistence.py` owns persistent-output marker verification.
- `utils/logging.py` emits structured phase-boundary events.
- `cli.py` delegates operations and contains no experiment logic.

### Generation

- `generation/engine.py` implements the predictor-aligned cached decoding loop.
- `generation/sampling.py` applies temperature, top-k, and top-p only for token
  selection. Use the fixed order temperature, top-k, then top-p.
- Probability metrics must receive raw logits, never sampling-transformed logits.
  The engine may prepare a short-lived immutable raw-logit workspace before
  sampling, then complete selected-token metrics from it after the token is
  retained.
- A prepared sampling path may reuse the raw distribution's descending
  vocabulary ordering because positive-temperature scaling preserves order. It
  must still construct dense probabilities in original vocabulary order and
  call `torch.multinomial` once per rollout generator. Preserve top-k tie
  behavior and do not sample directly from sorted probabilities.
- `top_k: null` means no top-k filtering. Do not translate it to an arbitrarily
  large top-k value.
- Unless configured otherwise in a future experiment, repetition penalty is
  exactly `1.0`.
- `rollout_state.py` tracks generated tokens, completion state, per-rollout RNG,
  and metadata without owning model tensors unnecessarily.
- `stopping.py` handles EOS, `<|im_end|>`, and maximum-token termination. Selected
  EOS and `<|im_end|>` rows are trimmed and not written.

### Model integration and instrumentation

- `models/qwen25vl.py` owns model/processor loading, multimodal input preparation,
  KV-cache-compatible inputs, decoder-layer discovery, and Qwen-specific details.
- `models/token_positions.py` constructs `I`, `P`, and `G_t` exactly as specified
  in the attention documentation.
- `models/instrumentation.py` must be a context manager that patches and restores
  decoder layers safely, including restoration after exceptions.
- Do not retain all layer attentions or hidden states through ordinary model
  outputs. Reduce each layer online and release its large tensors immediately.

For every decoder layer, instrumentation should:

1. Obtain post-softmax eager self-attention.
2. Select only the predictor query row `q_t`.
3. Convert selected attention to float32.
4. Accumulate head/layer sums for the three attention scenarios in batched
   device tensors.
5. Capture the layer output at `q_t`.
6. Compute the layer hidden-state/image-prototype cosine for the full
   microbatch and defer its device-to-host conversion until the forward ends.
7. Discard the layer attention tensor.

The image-token hidden states and layer prototypes are captured once on the
initial prompt pass, retained in float32, and reused at every decoding step.

### Metrics

- `metrics/probability.py`, `attention.py`, and `hidden_state.py` implement the
  corresponding documentation as pure functions.
- `metrics/probability.py` owns the short-lived validated batch workspace used
  to share one full-vocabulary ordering between sampling and raw-logit metrics.
  The workspace must not cache sampling-transformed values or survive a
  generation step.
- `accumulator.py` handles scenario-level online sums and valid-layer counts.
- `validation.py` enforces ranges, invariants, and invalid-input policies.
- Never implement KL support reduction.
- Use float32 or higher precision for metric calculations and stable log-domain
  expressions where available.
- Preserve mathematically defined infinity. Do not silently convert it to `None`.

### Dataset adapters

All datasets use a shared interface similar to:

~~~python
class DatasetAdapter(Protocol):
    def load_examples(self, config) -> Iterable[Example]: ...
    def build_messages(self, example: Example) -> list[dict]: ...
    def ground_truth(self, example: Example) -> dict: ...
    def score(self, example: Example, response: str) -> dict | None: ...
~~~

The shared `Example` representation must support multiple images so MMMU-Pro
does not require a later structural redesign. Adapters own loading, split and
filter selection, prompt construction, image loading, target metadata, and
optional response scoring.

### Storage

Use Parquet for primary experiment data rather than deeply nested JSONL.

The intended output layout is:

~~~text
outputs/<experiment_group_id>/
|-- experiment_manifest.json
`-- <run_id>/
    |-- manifest.json
    |-- examples.parquet
    |-- generations/
    |   `-- part-*.parquet
    |-- tokens/
    |   `-- part-*.parquet
    |-- token_probability_metrics/
    |   `-- part-*.parquet
    |-- token_attention_metrics/
    |   `-- part-*.parquet
    |-- token_hidden_state_metrics/
    |   `-- part-*.parquet
    |-- scores/
    |   `-- part-*.parquet
    |-- failures.jsonl
    `-- checkpoints/
        `-- shard-*.json
~~~

- An experiment group contains one independent run for each configured
  dataset-by-strategy pair. The initial group contains six runs.
- `experiment_manifest.json` records the expected run matrix, shared
  configuration identity, run IDs, and per-run status.
- Each run directory contains exactly one dataset/split/filter and one sampling
  strategy. Every row in that directory must match those values in its run
  manifest.
- `examples` contains one row per dataset example.
- `generations` contains one row per rollout, including generated token IDs and
  decoded text.
- `tokens` is the canonical token table and contains one row per retained
  generated token. Token steps are one-based so step 1 corresponds to `y_1` in
  the metric documentation.
- `token_probability_metrics`, `token_attention_metrics`, and
  `token_hidden_state_metrics` each contain exactly one row for every row in
  `tokens`, keyed by `(run_id, dataset, split, sample_id, strategy,
  rollout_index, step)`.
- A family-metric row must exist even when every metric in that row is undefined
  and stored as null. A missing row means incomplete or corrupt output, not an
  undefined metric.
- Family-specific validity and audit fields stay with their metric family.
  Common token identity, text, predictor position, and context length stay only
  in `tokens`.
- `scores` is separate from immutable generations and contains one row per
  rollout and scorer version. Online and offline scoring write the same schema,
  allowing rollouts to be rescored without rewriting generation data.
- Metric columns use their documented names within their family table. Attention
  columns use `<scenario>__<metric>`; storage-safe parameter suffixes replace a
  decimal point with `p`, such as `renyi_entropy_0p5`.
- Store metric values as nullable float32 unless
  `docs/OUTPUT_SCHEMA.md` assigns a more specific type. Parquet null represents
  an undefined value. Mathematically defined infinity remains IEEE infinity and
  must not be converted to null.
- Write immutable Parquet part files using a temporary file followed by atomic
  rename. Do not repeatedly append to one large Parquet file.
- A shard commit covers its generation, token, probability, attention, and
  hidden-state parts together. Validate key equality and row counts before
  atomically publishing the checkpoint, which is the final commit marker.
- Readers and resume logic must consume only checkpoint-committed shards.
- Resume identity is `(dataset, split, sample_id, strategy, rollout_index)`.
- Record failures separately without marking failed rollouts complete.

The experiment manifest must capture the `experiment_group_id`, shared model
configuration, expected dataset-by-strategy matrix, and all six run IDs. Each run
manifest must capture its parent `experiment_group_id`, exactly one resolved
dataset revision/split/filter, exactly one sampling strategy, model and tokenizer
revision, Git commit, Python/PyTorch/Transformers/CUDA versions, GPU and dtype,
seed derivation version, and metric-document hashes. It must also capture output
and metric schema versions, exact layer ranges, metric-column inventory,
stop-token IDs, prompt-template hashes, scorer versions, Parquet settings, and
the committed-shard inventory.

## Predictor-aligned decoding contract

For each retained generated token `y_t`, one output row combines values from the
same predictive computation:

~~~text
probability metrics: raw distribution predicting y_t
attention metrics:   q_t attention used while predicting y_t
hidden metrics:      q_t layer outputs used while predicting y_t
~~~

The query position is:

~~~text
q_1 = final unmasked prompt position
q_t = y_{t-1}, t > 1
~~~

The loop is:

1. Run the prompt to obtain the raw distribution and predictor state for `y_1`.
2. Validate the raw-logit batch and prepare its reusable vocabulary ordering.
3. Apply the strategy only to a copy of the raw logits.
4. Sample `y_1` with the rollout's independent generator.
5. If it is retained, compute raw-logit probability metrics from the prepared
   workspace and write the aligned probability, attention, and hidden rows.
6. Feed `y_1` through the KV cache to obtain the predictive computation for
   `y_2`.
7. Continue until termination.

Do not shift attention or hidden-state metrics to the state obtained after
processing the token named by the same row.

## GPU and batching policy

- Run one dataset example at a time; do not initially batch unrelated examples.
- `rollouts_per_example` and `rollout_microbatch_size` are different settings.
- Start conservatively and benchmark with real visual sequence lengths:
  - RTX 4090 24 GB: use the conservative profile initially; the dedicated
    `rtx_4090_mb32` profile is validated for the current one-example-at-a-time,
    32-rollout workload.
  - A100 40 GB: microbatch 4-8.
  - A100 80 GB: microbatch 8-16.
  - H100 80 GB: microbatch 16, then test 32.
- Use BF16 model weights, float32 metric reductions, `torch.inference_mode()`,
  eager attention, and one explicit CUDA device.
- Do not use quantization or `torch.compile` in the initial implementation.
- Avoid automatic CPU offload for a model that fits on the selected GPU.
- Because initial eager prompt attention is quadratic, do not assume that a 3B
  model implies all 32 rollouts can safely share one prompt batch.
- Keep final scenario attention metrics CPU-batched after one device-to-host
  transfer. Moving those reductions fully onto the RTX 4090 regressed measured
  decode time and was reverted; require new A/B evidence before changing this
  boundary.

## Determinism

Each rollout receives a stable seed derived from:

~~~text
base_seed, dataset, sample_id, strategy, rollout_index
~~~

Use an independent `torch.Generator` per rollout. Results must not change merely
because rollout microbatch size changes.

## Testing requirements

At minimum, add tests for:

1. Predictor alignment across logits, attention, hidden state, and selected token.
2. Hand-computed probability metrics, including KL without support reduction.
3. Attention ranges, group behavior, and ratio identities.
4. Qwen special-token, image-span, padding, and `G_t` position grouping.
5. Hidden-state norm thresholds and valid-layer aggregation.
6. Sampling transformations and proof that they do not alter raw-logit metrics.
   Include exact prepared-versus-public sampling comparisons for both production
   strategies, generator states, tied logits, top-k, top-p edges, and `-inf`.
7. Reproducibility across rollout microbatch sizes.
8. One-to-one key equality across `tokens` and all three metric-family tables.
9. Atomic multi-table shard output, orphan-part handling, and resume behavior.
10. Online and offline scoring producing the same versioned score rows.
11. A minimal real-Qwen smoke test before large runs.

Before production, run one example with two short rollouts and inspect every
stored token row manually or through invariant checks.

## Implementation order

1. Scaffold the package, test harness, typed domain records, configuration, and
   six-run experiment-group planner.
2. Implement pure probability, attention, and hidden metrics plus sampling,
   stable seeds, and stopping, all with unit tests.
3. Implement Arrow schemas, normalized Parquet writers, group and run manifests,
   checkpoint-last transactions, failures, and resume using synthetic records.
4. Implement the dataset protocol, MathVerse adapter, prompt construction, and
   tokenizer-based `I`, `P`, and `G_t` grouping tests.
5. Implement the Qwen facade, batch-size-one prompt prefill, restorable 36-layer
   instrumentation, image prototypes, and online metric accumulation.
6. Implement cached predictor-aligned generation, cache replication, active-row
   compaction, independent rollout RNG, and completed-rollout bundles.
7. Integrate one-run execution, CLI commands, group orchestration, and core
   transactional storage; prove the path first with deterministic test doubles.
8. Migrate answer normalization and add versioned offline and online scoring.
9. Pass the two-rollout real-Qwen MathVerse smoke test, resume/OOM fault tests,
   schema validation, and understood legacy-overlap comparison.
10. Add and validate MathVista and MMMU-Pro adapters and scorers without changing
    shared types or schemas.
11. Benchmark A100/H100 microbatch sizes, freeze production configurations, run a
    final interrupted-resume rehearsal, and only then start production.
