# VisConf Repository Schema

This document is the normative code-architecture contract for VisConf. It
defines repository structure, module boundaries, public data types, configuration
resolution, execution flow, dependency direction, and implementation gates.

Metric mathematics is normative in:

- `PROBABILITY_CONFIDENCE_METRICS.md`
- `ATTENTION_METRICS.MD`
- `VISUAL_HIDDEN_STATE_METRICS.MD`

Physical output is normative in `OUTPUT_SCHEMA.md`. If this document conflicts
with a metric document or the output schema, the more specialized document wins
and this repository schema must be updated.

## Locked experiment model

The initial experiment group contains the Cartesian product of:

~~~text
datasets   = [mathverse, mathvista, mmmu_pro]
strategies = [diverse, concentrated]
~~~

Default dataset selections are:

| Dataset | Source selection | Filters |
| --- | --- | --- |
| MathVerse | `testmini` | `problem_version: Vision Intensive` |
| MathVista | `testmini` | all rows |
| MMMU-Pro | standard four-option test source | all rows |


This produces six independent runs. Each `run_id` fixes exactly one dataset,
one resolved split/filter selection, and one sampling strategy.

Each run performs 32 rollouts per selected example. Across the two strategy runs,
each dataset example receives 64 rollouts. All six runs share one
`experiment_group_id`, but each has independent output, checkpoints, failures,
and resume state.

Sampling is fixed initially as:

| Strategy | Temperature | Top-p | Top-k | Repetition penalty |
| --- | ---: | ---: | ---: | ---: |
| `diverse` | 1.0 | 0.99 | null | 1.0 |
| `concentrated` | 0.7 | 0.8 | 20 | 1.0 |

The model is Qwen2.5-VL-3B with 36 decoder layers. Attention uses the eager
implementation. Model weights use BF16 and metric reductions use float32 or
higher precision.

Layer ranges are:

| Aggregate | Layers |
| --- | --- |
| All layers | 1–36 |
| Early visual integration | 1–21 |
| Visual reasoning | 22–34 |
| Last layer | 36 |

Layers 35–36 belong to the all-layer aggregate. Layer 36 also supplies the
last-layer hidden metric. Layers 35–36 do not belong to either fixed phase
range.

## Goals

- Generate predictor-aligned multimodal rollouts.
- Collect all normative probability, attention, and hidden-state metrics for
  every retained generated token.
- Preserve raw model logits separately from sampling transformations.
- Make results invariant to rollout microbatch size.
- Keep model integration, pure metrics, datasets, scoring, and storage isolated.
- Support atomic multi-table output and reliable resume.
- Support online or offline versioned scoring without rewriting generation data.
- Run on one explicit A100 or H100 CUDA device without CPU offload.

## Non-goals for the initial implementation

- Training or fine-tuning.
- Quantization, `torch.compile`, distributed inference, or multi-GPU sharding.
- Automatic device maps or CPU offload.
- Cross-example batching.
- General support for arbitrary vision-language model families.
- Persisting per-head or per-layer attention tensors.
- Persisting individual per-layer hidden cosine values.
- Reproducing every legacy metric from the previous single-file runner.
- Maintaining a second authoritative wide token-metrics table.

## Dependency direction

Dependencies flow downward:

~~~text
cli
 |
 +--> config --> planning
 |
 +--> runner
       |
       +--> datasets
       +--> models --> generation
       |               |
       |               +--> metrics
       |
       +--> scoring
       +--> storage

types and small utils may be imported by every layer
~~~

The following rules are mandatory:

- `metrics` imports no dataset, model-loading, storage, CLI, or filesystem code.
- `models` imports no dataset adapter or storage writer.
- `datasets` imports no generation engine, instrumentation, or metric function.
- `storage` imports no Transformers or Qwen implementation.
- `generation` does not load datasets or write files.
- `runner` orchestrates components but implements no metric formula.
- `cli` parses commands and delegates; it implements no experiment logic.
- Public cross-module values use types from `visconf.types` or explicitly
  documented protocols, not anonymous dictionaries.
- No module may depend on the legacy runner at runtime.

## Target repository tree

Create files incrementally; this is the final target, not a request for empty
placeholders.

~~~text
VisConf/
|-- pyproject.toml
|-- README.md
|-- AGENTS.MD
|-- configs/
|   |-- experiment_group.yaml
|   |-- experiment_group_4090.yaml
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
|       `-- h100.yaml
|-- docs/
|   |-- ATTENTION_METRICS.MD
|   |-- LEGACY_COMPARISON.md
|   |-- PRODUCTION_RUNBOOK.md
|   |-- PROBABILITY_CONFIDENCE_METRICS.md
|   |-- VISUAL_HIDDEN_STATE_METRICS.MD
|   |-- OUTPUT_SCHEMA.md
|   `-- REPO_SCHEMA.MD
|-- environment/
|   `-- runpod-qwen3vl-metrics.lock
|-- src/visconf/
|   |-- __init__.py
|   |-- benchmark.py
|   |-- cli.py
|   |-- config.py
|   |-- planning.py
|   |-- production.py
|   |-- runner.py
|   |-- types.py
|   |-- generation/
|   |   |-- __init__.py
|   |   |-- engine.py
|   |   |-- sampling.py
|   |   |-- rollout_state.py
|   |   `-- stopping.py
|   |-- models/
|   |   |-- __init__.py
|   |   |-- qwen25vl.py
|   |   |-- instrumentation.py
|   |   `-- token_positions.py
|   |-- metrics/
|   |   |-- __init__.py
|   |   |-- probability.py
|   |   |-- attention.py
|   |   |-- hidden_state.py
|   |   |-- accumulator.py
|   |   `-- validation.py
|   |-- datasets/
|   |   |-- __init__.py
|   |   |-- base.py
|   |   |-- mathverse.py
|   |   |-- mathvista.py
|   |   `-- mmmu_pro.py
|   |-- scoring/
|   |   |-- __init__.py
|   |   |-- engine.py
|   |   |-- answer_normalization.py
|   |   `-- identity.py
|   |-- storage/
|   |   |-- __init__.py
|   |   |-- schema.py
|   |   |-- parquet_writer.py
|   |   |-- transaction.py
|   |   |-- manifest.py
|   |   |-- resume.py
|   |   `-- persistence.py
|   `-- utils/
|       |-- __init__.py
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
    |-- run_experiment_group.sh
    |-- run_real_generation_smoke.py
    |-- run_resume_rehearsal.sh
    |-- run_single_run.sh
    |-- run_smoke_test.sh
    `-- verify_persistent_storage.sh
~~~

## Configuration model

Configuration has two levels.

### Experiment-group configuration

The group configuration selects shared settings and lists dataset and strategy
configurations to expand:

~~~yaml
experiment_group:
  id: null
  output_root: outputs
  base_seed: 42

model: qwen2_5_vl_3b
hardware: h100

datasets:
  - mathverse
  - mathvista
  - mmmu_pro

strategies:
  - diverse
  - concentrated

generation:
  rollouts_per_example: 32
  rollout_microbatch_size: 16
  max_new_tokens: 1024

scoring:
  mode: offline

storage:
  compression: zstd
  compression_level: 3
  target_row_group_size: 65536
~~~

The group configuration is not passed directly to the runner.

### Resolved run configuration

`planning.py` expands the group matrix and produces one immutable
`ResolvedRunConfig` per dataset-by-strategy cell. It contains no lists of
datasets or strategies.

A resolved run configuration includes:

- `experiment_group_id` and `run_id`.
- Exactly one resolved dataset adapter configuration, revision, split, and
  filter.
- Exactly one sampling strategy and its numeric parameters.
- Model, tokenizer, prompt, generation, hardware, scoring, storage, and seed
  settings.
- Output and metric schema versions.
- Hashes of the resolved configuration and normative documents.

The runner accepts only a `ResolvedRunConfig`. It must reject a configuration
containing multiple datasets or strategies.

### Configuration resolution rules

- Includes or named config fragments are fully resolved before hashing.
- Paths are expanded and normalized, but the manifest retains both configured
  and resolved forms where useful.
- `top_k: null` remains null and means no top-k filtering.
- Temperature must be positive.
- Top-p must be in `(0, 1]`.
- Top-k, when present, must be positive.
- Repetition penalty must equal 1.0 for the initial experiment.
- Rollouts per example and microbatch size must be positive.
- Microbatch size may exceed neither rollouts per example nor configured hardware
  limits without an explicit override.
- Model configuration must report 36 decoder layers.
- Attention implementation must be `eager`.
- Automatic CPU offload and quantization must be disabled.
- The resolved run's dataset and strategy must match its cell in the experiment
  manifest.

### Identity creation

If the user does not supply an `experiment_group_id`, planning creates one from
a UTC timestamp plus a short group-configuration hash. Each run receives a unique
opaque `run_id` and a human-readable label containing dataset and strategy.

Identity values are recorded, not reconstructed. Resuming loads IDs from existing
manifests; it never regenerates them from the current clock or reparses labels.


## Public domain types

`src/visconf/types.py` owns stable, model-independent domain types. Dataclasses
are frozen unless they represent intentionally mutable rollout state.

### Run and sampling identities

~~~python
@dataclass(frozen=True)
class RunCell:
    dataset: str
    split: str
    filter_id: str
    strategy: str

@dataclass(frozen=True)
class RolloutKey:
    run_id: str
    dataset: str
    split: str
    sample_id: str
    strategy: str
    rollout_index: int

@dataclass(frozen=True)
class TokenKey:
    rollout: RolloutKey
    step: int

@dataclass(frozen=True)
class SamplingConfig:
    name: str
    temperature: float
    top_p: float
    top_k: int | None
    repetition_penalty: float
~~~

`RunCell` equality is semantic and used to prove the six-cell experiment matrix
has no duplicates. `RunCell` is not a substitute for `run_id`.

### Dataset examples

~~~python
@dataclass(frozen=True)
class ExampleImage:
    source_ref: str | None
    image: PIL.Image.Image
    sha256: str
    width: int
    height: int
    mode: str

@dataclass(frozen=True)
class Example:
    dataset: str
    split: str
    sample_id: str
    source_row_index: int | None
    question: str
    images: tuple[ExampleImage, ...]
    ground_truth: Mapping[str, JsonValue]
    answer_type: str | None
    metadata: Mapping[str, JsonValue]
~~~

Images are ordered. An adapter must never collapse multiple images into one field.
`sample_id` must be stable across resume and must not depend on iteration order.

### Prompt and token grouping

~~~python
@dataclass(frozen=True)
class PromptRecord:
    rendered_prompt: str
    prompt_token_ids: tuple[int, ...]
    prompt_token_count: int

@dataclass(frozen=True)
class TokenGroups:
    image_positions: torch.LongTensor
    prompt_text_positions: torch.LongTensor
    prompt_last_position: int
    prompt_token_count: int
~~~

`TokenGroups` stores prompt-static `I` and `P`. The generation engine derives
`G_t` from retained token count at each step. Position tensors use logical
unpadded prompt coordinates.

### Output records

The following records mirror `OUTPUT_SCHEMA.md` and are converted to Arrow rows
only in storage code:

~~~python
@dataclass(frozen=True)
class GenerationRecord: ...

@dataclass(frozen=True)
class TokenRecord: ...

@dataclass(frozen=True)
class ProbabilityMetricRecord: ...

@dataclass(frozen=True)
class AttentionMetricRecord: ...

@dataclass(frozen=True)
class HiddenStateMetricRecord: ...

@dataclass(frozen=True)
class ScoreRecord: ...

@dataclass(frozen=True)
class FailureRecord: ...
~~~

Each record constructor validates its required key fields and types. Metric
record constructors do not silently replace infinity or NaN. Family validation
decides whether a value is a defined infinity or an invalid value before record
construction.

### Completed rollout bundle

~~~python
@dataclass(frozen=True)
class CompletedRollout:
    generation: GenerationRecord
    tokens: tuple[TokenRecord, ...]
    probability: tuple[ProbabilityMetricRecord, ...]
    attention: tuple[AttentionMetricRecord, ...]
    hidden_state: tuple[HiddenStateMetricRecord, ...]
~~~

Construction validates equal token-family lengths and identical token-key order.
This is the unit passed from generation to transactional storage.

## Mutable rollout state

`generation/rollout_state.py` owns the only mutable per-rollout decoding state:

~~~python
@dataclass
class RolloutState:
    key: RolloutKey
    generator: torch.Generator
    generated_token_ids: list[int]
    next_step: int
    active: bool
    stop_reason: StopReason | None
    terminating_token_id: int | None
~~~

It does not own model-wide prompt tensors, image prototypes, all-layer attention
tensors, or filesystem handles. RNG state belongs to its independent
`torch.Generator`.

A microbatch structure maps active tensor rows to `RolloutState` objects.
Whenever rows finish, cache selection and state ordering must be updated together.

## Dataset adapter contract

`datasets/base.py` defines:

~~~python
class DatasetAdapter(Protocol):
    name: str

    def load_examples(
        self,
        config: ResolvedDatasetConfig,
    ) -> Iterable[Example]: ...

    def build_messages(
        self,
        example: Example,
        prompt_config: PromptConfig,
    ) -> list[dict[str, object]]: ...

    def ground_truth(
        self,
        example: Example,
    ) -> Mapping[str, JsonValue]: ...

    def score(
        self,
        example: Example,
        response: str,
        scorer_config: ScorerConfig,
    ) -> ScoreResult | None: ...
~~~

Adapter responsibilities:

- Resolve dataset source, revision, split, and filters.
- Produce stable sample IDs.
- Load and normalize all ordered images to RGB where appropriate.
- Build messages without invoking the processor.
- Expose ground truth and dataset metadata.
- Implement or select dataset-appropriate scoring.

Adapters must not:

- Load the model.
- Tokenize through Qwen-specific code.
- Sample tokens.
- Compute confidence metrics.
- Write output files.

The repository implements MathVerse, MathVista, and MMMU-Pro through this shared
adapter contract. Their configured selections remain independently validated.

## Model facade contract

`models/qwen25vl.py` is the only module that knows Qwen2.5-VL processor and model
details. It owns:

- Loading `Qwen2_5_VLForConditionalGeneration` and `AutoProcessor`.
- Enforcing BF16, eager attention, evaluation mode, one explicit CUDA device, and
  no automatic offload.
- Applying the chat template and `qwen_vl_utils.process_vision_info`.
- Producing prompt tensors and a serializable `PromptRecord`.
- Calling and validating `prepare_inputs_for_generation()`.
- Managing Qwen multimodal position IDs, cache positions, rope deltas, and image
  inputs.
- Locating all 36 language decoder layers.
- Cloning and batch-repeating a batch-size-one prompt KV cache.
- Converting model output into raw next-token logits and cache state.
- Discovering tokenizer IDs and token pieces.

A narrow facade is exposed to generation:

~~~python
class QwenModelFacade:
    model: torch.nn.Module
    processor: AutoProcessor

    def prepare_example(...) -> PreparedQwenPrompt: ...
    def prefill(...) -> PrefillResult: ...
    def repeat_cache(...) -> Cache: ...
    def decode_step(...) -> DecodeResult: ...
    def decode_retained(...) -> str: ...
~~~

Generation must not inspect Qwen module paths, rope deltas, or processor internals
directly.

### Prepared prompt ownership

`PreparedQwenPrompt` contains device tensors required for the initial prefill,
the serializable prompt record, and token groups. Pixel values are released after
prefill. The reusable base KV cache and float32 image prototypes remain live only
while processing the current example.

## Token-position contract

`models/token_positions.py` constructs prompt-static groups exactly as defined
by the attention document:

- `I`: only unmasked `<|image_pad|>` positions.
- `P`: eligible unmasked prompt text excluding all tokenizer special IDs and all
  image-span positions.
- Image span delimiters and non-pad image-span positions belong to neither group.
- Padding belongs to no group.
- The first query is the final unmasked prompt position.
- `G_t` contains retained generated positions `1..t-1`, including the current
  self-attention key for `t > 1`.

The module validates image spans, special-token IDs, padding, multiple images,
and prompt-last position. It returns positions but computes no attention metric.

## Instrumentation contract

`models/instrumentation.py` provides a restorable context manager:

~~~python
with QwenInstrumentation(model, layer_ranges) as instrumentation:
    ...
~~~

Entering discovers and patches exactly 36 decoder layers. Exiting restores every
original forward method, including when generation raises.

For each layer call, instrumentation:

1. Forces eager post-softmax attention output locally.
2. Selects only the predictor query row.
3. Converts that row to float32.
4. Validates it before accumulation.
5. Adds head sums to every applicable attention scenario.
6. Captures the layer output at the predictor position.
7. Computes the normalized cosine against that layer's float32 image prototype.
8. Updates valid-layer sums and counts.
9. Drops references to full attention and temporary hidden tensors.

On the batch-size-one prompt prefill it additionally captures each layer's image
token states, mean-pools the image prototype in float32, and retains only the
prototype after reduction.

Instrumentation returns reduced observations through an explicit per-step
collector. It does not write records, sample tokens, or own rollout RNG.

### Instrumentation state

Instrumentation state is scoped to one prepared example and one active forward
pass. It is never module-global. A collector must be reset before every prefill
or decode step and must verify that all 36 expected layer calls occurred.

## Pure metric contracts

All metric functions run under `torch.inference_mode()` and accept tensors or
plain values only.

### Probability

~~~python
def compute_probability_metrics(
    raw_logits: torch.Tensor,
    selected_token_id: int,
) -> ProbabilityMetrics: ...
~~~

The function receives a one-dimensional raw-logit vector. Sampling-transformed
logits are a contract violation. It uses the full vocabulary and implements all
31 physical columns specified by `OUTPUT_SCHEMA.md`.

### Attention

~~~python
def compute_attention_metrics(
    scenario_vector: torch.Tensor,
    groups: StepTokenGroups,
) -> AttentionScenarioMetrics: ...
~~~

The scenario vector is already averaged across selected layer-head pairs.
Validation and group metric rules follow `ATTENTION_METRICS.MD`. The function
does not know layer modules or heads.

### Hidden state

~~~python
def aggregate_hidden_metrics(
    per_layer_cosines: Sequence[float | None],
) -> HiddenStateMetrics: ...
~~~

The sequence is one-based conceptually and has 36 entries. Aggregation uses valid
values only and applies fixed ranges 1–36, 1–21, 22–34, and layer 36.

### Validation

`metrics/validation.py` owns shared thresholds, finite checks, range checks, and
stable invalid-reason codes. It does not redefine metric formulas.

### Accumulation

`metrics/accumulator.py` owns online float32 scenario sums, per-layer cosine
sums, and valid-layer counts. It never stores full layer attention across calls.


## Sampling contract

`generation/sampling.py` exposes:

~~~python
def sample_next_token(
    raw_logits: torch.Tensor,
    config: SamplingConfig,
    generator: torch.Generator,
) -> int: ...
~~~

The function clones or derives selection logits without modifying
`raw_logits`. Transformations are applied in this fixed order:

1. Divide by temperature.
2. Apply top-k if `top_k is not None`.
3. Apply top-p.
4. Softmax and sample with the rollout's generator.

Top-p keeps the first token that reaches or crosses the cumulative threshold.
`top_k=None` applies no top-k filter. Repetition penalty is not applied because
its configured value is exactly 1.0.

Sampling is performed once per rollout row with that rollout's independent
generator. A batched `torch.multinomial` call using one shared generator is
forbidden because results would depend on active-row order or microbatch size.

## Stopping contract

`generation/stopping.py` receives selected token IDs and returns a typed stop
decision.

The stop-token ID set contains tokenizer EOS and `<|im_end|>` IDs after
deduplication. In the current checkpoint both names resolve to ID 151645.

- A selected stop token is not retained and receives no token or metric row.
- Its ID is stored as `terminating_token_id`.
- A retained token selected at the final allowed step is stored normally, then
  the rollout stops with `max_new_tokens`.
- An immediate stop produces a valid zero-token generation.
- Padding is never sampled as a synthetic replacement for a finished row.
  Finished rows are removed from the active microbatch.

## Generation engine contract

`generation/engine.py` executes rollouts for one prepared example and one
sampling strategy:

~~~python
class GenerationEngine:
    def generate_example(
        self,
        example: Example,
        prepared: PreparedQwenPrompt,
        rollout_keys: Sequence[RolloutKey],
        sampling: SamplingConfig,
    ) -> Iterable[CompletedRollout]: ...
~~~

All rollout keys must share run ID, dataset, split, sample ID, and strategy. Only
rollout index and seed may vary.

### Prompt prefill

For each example and run:

1. Reset instrumentation for a batch-size-one prompt pass.
2. Run the full multimodal prompt once with cache enabled.
3. Capture raw logits predicting `y_1`.
4. Capture predictor-aligned attention and hidden observations for `q_1`.
5. Capture all 36 float32 image prototypes.
6. Retain a batch-size-one base KV cache.
7. Release pixel values, full prompt attention tensors, and image-token hidden
   matrices.

The base prompt cache and prototypes live only until all 32 rollouts for that
example are complete.

### Rollout microbatches

Pending rollout indices are processed in configured microbatches. For each
microbatch:

1. Create one independently seeded `RolloutState` per rollout.
2. Clone and batch-repeat the immutable base KV cache.
3. Replicate the `y_1` distribution-level, attention, and hidden observations
   as read-only inputs for each rollout.
4. Sample each row separately with its own generator.
5. Complete selected-token probability metrics using the sampled ID and the same
   raw logits that predicted it.
6. Trim selected stop tokens without writing token-family rows.
7. For retained tokens, create one token row and all three family rows.
8. If maximum length is reached, finish after storing the retained token.
9. Select surviving cache rows in the same order as surviving rollout states.
10. Feed retained selected tokens through the cache to obtain observations
    predicting the next token.
11. Repeat until no active rows remain.

The engine yields completed rollouts; it does not write Parquet.

### Predictor alignment

For stored step `t`:

~~~text
selected token:      y_t
raw logits:          P(. | x, y_<t)
attention query:     q_1 = final prompt position, or q_t = y_(t-1)
hidden query:        the same q_t layer outputs
generated group:     G_t = {y_1, ..., y_(t-1)}
~~~

The engine must never attach attention or hidden observations obtained after
processing `y_t` to the row for `y_t`.

### Cache handling

The batch-size-one base cache is never mutated by a rollout microbatch. Cache
replication creates a distinct mutable cache for that microbatch. In the current
Transformers runtime, cache construction may use the supported cache conversion
and batch-repeat APIs, but those details remain inside the model facade.

Active-row compaction must apply the same indices to:

- KV cache.
- Attention mask.
- Position or cache-position state.
- Qwen rope-delta state where batch-specific.
- Rollout-state ordering.
- Instrumentation collector batch rows.

A cache-copy test compares the first two decoding steps against a conventional
independent prompt run.

## Seed derivation

`utils/seeds.py` derives a stable uint64 seed from:

~~~text
base_seed
dataset
sample_id
strategy
rollout_index
seed_derivation_version
~~~

The canonical byte encoding and hash algorithm are versioned and tested. Python's
process-randomized `hash()` must not be used.

Each rollout constructs one CUDA `torch.Generator` on the model device and seeds
it once. Dataset order, worker count, microbatch size, OOM fallback, or previously
completed rollouts must not change another rollout's generated sequence.

## Runner contract

`runner.py` executes exactly one `ResolvedRunConfig`:

~~~python
class RunExecutor:
    def execute(self, config: ResolvedRunConfig) -> RunResult: ...
~~~

Execution order:

1. Validate the resolved run and its experiment-manifest cell.
2. Initialize the run output directory and manifest.
3. Load or create the resume index from committed checkpoints.
4. Instantiate exactly one dataset adapter.
5. Load the model facade once.
6. Materialize examples in stable adapter order and atomically write
   `examples.parquet` before the first core shard.
7. Enter the instrumentation context.
8. Process one example at a time.
9. Derive the 32 rollout identities and remove committed ones.
10. Generate pending rollout microbatches.
11. Commit completed rollout bundles in core shards.
12. Optionally invoke online scoring after the core shard is committed.
13. Restore instrumentation and release model resources.
14. Atomically finalize the run manifest.
15. Update the parent experiment manifest.

The runner never changes dataset adapter or sampling strategy. A group launcher
invokes the runner separately for each cell.

### Example ordering and selection

Adapters expose a stable source order before optional configured shuffling.
Shuffling uses a separate deterministic dataset-selection RNG, not a rollout
generator. The manifest records selected sample IDs in order or a hash plus a
recoverable selection artifact.

Resume skips only committed rollout identities. It does not drop an entire
example merely because some of its rollouts are complete.

### OOM policy

An OOM in a microbatch may trigger a smaller microbatch size for the same pending
rollouts after collector and CUDA state are cleared. Seeds and rollout identities
remain unchanged.

The effective microbatch size and fallback are logged. Repeated failure at size
one creates failure records and leaves those rollout identities incomplete.
Successful rollouts from other microbatches remain committed.

## Planning contract

`planning.py` exposes two operations:

~~~python
def plan_experiment_group(
    group_config: ExperimentGroupConfig,
) -> ExperimentPlan: ...

def load_resolved_run(
    experiment_group_id: str,
    run_id: str,
) -> ResolvedRunConfig: ...
~~~

Planning:

- Resolves exactly three configured dataset cells and two strategy cells for the
  initial experiment.
- Validates that the Cartesian product has six unique `RunCell` values.
- Assigns one run ID and output directory per cell.
- Writes `experiment_manifest.json` atomically.
- Stores each resolved run configuration in its run manifest or an immutable
  planning artifact referenced by that manifest.
- Is idempotent when reopening an existing experiment group.
- Refuses to silently replace an existing cell with a newly generated run ID.

Group execution may be serial or scheduled externally. Group planning owns no
model and runs no dataset examples.

## Storage module contracts

Physical schemas and transactions follow `OUTPUT_SCHEMA.md`.

### schema.py

`storage/schema.py` defines the exact PyArrow schemas and ordered metric column
inventory. Schemas are constructed programmatically from fixed scenario, group,
and metric-name constants, then compared against the normative inventory in
tests.

It exposes no model or tensor types.

### parquet_writer.py

The Parquet writer converts validated domain records to Arrow arrays and writes
one immutable temporary part. It:

- Preserves float32 infinity.
- Uses null only for `None`.
- Rejects unexpected columns, missing required columns, unsafe casts, and NaN in
  fields whose metric validation did not explicitly permit it.
- Applies configured Zstandard, dictionary, statistics, and row-group settings.
- Closes and fsyncs output before returning file metadata.

### transaction.py

A core transaction receives one or more `CompletedRollout` bundles and writes
matching generation, token, probability, attention, and hidden-state parts.

Before commit it validates:

- Unique rollout and token keys.
- Equal token-family key sets and order.
- Contiguous one-based steps.
- Generation token-list equality.
- Run-level dataset, split, and strategy constants.
- Schema and row counts.

It renames all final parts and writes the checkpoint last. The checkpoint is the
only commit marker.

### manifest.py

This module owns typed experiment and run manifests, atomic JSON replacement,
configuration hashes, document hashes, environment capture, status transitions,
and committed-shard inventory.

Allowed run status transitions are:

~~~text
planned -> running -> complete
planned -> running -> failed
failed  -> running -> complete
~~~

A resumable failure does not require changing a run to `failed` while useful
pending work remains; failures are recorded independently.

### resume.py

Resume reads checkpoint files rather than globbing Parquet. It verifies checkpoint
schema versions and part metadata, then builds the completed rollout set.

Uncheckpointed parts are never treated as complete. Complete matching orphan sets
may be recovered only after full validation; incomplete sets are quarantined or
replaced.

## Scoring contracts

`scoring/engine.py` orchestrates adapter scoring and versioned score storage.
Core generation commits do not depend on scoring success.

Default initial mode is offline:

1. Finish or partially finish generation runs.
2. Read checkpoint-committed generation rows.
3. Load the matching example and adapter.
4. Score missing scorer keys.
5. Commit immutable score shards independently.

Online mode invokes the same scorer immediately after a core shard commit and
writes the same `ScoreRecord` schema.

`scoring/answer_normalization.py` replaces the current executable-loading
pattern. It is a normal importable module with explicit imports and unit tests.
No production code uses `exec()` to load scoring code.

Scorer identity contains:

- Stable scorer name.
- Semantic scorer version.
- Configuration hash.
- Code hash in the run manifest covering answer normalization and all three
  adapter scoring implementations.

Changing extraction or comparison logic requires a new scorer version and creates
new rows; it never overwrites old scores.

## CLI contract

The initial console entry point is `visconf` with subcommands:

~~~text
visconf plan  --config configs/experiment_group.yaml
visconf run   --group <experiment_group_id> --run <run_id>
visconf group --group <experiment_group_id>
visconf score --group <experiment_group_id> [--run <run_id>]
visconf validate --group <experiment_group_id>
visconf benchmark --group <experiment_group_id> --run <run_id>
visconf verify-storage --output-root <persistent_output_root> [--initialize]
~~~

- `plan` resolves and writes the six-run plan without loading model weights.
- `run` executes or resumes one run.
- `group` invokes incomplete runs sequentially; external schedulers may instead
  call `run` separately.
- `score` performs versioned offline scoring.
- `validate` verifies manifests, checkpoints, schemas, hashes, and relational
  invariants; `--production` additionally requires a benchmark report.
- `benchmark` measures configured microbatch candidates only on matching
  target hardware.
- `verify-storage` initializes or verifies the persistent-output marker.

Commands return nonzero on configuration, validation, or unhandled execution
failure. Logging is structured and always includes group ID, run ID, dataset,
strategy, sample ID when known, and rollout index when known.


## Resource lifetime and memory policy

The runner operates on one example at a time.

### Run lifetime

Live for the run:

- Loaded model and processor.
- Instrumentation patches.
- Dataset adapter.
- Storage session and resume index.
- Resolved run configuration.

### Example lifetime

Live only for one example:

- Prepared prompt tensors.
- Batch-size-one base KV cache.
- Float32 image prototypes for 36 layers.
- Prompt-static token groups.
- Pending rollout identities.

### Forward-pass lifetime

Live only during one layer or one forward:

- Full eager attention tensor returned by a layer.
- Selected predictor attention row.
- Layer output tensor.
- Per-forward scenario accumulators.

The full attention tensor is released immediately after its selected row is
reduced. Full layer attentions and hidden states are not returned and accumulated
through ordinary model output collections.

### Microbatch lifetime

Live only for one rollout microbatch:

- Repeated mutable KV cache.
- Active rollout states and generators.
- Current raw logits.
- Reduced observations.
- Completed rollout bundles not yet committed.

After shard commit, record buffers are released. CUDA cache clearing is not used
as routine memory management; it is reserved for OOM recovery and run teardown.

## Device and precision policy

- One explicitly selected CUDA device.
- Model weights and normal model forward computation in BF16.
- Raw logits converted to float32 before probability metrics and sampling.
- Selected attention rows converted to float32 before aggregation.
- Image prototypes retained in float32.
- Hidden cosine dot products and norms computed in float32 or higher.
- Parquet metric storage in float32 except explicitly typed integers.
- `torch.inference_mode()` around all model and metric execution.
- Model in evaluation mode; attention dropout disabled.
- No `device_map="auto"`, quantization, CPU offload, or `torch.compile`.

Device assertions run after loading and before the first example.

## Concurrency policy

The initial runner is a single GPU process with one model and one storage writer.

- Do not call the patched model concurrently from multiple threads.
- Do not process unrelated examples concurrently.
- CPU dataset decoding may prefetch only if it cannot mutate adapter ordering.
- Offline scoring may run as a separate process only after reading committed
  generation shards.
- Two run IDs must never write the same run directory.
- A filesystem run lock prevents concurrent writers for one run ID.
- Read-only validation may run concurrently with a stopped or completed writer,
  but not during an active part rename unless it uses checkpoint snapshots.

External orchestration may schedule different run IDs on different machines or
at different times, provided each run still uses one GPU and its own directory.

## Error model

Use typed exceptions with stable categories:

~~~text
ConfigError
PlanningError
DatasetError
PromptError
ModelContractError
InstrumentationError
MetricInputError
GenerationError
StorageValidationError
StorageCommitError
ScoringError
~~~

A caught error becomes a `FailureRecord` with the corresponding stable failure
stage. Exception messages may include diagnostic detail, but downstream logic
uses the stable category and reason code.

### Failure scope

- Configuration or planning failure stops before a run begins.
- Model loading or instrumentation failure stops the run.
- Dataset load failure may stop the run if the selected example set cannot be
  established.
- Per-example prompt failure records the example and leaves all its rollouts
  incomplete.
- Per-rollout generation failure leaves that rollout incomplete.
- Core storage failure does not mark any rollout in the uncommitted shard
  complete.
- Scoring failure does not invalidate committed generation.
- Metric absence caused by a documented undefined case creates null values, not a
  failure.
- Missing raw logits, predictor misalignment, wrong layer count, or wrong family
  key sets are contract failures, not nullable metrics.

## Logging and observability

Use Python structured logging with human-readable console rendering. Every event
includes available identity fields:

~~~text
experiment_group_id
run_id
dataset
split
strategy
sample_id
rollout_index
shard_id
stage
~~~

Required events include:

- Group planned.
- Run started or resumed.
- Environment and GPU summary.
- Example started and completed.
- Microbatch started and completed.
- OOM fallback.
- Core shard committed.
- Score shard committed.
- Failure recorded.
- Run completed.
- Group completed.

Progress bars are optional presentation and never serve as state.

## Testing strategy

Tests are divided into pure unit, contract, integration, and real-model smoke
tests.

### Pure metric tests

Probability tests cover:

- Uniform and hand-computed distributions.
- Selected modal and non-modal tokens.
- Tied competition rank.
- Full-support KL with no support reduction.
- Stable log-domain Rényi and Jensen-Shannon formulas.
- Underflow-sensitive tails.
- Defined infinity preservation.
- All physical metric names and types.

Attention tests cover:

- All five groups and three scenarios.
- Empty groups.
- Low total mass threshold.
- Zero entries and infinite `KL(U || p)`.
- Negative clipping tolerance.
- Invalid negative and non-finite rows.
- Ratio denominator threshold.
- Ratio identities.
- No full-vector renormalization.

Hidden tests cover:

- Cosine normalization and clipping.
- Near-zero predictor and prototype norms.
- Valid-layer-only aggregation.
- Ranges 1–36, 1–21, and 22–34.
- Layer 36 last-layer behavior.
- No-image behavior.

### Sampling and determinism tests

- Temperature, top-k, then top-p order.
- Null top-k performs no filtering.
- Raw logits remain byte-for-byte unchanged.
- Independent generator sequences.
- Stable seed vectors.
- Identical rollout outputs at microbatch sizes 1, 2, and 4 for a deterministic
  mock model.
- Active-row removal does not change surviving rollout sequences.
- OOM retry with smaller microbatch does not change sequences.

### Planning tests

- Three datasets times two strategies yields exactly six unique cells.
- Every cell receives a distinct run ID.
- Every resolved run has exactly one dataset and one strategy.
- All run manifests share one experiment group ID.
- Reopening a plan preserves IDs.
- Duplicate or missing cells fail validation.
- A runner rejects an unresolved group configuration.

### Token-position tests

Use the actual local tokenizer to cover:

- Special IDs.
- One and multiple image spans.
- `<|image_pad|>` selection.
- Excluded vision delimiters and image newlines.
- Left and right padding.
- Final unmasked prompt position.
- Dynamic `G_t`, including predictor self-attention.
- Trimmed stop tokens.

### Instrumentation tests

- All 36 layers are discovered and restored.
- Restoration occurs after an injected exception.
- Layer ranges receive correct contributions.
- Layers 35–36 contribute only to all-layer attention; layer 36 also supplies the
  last hidden value.
- Only predictor query rows enter accumulators.
- Prompt image prototypes are float32.
- Full attention references are released after reduction.
- Missing layer calls cause a contract failure.

### Output-schema tests

- PyArrow schemas exactly match `OUTPUT_SCHEMA.md`.
- Probability metric inventory has 31 columns.
- Attention metric inventory has 123 columns.
- Hidden metric inventory has 4 metric columns.
- Decimal parameter names use `p`.
- Null and infinity round-trip through Parquet.
- Zero-token rollouts create valid zero-row token-family parts.
- All token-family keys are exactly equal.
- Checkpoint is written last.
- Failure injection between every transaction phase never creates a false
  completion.
- Orphan detection and recovery are deterministic.
- Score versions coexist without overwrite.
- Every row matches its run manifest's dataset, split, and strategy.

### Predictor-alignment integration test

A small deterministic fake model emits unique markers in raw logits, attention,
and hidden output for each predictor position. The test samples known tokens and
proves every stored token row combines markers from the computation predicting
that token, not the computation after it.

### Real Qwen smoke test

Before any large run:

1. Use the local Qwen2.5-VL-3B checkpoint in `qwen3vl-metrics`.
2. Select one MathVerse example.
3. Generate two rollouts with a short maximum.
4. Run once with microbatch size one and once with size two.
5. Assert identical retained IDs per rollout.
6. Validate every Parquet schema and relational invariant.
7. Inspect predictor positions, first-step grouping, stop trimming, metric ranges,
   infinity/null handling, and layer-valid counts.
8. Restore every patched layer.
9. Compare overlapping metrics with the old runner while accepting intentional
   definition differences.

Production is blocked until this smoke test passes.

## Development environment

The current target WSL Conda environment is `qwen3vl-metrics`, observed with:

~~~text
Python       3.11.15
PyTorch      2.5.1+cu124
CUDA runtime 12.4
Transformers 4.57.0
Datasets     3.2.0
PyArrow      24.0.0
~~~

`qwen_vl_utils` is installed. The local Qwen checkpoint is configurable and
must not be hard-coded into pure modules.

`pyproject.toml` defines a `src` package, the `visconf` console entry point,
runtime dependencies, and pytest configuration. Configuration models use
Pydantic v2 in frozen mode after resolution. YAML loading uses safe loading.
Public domain records use standard frozen dataclasses.

Runtime versions are captured in every run manifest. Production setup should use
a locked environment file generated from the validated RunPod environment.

## Implementation phases

Every phase is gated: its exit criteria and all earlier tests must pass before
the next phase begins. Avoid placeholder modules for later phases.

### Phase 1: foundation, types, configuration, and planning

Create:

- `pyproject.toml`, package initializers, pytest configuration, and console-entry
  skeleton.
- Model-independent frozen records in `types.py`.
- Frozen Pydantic configuration models and safe YAML resolution.
- Group planning, run-cell validation, identity creation, and atomic planning
  manifests.
- The six initial config fragments.

Exit criteria:

- The package imports in `qwen3vl-metrics`.
- The initial config plans exactly six unique independent runs.
- Each resolved run contains one dataset/split/filter and one strategy.
- IDs, resolved configuration, and hashes survive reload.
- Replanning an existing group preserves IDs.
- Invalid multi-dataset or multi-strategy run configurations are rejected.

### Phase 2: pure metrics and decoding primitives

Implement:

- Probability, attention, and hidden-state metric functions.
- Metric validation and online accumulators.
- Temperature/top-k/top-p sampling.
- Stable seed derivation and per-rollout generators.
- Stop-token and maximum-length decisions.

Exit criteria:

- All hand-computed metric and edge-case tests pass.
- Physical metric inventories match `OUTPUT_SCHEMA.md`.
- Defined infinity is preserved and undefined values remain distinguishable.
- Raw logits are unchanged by sampling.
- Seed test vectors and independent-generator tests pass.

### Phase 3: records, Arrow schemas, storage, and resume

Implement storage before model generation, using synthetic
`CompletedRollout` fixtures:

- Exact PyArrow schemas for every primary table.
- Domain-record-to-Arrow conversion.
- Immutable Parquet part writing.
- Experiment and run manifests.
- Core and score transactions with checkpoint written last.
- Failure records, run locks, orphan handling, validation, and resume indexes.

Exit criteria:

- All table schemas and column inventories match `OUTPUT_SCHEMA.md`.
- Null and IEEE infinity round-trip through Parquet.
- Token-family key equality and zero-token rollout tests pass.
- Failure injection at every transaction stage never creates false completion.
- Resume uses checkpoints only and handles orphans deterministically.
- Synthetic six-run manifests remain isolated by run ID.

### Phase 4: dataset protocol, MathVerse, prompts, and token positions

Implement:

- Shared dataset adapter protocol.
- MathVerse source loading, revision/split/filter selection, stable sample IDs,
  image normalization, metadata, and ground truth.
- Configurable message and prompt construction.
- Qwen tokenizer-based `I`, `P`, prompt-last, and dynamic `G_t` grouping.

Exit criteria:

- MathVerse selection is stable and resumable.
- The shared example type already supports ordered multiple images.
- Prompt hashes and exact prompt token IDs are reproducible.
- Actual-tokenizer tests cover special tokens, image spans, padding, and
  generated-token self-attention.
- Dataset code imports no generation, metric, or storage implementation.

### Phase 5: Qwen facade and instrumentation

Implement:

- Explicit single-device BF16/eager model and processor loading.
- Multimodal prompt preparation and batch-size-one cached prefill.
- Qwen position, cache-position, rope-delta, and image-input handling.
- Restorable discovery and patching of all 36 decoder layers.
- Float32 image prototypes and online attention/hidden reductions.
- Batch-size-one base-cache cloning and repeat support.

Exit criteria:

- The local checkpoint reports and instruments exactly 36 layers.
- Patch restoration passes normal and exception tests.
- Layers 35–36 follow the documented aggregate rules.
- No full attention or per-layer hidden collection survives reduction.
- A short real prompt prefill returns raw logits, reduced aligned observations,
  prototypes, and a reusable base cache.
- Cache-copy results match an independent prompt run for the tested steps.

### Phase 6: cached predictor-aligned generation

Implement:

- Immutable base-cache replication per rollout microbatch.
- Mutable rollout state with one generator per rollout.
- Per-row sampling, selected-token probability completion, and stop trimming.
- Active-row cache compaction.
- Predictor-aligned token and three-family metric records.
- `CompletedRollout` construction.

Start with a deterministic model facade test double, then exercise the Qwen
facade.

Exit criteria:

- The fake-model predictor-alignment test passes.
- Microbatch sizes 1, 2, and 4 produce identical rollout sequences.
- Removing finished rows does not perturb surviving rows.
- EOS/`<|im_end|>`, immediate stop, and maximum-length cases pass.
- OOM retry at a smaller microbatch preserves seeds and sequences.
- Qwen two-step cached decoding matches independent decoding.

### Phase 7: runner, CLI, group orchestration, and core integration

Implement:

- Single-run execution and resume.
- Stable example traversal and pending-rollout selection.
- Core shard assembly and transactional commit.
- `plan`, `run`, `group`, and `validate` CLI commands.
- Parent experiment-manifest status updates.
- Structured logging and failure boundaries.

First prove the complete path with deterministic dataset and model test doubles.

Exit criteria:

- A synthetic group produces six isolated run directories.
- One run can be interrupted, resumed, and completed without affecting another.
- Partial example completion resumes only missing rollout indices.
- Committed generations reconstruct exactly from checkpoints.
- Run and group status transitions validate.
- No CLI module contains experiment logic.

### Phase 8: versioned scoring

Implement:

- Importable answer normalization with explicit imports.
- Dataset-appropriate scorer selection.
- Stable scorer names, versions, configuration hashes, and code hashes.
- Offline scoring command and optional post-commit online scoring.
- Independent score transactions and scoring resume.

Exit criteria:

- Online and offline modes produce identical score rows.
- Multiple scorer versions coexist without overwrite.
- Scoring failure cannot invalidate core generation.
- Numeric, multiple-choice, malformed-box, and abstention tests pass.
- Production code no longer loads scorer source with `exec()`.

### Phase 9: MathVerse real-model acceptance

Run the complete two-rollout MathVerse smoke test in `qwen3vl-metrics`, including
an intentionally interrupted transaction and an injected microbatch OOM fallback.

Exit criteria:

- Microbatch sizes one and two produce identical retained IDs per rollout.
- Every output table, key, type, null, infinity, and layer-valid count validates.
- Predictor positions and first-step groups are manually or programmatically
  audited.
- Stop-token rows are trimmed correctly.
- Interrupted resume produces no duplicate or missing rows.
- Overlapping legacy metrics are compared and all intentional differences are
  documented.
- Patched layers are restored and GPU memory is released.

### Phase 10: MathVista and MMMU-Pro

Validate the MathVista and MMMU-Pro adapters, prompt configurations, metadata
mappings, and versioned scorers against the shared contracts.

Exit criteria:

- All adapters emit the same shared types.
- Every dataset has its own diverse and concentrated run IDs.
- MMMU-Pro multiple-image examples require no shared-type or output-schema
  change.
- Adapter-specific smoke tests pass.
- The full group planner and validator still resolve exactly six independent
  runs.

### Phase 11: hardware benchmark and production readiness

Benchmark real visual sequence lengths:

- A100 40 GB: microbatches 4 and 8.
- A100 80 GB: microbatches 8 and 16.
- H100 80 GB: microbatch 16, then 32.

Record prompt and decode timing, peak memory, throughput, and OOM fallback.
Choose hardware-specific defaults without changing semantic configuration or
rollout seeds.

Exit criteria:

- Dataset revisions, splits, filters, prompts, model revision, scorer versions,
  `max_new_tokens`, and hardware defaults are frozen.
- The locked RunPod environment is recorded.
- Persistent output storage is verified.
- A final interrupted-run resume rehearsal passes.
- The production acceptance checklist below passes in full.

## Production acceptance checklist

Before creating the first large experiment group:

- Normative documents are hashed and unchanged from the validated smoke test.
- The group plan has exactly six unique runs.
- Every run has one dataset/split/filter and one strategy.
- Dataset revisions, filters, prompt templates, and example selections are
  frozen.
- Model and tokenizer revisions are frozen.
- All metric inventories and Arrow schemas validate.
- Predictor-alignment tests pass.
- Microbatch determinism tests pass.
- Transaction failure-injection tests pass.
- Real Qwen smoke test passes.
- Scorer versions and code hashes are fixed.
- Hardware microbatch defaults are benchmarked.
- RunPod persistent output root is verified.
- Resume is tested from an intentionally interrupted shard.
- Validation can reconstruct the committed rollout count from checkpoints only.

## Locked architectural decisions

The following are settled for the initial code:

- Six independent run IDs per experiment group.
- Exactly one dataset and one sampling strategy per run.
- Separate outputs and resume state for every run.
- Shared `experiment_group_id` and experiment manifest.
- 32 rollouts per example per run.
- Custom cached predictor-aligned decoding.
- Batch-size-one prompt prefill with per-microbatch cache replication.
- Independent rollout generators.
- Eager attention and online reduction.
- 36 model layers with fixed phase ranges 1–21 and 22–34.
- Normalized token, probability, attention, and hidden tables.
- Separate immutable versioned scores.
- One-based retained token steps.
- Trimmed stop-token rows.
- Parquet plus checkpoint-last commits.
- Offline scoring by default.
- One GPU and one example at a time.

Dataset source IDs, exact production splits/filters, prompt wording,
`max_new_tokens`, and hardware microbatch defaults remain configuration choices
that must be frozen and recorded before production. They do not require a
repository architecture change.

All source paths, split labels, and filters remain configurable in dataset YAML.
