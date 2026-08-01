# Output Schema

This document is the normative storage contract for VisConf output schema
version 1. Metric formulas and invalid-value rules remain normative in the three
metric documents; this file defines how their results are named, typed, related,
committed, and resumed.

## Design principles

- Primary experiment data is stored in immutable Parquet part files.
- The canonical retained-token row is separate from the three metric families.
- Every retained token has exactly one probability row, one attention row, and
  one hidden-state row, even when all metrics in a family are undefined.
- Probability, attention, and hidden metrics sharing a token key describe the
  same predictive computation.
- Generated stop tokens are not retained token rows.
- Undefined values are Parquet null. Mathematically defined infinity is stored as
  IEEE infinity, not null.
- Online and offline scoring use one versioned score schema and do not mutate
  generation data.
- Only checkpoint-committed shards are part of a run.

## Schema versions

The initial versions are:

~~~text
output_schema_version = 1
metric_schema_version = 1
seed_derivation_version = 1
~~~

The manifest records all three. A backward-incompatible key, type, meaning, or
column-name change requires a new output schema version. A metric definition
change requires a new metric schema version and updated hashes for the normative
metric documents.

## Experiment groups and runs

An experiment group is the comparison unit. Its configuration expands the
Cartesian product of configured datasets and sampling strategies into resolved
run specifications.

A `run_id` represents exactly one:

- Dataset adapter and dataset revision.
- Split and filter selection.
- Sampling strategy.

The initial experiment group therefore contains six runs:

~~~text
MathVerse  × diverse
MathVerse  × concentrated
MathVista  × diverse
MathVista  × concentrated
MMMU-Pro   × diverse
MMMU-Pro   × concentrated
~~~

Each run performs 32 rollouts per selected example. The two strategy runs for a
dataset therefore provide 64 total rollouts per example for comparison.

Every run has its own directory, manifest, checkpoints, failures, and resume
state. All six runs share one `experiment_group_id` and are listed in one
experiment manifest. Run IDs are opaque identities; dataset and strategy names
may appear in human-readable labels but must not be recovered by parsing a
`run_id`.

A launcher may execute several runs sequentially in one RunPod job and may keep
the model loaded between runs, but each run remains logically and transactionally
independent. It must not share completion state or output parts across run IDs.

Dataset and strategy remain present in every primary key even though they are
constant within a run. This keeps exported rows self-describing and prevents
unsafe cross-run merges.

## Output layout

~~~text
outputs/<experiment_group_id>/
|-- experiment_manifest.json
`-- <run_id>/
    |-- manifest.json
    |-- examples.parquet
    |-- generations/
    |   `-- part-<shard_id>.parquet
    |-- tokens/
    |   `-- part-<shard_id>.parquet
    |-- token_probability_metrics/
    |   `-- part-<shard_id>.parquet
    |-- token_attention_metrics/
    |   `-- part-<shard_id>.parquet
    |-- token_hidden_state_metrics/
    |   `-- part-<shard_id>.parquet
    |-- scores/
    |   `-- part-<score_shard_id>.parquet
    |-- failures.jsonl
    `-- checkpoints/
        |-- shard-<shard_id>.json
        `-- score-<score_shard_id>.json
~~~

Parquet part files are immutable after commit. Derived summaries and joined
analysis tables are not primary experiment data and must be written outside
these directories or under an explicitly marked derived-output directory.

## Keys and identity

### Example key

~~~text
(run_id, dataset, split, sample_id)
~~~

### Rollout key

~~~text
(run_id, dataset, split, sample_id, strategy, rollout_index)
~~~

### Token key

~~~text
(run_id, dataset, split, sample_id, strategy, rollout_index, step)
~~~

### Score key

~~~text
(
    run_id,
    dataset,
    split,
    sample_id,
    strategy,
    rollout_index,
    scorer_name,
    scorer_version
)
~~~

All key columns are non-null. Token `step` is one-based: step 1 is the retained
token `y_1` in the metric documents. `rollout_index` is zero-based within a
strategy. Within a run, every `dataset`, `split`, and `strategy` value must
equal the single resolved value in that run's manifest. A run must not mix
conventions, datasets, splits, filters, or strategies.

The resume identity for generation is the rollout key without `run_id`, because
the run directory and manifest already fix `run_id`. A rollout is complete only
when its core shard checkpoint is committed.

## Common Arrow types

| Logical value | Arrow/Parquet type |
| --- | --- |
| IDs, names, text, JSON | `string` |
| Token IDs, steps, positions, counts | `int32` |
| Rollout index | `int32` |
| Stable rollout seed | `uint64` |
| Metric scalar | nullable `float32` unless specified otherwise |
| Rank or nucleus-size metric | nullable `int32` |
| Timing | nullable `float64` |
| Flags | `bool`, nullable only where absence has meaning |
| UTC timestamp | `timestamp[us, tz=UTC]` |
| SHA-256 digest | lowercase hexadecimal `string` |

Strings use UTF-8. Arbitrary JSON fields use canonical JSON: UTF-8, sorted object
keys, no insignificant whitespace, and no NaN or infinity literals.

## Metric column naming

Within a metric-family table, columns use the documented metric name without an
additional family prefix.

Attention columns use:

~~~text
<scenario>__<documented_metric_name>
~~~

For example:

~~~text
all_layers_all_heads__img_attn_total
early_visual_integration__img_attn_total
visual_reasoning__attn_ratio_img_to_all
~~~

A metric parameter containing a decimal point uses `p` in the physical Parquet
column name:

| Documented name | Parquet column |
| --- | --- |
| `nucleus_size_0.9` | `nucleus_size_0p9` |
| `nucleus_size_0.95` | `nucleus_size_0p95` |
| `nucleus_size_0.99` | `nucleus_size_0p99` |
| `renyi_entropy_0.5` | `renyi_entropy_0p5` |

No other documented metric names are changed.

## experiment_manifest.json

The experiment manifest contains at least:

- `experiment_group_id`, creation time, group status, and output root.
- Group configuration hash and the shared model, prompt, metric, schema, and seed
  configuration.
- The expected dataset-by-strategy matrix.
- For each matrix cell: `run_id`, dataset, split/filter identity, strategy,
  relative run path, resolved run-configuration hash, and run status.
- The group-level Git commit and metric-document hashes used when the group was
  planned.

Stable group statuses are `planned`, `running`, `complete`, and `failed`.
The group is complete only when every expected run is complete. The experiment
manifest is updated through a temporary file and atomic rename. A run remains
independently resumable regardless of group status.

## manifest.json

The run manifest contains at least:

- `experiment_group_id`, `run_id`, creation time, completion status, and
  output location.
- One fully resolved run configuration containing exactly one dataset/split/filter
  and one sampling strategy.
- `output_schema_version`, `metric_schema_version`, and
  `seed_derivation_version`.
- Model and tokenizer identifiers, local paths if applicable, revisions, and
  configuration hash.
- The single dataset identifier, revision, split, filter, and source-file hashes
  where applicable.
- The selected example count and SHA-256 hash of the immutable
  `examples.parquet` selection artifact.
- Prompt-template configuration and hashes.
- The single sampling strategy definition, including temperature, top-p, top-k,
  and repetition penalty.
- Rollouts per example, prompt batch size, rollout microbatch size, prompt
  batching strategy, bucket window, scheduler algorithm version, and hardware
  maximum active decode rows.
- Model dtype, metric reduction dtype, attention implementation, and device.
- Python, PyTorch, Transformers, Datasets, PyArrow, CUDA, and
  `qwen_vl_utils` versions.
- GPU model, memory capacity, and relevant CUDA runtime information.
- Git commit and dirty-worktree status.
- Metric-document paths and SHA-256 hashes.
- Decoder layer count and exact aggregation ranges:
  `all_layers_all_heads=1..36`, `early_visual_integration=1..21`, and
  `visual_reasoning=22..34`.
- Tokenizer stop-token IDs and all symbolic names resolving to each ID.
- Ordered metric-column inventory with physical types.
- Scorer names, versions, configuration, and code hashes when scoring is enabled.
- Parquet compression, dictionary-encoding, and row-group settings.
- Committed core and score shard inventory.

The manifest is written through a temporary file and atomic rename whenever its
mutable run-status or shard inventory is updated.

## examples.parquet

There is exactly one row per example key.

| Column | Type | Null | Meaning |
| --- | --- | --- | --- |
| `run_id` | string | no | Run identity |
| `dataset` | string | no | Stable dataset adapter label |
| `split` | string | no | Resolved dataset split |
| `sample_id` | string | no | Stable dataset example identity |
| `source_row_index` | int64 | yes | Original row index when meaningful |
| `question` | string | no | Dataset question before prompt augmentation |
| `rendered_prompt` | string | no | Exact rendered chat-template text |
| `prompt_token_ids` | list<int32> | no | Exact unpadded prompt IDs |
| `prompt_token_count` | int32 | no | Length of `prompt_token_ids` |
| `ground_truth_json` | string | no | Canonical adapter-owned target data |
| `answer_type` | string | yes | Adapter scoring category |
| `images` | list<image_struct> | no | Ordered image metadata |
| `metadata_json` | string | no | Canonical dataset-specific metadata |

`image_struct` is:

~~~text
struct<
    source_ref: string nullable,
    sha256: string non-null,
    width: int32 non-null,
    height: int32 non-null,
    mode: string non-null
>
~~~

Raw image bytes are not copied into experiment output. Dataset revision and image
hashes provide identity; persistence of source data is an operational concern
recorded in configuration.

Frequently analyzed dataset-specific values may receive additional typed columns,
but the adapter must also retain them in `metadata_json`. Added columns must be
declared in the manifest.

## generations

There is exactly one row per successfully completed rollout key. Failed attempts
are recorded only in `failures.jsonl` and do not create a completed generation
row.

| Column | Type | Null | Meaning |
| --- | --- | --- | --- |
| Rollout key columns | as defined | no | Rollout identity |
| `shard_id` | string | no | Committed core shard |
| `rollout_seed` | uint64 | no | Stable per-rollout seed |
| `temperature` | float32 | no | Selection temperature |
| `top_p` | float32 | no | Selection top-p |
| `top_k` | int32 | yes | Null means no top-k filtering |
| `repetition_penalty` | float32 | no | Initially exactly 1.0 |
| `generated_token_ids` | list<int32> | no | Retained token IDs only |
| `generated_text` | string | no | Decode of retained token IDs |
| `num_retained_tokens` | int32 | no | Length of retained token list |
| `stop_reason` | string | no | `stop_token` or `max_new_tokens` |
| `terminating_token_id` | int32 | yes | Selected trimmed stop ID |
| `hit_max_new_tokens` | bool | no | True only for maximum-token stop |
| `prompt_token_count` | int32 | no | Unpadded prompt length |
| `wall_time_seconds` | float64 | yes | Rollout-attributed elapsed time |
| `tokens_per_second` | float64 | yes | Retained tokens per attributed second |
| `completed_at_utc` | timestamp | no | Completion time |

A selected EOS or `<|im_end|>` token is absent from
`generated_token_ids`, `generated_text`, `tokens`, and all metric tables.
Its ID is stored in `terminating_token_id`. Because the current checkpoint maps
EOS and `<|im_end|>` to the same ID, the row does not claim which symbolic name
caused the stop.

An immediate stop is a valid completed rollout: `num_retained_tokens=0`, the
token list is empty, and no token-family rows exist for that rollout.

## tokens

This is the canonical retained-token table. There is one row per token key.

| Column | Type | Null | Meaning |
| --- | --- | --- | --- |
| Token key columns | as defined | no | Token identity |
| `shard_id` | string | no | Committed core shard |
| `token_id` | int32 | no | Selected retained token ID |
| `token_piece` | string | no | Tokenizer vocabulary piece |
| `token_text` | string | no | Single-token decode with special tokens retained and cleanup disabled |
| `predictor_position` | int32 | no | Zero-based logical unpadded position of `q_t` |
| `context_length` | int32 | no | Unmasked token count used to predict this token |

`predictor_position` uses logical unpadded positions rather than tensor columns.
For retained step `t`, `context_length` equals prompt token count plus
`t - 1`, and the predictor position is `context_length - 1`.

## token_probability_metrics

There is exactly one row per token key.

Audit columns:

| Column | Type | Null | Meaning |
| --- | --- | --- | --- |
| Token key columns | as defined | no | Token identity |
| `shard_id` | string | no | Committed core shard |
| `metrics_valid` | bool | no | Probability-family input passed validation |
| `invalid_reason` | string | yes | Stable reason code when invalid |

Float32 metric columns:

~~~text
logp
kl_u_p
kl_p_u
gini
entropy
dist_perplexity
inverse_perplexity
max_prob
margin_top2
log_ratio_margin_top2
selected_dominance
selected_logrank
topk_mass_2
topk_mass_5
topk_mass_10
topk_mass_20
tail_mass_2
tail_mass_5
tail_mass_10
tail_mass_20
norm_entropy_concentration
renyi_entropy_0p5
renyi_entropy_1
renyi_entropy_2
renyi_entropy_4
renyi_entropy_inf
js_p_u
~~~

Nullable int32 metric columns:

~~~text
selected_rank
nucleus_size_0p9
nucleus_size_0p95
nucleus_size_0p99
~~~

The signed confidence convention from the probability document is preserved for
integer metrics. For example, `selected_rank` stores negative competition rank.

## token_attention_metrics

There is exactly one row per token key.

Shared group-count columns are non-null int32:

~~~text
n_image_tokens
n_prompt_text_tokens
n_generated_text_tokens
n_prompt_generated_text_tokens
n_all_attn_tokens
~~~

For each scenario, store a non-null Boolean validity flag:

~~~text
all_layers_all_heads__valid
early_visual_integration__valid
visual_reasoning__valid
~~~

For each of the three scenarios, instantiate all columns below using
`<scenario>__<metric>`. Every metric column is nullable float32.

Group prefixes:

~~~text
img_attn
prompt_text_attn
generated_text_attn
prompt_generated_text_attn
all_attn
~~~

Per-group suffixes:

~~~text
total
avg
gini
entropy
kl_u_p
kl_p_u
dist_perplexity
~~~

This produces 35 group metrics per scenario. Also instantiate these six
cross-group ratios per scenario:

~~~text
attn_ratio_img_to_prompt_text
attn_ratio_img_to_generated_text
attn_ratio_img_to_all
attn_ratio_prompt_text_to_all
attn_ratio_generated_text_to_all
attn_ratio_prompt_generated_text_to_all
~~~

Each scenario therefore has 41 attention metric columns and the table has 123
attention metric columns in total.

If a scenario is invalid, its validity flag is false and all 41 metrics for that
scenario are null. Empty-group and low-mass behavior for a valid scenario follows
the attention document: in particular, an empty group's total is zero while its
average and normalized distribution metrics are null.

The group counts are independent of attention scenario. At retained step `t`,
`n_generated_text_tokens = t - 1`.

## token_hidden_state_metrics

There is exactly one row per token key.

| Column | Type | Null | Meaning |
| --- | --- | --- | --- |
| Token key columns | as defined | no | Token identity |
| `shard_id` | string | no | Committed core shard |
| `valid_layers_all` | int16 | no | Valid layers among 1–36 |
| `valid_layers_early_visual_integration` | int16 | no | Valid layers among 1–21 |
| `valid_layers_visual_reasoning` | int16 | no | Valid layers among 22–34 |
| `last_layer_valid` | bool | no | Whether layer 36 is valid |

Nullable float32 metric columns:

~~~text
cosine_gen_imgproto_hidden_avg_all_layers
cosine_gen_imgproto_hidden_last_layer
cosine_gen_imgproto_hidden_early_visual_integration
cosine_gen_imgproto_hidden_visual_reasoning
~~~

Layers 35–36 contribute to the all-layer aggregate. Layer 36 supplies the
last-layer metric. Neither contributes to the fixed early-visual-integration or
visual-reasoning aggregates.

If no image tokens exist, all valid-layer counts are zero,
`last_layer_valid=false`, and all four metrics are null.

## scores

Scores are independent immutable parts so generations can be rescored without
rewriting core output. Online and offline scoring write the same rows.

| Column | Type | Null | Meaning |
| --- | --- | --- | --- |
| Score key columns | as defined | no | Versioned scorer identity |
| `score_shard_id` | string | no | Committed score shard |
| `is_correct` | bool | yes | Binary result or null on scorer abstention |
| `raw_final_answer` | string | yes | Directly extracted final-answer span |
| `extracted_answer` | string | yes | Normalized scorer answer |
| `scorer_method` | string | no | Stable extraction/comparison method |
| `score_details_json` | string | no | Canonical additional scorer output |
| `scored_at_utc` | timestamp | no | Scoring completion time |

Scoring failure creates a failure record and no score row for that scorer key.
Scoring resume identity is the score key without `run_id`.

## failures.jsonl

Each line is one UTF-8 JSON object containing at least:

~~~text
failure_id
run_id
attempt_id
stage
dataset
split
sample_id
strategy
rollout_index
scorer_name
scorer_version
exception_type
message
traceback
retryable
created_at_utc
~~~

Identity and scorer fields may be null when failure occurs before they are known.
Stable `stage` values include `dataset`, `prompt`, `model`, `generation`,
`metrics`, `storage`, and `scoring`.

A failure record never marks a rollout or score complete.

## Core shard commit protocol

One core shard contains matching part files for:

1. `generations`
2. `tokens`
3. `token_probability_metrics`
4. `token_attention_metrics`
5. `token_hidden_state_metrics`

All five use the same `shard_id`. Even a shard containing only zero-token
rollouts writes schema-correct zero-row files for all four token-level tables.

The writer must:

1. Write every part to a uniquely named temporary file in its destination
   directory.
2. Close and fsync each file.
3. Validate schemas, rollout identities, token-key equality, and row counts.
4. Atomically rename every temporary part to its final immutable name.
5. Write and fsync a temporary checkpoint JSON.
6. Atomically rename the checkpoint last.

The checkpoint is the commit marker. It contains:

- Run ID, shard ID, attempt ID, and schema versions.
- Completed rollout identities.
- Relative paths, byte sizes, SHA-256 hashes, and row counts for all five parts.
- Token-key count and minimum/maximum token step.
- Commit timestamp.

Readers must use checkpoint inventories rather than blindly globbing Parquet
directories. A part without a checkpoint is uncommitted. Resume logic may verify
and recover a complete matching part set or quarantine it before rerunning; it
must never treat an incomplete set as complete.

Score shards use the same temporary-file, validation, and checkpoint-last pattern
but commit independently from core generation shards.

## Required relational invariants

For each committed core shard:

- Every dataset, split, and strategy value equals the single resolved value in the
  run manifest.
- No row references another run ID or experiment-group run directory.
- Token keys are unique within `tokens`.
- The key set of each metric-family table exactly equals the `tokens` key set.
- The four token-level tables have equal row counts.
- Token steps for a rollout are contiguous integers from 1 through
  `num_retained_tokens`.
- Ordered `tokens.token_id` values exactly equal
  `generations.generated_token_ids`.
- The number of token rows for a rollout equals
  `generations.num_retained_tokens`.
- `stop_reason=max_new_tokens` implies `terminating_token_id` is null and
  `hit_max_new_tokens=true`.
- `stop_reason=stop_token` implies `terminating_token_id` is non-null and
  `hit_max_new_tokens=false`.
- Every metric column and physical type matches the manifest inventory.
- No failed rollout identity appears in a checkpoint unless a later successful
  attempt completed it.

These invariants are checked before commit and covered by automated tests.

## Recommended Parquet settings

Initial defaults:

~~~text
compression = zstd
compression_level = 3
use_dictionary = true for strings
write_statistics = true
target_row_group_size = 65536 rows
~~~

Settings remain configurable and are recorded in the manifest. Do not reduce
float metrics to float16 for storage.

## Logical wide analysis view

The normalized tables are canonical. A logical `token_metrics_wide` view may
join `tokens` with all three family tables on the token key for exploratory
analysis or model training.

The view should be generated in DuckDB, Polars, or another analysis layer rather
than persisted as a second authoritative copy. Any exported materialization is
derived data and must record the source run ID, committed-shard set, and schema
versions.

Prompt-batching settings are manifest configuration, not primary table schema.
Run and experiment manifests record prompt batch size, rollout microbatch size,
strategy, bounded bucket window, scheduler algorithm version, and the hardware
maximum active decode rows. These settings do not change output or metric schema
versions because table keys and physical columns are unchanged.
