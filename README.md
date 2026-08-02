# VisConf

VisConf runs predictor-aligned visual-confidence experiments for Qwen2.5-VL-3B.
The initial group contains six independent runs: MathVerse, MathVista, and
MMMU-Pro crossed with the diverse and concentrated sampling strategies.
The defaults use the Vision Intensive subset of MathVerse testmini, all
MathVista testmini rows, and all rows in the MMMU-Pro standard test source;
that source may contain between two and nine choices per question. Dataset YAML
files can override these selections.

The normative contracts are [AGENTS.MD](AGENTS.MD),
[docs/REPO_SCHEMA.MD](docs/REPO_SCHEMA.MD),
[docs/OUTPUT_SCHEMA.md](docs/OUTPUT_SCHEMA.md), and the three metric documents
in `docs/`. Production setup and acceptance are documented in
[docs/PRODUCTION_RUNBOOK.md](docs/PRODUCTION_RUNBOOK.md).

## Environment

Use the WSL Conda environment supplied for this repository:

~~~bash
conda activate qwen3vl-metrics
python -m pip install -e .
~~~

The frozen RunPod package inventory is
`environment/runpod-qwen3vl-metrics.lock`.

The full portable Conda export is
`environment/qwen3vl-metrics.yml`; the verbose Conda and pip inventory is
`environment/qwen3vl-metrics.verbose.txt`. The export omits only the local
editable `visconf` install and machine-specific prefix; recreate the environment
with the export, then run `python -m pip install -e .` from the repository.

## Core workflow

~~~bash
visconf plan --config configs/experiment_group.yaml
visconf validate --group <experiment_group_id>
visconf run --group <experiment_group_id> --run <run_id>
visconf group --group <experiment_group_id>
visconf score --group <experiment_group_id>
~~~

For local RTX 4090 execution, use the RTX 4090 24 GB profile:

~~~bash
visconf plan --config configs/experiment_group_4090.yaml
python scripts/run_real_generation_smoke.py
~~~

To plan the complete six-run matrix on the RTX 4090 with all 32 rollouts in
one microbatch and a 1,024-token generation limit, use the dedicated full-run
profile:

~~~bash
visconf plan \
  --config configs/experiment_group_4090_full_mb32.yaml \
  --group-id <experiment_group_id>
visconf group --group <experiment_group_id>
~~~

This profile resolves its output root to `VisConf/outputs/`, uses microbatch 32,
and creates one independently resumable run for each of the six
dataset-by-strategy cells.

Capture a runner-backed RTX 4090 efficiency baseline over one real prompt,
with 16 rollouts, microbatch 8, and a 1,024-token generation limit:

~~~bash
scripts/run_efficiency_benchmark.sh \
  <output_root>
~~~

Unlike the timing-only `visconf benchmark` command, this test uses the normal
runner and commits a complete output directory: examples, generations, tokens,
all metric-family Parquet files, a checkpoint, and updated manifests. Additional
script arguments override the defaults.

To exercise the validated 32-rollout RTX 4090 shape through the same complete
output transaction, run:

~~~bash
scripts/run_efficiency_benchmark.sh \
  outputs/benchmarks/rtx4090-mb32 \
  --config configs/experiment_group_4090_full_mb32.yaml \
  --group-id benchmark-rtx4090-mb32 \
  --rollouts 32 \
  --microbatch 32 \
  --max-new-tokens 1024
~~~

Each dataset-by-strategy cell has its own immutable `run_id`, output directory,
checkpoint inventory, and resume state. Core shards commit generation, token,
probability, attention, and hidden-state tables atomically. Scores are versioned
and committed independently.

During decoding, VisConf validates each raw-logit microbatch once and shares its
descending vocabulary ordering between selection and probability metrics.
Sampling still applies temperature, top-k, then top-p to a copy, constructs
dense probabilities in original vocabulary order, and uses one independent
generator per rollout. Attention and hidden observations are accumulated across
the microbatch on-device; final attention metric formulas are CPU-batched after
one transfer. These implementation optimizations do not change metric formulas,
Parquet schemas, or rollout identities.

Operational commands are:

~~~bash
visconf benchmark --group <experiment_group_id> --run <run_id>
visconf verify-storage --output-root <persistent_output_root> --initialize
visconf validate --group <experiment_group_id> --production \
  --benchmark-report <benchmark_report.json>
~~~

Hardware benchmarking intentionally refuses to run when the active GPU does not
match the resolved A100, H100, or RTX 4090 profile. Target measurements and persistent-volume
restart verification therefore run on the actual RunPod host, using the scripts
in `scripts/`.

## Prompt-batched decoding

Generation can prefill and decode multiple independent prompts in one scheduling
unit. prompt_batch_size is the prompt axis and rollout_microbatch_size is the
per-prompt rollout axis; max_active_decode_rows bounds their product. Supported
schedulers are contiguous and bounded token_count_bucketed. The latter uses
exact prompt-token and eligible image-pad counts with canonical source ordinal
as its deterministic tie-breaker. Output identities, per-example shards,
checkpoint-last publication, metric keys, and rollout seeds are unchanged.
The primary frozen production configuration uses the A100 80 GB profile with prompt_batch_size 32 and rollout_microbatch_size 32.
