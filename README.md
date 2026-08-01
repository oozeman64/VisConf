# VisConf

VisConf runs predictor-aligned visual-confidence experiments for Qwen2.5-VL-3B.
The initial group contains six independent runs: MathVerse, MathVista, and
MMMU-Pro crossed with the diverse and concentrated sampling strategies.
The defaults use the Vision Intensive subset of MathVerse testmini, all
MathVista testmini rows, and all rows in the MMMU-Pro standard four-option test
source. Dataset YAML files can override these selections.

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

For local RTX 4090 execution, use the conservative 24 GB profile:

~~~bash
visconf plan --config configs/experiment_group_4090.yaml
python scripts/run_real_generation_smoke.py
~~~

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

Each dataset-by-strategy cell has its own immutable `run_id`, output directory,
checkpoint inventory, and resume state. Core shards commit generation, token,
probability, attention, and hidden-state tables atomically. Scores are versioned
and committed independently.

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
