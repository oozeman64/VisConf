# Production runbook

The initial production matrix is frozen by the files under `configs/`. The
selected group uses A100 80 GB, 32 rollouts per example, prompt_batch_size 32,
rollout_microbatch_size 32, and a maximum of 1024 retained tokens. Dataset and local model
artifact revisions are explicit strings in their fragment files.

Do not copy benchmark numbers from another GPU. The benchmark command verifies
that the active device name and memory match the selected hardware profile.

## Recommended reusable RunPod setup

For the least repeated setup, use a private custom Pod template plus a RunPod
network volume mounted at `/workspace`:

- Bake Python, CUDA-compatible PyTorch, Conda, and the locked Python packages
  into the template image. Keep model weights, datasets, repository state, and
  outputs off the container disk.
- Store the persistent project at this layout so the checked-in relative paths
  resolve without edits:

~~~text
/workspace/VisualPO/
|-- VisConf/
|-- Qwen2.5-VL-3B-Instruct/
|-- MathVerse/
|-- MathVista/
`-- MMMU_Pro/
~~~

- Mount the network volume at `/workspace`. Expose `22/tcp` for SSH and
  optionally `8888/http` for JupyterLab. Keep API tokens in RunPod secrets or
  template environment variables, never in the repository.
- Set `CONDA_ENVS_PATH=/workspace/.conda/envs`,
  `HF_HOME=/workspace/.cache/huggingface`, and
  `PIP_CACHE_DIR=/workspace/.cache/pip` in the template so environments and
  download caches also survive replacement Pods.
- Use the same pinned template image for every rental. Keep the Pod alive with
  the template's normal idle command; connect by SSH and launch experiments in
  `tmux` or another persistent terminal multiplexer.

If maintaining a custom image is not yet worthwhile, use an official RunPod
PyTorch template and create `qwen3vl-metrics` once on the network volume.
Reuse exactly the same template afterward, activate the existing environment,
and run `python -m pip install -e .` after checking out the desired commit.

On every new Pod:

~~~bash
cd /workspace/VisualPO/VisConf
source /opt/conda/etc/profile.d/conda.sh
conda activate qwen3vl-metrics
git fetch --all --prune
git checkout <validated-commit>
python -m pip install -e .
scripts/verify_persistent_storage.sh /workspace/visconf-output
~~~

Use a network volume when you expect to terminate and rent different Pods.
A Pod volume disk survives stop/start only while that Pod exists; it is deleted
with the Pod.

The RTX 4090 development profile uses prompt_batch_size 16 and
rollout_microbatch_size 32. Validate representative prompt lengths before
long runs; the primary production group remains A100 80 GB by default.

`configs/experiment_group_4090_full_mb32.yaml` is the separately validated
local full-run profile. It uses 32 rollouts, microbatch 32, a 1,024-token limit,
and writes beneath `VisConf/outputs/`. Validate representative prompt lengths
before long runs; the primary A100 80 GB production selection remains separate.

## 1. Recreate the locked environment

Create and activate the Conda environment named `qwen3vl-metrics`, then install
the exact versions recorded in
`environment/runpod-qwen3vl-metrics.lock`. Keep the model, dataset sources,
repository, and output root on persistent RunPod storage.

## 2. Prove persistence across a restart

Initialize the marker once:

~~~bash
scripts/verify_persistent_storage.sh /workspace/visconf-output --initialize
~~~

Restart the pod or detach and reattach the volume, then verify the existing
marker:

~~~bash
scripts/verify_persistent_storage.sh /workspace/visconf-output
~~~

The second command must return the same `marker_id`. A marker created and
verified in one uninterrupted container session is not sufficient evidence of
persistence.

## 3. Plan the frozen six-run group

~~~bash
scripts/run_experiment_group.sh \
  configs/experiment_group.yaml \
  <experiment_group_id> \
  /workspace/visconf-output
~~~

For a pre-production plan, stop the script after its `visconf plan` command.
The manifest must contain six unique dataset-by-strategy cells and run IDs.

## 4. Benchmark the selected target GPU

Choose one planned run with a representative visual example and run:

~~~bash
scripts/benchmark_hardware.sh \
  <experiment_group_id> \
  <run_id> \
  /workspace/visconf-output
~~~

The selected hardware profile measures explicit prompt/rollout batch shapes,
including a prompt-batch-one baseline and measured multi-prompt candidates where
configured. Each report records prompt time, decode time, total time, peak
allocated memory, retained-token throughput, OOM fallback count, environment
details, and a retained-ID hash.

For a runner-backed local RTX 4090 check that also commits the complete output
transaction, use:

~~~bash
scripts/run_efficiency_benchmark.sh \
  outputs/benchmarks/rtx4090-mb32 \
  --config configs/experiment_group_4090_full_mb32.yaml \
  --group-id benchmark-rtx4090-mb32 \
  --rollouts 32 \
  --microbatch 32 \
  --max-new-tokens 1024
~~~

The command must finish with 32 committed rollouts, no orphan parts, and equal
row counts for tokens and all three token-metric families. Its output includes
examples and generation Parquet files, atomic core-table shards, a checkpoint,
and updated manifests; it is therefore suitable for end-to-end regression
comparison, unlike a timing-only hardware probe.

A production default is accepted only when every configured exact batch shape
completes without fallback and retained-ID hashes agree. If the larger candidate
OOMs, lower the hardware profile limit and re-plan; do not silently change an
existing resolved run.

## 5. Run acceptance and interrupted-resume rehearsal

~~~bash
scripts/run_smoke_test.sh
scripts/run_resume_rehearsal.sh
~~~

The rehearsal injects a transaction interruption, quarantines uncommitted parts,
resumes the same identities, and reconstructs completion from checkpoints.

## 6. Validate production readiness

~~~bash
python -m visconf validate \
  --group <experiment_group_id> \
  --output-root /workspace/visconf-output \
  --production \
  --benchmark-report /workspace/visconf-output/<experiment_group_id>/benchmarks/<run_id>.json
~~~

This checks frozen revisions, normative-document hashes, six isolated run cells,
schema and scorer inventories, hardware defaults, benchmark completion,
checkpoint integrity, orphan absence, the environment lock, and the persistent
storage marker.

## 7. Launch or resume

Run one cell:

~~~bash
scripts/run_single_run.sh \
  <experiment_group_id> \
  <run_id> \
  /workspace/visconf-output
~~~

Or run incomplete cells sequentially with `visconf group`. Reusing the same
group and run IDs resumes exclusively from checkpoint-committed rollout keys.

For any frozen production prompt-batch shape, require a representative
target-GPU benchmark showing that the exact chosen prompt/rollout shape fits
without fallback, improves useful retained-token throughput, and matches the
prompt-batch-one retained-ID hash. Compare contiguous and bounded
token-count-bucketed runs using the same prompt identities. The RTX 4090
development profile currently selects (16,32); H100 remains prompt-batch one
until equivalent evidence exists.
