# RTX 4090 diagnostic report

Date: 2026-08-03 (Asia/Karachi)

Repository: `/home/fedosr/VisualPO/VisConf`

Commit: `526eb2cac662cffa19499b16a4149242f7eda3a3`

## Executive result

The RTX 4090 side is healthy. Raw BF16 compute, host/device transfer, synchronization, plain Qwen generation, and the matched VisConf benchmark all complete successfully. The matched VisConf run reached `2,388.63 decode tokens/s` and `1,024.41 end-to-end tokens/s` for `8 prompts × 32 rollouts`, `20` new tokens.

Nothing in the RTX 4090 measurements indicates a local driver, power, thermal, CPU-quota, or raw-GPU-compute problem. A100-side artifacts were not present, so cross-machine diffs cannot yet be computed.

## Runtime identity

- Python: 3.11.15, `/home/fedosr/miniconda3/envs/qwen3vl-metrics/bin/python`
- PyTorch: `2.5.1+cu124`; CUDA build/runtime: `12.4`
- Transformers: `4.57.0`
- GPU: NVIDIA GeForce RTX 4090, compute capability `(8, 9)`
- BF16 supported: `True`
- cuDNN: `90100`
- TF32 matmul: `False`; TF32 cuDNN: `True`
- Imported VisConf source: `/home/fedosr/VisualPO/VisConf/src/visconf/__init__.py`
- Model: `/home/fedosr/VisualPO/Qwen2.5-VL-3B-Instruct`

The initially selected `vgpo` environment was Python 3.10.20. It cannot import this checkout's `enum.StrEnum` dependency, so the compatible Python 3.11 `qwen3vl-metrics` environment was used for all project-dependent tests. Identity and environment logs were corrected to reflect that environment.

## Results

| Test | Result |
|---|---:|
| BF16 GEMM, 8192², 20 iterations | 144.23 TFLOP/s |
| D2H transfer, 256 MiB, pinned memory | 11.72 GiB/s |
| 10,000 `torch.cuda.synchronize()` calls | 2.61 µs/call |
| Probability/sampling pipeline, batch 64, vocab 151,936 | 71.18 steps/s; 4,555.66 rows/s |
| Plain Qwen, batch 32 × 64 tokens | 1,266.96 generated tokens/s |
| VisConf matched benchmark, 8 × 32 × 20 tokens | 2,388.63 decode tokens/s; 1,024.41 end-to-end tokens/s |

The probability pipeline initially failed under Python 3.10 before doing GPU work; it passed unchanged under the compatible Python 3.11 environment.

## GPU and container checks

- Driver: `591.86`; `nvidia-smi` reports CUDA `13.1`.
- GPU memory: `24,564 MiB` total.
- Power limit: `450 W`; no software power cap, hardware slowdown, thermal slowdown, or power-brake events were reported.
- Idle snapshot: P8, 33°C, 480 MHz SM, 14.1 W. This is expected while idle.
- During the benchmark telemetry: P0/P2/P3/P5/P8 observed, SM clocks up to 2,775 MHz, power up to 181.23 W, temperature up to 43°C. Aggregate telemetry over 95 samples: GPU utilization 8–80% (mean 38.77%), memory utilization 0–41% (mean 8.83%).
- CPU: 32 logical CPUs, Intel i9-14900K; allowed CPUs `0-31`, memory node `0`.
- Cgroup v1 quota: `-1` with period `100000`, meaning no CPU quota was imposed. The checklist's `taskset -pc $$` line was malformed by the Windows/WSL wrapper and returned an invalid PID; `/proc/self/status` supplied the authoritative affinity.
- MIG: not applicable to this RTX 4090; `nvidia-smi` reports MIG `N/A`.

## Matched VisConf benchmark details

- Config: `configs/experiment_group_4090_full_mb32.yaml`
- Hardware profile: `configs/hardware/rtx_4090.yaml`
- Shape: `8x32`
- Retained tokens: `5,090`
- Decode time: `2.1309 s`
- Total time: `4.9687 s`
- Prompt time: `0.7407 s`
- Peak allocated/reserved: `10.86 / 13.07 GiB`
- OOM fallbacks: `0`
- Status: `complete`

The second telemetry-coupled repeat was also complete: `2,447.35 decode tokens/s`, with the same peak memory and no OOM fallback.

## Nsight and cross-machine comparison

- `nsys` is not installed or not on PATH, so no Nsight Systems trace was produced.
- `diagnostics/a100/` is absent. The requested `pip-freeze` and `torch-environment` diffs were executed and recorded as missing-file errors in `cross-machine-diffs.txt`; no A100 conclusions can be drawn from this side alone.

## Captured artifacts

- `identity.txt`
- `pip-freeze.txt`
- `torch-environment.txt`
- `gpu-system.txt`
- `cpu-container.txt`
- `bf16-gemm.txt`
- `transfer-sync.txt`
- `probability-pipeline.txt`
- `plain-qwen.txt`
- `benchmark-console.txt`
- `benchmark.json`
- `gpu-telemetry.csv`
- `telemetry-benchmark-console.txt`
- `telemetry-benchmark.json`
- `nsys-availability.txt`
- `cross-machine-diffs.txt`
