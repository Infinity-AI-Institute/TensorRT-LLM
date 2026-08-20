(perf-analysis)=

# Performance Analysis

NVIDIA Nsight Systems reports at the application level are highly informative. Metric sampling capabilities have increased over generations and provide a clean middle-ground between timing analysis and kernel-level deep dives with NVIDIA Nsight Compute.

Given the potential long runtimes of Large Languages Models (LLMs) and the diversity of workloads a model may experience during a single inference pass or binary execution, NVIDIA has added features to TensorRT LLM to get the most out of Nsight Systems capabilities. This document outlines those features as well as provides examples of how to best utilize them to understand your application.


## Feature Descriptions

The main functionality:
  * Relies on toggling the CUDA profiler runtime API on and off.
  * (PyTorch workflow only) Toggling the PyTorch profiler on and off.
  * Provides a means to understand which regions a user may want to focus on.

Toggling the CUDA profiler runtime API on and off:
  * Allows users to know specifically what the profiled region corresponds to.
  * Results in smaller files to post-process (for metric extraction or similar).

(PyTorch workflow only) Toggling the PyTorch profiler on and off:
  * Help users to analyze the performance breakdown in the model.
  * Results in smaller files to post-process (for metric extraction or similar).


## Coordinating with NVIDIA Nsight Systems Launch

Consult the Nsight Systems User Guide for full overview of options.

On the PyTorch workflow, basic NVTX markers are by default provided. On the C++/TensorRT workflow, append `--nvtx` when calling `scripts/build_wheel.py` script to compile, and clean build the code.

### Only collect specific iterations

To reduce the Nsight Systems profile size, and ensure that only specific iterations are collected, set environment variable `TLLM_PROFILE_START_STOP=A-B`, and append `-c cudaProfilerApi` to `nsys profile` command.


### Enable more NVTX markers for debugging

Set environment variable `TLLM_NVTX_DEBUG=1`.

### Enable garbage collection (GC) NVTX markers

Set environment variable `TLLM_PROFILE_RECORD_GC=1`.


### Enable GIL information in NVTX markers

Append “python-gil” to Nsys “-t” option.


## Coordinating with PyTorch profiler (PyTorch workflow only)

### Collect PyTorch profiler results

1. Set environment variable `TLLM_PROFILE_START_STOP=A-B` to specify the range of the iterations to be collected.
2. Set environment variable `TLLM_TORCH_PROFILE_TRACE=<path>`, and the results will be saved to `<path>`.

For VisualGen, use `TLLM_PROFILE_VISUAL_GEN_START_STOP` instead. Numeric ranges select
per-request denoise steps, while `predenoise`, `postdenoise`, and `all` select the
corresponding generation phases. For example:

```bash
TLLM_PROFILE_VISUAL_GEN_START_STOP=0-4 \
TLLM_TORCH_PROFILE_TRACE=/tmp/visual-gen-trace.json \
python examples/visual_gen/quickstart_example.py
```

Each process writes its trace to a rank-specific path such as
`/tmp/visual-gen-trace-rank-0.json`. If a process captures more than one
window, later traces add a window suffix such as
`/tmp/visual-gen-trace-rank-0-window-1.json`.

These ranges use the existing VisualGen CUDA/Nsight boundaries. `all` captures
the complete request from text encoding through VAE decode. `predenoise`
captures text encoding, latent preparation, and denoise-loop setup;
`postdenoise` captures VAE decode and the remaining request work.

Two contracts worth knowing: a numeric range never extends past the denoise loop
it selects (a stop index beyond the last step closes at the last step instead),
and on a pipeline that runs more than one denoise loop per request, the per-loop
modes apply to each loop — a numeric range writes one trace per loop, while
`postdenoise` arms after the first loop rather than the last. All windows open at
the pipeline's inference entry point, so executor-side request preparation falls
outside them even though it counts toward the reported `generation` latency.

For the per-mode specifics, see `parse_profile_range` in
`tensorrt_llm/_torch/visual_gen/profiler.py`.

### Visualize the PyTorch profiler results

Use [chrome://tracing/](chrome://tracing/) to inspect the saved profile.


## Examples

Consult the Nsight Systems User Guide for full overview of MPI-related options.

### Profiling specific iterations on a `trtllm-bench`/`trtllm-serve` run

Say we want to profile iterations 100 to 150 on a `trtllm-bench`/`trtllm-serve` run, we want to collect as much information as possible for debugging, such as GIL, debugging NVTX markers, etc:

```bash
#!/bin/bash

# Prepare dataset for the benchmark
trtllm-bench --model ${MODEL_PATH} \
    prepare-dataset \
    --output dataset.txt \
    token-norm-dist \
    --num-requests=${NUM_SAMPLES} \
    --input-mean=1000 --output-mean=1000 --input-stdev=0 --output-stdev=0

# Benchmark and profile
TLLM_PROFILE_START_STOP=100-150 nsys profile \
  -o trace -f true \
  -t 'cuda,nvtx,python-gil' -c cudaProfilerApi \
  --cuda-graph-trace node \
  -e TLLM_PROFILE_RECORD_GC=1,TLLM_LLMAPI_ENABLE_NVTX=1,TLLM_TORCH_PROFILE_TRACE=trace.json \
  --trace-fork-before-exec=true \
  trtllm-bench \ # or trtllm-serve command
    --model deepseek-ai/DeepSeek-V3 \
    --model_path ${MODEL_PATH} \
    throughput \
    --dataset /tmp/dataset.txt --warmup 0 \
    --backend pytorch \
    --streaming
```

The Nsight Systems reports will be saved to `trace.nsys-rep`. Use NVIDIA Nsight Systems application to open it.

The PyTorch profiler results will be saved to `trace.json`. Use [chrome://tracing/](chrome://tracing/) to inspect the saved profile.

## PyExecutor mixed-cycle host timing ledger

The overlap scheduler can optionally write one low-overhead JSONL record for
each completed PyExecutor host iteration. The ledger is intended to explain
mixed prefill/decode gaps before collecting a more intrusive system trace. It
is disabled by default and the disabled path does not read a clock, create a
thread, or open a file.

Set a path before launching `trtllm-serve`. Use the placeholders when running
more than one rank or process so writers never share a file:

```bash
TLLM_PYEXECUTOR_TIMING_LEDGER_PATH=/tmp/pyexecutor-{pid}-rank{rank}.jsonl \
trtllm-serve MODEL --config config.yaml
```

`TLLM_PYEXECUTOR_TIMING_LEDGER_QUEUE_SIZE` changes the bounded asynchronous
writer queue from its default of 8192 records. The executor never blocks when
the queue is full. The final `ledger_end` row reports `dropped_records`; do not
analyze an incomplete ledger as full workload coverage.

Every iteration row contains:

- `iteration_id`, monotonic `start_ns`/`end_ns`, `total_ns`, and mutually
  exclusive host durations for dependency waits, request update/admission,
  scheduling, KV/resource preparation, input packing, model/graph launch, and
  output handling. `other_ns` is the unclassified remainder;
  `phase_overflow_ns` must remain zero and exposes any accidental overlap.
- active, queued, admitted, context, and decode request counts plus scheduled
  tokens.
- whether a CUDA graph was selected, its batch and draft-length key, and the
  number of padded request rows.
- scheduled MTP draft slots. With overlap scheduling, acceptance becomes known
  while processing the preceding batch, so `completed_batch_iteration_id`
  identifies the batch for `completed_mtp_draft_tokens` and
  `completed_mtp_accepted_tokens`.

The durations are host-wall attribution from `time.perf_counter_ns`; the
instrumentation adds no CUDA events and never synchronizes the device.
Consequently, `graph_launch_ns` is the host span that enqueues an eager forward
or CUDA graph replay, not GPU execution time. `dependency_wait_ns` measures
only waits already required by the executor. Asynchronous GPU work can outlive
the host phase that launched it. Join the ledger to TensorRT-LLM iteration IDs
and use Nsight Systems when GPU overlap or stream-idle time is the question.

The initial implementation covers the single-stage overlap scheduler. Pipeline
parallel and `disable_overlap_scheduler=True` loops are not yet attributed.
Always compare an exact workload with the ledger disabled and enabled before
using it for timing diagnosis; the enabled path performs clock reads, builds a
small Python record, and enqueues it to a background writer.

## MoE Expert Load Balance Analysis (Perfect Router)

For Mixture-of-Experts (MoE) models, performance can vary significantly based on how tokens are routed to experts. Uneven expert load distribution can cause some GPUs to be overloaded while others are underutilized, leading to suboptimal throughput.

TensorRT-LLM provides the `ENABLE_PERFECT_ROUTER` environment variable to help analyze and isolate expert load balancing issues from kernel performance.

### What It Does

When enabled, this feature **bypasses the learned router** and replaces it with pre-computed, perfectly load-balanced routing logits. This creates an idealized scenario where tokens are distributed evenly across all experts and GPUs.

Key behaviors:
- The learned gate/router is still computed (to maintain realistic timing)
- The gate output is **discarded** and replaced with ideal balanced logits
- Logits are pre-computed and cached for common batch sizes to minimize overhead
- Works with all MoE backends (CUTLASS, TRTLLM, TRITON)

```{warning}
This feature is for **performance analysis only**. It produces **incorrect model outputs** because the learned router decisions are discarded. Never use this in production inference.
```

### When to Use It

Use `ENABLE_PERFECT_ROUTER` when you want to:

1. **Establish performance upper bounds**: Measure the theoretical best-case MoE throughput when expert loads are perfectly balanced.

2. **Isolate routing bottlenecks**: Compare performance with vs. without perfect routing to determine if the learned router is causing load imbalance issues.

3. **Test different load balancing strategies**: Validate that MoE kernels and communication patterns behave correctly with balanced loads before implementing custom routing logic.

4. **Benchmark kernel efficiency**: Remove routing variability to get consistent, reproducible kernel performance measurements.

### How to Enable

Set the environment variable before running your workload. This works with both `trtllm-bench` and `trtllm-serve`:

```bash
export ENABLE_PERFECT_ROUTER=1
```

### Example Workflow

```bash
# Step 1: Benchmark with normal (learned) routing
trtllm-bench ...
# or
trtllm-serve ...

# Step 2: Benchmark with perfect routing (upper bound)
ENABLE_PERFECT_ROUTER=1 trtllm-bench ...
# or
ENABLE_PERFECT_ROUTER=1 trtllm-serve ...

# Step 3: Compare the throughput numbers
# If perfect router shows >10% improvement, routing imbalance is significant
```

### Interpreting Results

| Scenario | Interpretation |
|----------|----------------|
| Similar performance with/without perfect router | Router load balancing is not a bottleneck; focus optimization efforts elsewhere |
| Significant improvement with perfect router | The learned router is causing load imbalance; consider router optimization or load balancing strategies |

### Supported Models

```{note}
This feature currently requires model-specific integration. The plumbing to support perfect routing must be added to each MoE model implementation. If you need this feature for a model that doesn't yet support it, you will need to add the integration following the pattern used in existing implementations.
```

```{note}
The perfect router logits are specifically designed for `RenormalizeMoeRoutingMethod` (TopK first, then Softmax). Models using other routing methods such as `DefaultMoeRoutingMethod` or `DeepSeekV3MoeRoutingMethod` would require adapting the logit generation logic to match their routing behavior.
```

Currently supported:
- GPT-OSS (uses `RenormalizeMoeRoutingMethod`)
- DeepSeek-V3 / DeepSeek-R1 (uses `DeepSeekV3MoeRoutingMethod`)
