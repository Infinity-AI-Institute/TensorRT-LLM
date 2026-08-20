# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Probe cold versus warm Triton SSD behavior at captured C160 shapes.

This script intentionally loads only the Python/Triton Mamba kernel sources
from a TensorRT-LLM checkout. It does not need a built TensorRT-LLM wheel.
The default shapes mirror the two mixed iterations adjacent to the residual
JIT event in the C160 lifecycle profile.

Example:
    TRITON_CACHE_DIR=/tmp/c160-triton-cache \
      python scripts/c160_mamba_jit_probe.py --json /tmp/result.json
"""

import argparse
import gc
import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
import time
import types
from pathlib import Path

import torch

CHUNK_SIZE = 128
NHEADS = 128
HEAD_DIM = 64
NGROUPS = 8
DSTATE = 128


def _load_mamba_sources(source_root: Path):
    mamba_dir = source_root / "tensorrt_llm/_torch/modules/mamba"
    package_name = "_c160_mamba_jit_probe"
    package = types.ModuleType(package_name)
    package.__path__ = [str(mamba_dir)]
    sys.modules[package_name] = package
    modules = {}
    for name in (
        "softplus",
        "ssd_bmm",
        "ssd_chunk_scan",
        "ssd_chunk_state",
        "ssd_state_passing",
    ):
        modules[name] = importlib.import_module(f".{name}", package_name)
    return modules


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(source_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _cache_inventory(cache_dir: Path) -> dict:
    files = [path for path in cache_dir.rglob("*") if path.is_file()]
    return {
        "files": len(files),
        "bytes": sum(path.stat().st_size for path in files),
        "cubins": sorted(
            str(path.relative_to(cache_dir)) for path in files if path.suffix == ".cubin"
        ),
        "suffixes": {
            suffix: sum(1 for path in files if path.suffix == suffix)
            for suffix in sorted({path.suffix for path in files})
        },
    }


def _sequence_metadata(lengths: list[int], device: torch.device):
    cumulative = [0]
    for length in lengths:
        cumulative.append(cumulative[-1] + length)
    cu_seqlens = torch.tensor(cumulative, dtype=torch.int32, device=device)
    seq_idx = torch.repeat_interleave(
        torch.arange(len(lengths), dtype=torch.int32, device=device),
        torch.tensor(lengths, dtype=torch.int64, device=device),
    ).unsqueeze(0)

    total_seqlen = cumulative[-1]
    logical_chunks = math.ceil(total_seqlen / CHUNK_SIZE)
    logical_chunks += sum(1 for sequence_end in cumulative[1:-1] if sequence_end % CHUNK_SIZE > 0)
    chunk_indices = list(range(logical_chunks))
    chunk_offsets = [0] * logical_chunks
    insertions = 0
    for sequence_start, sequence_end in zip(cumulative[1:-1], cumulative[2:]):
        if sequence_start % CHUNK_SIZE > 0:
            insertions += 1
        start = sequence_start // CHUNK_SIZE + insertions
        end = sequence_end // CHUNK_SIZE + insertions
        if sequence_end % CHUNK_SIZE > 0:
            end += 1
        for index in range(start, end):
            chunk_indices[index] -= insertions
        chunk_offsets[start] = sequence_start % CHUNK_SIZE
    return (
        cu_seqlens,
        seq_idx,
        torch.tensor(chunk_indices, dtype=torch.int32, device=device),
        torch.tensor(chunk_offsets, dtype=torch.int32, device=device),
    )


def _run_ssd(modules, lengths: list[int]) -> dict:
    device = torch.device("cuda")
    seqlen = sum(lengths)
    nchunks = math.ceil(seqlen / CHUNK_SIZE)
    cu_seqlens, seq_idx, chunk_indices, chunk_offsets = _sequence_metadata(lengths, device)

    # Dtypes mirror the captured Nemotron-H configuration: BF16 model
    # activations, FP32 A/D/dt-bias parameters, and FP16 Mamba state cache.
    x = torch.zeros((1, seqlen, NHEADS, HEAD_DIM), dtype=torch.bfloat16, device=device)
    dt = torch.zeros((1, seqlen, NHEADS), dtype=torch.bfloat16, device=device)
    b = torch.zeros((1, seqlen, NGROUPS, DSTATE), dtype=torch.bfloat16, device=device)
    c = torch.zeros_like(b)
    a = torch.full((NHEADS,), -1.0, dtype=torch.float32, device=device)
    d = torch.ones((NHEADS,), dtype=torch.float32, device=device)
    dt_bias = torch.zeros((NHEADS,), dtype=torch.float32, device=device)
    initial_states = torch.zeros(
        (len(lengths), NHEADS, HEAD_DIM, DSTATE),
        dtype=torch.float16,
        device=device,
    )
    out = torch.empty_like(x)

    torch.cuda.synchronize()
    started = time.perf_counter()
    torch.cuda.nvtx.range_push(
        f"ssd:{seqlen}:{len(lengths)}:{','.join(str(length) for length in lengths)}"
    )
    try:
        d_a_cumsum, dt_out = modules["ssd_chunk_state"]._chunk_cumsum_fwd(
            dt,
            a,
            CHUNK_SIZE,
            dt_bias=dt_bias,
            dt_softplus=True,
        )
        states = modules["ssd_chunk_state"]._chunk_state_fwd(
            b, x, dt_out, d_a_cumsum, seq_idx=seq_idx, states_in_fp32=True
        )
        states_flat, final_states = modules["ssd_state_passing"]._state_passing_fwd(
            states.flatten(-2),
            d_a_cumsum,
            initial_states=initial_states.flatten(-2),
            seq_idx=seq_idx,
            chunk_size=CHUNK_SIZE,
            out_dtype=torch.float16,
            is_cont_batched=True,
            chunk_offsets=chunk_offsets,
        )
        states = states_flat.unflatten(-1, (HEAD_DIM, DSTATE))
        cb = modules["ssd_bmm"]._bmm_chunk_fwd(
            c,
            b,
            CHUNK_SIZE,
            seq_idx=seq_idx,
            output_dtype=torch.float32,
        )
        modules["ssd_chunk_scan"]._chunk_scan_fwd(
            cb,
            x,
            dt_out,
            d_a_cumsum,
            c,
            states,
            D=d,
            seq_idx=seq_idx,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            initial_states=initial_states,
            out=out,
        )
        varlen_states = modules["ssd_chunk_state"].chunk_state_varlen(
            b.squeeze(0),
            x.squeeze(0),
            dt_out.squeeze(0),
            d_a_cumsum.squeeze(0),
            cu_seqlens,
            states.squeeze(0),
            initial_states=initial_states,
        )
        torch.cuda.synchronize()
    finally:
        torch.cuda.nvtx.range_pop()
    wall_seconds = time.perf_counter() - started

    result = {
        "context_lengths": lengths,
        "context_tokens": seqlen,
        "context_requests": len(lengths),
        "physical_chunks": nchunks,
        "logical_chunks": len(chunk_indices),
        "wall_seconds": wall_seconds,
        "finite": bool(
            torch.isfinite(final_states).all().item()
            and torch.isfinite(varlen_states).all().item()
            and torch.isfinite(out).all().item()
        ),
    }
    del (
        x,
        dt,
        b,
        c,
        a,
        d,
        dt_bias,
        initial_states,
        out,
        d_a_cumsum,
        dt_out,
        states,
        states_flat,
        final_states,
        cb,
        varlen_states,
        cu_seqlens,
        seq_idx,
        chunk_indices,
        chunk_offsets,
    )
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--json", type=Path, required=True)
    return parser.parse_args()


def main():
    args = _parse_args()
    source_root = args.source_root.resolve()
    cache_dir = Path(os.environ.get("TRITON_CACHE_DIR", "~/.triton/cache")).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    modules = _load_mamba_sources(source_root)
    source_files = {
        name: {
            "path": str(source_root / f"tensorrt_llm/_torch/modules/mamba/{name}.py"),
            "sha256": _sha256(source_root / f"tensorrt_llm/_torch/modules/mamba/{name}.py"),
        }
        for name in (
            "ssd_bmm",
            "ssd_chunk_scan",
            "ssd_chunk_state",
            "ssd_state_passing",
        )
    }
    result = {
        "diagnostic_scope": "source-level Triton SSD; not an E2E throughput run",
        "source_root": str(source_root),
        "source_revision": _git_revision(source_root),
        "source_files": source_files,
        "torch_version": torch.__version__,
        "triton_version": importlib.import_module("triton").__version__,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "triton_cache_dir": str(cache_dir),
        "runs": [],
    }
    shapes = [
        # Exact least_requests=False split for max_batch_size=192:
        # 191 full sequences of 4096 // 191 tokens plus one remainder.
        ("existing-4096-warmup", [21] * 191 + [85]),
        ("captured-32288-cold", [8016, 8016, 8016, 8016, 224]),
        ("captured-32288-warm", [8016, 8016, 8016, 8016, 224]),
        ("captured-32255-cold", [8016, 8016, 8016, 8016, 191]),
        ("captured-32255-warm", [8016, 8016, 8016, 8016, 191]),
    ]
    for label, lengths in shapes:
        before = _cache_inventory(cache_dir)
        run = _run_ssd(modules, lengths)
        after = _cache_inventory(cache_dir)
        new_cubins = sorted(set(after["cubins"]) - set(before["cubins"]))
        run.update(
            {
                "label": label,
                "cache_before": before,
                "cache_after": after,
                "cache_new_files": after["files"] - before["files"],
                "cache_new_bytes": after["bytes"] - before["bytes"],
                "cache_new_cubins": new_cubins,
            }
        )
        result["runs"].append(run)
        print(json.dumps(run, sort_keys=True), flush=True)

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
