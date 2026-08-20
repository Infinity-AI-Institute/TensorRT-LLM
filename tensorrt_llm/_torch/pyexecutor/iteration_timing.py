# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

ITERATION_TIMING_PATH_ENV: Final = "TLLM_PYEXECUTOR_TIMING_LEDGER_PATH"
ITERATION_TIMING_QUEUE_SIZE_ENV: Final = "TLLM_PYEXECUTOR_TIMING_LEDGER_QUEUE_SIZE"
_DEFAULT_QUEUE_SIZE: Final = 8192
_WRITER_JOIN_TIMEOUT_SECONDS: Final = 5
_SCHEMA: Final = "trtllm_pyexecutor_host_timing/v1"
_PHASE_NAMES: Final = (
    "dependency_wait_ns",
    "request_update_admission_ns",
    "scheduler_ns",
    "kv_resource_prepare_ns",
    "input_pack_ns",
    "graph_launch_ns",
    "output_handling_ns",
)


@dataclass(slots=True)
class IterationTiming:
    """Mutable timing record for one PyExecutor host iteration.

    All timestamps use ``time.perf_counter_ns``. Phase spans are host-wall
    intervals and must not overlap. They intentionally do not synchronize CUDA.
    """

    iteration_id: int
    rank: int
    overlap_scheduler: bool
    is_warmup: bool
    start_ns: int = field(default_factory=time.perf_counter_ns)
    phase_ns: dict[str, int] = field(default_factory=lambda: {name: 0 for name in _PHASE_NAMES})
    admitted_requests: int = 0
    active_requests: int = 0
    queued_requests: int = 0
    waiting_requests: int = 0
    inbound_requests: int = 0
    context_requests: int = 0
    decode_requests: int = 0
    scheduled_tokens: int = 0
    scheduled_mtp_draft_tokens: int = 0
    completed_batch_iteration_id: int | None = None
    completed_mtp_draft_tokens: int | None = None
    completed_mtp_accepted_tokens: int | None = None
    cuda_graph_used: bool = False
    cuda_graph_batch_size: int | None = None
    cuda_graph_draft_len: int | None = None
    graph_padding_rows: int = 0

    def add_elapsed(self, phase: str, start_ns: int) -> None:
        """Add one non-overlapping host interval to ``phase``."""
        self.phase_ns[phase] += time.perf_counter_ns() - start_ns

    def set_batch_shape(self, scheduled_requests, tokens_per_gen_step: int) -> None:
        """Snapshot request counts and scheduled token work before forward."""
        self.context_requests = scheduled_requests.num_context_requests
        self.decode_requests = scheduled_requests.num_generation_requests
        context_tokens = sum(
            request.context_chunk_size for request in scheduled_requests.context_requests
        )
        self.scheduled_tokens = context_tokens + self.decode_requests * tokens_per_gen_step
        self.scheduled_mtp_draft_tokens = sum(
            max(request.num_draft_tokens, 0) for request in scheduled_requests.generation_requests
        )

    def set_graph_shape(
        self, *, used: bool, batch_size: int | None, draft_len: int | None, padding_rows: int
    ) -> None:
        """Record the selected CUDA graph bucket without touching the device."""
        self.cuda_graph_used = used
        self.cuda_graph_batch_size = batch_size
        self.cuda_graph_draft_len = draft_len
        self.graph_padding_rows = padding_rows

    def set_completed_mtp(self, batch_iteration_id: int, sample_requests) -> None:
        """Snapshot MTP verification counters after sampler request update."""
        drafted = 0
        accepted = 0
        for request in sample_requests:
            drafted += max(int(getattr(request, "py_num_draft_tokens_verified", 0)), 0)
            accepted += max(int(getattr(request, "py_num_accepted_draft_tokens", 0)), 0)
        self.completed_batch_iteration_id = batch_iteration_id
        self.completed_mtp_draft_tokens = drafted
        self.completed_mtp_accepted_tokens = accepted

    def finish(self) -> dict[str, object]:
        """Freeze the record into a JSON-serializable dictionary."""
        end_ns = time.perf_counter_ns()
        total_ns = end_ns - self.start_ns
        phase_sum_ns = sum(self.phase_ns.values())
        return {
            "schema": _SCHEMA,
            "clock": "time.perf_counter_ns",
            "iteration_id": self.iteration_id,
            "rank": self.rank,
            "overlap_scheduler": self.overlap_scheduler,
            "is_warmup": self.is_warmup,
            "start_ns": self.start_ns,
            "end_ns": end_ns,
            "total_ns": total_ns,
            **self.phase_ns,
            "phase_sum_ns": phase_sum_ns,
            "other_ns": max(total_ns - phase_sum_ns, 0),
            "phase_overflow_ns": max(phase_sum_ns - total_ns, 0),
            "admitted_requests": self.admitted_requests,
            "active_requests": self.active_requests,
            "queued_requests": self.queued_requests,
            "waiting_requests": self.waiting_requests,
            "inbound_requests": self.inbound_requests,
            "context_requests": self.context_requests,
            "decode_requests": self.decode_requests,
            "scheduled_tokens": self.scheduled_tokens,
            "scheduled_mtp_draft_tokens": self.scheduled_mtp_draft_tokens,
            "completed_batch_iteration_id": self.completed_batch_iteration_id,
            "completed_mtp_draft_tokens": self.completed_mtp_draft_tokens,
            "completed_mtp_accepted_tokens": self.completed_mtp_accepted_tokens,
            "cuda_graph_used": self.cuda_graph_used,
            "cuda_graph_batch_size": self.cuda_graph_batch_size,
            "cuda_graph_draft_len": self.cuda_graph_draft_len,
            "graph_padding_rows": self.graph_padding_rows,
        }


class IterationTimingLedger:
    """Asynchronously write opt-in PyExecutor host timing records as JSONL."""

    _STOP: Final = object()

    def __init__(
        self,
        path: str,
        *,
        rank: int,
        overlap_scheduler: bool,
        queue_size: int = _DEFAULT_QUEUE_SIZE,
    ) -> None:
        self.rank = rank
        self.overlap_scheduler = overlap_scheduler
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8", buffering=1)
        self._queue: queue.Queue[dict[str, object] | object] = queue.Queue(maxsize=queue_size)
        self._dropped_records = 0
        self._closed = False
        self._writer_error: OSError | None = None
        self._writer = threading.Thread(
            target=self._write_loop, daemon=True, name="pyexecutor_timing_ledger"
        )
        self._writer.start()

    @property
    def dropped_records(self) -> int:
        return self._dropped_records

    @property
    def writer_error(self) -> OSError | None:
        return self._writer_error

    def start_iteration(self, iteration_id: int, is_warmup: bool) -> IterationTiming:
        return IterationTiming(
            iteration_id=iteration_id,
            rank=self.rank,
            overlap_scheduler=self.overlap_scheduler,
            is_warmup=is_warmup,
        )

    def emit(self, timing: IterationTiming) -> None:
        """Enqueue a completed record without blocking the executor thread."""
        if self._closed:
            return
        try:
            self._queue.put_nowait(timing.finish())
        except queue.Full:
            self._dropped_records += 1

    def close(self) -> None:
        """Flush queued records and write a terminal drop-count record."""
        if self._closed:
            return
        self._closed = True
        while True:
            try:
                self._queue.put_nowait(self._STOP)
                break
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    continue
                self._dropped_records += 1
        self._writer.join(timeout=_WRITER_JOIN_TIMEOUT_SECONDS)

    def _write_loop(self) -> None:
        try:
            while True:
                record = self._queue.get()
                if record is self._STOP:
                    break
                self._file.write(json.dumps(record, separators=(",", ":")) + "\n")
            self._file.write(
                json.dumps(
                    {
                        "schema": _SCHEMA,
                        "record_type": "ledger_end",
                        "rank": self.rank,
                        "dropped_records": self._dropped_records,
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
        except OSError as error:
            self._writer_error = error
        finally:
            self._file.close()


def create_iteration_timing_ledger(
    *, rank: int, overlap_scheduler: bool
) -> IterationTimingLedger | None:
    """Create the opt-in ledger, or return ``None`` for the default no-op."""
    path_template = os.environ.get(ITERATION_TIMING_PATH_ENV)
    if not path_template:
        return None
    queue_size = int(os.environ.get(ITERATION_TIMING_QUEUE_SIZE_ENV, _DEFAULT_QUEUE_SIZE))
    if queue_size <= 0:
        raise ValueError(f"{ITERATION_TIMING_QUEUE_SIZE_ENV} must be greater than zero")
    path = path_template.format(rank=rank, pid=os.getpid())
    return IterationTimingLedger(
        path, rank=rank, overlap_scheduler=overlap_scheduler, queue_size=queue_size
    )
