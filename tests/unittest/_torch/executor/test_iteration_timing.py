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

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[4] / "tensorrt_llm/_torch/pyexecutor/iteration_timing.py"
)
_SPEC = importlib.util.spec_from_file_location("iteration_timing_standalone", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
iteration_timing = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = iteration_timing
_SPEC.loader.exec_module(iteration_timing)

ITERATION_TIMING_PATH_ENV = iteration_timing.ITERATION_TIMING_PATH_ENV
ITERATION_TIMING_QUEUE_SIZE_ENV = iteration_timing.ITERATION_TIMING_QUEUE_SIZE_ENV
IterationTiming = iteration_timing.IterationTiming
create_iteration_timing_ledger = iteration_timing.create_iteration_timing_ledger


def test_default_factory_is_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ITERATION_TIMING_PATH_ENV, raising=False)

    assert create_iteration_timing_ledger(rank=0, overlap_scheduler=True) is None


def test_record_captures_batch_graph_and_mtp_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    timing = IterationTiming(
        iteration_id=17, rank=2, overlap_scheduler=True, is_warmup=False, start_ns=100
    )
    scheduled = SimpleNamespace(
        num_context_requests=2,
        num_generation_requests=3,
        context_requests=[
            SimpleNamespace(context_chunk_size=10),
            SimpleNamespace(context_chunk_size=20),
        ],
        generation_requests=[
            SimpleNamespace(num_draft_tokens=3),
            SimpleNamespace(num_draft_tokens=3),
            SimpleNamespace(num_draft_tokens=0),
        ],
    )
    timing.set_batch_shape(scheduled, tokens_per_gen_step=4)
    timing.set_graph_shape(used=True, batch_size=160, draft_len=3, padding_rows=1)
    timing.set_completed_mtp(
        16,
        [
            SimpleNamespace(py_num_draft_tokens_verified=3, py_num_accepted_draft_tokens=2),
            SimpleNamespace(py_num_draft_tokens_verified=3, py_num_accepted_draft_tokens=1),
        ],
    )
    clock_values = iter((130, 200))
    monkeypatch.setattr(iteration_timing.time, "perf_counter_ns", clock_values.__next__)
    timing.add_elapsed("scheduler_ns", 110)

    record = timing.finish()

    assert record["scheduler_ns"] == 20
    assert record["total_ns"] == 100
    assert record["other_ns"] == 80
    assert record["phase_overflow_ns"] == 0
    assert record["scheduled_tokens"] == 42
    assert record["scheduled_mtp_draft_tokens"] == 6
    assert record["completed_batch_iteration_id"] == 16
    assert record["completed_mtp_draft_tokens"] == 6
    assert record["completed_mtp_accepted_tokens"] == 3
    assert record["cuda_graph_batch_size"] == 160
    assert record["graph_padding_rows"] == 1


def test_async_jsonl_writer_expands_rank_and_reports_drops(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_template = str(tmp_path / "timing-rank{rank}.jsonl")
    monkeypatch.setenv(ITERATION_TIMING_PATH_ENV, output_template)
    monkeypatch.setenv(ITERATION_TIMING_QUEUE_SIZE_ENV, "4")
    ledger = create_iteration_timing_ledger(rank=3, overlap_scheduler=True)
    assert ledger is not None

    timing = ledger.start_iteration(iteration_id=9, is_warmup=False)
    timing.active_requests = 160
    ledger.emit(timing)
    ledger.close()

    records = [
        json.loads(line) for line in (tmp_path / "timing-rank3.jsonl").read_text().splitlines()
    ]
    assert records[0]["schema"] == "trtllm_pyexecutor_host_timing/v1"
    assert records[0]["iteration_id"] == 9
    assert records[0]["active_requests"] == 160
    assert records[-1] == {
        "schema": "trtllm_pyexecutor_host_timing/v1",
        "record_type": "ledger_end",
        "rank": 3,
        "dropped_records": 0,
    }


def test_queue_size_must_be_positive(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ITERATION_TIMING_PATH_ENV, str(tmp_path / "timing.jsonl"))
    monkeypatch.setenv(ITERATION_TIMING_QUEUE_SIZE_ENV, "0")

    with pytest.raises(ValueError, match="must be greater than zero"):
        create_iteration_timing_ledger(rank=0, overlap_scheduler=True)
