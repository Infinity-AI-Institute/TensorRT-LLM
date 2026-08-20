# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch


SOURCE = (
    Path(__file__).parents[5]
    / "tensorrt_llm/_torch/modules/mamba/replay_selective_state_update.py"
)
BASELINE_TABLE_SHA256 = "03867379fdcd69630512d319b2eadf7928d28eeb48ab2f57b674ac7e1544fab3"


def _table():
    module = ast.parse(SOURCE.read_text())
    for node in module.body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "_DEFAULT_TUNING":
            return ast.literal_eval(node.value)
    raise AssertionError("_DEFAULT_TUNING not found")


def _resolver_namespace():
    module = ast.parse(SOURCE.read_text())
    selected = []
    for node in module.body:
        if isinstance(node, ast.Import) and any(alias.name == "os" for alias in node.names):
            selected.append(node)
        elif isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "_DEFAULT_TUNING":
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(
            getattr(target, "id", None) == "_ISSUE18_TUNING_ENV" for target in node.targets
        ):
            selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in (
            "_issue18_tuning_enabled",
            "_resolve_tuning",
        ):
            selected.append(node)
    namespace = {}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


def _canonical_digest(table):
    payload = [[key[0], key[1], table[key]] for key in sorted(table)]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def test_issue18_adds_only_the_new_high_batch_cell():
    table = _table()
    entries = table[("fp16", "SR")]
    candidate = entries[-1]
    assert candidate[0] == 32768
    assert candidate[1] == "persistent_main"
    assert candidate[2]["_block_size_m_nowrite"] == 32
    assert candidate[2]["_heads_per_block"] == 4

    baseline = dict(table)
    baseline[("fp16", "SR")] = entries[:-1]
    assert _canonical_digest(baseline) == BASELINE_TABLE_SHA256


def test_issue18_threshold_selection_and_stock_rollback():
    resolve = _resolver_namespace()["_resolve_tuning"]

    _, knobs = resolve(1024, 16, "fp16", "SR")
    assert knobs["_block_size_m_nowrite"] == 64
    assert knobs["_heads_per_block"] == 8

    for raw_batch in (160, 192):
        _, knobs = resolve(raw_batch, 128, "fp16", "SR")
        assert knobs["_block_size_m_nowrite"] == 32
        assert knobs["_heads_per_block"] == 4

    with patch.dict(os.environ, {"TRTLLM_REPLAY_SSU_B300_TP1_TUNING": "0"}):
        _, rollback_knobs = resolve(192, 128, "fp16", "SR")
        assert rollback_knobs["_block_size_m_nowrite"] == 64
        assert rollback_knobs["_heads_per_block"] == 8

    with patch.dict(os.environ, {"TRTLLM_REPLAY_SSU_B300_TP1_TUNING": "invalid"}):
        try:
            resolve(192, 128, "fp16", "SR")
        except ValueError as error:
            assert "must be 0 or 1" in str(error)
        else:
            raise AssertionError("invalid rollback selector did not fail closed")
