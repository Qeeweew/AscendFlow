"""Prefill profiling inputs preserve real cache rows and early causal masks."""

import csv
import ctypes as ct
import json
from decimal import Decimal

import numpy as np
import pytest

from benchmarks.attention.sparse_reference import gather_oracle


@pytest.mark.parametrize("ratio", (1, 2))
def test_full_prefill_measurement_inputs_match_paged_oracle(ratio):
    from benchmarks.attention.sparse_performance import make_inputs
    from benchmarks.validation.dtypes import decode

    policy = {
        "B": 1,
        "T": 8,
        "H": 2,
        "D": 4,
        "PAGE": 4,
        "RATIO": ratio,
        "MAX_SEQ": 16,
        "ORI_BLOCKS": 4,
        "CMP_BLOCKS": 4,
        "WINDOW": 3,
        "TOPK": 4,
        "S": 7,
        "DTYPE": "bf16",
        "SCALE": 0.5,
    }
    arrays, patches, valid = make_inputs(policy, 8, 41)
    data = dict(arrays)
    for name, page in (("ori_kv", 4), ("cmp_kv", 4 // ratio)):
        shape = (4, page, 1, 4)
        raw = bytearray(int(np.prod(shape)) * 2)
        for offset, value in patches[name]:
            raw[offset : offset + len(value)] = value
        data[name] = decode(bytes(raw), "bf16", shape)
    rows, bias = gather_oracle(data, policy)
    np.testing.assert_array_equal(np.isfinite(bias), valid)
    assert np.isfinite(rows).all()
    assert np.count_nonzero(valid[0]) == 1 + (ratio == 1)


def test_prefill_cache_patch_staging_preserves_zero_holes_and_guard():
    from benchmarks.attention.sparse_performance import GUARD, zero_cache

    class Memory:
        def allocate_uninitialized(self, size):
            self.raw = ct.create_string_buffer(size)
            return ct.c_void_p(ct.addressof(self.raw))

        def call(self, name, signature, *args):
            if name == "aclrtMemset":
                ct.memset(args[0], args[2], args[3])
            else:
                ct.memmove(args[0], args[2], args[3])

    runtime = Memory()
    size = 3 * 1024 * 1024 + 97
    patches = [(0, b"abc"), (1048575, b"cross-boundary"), (1048590, b"def"), (size - 3, b"xyz")]
    expected = bytearray(size)
    for offset, value in sorted(patches):
        expected[offset : offset + len(value)] = value
    pointer, _identity = zero_cache(runtime, size, patches)
    assert ct.string_at(pointer, size + len(GUARD)) == expected + GUARD


@pytest.mark.parametrize("reference_start", (150, 500))
def test_profiling_rejects_reference_tasks_after_timing_starts(tmp_path, reference_start):
    from ascendflow.diagnostics import CompileError
    from benchmarks.attention.profile_sparse import summarize

    keys = (
        "Op Name",
        "OP Type",
        "Task Type",
        "Task Start Time(us)",
        "Task Duration(us)",
        "Block Num",
    )
    with (tmp_path / "op_summary_0.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(keys)
        writer.writerows(
            [
                ("warmup", "sparse_mla", "MIX_AIC", 100, 10, 20),
                ("sample", "sparse_mla", "MIX_AIC", 200, 12, 20),
                ("reference", "BatchMatMul", "AI_CORE", reference_start, 30, 20),
            ]
        )
    with pytest.raises(CompileError, match="unexpected task during measured calls"):
        summarize(tmp_path, 1, 1, "ascendflow")


@pytest.mark.parametrize("gap_us", ("0.020", "-0.020"))
def test_gather_eviction_boundary_preserves_precision_at_epoch_timestamps(tmp_path, gap_us):
    from ascendflow.diagnostics import CompileError
    from benchmarks.attention.gather_dma import NAMES, summarize

    measure = tmp_path / "measure"
    profile = tmp_path / "profile"
    measure.mkdir()
    profile.mkdir()
    (measure / "device.json").write_text(
        json.dumps(
            {
                "status": "measured",
                "batches": [64],
                "warmup": 0,
                "repeats": 1,
                "query_tokens": 4,
                "topk": 512,
                "row_bytes": 1024,
                "launch_sequence": [{"kernel": name} for name in NAMES],
            }
        )
    )
    epoch = Decimal("1792645861338118.182")
    rows = [("state_sum", "AI_VECTOR_CORE", epoch - 1000, 10)]
    for i, name in enumerate(NAMES):
        start = epoch + i * 1000
        rows.extend(
            [
                ("state_sum", "AI_VECTOR_CORE", start, "764.376"),
                (name, "AI_VECTOR_CORE", start + Decimal("764.376") + Decimal(gap_us), 1),
            ]
        )
    with (profile / "op_summary_0.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(("OP Type", "Task Type", "Task Start Time(us)", "Task Duration(us)"))
        writer.writerows(rows)
    if gap_us.startswith("-"):
        with pytest.raises(CompileError, match="eviction overlapped gather"):
            summarize(tmp_path)
        assert not (tmp_path / "results.json").exists()
    else:
        summarize(tmp_path)
        results = json.loads((tmp_path / "results.json").read_text())["results"]
        assert len(results) == 6
        assert all(result["median_us"] == 1 for result in results)
