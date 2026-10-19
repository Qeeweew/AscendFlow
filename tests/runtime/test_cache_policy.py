"""Runtime cache aliases retain the original physical allocation identity."""

from pathlib import Path

import pytest
from xdsl.dialects.builtin import StringAttr

from ascendflow import language as T
from ascendflow.analysis.structured_bounds import admit_memory_bounds
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.build import build_file
from ascendflow.diagnostics import CompileError
from ascendflow.dialects.io import parse
from ascendflow.dialects.physical import BufferOp
from examples.gemm.dense import dense_gemm


def test_cache_alias_binding_changes_launcher_and_preserves_kernel_and_storage(tmp_path):
    artifacts = {}
    source = Path(__file__).parents[2] / "examples/gemm/dense.py"
    for cache in ("normal", "bypass"):
        directory = tmp_path / cache
        build_file(
            source,
            directory,
            entry="dense_gemm",
            parameters={
                "M": 64,
                "N": 192,
                "K": 32,
                "BM": 16,
                "BN": 16,
                "BK": 32,
                "CK": 16,
                "GROUPS": 3,
                "OUTPUT_CACHE": cache,
            },
        )
        memory = collect_native_memory(parse((directory / "after.xdsl").read_text()))
        roots = [op for op in memory.module.walk() if isinstance(op, BufferOp) and op.external.data]
        assert len(roots) == 3
        output = next(op for op in roots if op.sym_name.data == "C")
        assert (output.cache.data if output.cache else "normal") == cache
        accesses = [
            a
            for native in memory.templates
            for a in native.accesses
            if a.address.root is output.ref
        ]
        assert accesses and all(a.address.operation.source is output.ref for a in accesses)
        artifacts[cache] = (directory / "dense_gemm.asc").read_text()
    assert artifacts["normal"].split("\n}\n", 1)[0] == artifacts["bypass"].split("\n}\n", 1)[0]
    assert "rtGetL2CacheOffset" not in artifacts["normal"]
    assert "rtGetL2CacheOffset(uint32_t(npu_device), &npu_l2_offset)" in artifacts["bypass"]
    assert "if (npu_status != 0) return npu_status;" in artifacts["bypass"]
    assert "uintptr_t(arg_C) + npu_l2_offset" in artifacts["bypass"]


def test_cache_bypass_is_not_a_local_storage_policy():
    from ascendflow.passes.lower_native import lower_native_structure

    memory = collect_native_memory(lower_native_structure(dense_gemm.specialize()).module)
    local = next(
        op for op in memory.module.walk() if isinstance(op, BufferOp) and not op.external.data
    )
    local.attributes["cache"] = StringAttr("bypass")
    with pytest.raises(CompileError, match="GM bypass"):
        admit_memory_bounds(memory)


def test_unknown_cache_policy_is_rejected_without_a_default():
    source = T.from_source("""
        def invalid(A:T.Tensor((32,32),T.f16,cache="unknown")):
            with T.Kernel(1,kind="cube"):
                x=T.alloc_l1((32,32),T.f16)
                T.copy(A,x)
    """)
    with pytest.raises(CompileError, match="cache policy"):
        source.to_ir()
