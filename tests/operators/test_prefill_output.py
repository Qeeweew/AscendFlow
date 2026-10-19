"""Bulk output publication preserves both products and each AIV's real rows."""

import pytest

from ascendflow import language as N
from ascendflow.diagnostics import CompileError
from ascendflow.passes.structured_sync import synchronize_structure
from ascendflow.target.ascend910b import Actor
from examples.kda.prefill_output import prefill_output
from tests.models.structured_events import _Interpret, check_compiled_events


@pytest.fixture(scope="module")
def output():
    return synchronize_structure(prefill_output.specialize(CHUNKS=4, NC=4, H=1, GROUPS=1))


@pytest.mark.parametrize("tail", [1, 13, 32, 33, 47, 64])
def test_bulk_output_covers_full_chunks_and_zero_or_partial_peer_tail(output, tail, monkeypatch):
    native = _Interpret.native

    def metadata(self, operation, values, invocation, group, actor, coordinates):
        native(self, operation, values, invocation, group, actor, coordinates)
        if operation.native.data == "index_load":
            root, offset, _ = values[operation.refs[0]]
            chunk = offset // 8
            assert root.sym_name.data in {"Starts", "Lengths"} and 0 <= chunk < 4
            values[operation.value] = (
                chunk * 64 if root.sym_name.data == "Starts" else tail if chunk == 3 else 64
            )

    monkeypatch.setattr(_Interpret, "native", metadata)
    checked = check_compiled_events(output, {"TOKENS": 192 + tail}, max_commands=20000)
    assert checked.bytes_read and checked.bytes_written and checked.consumptions
    assert output.bounds.local_bytes[Actor.AIC, "L0C"] == 128 * 1024
    assert output.bounds.local_bytes[Actor.AIV0, "UB"] <= 192 * 1024


@pytest.mark.parametrize("row,column,height", [(0, 8, 32), (16, 0, 16)])
def test_accumulator_slices_reject_unaligned_columns_and_partial_rows(row, column, height):
    source = N.from_source(f"""
def invalid(A:N.Tensor(({height},16),N.f16),B:N.Tensor((16,16),N.f16),
            Out:N.Tensor(({height},16),N.f32,access="write")):
    with N.Kernel(1,kind="cube"):
        a1=N.alloc_l1(({height},16),N.f16)
        b1=N.alloc_l1((16,16),N.f16)
        a0=N.alloc_l0a(({height},16),N.f16)
        b0=N.alloc_l0b((16,16),N.f16)
        c0=N.alloc_l0c((32,32),N.f32)
        N.copy(A,a1)
        N.copy(B,b1)
        N.copy(a1,a0)
        N.copy(b1,b0)
        N.matmul(a0,b0,accum=c0[{row}:{row}+{height},{column}:{column}+16],initialize=True)
        N.copy(c0[{row}:{row}+{height},{column}:{column}+16],Out)
""")
    with pytest.raises(CompileError, match="fractal"):
        synchronize_structure(source)
