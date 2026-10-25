"""Packed GM matrix blocks preserve their real row gaps and NZ layout."""

import pytest

from ascendflow import compile
from ascendflow import language as T
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.diagnostics import CompileError
from ascendflow.dialects.io import parse
from tests.models.structured_events import check_structured_events


@pytest.mark.parametrize("padded", (False, True))
def test_nz_import_reads_exact_column_blocks(padded):
    size = 64 if padded else 32
    padding = ",pad_value=0" if padded else ""
    definition = T.from_source(f"""
def packed(W:T.Tensor((2,64,64),T.bf16,layout="NZ")):
    with T.Kernel(kind="cube"):
        b1=T.alloc_l1(({size},{size}),T.bf16)
        b0=T.alloc_l0b(({size},{size}),T.bf16)
        value=T.copy(W[1,16:48,16:48],b1{padding})
        T.copy(value,b0)
""")
    result = compile(definition)
    memory = collect_native_memory(result.plan)
    before = collect_native_memory(parse(result.before_ir))
    checked = check_structured_events(memory, logical_reference=before, invocations=2)
    copies = [
        c for c in checked.natives if c.occurrence.operation.native.data.startswith("copy_nz")
    ]
    assert len(copies) == 2
    for command in copies:
        reads = [s for s in command.spans if s.root.sym_name.data == "W"]
        assert [(s.start, s.end) for s in reads] == [(10752, 11776), (12800, 13824)]
        writes = [s for s in command.spans if s.root.ref.type.bank.data == "L1"]
        assert sum(s.end - s.start for s in writes) == size * size * 2
    assert "asc_copy_gm2l1((__cbuf__ void*)" in result.asc
    assert "asc_copy_gm2l1_nd2nz(" not in result.asc
    assert ("asc_fill_l1" in result.asc) == padded


@pytest.mark.parametrize("resident_columns", (512, 768))
def test_prepacked_nz_wide_panel_uses_raw_dma_limits(resident_columns):
    result = compile(
        T.from_source(f"""
def packed(W:T.Tensor((1024,1024),T.bf16,layout="NZ")):
    with T.Kernel(kind="cube"):
        b1=T.alloc_l1((128,{resident_columns}),T.bf16)
        b0=T.alloc_l0b((128,32),T.bf16)
        value=T.copy(W[128:256,256:768],b1,pad_value=0)
        T.copy(value[:,:32],b0)
""")
    )
    checked = check_structured_events(
        collect_native_memory(result.plan),
        logical_reference=collect_native_memory(parse(result.before_ir)),
    )
    copy = next(
        c for c in checked.natives if c.occurrence.operation.native.data.startswith("copy_nz")
    )
    reads = [s for s in copy.spans if s.root.sym_name.data == "W"]
    assert [(s.start, s.end) for s in reads] == [
        (528384 + block * 32768, 532480 + block * 32768) for block in range(32)
    ]
    assert "uint16_t(32), uint16_t(128), uint16_t(896), uint16_t(0)" in result.asc


@pytest.mark.parametrize("shape", ("(15,16)", "(16,17)", "(16,)", "(n,16)"))
def test_external_nz_rejects_unrepresentable_layout(shape):
    with pytest.raises(CompileError):
        compile(
            T.from_source(f"""
def packed(W:T.Tensor({shape},T.bf16,layout="NZ"),n:T.i64):
    with T.Kernel(kind="cube"):
        pass
""")
        )


def test_external_nz_has_no_runtime_row_conversion():
    with pytest.raises(CompileError, match="valid_rows requires"):
        compile(
            T.from_source("""
def packed(W:T.Tensor((16,16),T.bf16,layout="NZ"),n:T.i64):
    with T.Kernel(kind="cube"):
        b1=T.alloc_l1((16,16),T.bf16)
        T.copy(W,b1,valid_rows=T.min(n,16),pad_value=0)
""")
        )


@pytest.mark.parametrize("dtype,access", (("T.f32", "read"), ("T.bf16", "write")))
def test_external_nz_requires_immutable_b16(dtype, access):
    with pytest.raises(CompileError, match="external NZ"):
        compile(
            T.from_source(f'''
def packed(W:T.Tensor((16,16),{dtype},layout="NZ",access="{access}")):
    with T.Kernel(kind="cube"):
        pass
''')
        )
