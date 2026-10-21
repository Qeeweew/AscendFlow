"""Runtime NZ overlays preserve prior rows, holes and initialization duties."""

import pytest

from ascendflow import compile
from ascendflow import language as N
from ascendflow.diagnostics import CompileError
from ascendflow.passes.structured_sync import synchronize_structure
from tests.models.structured_events import check_compiled_events

SOURCE = """
def overlay(A:N.Tensor((32,64),N.bf16), B:N.Tensor((32,64),N.bf16), n:N.int64):
    with N.Kernel(kind="cube"):
        x=N.alloc_l1((32,64),N.bf16)
        y=N.alloc_l0a((32,64),N.bf16)
        count=N.min(n,27)
        N.copy(A,x)
        N.copy(B[:27,:],x[5:32,:],valid_rows=count)
        N.copy(x,y)
"""

INITIALIZE_SUBVIEW = """
def initialize(A:N.Tensor((16,64),N.bf16), B:N.Tensor((16,64),N.bf16), n:N.int64):
    with N.Kernel(kind="cube"):
        x=N.alloc_l1((32,64),N.bf16)
        y=N.alloc_l0a((16,64),N.bf16)
        N.copy(A,x[:16,:],valid_rows=N.min(n,16))
        N.copy(B,x[:16,:])
        N.copy(x[:16,:],y)
"""


@pytest.fixture(scope="module")
def plan():
    return synchronize_structure(N.from_source(SOURCE))


@pytest.mark.parametrize("count", (0, 1, 3, 15, 16, 17, 27, 32))
def test_dynamic_row_prefix_has_different_gm_and_nz_physical_geometry(plan, count):
    execution = check_compiled_events(plan, {"n": count}, invocations=2)
    for invocation in (0, 1):
        transfer = next(
            c
            for c in execution.natives
            if c.occurrence.invocation == invocation
            and c.occurrence.operation.native.data == "nd2nz_overlay"
        )
        writes = [s for s in transfer.spans if s.root.ref.type.bank.data == "L1"]
        reads = [s for s in transfer.spans if s.root.sym_name.data == "B"]
        assert sum(s.end - s.start for s in reads) == min(count, 27) * 128
        assert [(s.start, s.end) for s in writes] == (
            [(block * 1024 + 160, block * 1024 + (5 + min(count, 27)) * 32) for block in range(4)]
            if count
            else []
        )


def test_partial_overlay_does_not_define_the_remaining_nz_rows():
    with pytest.raises(CompileError, match="initialized writer coverage"):
        synchronize_structure(N.from_source(SOURCE.replace("        N.copy(A,x)\n", "")))


def test_overlay_capacity_checks_real_rows_after_the_dynamic_destination_origin():
    unsafe = SOURCE.replace("N.min(n,27)", "N.min(n,28)")
    with pytest.raises(CompileError, match="axis|extent|bounds|capacity"):
        synchronize_structure(N.from_source(unsafe))


def test_overlay_emits_runtime_nd2nz_without_a_second_clear():
    result = compile(N.from_source(SOURCE))
    assert "asc_fill_l1" not in result.asc
    assert "uint16_t(" in result.asc


def test_complete_write_after_runtime_prefix_defines_the_same_pitched_subview():
    result = synchronize_structure(N.from_source(INITIALIZE_SUBVIEW))
    memory = result.admission.memory
    writes = [
        (template, access)
        for template in memory.templates
        for access in template.accesses
        if access.mode.writes and access.address.geometry.bank == "L1"
    ]
    assert len(writes) == 2
    (prefix, partial), (complete, full) = writes
    # This must collide in the old complete-definition deduplication key.
    assert prefix.frame == complete.frame
    assert partial.address.root is full.address.root
    assert partial.address.offset == full.address.offset
    assert partial.address.geometry == full.address.geometry
    assert partial.address.valid_rows is not None
    assert full.address.valid_rows is None
    read = next(
        read for read in result.admission.reads if read.access.address.geometry.bank == "L1"
    )
    assert read.writers == (complete.operation,)
    # The prefix still participates in the real WAW ordering and byte model.
    for count in (0, 1, 15, 16, 32):
        execution = check_compiled_events(result, {"n": count}, invocations=2)
        for invocation in (0, 1):
            transfer = next(
                c
                for c in execution.natives
                if c.occurrence.invocation == invocation
                and c.occurrence.operation.native.data == "nd2nz_overlay"
            )
            spans = [s for s in transfer.spans if s.root.ref.type.bank.data == "L1"]
            assert [(s.start, s.end) for s in spans] == (
                [(block * 1024, block * 1024 + min(count, 16) * 32) for block in range(4)]
                if count
                else []
            )


def test_runtime_prefix_alone_still_cannot_initialize_the_pitched_subview():
    source = INITIALIZE_SUBVIEW.replace("        N.copy(B,x[:16,:])\n", "")
    with pytest.raises(CompileError, match="initialized writer coverage"):
        synchronize_structure(N.from_source(source))
