"""Runtime NZ overlays preserve prior rows, holes and initialization duties."""

import pytest

from ascendflow import compile
from ascendflow import language as N
from ascendflow.diagnostics import CompileError
from ascendflow.passes.structured_sync import synchronize_structure
from tests.models.structured_events import EventModelError, check_compiled_events

SOURCE = """
def overlay(A:N.Tensor((32,64),N.bf16), B:N.Tensor((32,64),N.bf16), n:N.i64):
    with N.Kernel(kind="cube"):
        x=N.alloc_l1((32,64),N.bf16)
        y=N.alloc_l0a((32,64),N.bf16)
        count=N.min(n,27)
        N.copy(A,x)
        N.copy(B[:27,:],x[5:32,:],valid_rows=count)
        N.copy(x,y)
"""

INITIALIZE_SUBVIEW = """
def initialize(A:N.Tensor((16,64),N.bf16), B:N.Tensor((16,64),N.bf16), n:N.i64):
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
    plan = synchronize_structure(N.from_source(unsafe))
    check_compiled_events(plan, {"n": 27})
    with pytest.raises(EventModelError, match="row count|axis"):
        check_compiled_events(plan, {"n": 28})


def test_overlay_emits_runtime_nd2nz_without_a_second_clear():
    result = compile(N.from_source(SOURCE))
    assert "asc_fill_l1" not in result.asc
    assert "uint16_t(" in result.asc


FP32_COLUMN = SOURCE.replace("N.bf16", "N.f32").replace(
    "N.copy(B[:27,:],x[5:32,:],valid_rows=count)",
    "N.copy(B[:27,:8],x[5:32,8:16],valid_rows=count)",
)


def test_fp32_overlay_can_select_one_eight_column_block():
    plan = synchronize_structure(N.from_source(FP32_COLUMN))
    for count in (0, 1, 27):
        execution = check_compiled_events(plan, {"n": count}, invocations=2)
        for invocation in (0, 1):
            copy = next(
                c
                for c in execution.natives
                if c.occurrence.invocation == invocation
                and c.occurrence.operation.native.data == "nd2nz_overlay"
            )
            writes = [s for s in copy.spans if s.root.ref.type.bank.data == "L1"]
            assert [(s.start, s.end) for s in writes] == (
                [(1024 + 5 * 32, 1024 + (5 + count) * 32)] if count else []
            )
            reads = [s for s in copy.spans if s.root.sym_name.data == "B"]
            assert sum(s.end - s.start for s in reads) == count * 8 * 4


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


JOINED = """
def joined(A:N.Tensor((128,128),N.bf16), B:N.Tensor((128,128),N.bf16), n:N.i64):
    with N.Kernel(kind="cube"):
        x=N.alloc_l1((128,128),N.bf16)
        y=N.alloc_l0a((128,128),N.bf16)
        split=N.min(n,128)
        N.copy(A,x,valid_rows=split)
        N.copy(B,x[split:split+128,:],valid_rows=128-split)
        N.copy(x,y)
"""


@pytest.fixture(scope="module", params=[("bf16", 16), ("f32", 8)])
def joined(request):
    dtype, c0 = request.param
    return compile(N.from_source(JOINED.replace("N.bf16", f"N.{dtype}"))), c0


@pytest.mark.parametrize("count", [0, 1, 7, 63, 127, 128, 256])
def test_adjacent_runtime_fragments_define_nz_without_clearing(joined, count):
    from ascendflow.analysis.structured_memory import collect_native_memory
    from ascendflow.dialects.io import parse
    from tests.models.structured_events import check_structured_events

    joined, c0 = joined
    execution = check_structured_events(
        collect_native_memory(joined.plan),
        {"n": count},
        logical_reference=collect_native_memory(parse(joined.before_ir)),
        invocations=2,
    )
    assert "asc_fill_l1" not in joined.asc
    assert joined.report["analysis_work"]["admission"]["l1_region_transfers"] > 0
    split = min(count, 128)
    for invocation in (0, 1):
        copies = [
            c
            for c in execution.natives
            if c.occurrence.invocation == invocation
            and c.occurrence.operation.native.data == "nd2nz_overlay"
        ]
        for copy, start, end in zip(copies, (0, split), (split, 128), strict=True):
            writes = [s for s in copy.spans if s.root.ref.type.bank.data == "L1"]
            assert [(s.start, s.end) for s in writes] == (
                [(k * 4096 + start * 32, k * 4096 + end * 32) for k in range(128 // c0)]
                if start != end
                else []
            )


def test_a_hole_between_runtime_fragments_stays_undefined():
    source = JOINED.replace("N.min(n,128)", "N.min(n,127)").replace(
        "x[split:split+128,:],valid_rows=128-split",
        "x[split+1:split+129,:],valid_rows=127-split",
    )
    with pytest.raises(CompileError, match="initialized writer coverage"):
        compile(N.from_source(source))


def test_partial_window_cannot_initialize_the_full_cube_operand():
    source = JOINED.replace(
        "split=N.min(n,128)", "visible=N.min(n,128)\n        split=visible//2"
    ).replace("valid_rows=128-split", "valid_rows=visible-split")
    with pytest.raises(CompileError, match="initialized writer coverage"):
        compile(N.from_source(source))


def test_column_holes_are_not_defined_by_a_nz_bounding_box():
    source = JOINED.replace(
        "        split=N.min(n,128)\n        N.copy(A,x,valid_rows=split)\n"
        "        N.copy(B,x[split:split+128,:],valid_rows=128-split)",
        "        N.copy(A[:,:16],x[:,:16])\n        N.copy(B[:,32:128],x[:,32:128])",
    )
    with pytest.raises(CompileError, match="initialized writer coverage"):
        compile(N.from_source(source))


@pytest.mark.parametrize("dtype", ["bf16", "f32"])
def test_fixed_column_tiles_define_a_resident_plane_without_clear_or_unroll(dtype):
    from ascendflow.analysis.structured_memory import collect_native_memory
    from ascendflow.dialects.io import parse
    from tests.models.structured_events import check_structured_events

    source = JOINED.replace(
        "        split=N.min(n,128)\n        N.copy(A,x,valid_rows=split)\n"
        "        N.copy(B,x[split:split+128,:],valid_rows=128-split)",
        "        for chunk in N.Pipelined(4):\n"
        "            N.copy(A[:,chunk*32:(chunk+1)*32],x[:,chunk*32:(chunk+1)*32])",
    )
    result = compile(N.from_source(source.replace("N.bf16", f"N.{dtype}")))
    assert "asc_fill_l1" not in result.asc
    check_structured_events(
        collect_native_memory(result.plan),
        {"n": 0},
        logical_reference=collect_native_memory(parse(result.before_ir)),
        invocations=2,
    )


def test_l1_allocation_layout_cannot_be_changed_after_lowering():
    from xdsl.dialects.builtin import ArrayAttr, IntAttr

    from ascendflow.analysis.structured_memory import collect_native_memory
    from ascendflow.dialects.physical import BufferOp
    from ascendflow.passes.lower_native import lower_native_structure

    structure = lower_native_structure(N.from_source(JOINED))
    root = next(
        op for op in structure.module.walk() if isinstance(op, BufferOp) and op.matrix_shape
    )
    # Same byte capacity, different NZ leading dimension. Repeated AddressOp
    # axis metadata cannot silently give this allocation another layout.
    root.attributes["matrix_shape"] = ArrayAttr([IntAttr(64), IntAttr(256)])
    with pytest.raises(CompileError, match="allocation layout"):
        collect_native_memory(structure.module)


def test_conditional_plane_writes_compose_without_expanding_the_loop():
    from ascendflow.analysis.structured_memory import collect_native_memory
    from ascendflow.dialects.io import parse
    from tests.models.structured_events import check_structured_events

    source = """
def planes(A:N.Tensor((2,32,64),N.bf16)):
    with N.Kernel(kind="cube"):
        x=N.alloc_l1((2,32,64),N.bf16)
        y=N.alloc_l0a((32,64),N.bf16)
        for plane in N.serial(4):
            if plane<2:
                for chunk in N.Pipelined(4):
                    N.copy(A[plane,:,chunk*16:(chunk+1)*16],x[plane,:,chunk*16:(chunk+1)*16])
        for plane in N.serial(2):
            N.copy(x[plane,:,:],y)
"""
    result = compile(N.from_source(source))
    assert "asc_fill_l1" not in result.asc
    check_structured_events(
        collect_native_memory(result.plan),
        logical_reference=collect_native_memory(parse(result.before_ir)),
        invocations=2,
    )
    # A sibling branch cannot borrow the other branch's current-iteration
    # write merely because their enclosing loop has a fixed count.
    invalid = (
        source[: source.index("        for plane in N.serial(4):")]
        + """
        for plane in N.serial(2):
            if plane<1:
                N.copy(A[plane,:,:],x[plane,:,:])
            else:
                N.copy(x[plane,:,:],y)
"""
    )
    with pytest.raises(CompileError, match="initialized writer coverage"):
        compile(N.from_source(invalid))
