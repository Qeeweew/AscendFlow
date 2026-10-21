"""Actual row-prefix bytes, complete NZ initialization and native correspondence."""

import pytest
from xdsl.dialects import arith
from xdsl.dialects.builtin import IndexType

from ascendflow import language as N
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.diagnostics import CompileError
from ascendflow.passes.structured_sync import synchronize_structure
from tests.fixtures.runtime_rows import fp32_padding, matrix_import, row_output
from tests.models.structured_events import EventModelError, check_compiled_events
from tests.support.structured import native_memory


@pytest.fixture(scope="module", params=["f16", "f32", "output"])
def rows(request):
    return synchronize_structure(
        row_output() if request.param == "output" else matrix_import(request.param)
    )


@pytest.mark.parametrize("count", (0, 1, 3, 4, 7, 15, 16, 17))
def test_bounded_rows_keep_exact_bytes_for_empty_partial_and_full_transfers(rows, count):
    checked = check_compiled_events(rows, {"n": count}, invocations=2)
    for command in checked.natives:
        for span in command.spans:
            if span.root.sym_name.data in {"A", "O"} and span.root.extent_count is not None:
                stride = 16 * (2 if span.root.ref.type.dtype.data == "f16" else 4)
                assert 0 <= span.start < span.end <= count * stride
        if command.occurrence.operation.native.data == "nd2nz_rows":
            itemsize = 2 if command.spans[0].root.ref.type.dtype.data == "f16" else 4
            reads = [s for s in command.spans if s.root.sym_name.data == "A"]
            writes = [s for s in command.spans if s.root.ref.type.bank.data == "L1"]
            assert sum(s.end - s.start for s in reads) == min(count, 16) * 16 * itemsize
            assert sum(s.end - s.start for s in writes) == 16 * 16 * itemsize


@pytest.mark.parametrize("runtime", [False, True])
@pytest.mark.parametrize("pitch", [19, 65536])
def test_fp32_padding_defines_all_nz_bytes_but_only_reads_real_gm_rows(runtime, pitch):
    from ascendflow import compile
    from ascendflow.dialects.io import parse
    from tests.models.structured_events import check_structured_events

    result = compile(fp32_padding(runtime=runtime, pitch=pitch))
    before = collect_native_memory(parse(result.before_ir))
    after = collect_native_memory(result.plan)
    for n in (0, 1, 3, 4):
        execution = check_structured_events(
            after, {"n": n}, logical_reference=before, invocations=2
        )
        for command in execution.natives:
            if command.occurrence.operation.native.data not in {"nd2nz_pad", "nd2nz_rows"}:
                continue
            actual_rows = min(n, 3) if runtime else 3
            reads = [s for s in command.spans if s.root.sym_name.data == "A"]
            writes = [s for s in command.spans if s.root.ref.type.bank.data == "L1"]
            assert [(s.start, s.end) for s in reads] == [
                (row * pitch * 4, row * pitch * 4 + 5 * 4) for row in range(actual_rows)
            ]
            assert [(s.start, s.end) for s in writes] == [(0, 16 * 16 * 4)]
    assert "asc_fill_l1" in result.asc and "asc_sync_pipe(PIPE_MTE2)" in result.asc
    if pitch > 65535:
        assert f"npu_row * {pitch * 4}" in result.asc
    else:
        assert f", {pitch}, 16, 1, 0);" in result.asc


def test_dynamic_row_capacity_is_an_input_contract_checked_by_the_independent_model():
    definition = matrix_import()
    unsafe = N.from_source(definition.source.replace("N.min(16,n)", "n"), disjoint_external=True)
    plan = synchronize_structure(unsafe)
    for n in (0, 8, 16):
        check_compiled_events(plan, {"n": n})
    with pytest.raises(EventModelError, match="row count.*capacity"):
        check_compiled_events(plan, {"n": 17})


@pytest.mark.parametrize("count", (-1, 17))
def test_row_count_cannot_be_negative_or_exceed_its_static_capacity(count):
    source = matrix_import().source.replace("N.min(16,n)", str(count))
    with pytest.raises(CompileError, match="static row count.*declared capacity"):
        synchronize_structure(N.from_source(source, disjoint_external=True))


@pytest.mark.parametrize("count", (0, 2))
def test_constant_row_counts_share_the_same_address_and_instruction_operand(count):
    source = row_output().source.replace("N.min(4,n)", str(count))
    source = source.replace("n:N.int64", "n:N.int64(minimum=2)")
    checked = check_compiled_events(
        synchronize_structure(N.from_source(source, disjoint_external=True)), {"n": 2}
    )
    spans = [
        span
        for command in checked.natives
        for span in command.spans
        if span.root.sym_name.data == "O"
    ]
    assert sum(span.end - span.start for span in spans) == count * 16 * 4


def test_empty_rows_do_not_legalize_undefined_original_scalar_arithmetic():
    source = row_output().source.replace("n:N.int64", "n:N.int64,d:N.int64")
    source = source.replace("N.min(4,n)", "(n//0)*0")
    with pytest.raises(CompileError, match="divisor|division"):
        synchronize_structure(N.from_source(source, disjoint_external=True))


def test_row_count_in_the_address_must_match_the_actual_native_operand():
    memory = native_memory(row_output())
    prefix = next(t for t in memory.templates if t.spec.native == "copy_rows_prefix")
    address = prefix.accesses[1].address.operation
    zero = arith.ConstantOp.from_int_and_width(0, IndexType())
    address.parent.insert_op_before(zero, address)
    address.operands = (*address.operands[:-1], zero.result)
    with pytest.raises(CompileError, match="address and instruction counts disagree"):
        collect_native_memory(memory.module)


def test_a_partial_output_write_does_not_initialize_its_static_upper_envelope():
    source = N.from_source(
        """
def partial(O:N.Tensor((4,16),N.f32,access="write"),n:N.int64):
    with N.Kernel():
        x=N.alloc_ub((4,16),N.f32)
        y=N.alloc_ub((4,16),N.f32)
        N.fill(x,1.0)
        N.copy(x,O,valid_rows=N.min(4,n))
        N.copy(O,y)
""",
        disjoint_external=True,
    )
    with pytest.raises(CompileError, match="initialized writer coverage"):
        synchronize_structure(source)


def test_nd2nz_single_row_preserves_nz_column_pitch_and_holes():
    from ascendflow import compile

    source = N.from_source("""
def fragment(A:N.Tensor((16,32),N.bf16)):
    with N.Kernel(kind="cube"):
        x=N.alloc_l1((16,32),N.bf16)
        N.copy(A[3:4,:],x[5:6,:])
""")
    result = compile(source)
    from ascendflow.dialects.io import parse
    from tests.models.structured_events import check_structured_events

    execution = check_structured_events(
        collect_native_memory(result.plan),
        logical_reference=collect_native_memory(parse(result.before_ir)),
        invocations=2,
    )
    writes = [s for c in execution.natives for s in c.spans if s.root.ref.type.bank.data == "L1"]
    assert [(s.start, s.end) for s in writes] == [(160, 192), (672, 704)] * 2
    assert "1, 1, 32, 0, 32, 16, 1, 0" in result.asc


def test_nd2nz_row_fragment_does_not_define_other_nz_rows():
    source = N.from_source("""
def fragment(A:N.Tensor((16,32),N.bf16)):
    with N.Kernel(kind="cube"):
        x=N.alloc_l1((16,32),N.bf16)
        y=N.alloc_l0a((16,32),N.bf16)
        N.copy(A[3:4,:],x[5:6,:])
        N.copy(x,y)
""")
    with pytest.raises(CompileError, match="initialized writer coverage"):
        synchronize_structure(source)


@pytest.mark.parametrize("column,accepted", [(0, True), (32, False), (64, False)])
def test_runtime_row_tiles_require_defined_columns_without_row_holes(column, accepted):
    source = N.from_source(f"""
def rows(A:N.Tensor((16,128),N.f32)):
    with N.Kernel():
        x=N.alloc_ub((16,128),N.f32)
        y=N.alloc_ub((4,64),N.f32)
        N.copy(A[:,0:64],x[:,0:64])
        for block in N.serial(4):
            N.copy(x[block*4:block*4+4,{column}:{column + 64}],y)
""")
    if accepted:
        plan = synchronize_structure(source)
        check_compiled_events(plan, invocations=2)
    else:
        with pytest.raises(CompileError, match="initialized writer coverage"):
            synchronize_structure(source)


@pytest.mark.parametrize("other_rows,accepted", [(16, True), (8, False), (0, False)])
def test_branch_merge_initialization_requires_both_complete_physical_writes(other_rows, accepted):
    other = f"N.copy(A[0:{other_rows},:],x[0:{other_rows},:])" if other_rows else "pass"
    source = N.from_source(f"""
def branches(A:N.Tensor((16,16),N.f32),choice:N.int64):
    with N.Kernel():
        x=N.alloc_ub((16,16),N.f32)
        y=N.alloc_ub((16,16),N.f32)
        if choice == 0:
            N.copy(A,x)
        else:
            {other}
        N.copy(x,y)
""")
    if accepted:
        plan = synchronize_structure(source)
        for choice in (0, 1):
            check_compiled_events(plan, {"choice": choice}, invocations=2)
    else:
        with pytest.raises(CompileError, match="initialized writer coverage"):
            synchronize_structure(source)
