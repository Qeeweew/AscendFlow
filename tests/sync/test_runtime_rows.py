"""Actual row-prefix bytes, complete NZ initialization and native correspondence."""

import pytest
from xdsl.dialects import arith
from xdsl.dialects.builtin import IndexType

from ascendflow import language as N
from ascendflow.analysis.structured_bounds import admit_memory_bounds
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.diagnostics import CompileError
from ascendflow.passes.structured_sync import synchronize_structure
from tests.fixtures.runtime_rows import matrix_import, row_output
from tests.models.structured_events import check_compiled_events
from tests.support.structured import native_memory


@pytest.fixture(scope="module", params=[matrix_import, row_output])
def rows(request):
    return synchronize_structure(request.param())


@pytest.mark.parametrize("count", (0, 1, 3, 4, 7, 15, 16, 17))
def test_bounded_rows_keep_exact_bytes_for_empty_partial_and_full_transfers(rows, count):
    checked = check_compiled_events(rows, {"n": count}, invocations=2)
    for command in checked.natives:
        for span in command.spans:
            if span.root.sym_name.data in {"A", "O"} and span.root.extent_count is not None:
                stride = 32 if span.root.sym_name.data == "A" else 64
                assert 0 <= span.start < span.end <= count * stride


def test_runtime_rows_cannot_bypass_the_capacity_contract():
    definition = matrix_import()
    unsafe = N.from_source(definition.source.replace("N.min(16,n)", "n"), disjoint_external=True)
    with pytest.raises(CompileError, match="runtime row count.*declared capacity"):
        admit_memory_bounds(native_memory(unsafe))


@pytest.mark.parametrize("count", (-1, 17))
def test_row_count_cannot_be_negative_or_exceed_its_static_capacity(count):
    source = matrix_import().source.replace("N.min(16,n)", str(count))
    with pytest.raises(CompileError, match="runtime row count.*declared capacity"):
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
    source = source.replace("N.min(4,n)", "(n//d)*0")
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
