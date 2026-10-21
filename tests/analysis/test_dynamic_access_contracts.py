"""Legal dynamic source accesses need no universal range proof.

The independent finite model executes actual metadata and checks byte accesses,
versions, and asynchronous retirement against the pre-sync native structure.
"""

import pytest
from xdsl.utils.exceptions import VerifyException

from ascendflow import compile
from ascendflow import language as N
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.diagnostics import CompileError, ErrorKind
from ascendflow.dialects.io import parse
from tests.models.structured_events import _Interpret, check_structured_events


def execution(monkeypatch, definition, inputs, parameters=None):
    original = _Interpret.native

    def metadata(self, operation, values, *args):
        original(self, operation, values, *args)
        if operation.native.data == "index_load":
            root, offset, _ = values[operation.refs[0]]
            width = self.templates[operation].accesses[0].address.geometry.nbytes
            values[operation.value] = inputs[root.sym_name.data][offset // width]

    monkeypatch.setattr(_Interpret, "native", metadata)
    result = compile(N.from_source(definition, disjoint_external=True))
    trace = check_structured_events(
        collect_native_memory(result.plan),
        parameters,
        logical_reference=collect_native_memory(parse(result.before_ir)),
    )
    return result, trace


def reads(trace, root):
    return [
        (span.start, span.end)
        for command in trace.natives
        for span in command.spans
        if span.root.sym_name.data == root and "R" in span.mode
    ]


@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.parametrize("split", [False, True])
def test_complete_local_initialization_needs_no_dynamic_read_bounds(monkeypatch, index, split):
    initialize = (
        "        N.fill(x[0,:],1.0)\n        N.fill(x[1,:],2.0)"
        if split
        else "        N.fill(x,1.0)"
    )
    _, trace = execution(
        monkeypatch,
        f"""
def local(I:N.Tensor((1,),N.i32), O:N.Tensor((64,),N.f32,access="write")):
    with N.Kernel():
        x=N.alloc_ub((2,64),N.f32)
{initialize}
        row=N.load_index(I)
        N.copy(x[row,:],O)
""",
        {"I": [index]},
    )
    local_reads = [
        (span.start, span.end)
        for command in trace.natives
        for span in command.spans
        if span.root.ref.type.bank.data == "UB" and "R" in span.mode
    ]
    assert local_reads == [(index * 256, (index + 1) * 256)]


def test_partial_local_initialization_does_not_define_arbitrary_dynamic_reads():
    with pytest.raises(CompileError, match="initialized writer coverage"):
        compile(
            N.from_source(
                """
def local(I:N.Tensor((1,),N.i32), O:N.Tensor((64,),N.f32,access="write")):
    with N.Kernel():
        x=N.alloc_ub((2,64),N.f32)
        N.fill(x[0,:],1.0)
        row=N.load_index(I)
        N.copy(x[row,:],O)
""",
                disjoint_external=True,
            )
        )


@pytest.mark.parametrize("index", [-1, 0, 31])
def test_unhinted_signed_metadata_preserves_guarded_gather(monkeypatch, index):
    result, trace = execution(
        monkeypatch,
        """
def gather(X:N.Tensor((32,64),N.f32), I:N.Tensor((1,),N.i32),
           O:N.Tensor((64,),N.f32,access="write")):
    with N.Kernel():
        row=N.alloc_ub((64,),N.f32)
        N.fill(row,0.0)
        index=N.load_index(I)
        if index >= 0:
            N.copy(X[index,:],row)
        N.copy(row,O)
""",
        {"I": [index]},
    )
    assert reads(trace, "I") == [(0, 4)]
    assert reads(trace, "X") == ([] if index < 0 else [(index * 256, (index + 1) * 256)])
    assert "volatile int32_t" in result.asc


def test_data_dependent_product_and_divisor_are_executable(monkeypatch):
    _, trace = execution(
        monkeypatch,
        """
def gather(X:N.Tensor((32,64),N.f32), I:N.Tensor((4,),N.i64),
           D:N.Tensor((1,),N.i32), O:N.Tensor((4,64),N.f32,access="write")):
    with N.Kernel():
        row=N.alloc_ub((64,),N.f32)
        divisor=N.load_index(D)
        for i in N.serial(4):
            value=N.load_index(I[i:i+1])
            index=(value*i)//divisor
            N.copy(X[index,:],row)
            N.copy(row,O[i,:])
""",
        {"I": [7, 9, 11, 13], "D": [2]},
    )
    assert reads(trace, "X") == [(i * 256, (i + 1) * 256) for i in (0, 4, 11, 19)]


@pytest.mark.parametrize("n", [0, 1, 4])
def test_scalar_carry_and_conditional_result_retain_actual_addresses(monkeypatch, n):
    _, trace = execution(
        monkeypatch,
        """
def carry(X:N.Tensor((32,64),N.f32), I:N.Tensor((1,),N.i32),
          O:N.Tensor((64,),N.f32,access="write"), n:N.int64):
    with N.Kernel():
        row=N.alloc_ub((64,),N.f32)
        N.fill(row,0.0)
        index=N.load_index(I)
        for i in N.serial(n):
            if i < 2:
                index=index+1
            else:
                index=index+2
            N.copy(X[index,:],row)
        N.copy(row,O)
""",
        {"I": [3]},
        {"n": n},
    )
    assert reads(trace, "X") == [(i * 256, (i + 1) * 256) for i in (4, 5, 7, 9)[:n]]


@pytest.mark.parametrize("index", [-1, 0, 1])
def test_boolean_short_circuit_does_not_read_inactive_metadata(monkeypatch, index):
    _, trace = execution(
        monkeypatch,
        """
def lazy(X:N.Tensor((32,64),N.f32), I:N.Tensor((1,),N.i32),
         Flags:N.Tensor((2,),N.i32), O:N.Tensor((64,),N.f32,access="write")):
    with N.Kernel():
        row=N.alloc_ub((64,),N.f32)
        N.fill(row,0.0)
        index=N.load_index(I)
        if index >= 0 and N.load_index(Flags[index:index+1]) > 0:
            N.copy(X[index,:],row)
        N.copy(row,O)
""",
        {"I": [index], "Flags": [0, 1]},
    )
    assert reads(trace, "Flags") == ([] if index < 0 else [(index * 4, (index + 1) * 4)])
    assert reads(trace, "X") == ([(256, 512)] if index == 1 else [])


@pytest.mark.parametrize("index", [-1, 0])
def test_conditional_expression_does_not_speculate_load_or_division(monkeypatch, index):
    _, trace = execution(
        monkeypatch,
        """
def lazy(X:N.Tensor((32,64),N.f32), I:N.Tensor((1,),N.i32),
         D:N.Tensor((1,),N.i32), O:N.Tensor((64,),N.f32,access="write")):
    with N.Kernel():
        row=N.alloc_ub((64,),N.f32)
        index=N.load_index(I)
        selected=0 if index < 0 else 8//N.load_index(D[index:index+1])
        N.copy(X[selected,:],row)
        N.copy(row,O)
""",
        {"I": [index], "D": [0 if index < 0 else 2]},
    )
    row = 0 if index < 0 else 4
    assert reads(trace, "D") == ([] if index < 0 else [(0, 4)])
    assert reads(trace, "X") == [(row * 256, (row + 1) * 256)]


@pytest.mark.parametrize("expression", ["n//0", "(-1)//(n+1)", "((9223372036854775807+1)+n)//2"])
def test_static_illegal_original_arithmetic_is_still_rejected(expression):
    source = N.from_source(
        """
def invalid(n:N.int64):
    with N.Kernel():
        bad=EXPRESSION
        gone=bad-bad
""".replace("EXPRESSION", expression)
    )
    with pytest.raises((CompileError, VerifyException, ValueError)):
        compile(source)


@pytest.mark.parametrize("mode", ["unknown", "budget"])
def test_unknown_optional_ranges_do_not_gate_an_ordinary_gather(monkeypatch, mode):
    from ascendflow.indexing import IndexRanges

    def unknown(self, expr, domains):
        if mode == "budget":
            raise CompileError("optional range budget", kind=ErrorKind.ANALYSIS_LIMIT)
        return expr, expr

    monkeypatch.setattr(IndexRanges, "interval", unknown)
    _, trace = execution(
        monkeypatch,
        """
def gather(X:N.Tensor((32,64),N.f32), I:N.Tensor((1,),N.i32),
           O:N.Tensor((64,),N.f32,access="write")):
    with N.Kernel():
        row=N.alloc_ub((64,),N.f32)
        index=N.load_index(I)
        N.copy(X[index,:],row)
        N.copy(row,O)
""",
        {"I": [31]},
    )
    assert reads(trace, "X") == [(31 * 256, 32 * 256)]


@pytest.mark.parametrize("index", [0, 1, 3])
def test_byte_addressed_dma_preserves_the_exact_unaligned_gm_slice(monkeypatch, index):
    result, trace = execution(
        monkeypatch,
        """
def unaligned(X:N.Tensor((67,),N.f32), I:N.Tensor((1,),N.i32),
              O:N.Tensor((67,),N.f32,access="write")):
    with N.Kernel():
        row=N.alloc_ub((64,),N.f32)
        index=N.load_index(I)
        N.copy(X[index:index+64],row)
        N.copy(row,O[index:index+64])
""",
        {"I": [index]},
    )
    assert reads(trace, "X") == [(index * 4, (index + 64) * 4)]
    writes = [s for c in trace.natives for s in c.spans if s.root.sym_name.data == "O"]
    assert [(s.start, s.end) for s in writes] == [(index * 4, (index + 64) * 4)]
    assert "asc_copy_gm2ub_align(" in result.asc
    assert "asc_copy_ub2gm_align(" in result.asc


def test_inactive_projection_is_not_a_source_definedness_obligation(monkeypatch):
    _, trace = execution(
        monkeypatch,
        """
def guarded(X:N.Tensor((2,64),N.f32), O:N.Tensor((4,64),N.f32,access="write")):
    with N.Kernel():
        row=N.alloc_ub((64,),N.f32)
        for i in N.serial(4):
            N.fill(row,0.0)
            if i > 0:
                N.copy(X[(i-1)//2,:],row)
            N.copy(row,O[i,:])
""",
        {},
    )
    assert reads(trace, "X") == [(0, 256), (0, 256), (256, 512)]


@pytest.mark.parametrize("value", [-1, 0, 33])
def test_optimization_opaque_integer_operation_is_an_executable_address(monkeypatch, value):
    result, trace = execution(
        monkeypatch,
        """
def bit_address(X:N.Tensor((32,64),N.f32), I:N.Tensor((1,),N.i32),
                O:N.Tensor((64,),N.f32,access="write")):
    with N.Kernel():
        row=N.alloc_ub((64,),N.f32)
        index=N.load_index(I)&31
        N.copy(X[index,:],row)
        N.copy(row,O)
""",
        {"I": [value]},
    )
    index = value & 31
    assert reads(trace, "X") == [(index * 256, (index + 1) * 256)]
    memory = collect_native_memory(result.plan)
    access = next(
        a for t in memory.templates for a in t.accesses if a.address.root.owner.sym_name.data == "X"
    )
    assert any(s.kind.value == "ssa" for s in access.address.offset.variables)
