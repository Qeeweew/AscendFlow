"""Executed scalar branch results retain bounds without exporting arm facts."""

import pytest

from ascendflow import compile
from ascendflow import language as N
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.diagnostics import CompileError
from ascendflow.dialects.io import parse
from tests.models.structured_events import check_structured_events


def check(result, parameters):
    return check_structured_events(
        collect_native_memory(result.plan),
        parameters,
        logical_reference=collect_native_memory(parse(result.before_ir)),
    )


def choose(body):
    return N.from_source(
        """
def choose(X:N.Tensor((128,64),N.f32), I:N.Tensor((1,),N.i32),
           O:N.Tensor((64,),N.f32,access="write"), n:N.i64):
    with N.Kernel():
        row=N.alloc_ub((64,),N.f32)
        n=N.min(n,64)
        BODY
        N.copy(X[start,:],row)
        N.copy(row,O)
""".replace("        BODY", body),
        disjoint_external=True,
    )


def test_conditional_metadata_result_has_real_addresses_and_an_exit_range(monkeypatch):
    from tests.models.structured_events import _Interpret

    definition = choose("""        if n > 0:
            start=N.load_index(I,minimum=0,maximum=64)
        else:
            start=32""")
    result = compile(definition)
    original = _Interpret.native

    def metadata(self, operation, values, *args):
        original(self, operation, values, *args)
        if operation.native.data == "index_load":
            values[operation.value] = 64

    monkeypatch.setattr(_Interpret, "native", metadata)
    for n, offset in ((0, 32), (1, 64)):
        trace = check(result, {"n": n})
        spans = [s for c in trace.natives for s in c.spans if s.root.sym_name.data == "X"]
        assert [(s.start, s.end) for s in spans] == [(offset * 256, (offset + 1) * 256)]


def test_optional_load_range_does_not_escape_the_unselected_outer_path():
    from ascendflow.indexing import arguments
    from ascendflow.passes.lower_native import lower_native_structure

    definition = choose("""        if n > 0:
            start=N.load_index(I,minimum=n+1,maximum=64)
        else:
            start=0
        N.copy(X[n+100,:],row)""")
    memory = collect_native_memory(lower_native_structure(definition).module)
    load = next(t.operation for t in memory.templates if t.spec.native == "index_load")
    symbol = memory.control.environment.expression(load.value).args[0]
    outside = [
        t
        for t in memory.templates
        if t.spec.native == "copy" and not any(f.branch is not None for f in t.frame.ancestors())
    ]
    assert outside
    assert all(
        symbol not in arguments(value)
        for t in outside
        for domain in t.frame.domains()
        for value in domain
    )


def test_one_unsafe_arm_still_rejects_the_join():
    definition = choose("""        if n > 0:
            N.copy(X[128,:],row)
            start=0
        else:
            start=0""")
    with pytest.raises(CompileError, match="tensor extent|axis|bounds"):
        compile(definition)


def test_raw_invalid_arithmetic_in_an_arm_is_not_erased_by_equal_results():
    definition = choose("""        if n > 0:
            bad=(-1)//(n+1)
            start=bad-bad
        else:
            start=0""")
    with pytest.raises(CompileError, match="nonnegative dividend|nonnegative numerator"):
        compile(definition)


def test_joined_nonnegative_result_can_be_divided_without_loosening_source_checks():
    definition = choose("""        if n > 0:
            start=64
        else:
            start=0
        start=start//2""")
    result = compile(definition)
    for n in (0, 1):
        check(result, {"n": n})
