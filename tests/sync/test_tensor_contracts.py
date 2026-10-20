"""Source permissions and disjoint external writes have distinct consumers."""

from itertools import product

import pytest
from xdsl.dialects.builtin import StringAttr

from ascendflow import compile
from ascendflow import language as N
from ascendflow.analysis.hazards import HazardKind
from ascendflow.analysis.structured_dependencies import build_structured_dependencies
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.diagnostics import CompileError
from ascendflow.dialects.io import parse
from ascendflow.dialects.source import AllocationOp
from ascendflow.passes.lower_native import lower_native_structure
from tests.models.structured_events import EventModelError, check_structured_events

PERMUTED = """
def permuted(X:N.Tensor((8,64),N.f32),
             Y:N.Tensor((8,64),N.f32,access="write",write_disjoint=True),
             p:N.int64):
    with N.Kernel(3) as group:
        x=N.alloc_ub((64,))
        for i in N.Pipelined(N.ceildiv(N.max(0,8-group),3)):
            flat=group+i*3
            row=(flat*(2*p+1))%8
            N.copy(X[row,:],x)
            N.copy(x,Y[row,:])
"""


def test_disjoint_output_contract_accepts_symbolic_permutations_and_keeps_ub_release():
    source = N.from_source(PERMUTED, disjoint_external=True)
    memory = collect_native_memory(lower_native_structure(source).module)
    dependencies = build_structured_dependencies(memory)
    assert dependencies.work.disjoint_writes == 1
    assert not any(need.cell.root.owner.sym_name.data == "Y" for need in dependencies.needs)
    assert any(need.kind is HazardKind.WAR for need in dependencies.required)
    result = compile(source)
    final = collect_native_memory(result.plan)
    original = collect_native_memory(parse(result.before_ir))
    for p in (0, 1, 2, 7):
        check_structured_events(
            final, {"p": p}, logical_reference=original, groups=(0, 1, 2), invocations=2
        )


def test_disjoint_write_declaration_does_not_assert_its_own_truth():
    source = PERMUTED.replace("row=(flat*(2*p+1))%8", "row=flat//2")
    result = compile(N.from_source(source, disjoint_external=True))
    with pytest.raises(EventModelError, match="write_disjoint contract violated"):
        check_structured_events(collect_native_memory(result.plan), {"p": 0}, groups=(0, 1, 2))


def test_unknown_output_permutation_keeps_order_without_a_grid_inverse():
    source = N.from_source(PERMUTED.replace(",write_disjoint=True", ""), disjoint_external=True)
    dependencies = build_structured_dependencies(
        collect_native_memory(lower_native_structure(source).module)
    )
    assert dependencies.work.alias_regions
    assert any(
        n.kind is HazardKind.WAW and n.cell.root.owner.sym_name.data == "Y"
        for n in dependencies.required
    )
    artifact = compile(source)
    final = collect_native_memory(artifact.plan)
    original = collect_native_memory(parse(artifact.before_ir))
    for p in (0, 1, 2, 7):
        check_structured_events(
            final, {"p": p}, logical_reference=original, groups=(0, 1, 2), invocations=2
        )


def test_disjoint_writes_do_not_remove_raw_or_war_on_a_read_write_tensor():
    source = N.from_source(
        """
    def state(Y:N.Tensor((4,64),N.f32,access="read_write",write_disjoint=True),
              a:N.int64,b:N.int64,c:N.int64):
        with N.Kernel():
            x=N.alloc_ub((64,))
            z=N.alloc_ub((64,))
            N.copy(Y[a%4,:],x)
            N.copy(x,Y[b%4,:])
            N.copy(Y[c%4,:],z)
    """,
        disjoint_external=True,
    )
    memory = collect_native_memory(lower_native_structure(source).module)
    dependencies = build_structured_dependencies(memory)
    assert dependencies.work.disjoint_writes == 0
    assert {n.kind for n in dependencies.required if n.cell.root.owner.sym_name.data == "Y"} >= {
        HazardKind.RAW,
        HazardKind.WAR,
    }
    result = compile(source)
    final, original = (
        collect_native_memory(result.plan),
        collect_native_memory(parse(result.before_ir)),
    )
    for a, b, c in product((0, 1), repeat=3):
        check_structured_events(final, {"a": a, "b": b, "c": c}, logical_reference=original)


def test_native_effects_enforce_read_only_even_for_source_ir_input():
    source = N.from_source(PERMUTED, disjoint_external=True).to_ir()
    output = next(
        op for op in source.walk() if isinstance(op, AllocationOp) and op.sym_name.data == "Y"
    )
    output.attributes["access"] = StringAttr("read")
    del output.attributes["write_disjoint"]
    source.verify()
    with pytest.raises(CompileError, match="writes a read-only tensor"):
        compile(source)


@pytest.mark.parametrize(
    "options", ['access="read",write_disjoint=True', 'access="write",write_disjoint=1']
)
def test_disjoint_write_contract_rejects_permission_and_type_mismatches(options):
    source = PERMUTED.replace('access="write",write_disjoint=True', options)
    with pytest.raises(CompileError, match="write_disjoint"):
        N.from_source(source, disjoint_external=True).to_ir()
