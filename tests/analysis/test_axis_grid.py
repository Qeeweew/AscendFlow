"""Runtime ND digit permutations retain actual bytes and disjoint owners."""

import pytest

from ascendflow import compile
from ascendflow import language as N
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.dialects.io import parse
from tests.models.structured_events import EventModelError, check_structured_events

SOURCE = """
def permuted(X:N.Tensor((4,n,64),N.f32),Y:N.Tensor((4,n,64),N.f32,access="write"),
             n:N.int64(minimum=1)):
    with N.Kernel(3) as group:
        x=N.alloc_ub((64,))
        for t in N.serial(N.ceildiv(N.max(0,4*n-group),3)):
            flat=group+t*3
            head=flat%4
            row=flat//4
            N.copy(X[head,row,:],x)
            N.copy(x,Y[head,row,:])
"""


def test_axis_permutation_retains_whole_loop_cell_separation():
    result = compile(N.from_source(SOURCE, disjoint_external=True))
    memory = collect_native_memory(result.plan)
    original = collect_native_memory(parse(result.before_ir))
    for n in (1, 2, 5, 11):
        checked = check_structured_events(
            memory, {"n": n}, groups=(0, 1, 2), logical_reference=original
        )
        assert checked.bytes_read and checked.bytes_written


def _reject_parallel_contract(source, n):
    result = compile(N.from_source(source, disjoint_external=True))
    memory = collect_native_memory(result.plan)
    with pytest.raises(EventModelError, match="unordered physical WAW"):
        check_structured_events(memory, {"n": n}, groups=(0, 1, 2))


def test_duplicate_owner_mapping_violates_the_source_parallel_contract():
    invalid = SOURCE.replace("head=flat%4", "head=0").replace("row=flat//4", "row=flat% n")
    _reject_parallel_contract(invalid, 2)


def test_cyclic_prefix_translation_is_bijective_only_with_a_unique_source_coordinate():
    shifted = SOURCE.replace(
        "head=flat%4", "mapped=(flat+N.max(0,4*n-3))%(4*n)\n            head=mapped%4"
    ).replace("row=flat//4", "row=mapped//4")
    result = compile(N.from_source(shifted, disjoint_external=True))
    memory = collect_native_memory(result.plan)
    original = collect_native_memory(parse(result.before_ir))
    for n in (1, 2, 5, 11):
        check_structured_events(memory, {"n": n}, groups=(0, 1, 2), logical_reference=original)
    _reject_parallel_contract(shifted.replace("mapped=(flat+", "mapped=(flat//2+"), 2)
    # Translation by a different amount on each owner is not one global
    # bijection: n=1 maps both group 0 and group 2 to cell 0.
    _reject_parallel_contract(shifted.replace("N.max(0,4*n-3)", "group"), 1)


def test_contracted_child_digits_keep_their_actual_bounded_range():
    source = """
    def nested(X:N.Tensor((4,n,64),N.f32),
               Y:N.Tensor((4,n,64),N.f32,access="write"),n:N.int64(minimum=16,multiple_of=16)):
        with N.Kernel(3) as group:
            x=N.alloc_ub((64,))
            for t in N.serial(n//4):
                flat=group+t*3
                for r in N.serial(2):
                    N.copy(X[flat%4,2*(flat//4)+r,:],x)
                    N.copy(x,Y[flat%4,2*(flat//4)+r,:])
    """
    result = compile(N.from_source(source, disjoint_external=True))
    memory = collect_native_memory(result.plan)
    original = collect_native_memory(parse(result.before_ir))
    for n in (16, 32):
        check_structured_events(memory, {"n": n}, groups=(0, 1, 2), logical_reference=original)
    # r=2 collides with the next tile's r=0 on a different group. Every
    # physical address remains in bounds. The independent model checks the
    # algorithm's parallel contract; normal AutoSync does not prove it.
    _reject_parallel_contract(source.replace("N.serial(2)", "N.serial(3)"), 16)
