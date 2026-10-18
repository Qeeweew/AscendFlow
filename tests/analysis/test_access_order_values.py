"""Read values are checked by independent logical/physical byte interpretation.

Production access order deliberately does not select a unique numerical
writer. These replace assertions about the deleted whole-program value graph
with actual source-value and event-consumption checks on the public compiler.
"""

import pytest

from ascendflow import language as N
from ascendflow.analysis.hazards import HazardKind
from ascendflow.passes.structured_sync import synchronize_structure
from tests.fixtures.nested import shared_l0c_qk_pv
from tests.models.structured_events import check_compiled_events


@pytest.fixture(scope="module")
def scratch():
    return synchronize_structure(
        N.from_source(
            """
    def scratch(X:N.Tensor((n,64),N.f32),A:N.Tensor((n,64),N.f32,access="write"),
                B:N.Tensor((n,64),N.f32,access="write"),n:N.int64):
        with N.Kernel():
            x=N.alloc_ub((64,))
            for g in N.Pipelined(n,stages=2):
                with N.Stage(0):
                    N.copy(X[g,:],x)
                    N.copy(x,A[g,:])
                with N.Stage(1):
                    N.fill(x,1.0)
                    N.copy(x,B[g,:])
    """,
            disjoint_external=True,
        )
    )


@pytest.mark.parametrize("count", (0, 1, 3))
def test_two_stage_scratch_preserves_actual_source_read_values(scratch, count):
    assert scratch.dependencies.order.packet_partitions
    check_compiled_events(scratch, {"n": count}, invocations=2)


@pytest.fixture(scope="module")
def shared():
    return synchronize_structure(shared_l0c_qk_pv)


@pytest.mark.parametrize("count", (0, 1, 2))
def test_shared_qk_pv_preserves_first_clear_and_recursive_values(shared, count):
    recurrences = [
        n
        for n in shared.dependencies.needs
        if n.kind is HazardKind.RAW
        and n.cell.root.type.bank.data == "L0C"
        and n.source.operation is n.consumer.operation
    ]
    assert recurrences and all(n.distance == 1 for n in recurrences)
    check_compiled_events(shared, {"n": count}, invocations=2)


@pytest.fixture(scope="module")
def recurrence():
    return synchronize_structure(
        N.from_source(
            """
    def reduction(A:N.Tensor((n,16,16),N.f16),B:N.Tensor((n,16,16),N.f16),
                  O:N.Tensor((n,16,16),N.f32,access="write",write_disjoint=True),n:N.int64):
        with N.Kernel(kind="cube"):
            a1=N.alloc_l1((16,16),N.f16)
            b1=N.alloc_l1((16,16),N.f16)
            a0=N.alloc_l0a((16,16),N.f16)
            b0=N.alloc_l0b((16,16),N.f16)
            c0=N.alloc_l0c((16,16),N.f32)
            for k in N.Pipelined(n,stages=3):
                with N.Stage(0):
                    av=N.copy(A[k,:,:],a1)
                    bv=N.copy(B[k,:,:],b1)
                with N.Stage(1):
                    aa=N.copy(av,a0)
                    bb=N.copy(bv,b0,transpose=True)
                with N.Stage(2):
                    N.matmul(aa,bb,accum=c0,initialize=(k==0))
                    N.copy(c0,O[k,:,:])
    """,
            disjoint_external=True,
        )
    )


@pytest.mark.parametrize("count", (0, 1, 2, 5))
def test_runtime_first_clear_preserves_empty_single_and_recurrent_values(recurrence, count):
    check_compiled_events(recurrence, {"n": count}, invocations=2)


def test_periodic_native_reset_does_not_require_activation_history_closure():
    # Primitive mutation is a target-contract unit fixture, not a second DSL
    # compile path. Definedness/first-reset counterexamples stay in the native
    # admission suite; this checks order generation accepts their real modes.
    from ascendflow.analysis.structured_dependencies import build_structured_dependencies
    from tests.analysis.test_conditional_accesses import _admit, _retarget, _structure

    for period in (1, 2, 3):
        structure = _structure()
        _retarget(structure, period)
        admitted = _admit(structure)
        result = build_structured_dependencies(admitted.memory, admission=admitted)
        assert not hasattr(result, "values")
