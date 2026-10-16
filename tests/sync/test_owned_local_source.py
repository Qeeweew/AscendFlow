"""Owned source facts cross lowering and local synchronization exactly once."""

import pytest
from xdsl.dialects.builtin import ArrayAttr, FloatAttr, IntAttr, UnitAttr, f32, f64

from examples.gemm.batched import batched_matmul
from examples.vector.scale import scale
from tests.sync.test_structured_local_sync import _forbid_old_solver
from xdsl_npu import language as N
from xdsl_npu import loop_model
from xdsl_npu._physical_builder import Kernel
from xdsl_npu.analysis import dependencies, native
from xdsl_npu.backend.local import emit_local
from xdsl_npu.compilation_facts import PhysicalFacts
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.dialects.io import dump, parse
from xdsl_npu.dialects.physical import LoopChannelOp, LoopSyncOp, TaskOp
from xdsl_npu.dialects.source import ProgramOp
from xdsl_npu.lowering import lower_source
from xdsl_npu.passes import insert_sync
from xdsl_npu.passes.insert_sync import lower_and_synchronize_local, synchronize_local

SCALE = scale.specialize(B=64)
CUBE = batched_matmul.specialize(M=16, N=16, K=16, GROUPS=1, L1_SLOTS=2, L0_SLOTS=2, C_SLOTS=2)


def _count_admission(monkeypatch):
    contracts, loops, scans = [], [], []
    contract, extract, analyze = (
        native.native_contract,
        loop_model.extract_loop,
        dependencies.analyze_dependencies,
    )

    def native_contract(*args, **kwargs):
        contracts.append(args[0])
        return contract(*args, **kwargs)

    def extract_loop(module, **kwargs):
        result = extract(module, **kwargs)
        loops.append((module, result))
        return result

    def analyze_dependencies(loop, **kwargs):
        result = analyze(loop, **kwargs)
        scans.append((loop, result))
        return result

    monkeypatch.setattr(native, "native_contract", native_contract)
    monkeypatch.setattr(loop_model, "extract_loop", extract_loop)
    monkeypatch.setattr(dependencies, "analyze_dependencies", analyze_dependencies)
    return contracts, loops, scans


@pytest.mark.parametrize("entry,task_count", [(SCALE, 3), (CUBE, 6)])
@pytest.mark.parametrize("typed_ir", [False, True])
def test_source_handoff_owns_one_loop_one_scan_and_one_contract_per_native(
    entry, task_count, typed_ir, monkeypatch
):
    source = entry.to_ir() if typed_ir else entry
    before = dump(source) if typed_ir else None
    contracts, loops, scans = _count_admission(monkeypatch)
    _forbid_old_solver(monkeypatch)
    plans = [lower_and_synchronize_local(source) for _ in range(2)]
    assert len(contracts) == 2 * task_count
    assert len(loops) == len(scans) == 2
    for plan, (owned, loop), (scanned_loop, analysis) in zip(plans, loops, scans, strict=True):
        assert plan.module is owned
        assert plan.loop is loop is scanned_loop
        assert plan.dependencies is analysis
        assert len(plan.selection.tasks) == task_count
        assert all(task.operation in tuple(owned.walk()) for task in plan.selection.tasks)
        work = plan.analysis_work
        assert work["loop_extractions"] == work["dependency_builds"] == 1
        assert work["loop_reuses"] == 1
        assert work["dependency_records"] == len(analysis.obligations)
        if entry is CUBE:
            # Its borrowed Tile values already used these exact obligations
            # during source admission, before synchronization consumes them.
            assert work["dependency_reuses"] == 1
        assert emit_local(plan)
    assert plans[0].module is not plans[1].module
    assert plans[0].loop is not plans[1].loop
    assert plans[0].dependencies is not plans[1].dependencies
    assert plans[0].selection.tasks[0].operation is not plans[1].selection.tasks[0].operation
    if typed_ir:
        assert dump(source) == before
        assert all(plan.source_ir == before for plan in plans)


def test_owned_snapshots_show_source_and_unsynchronized_physical_ir():
    source = SCALE.to_ir()
    original = dump(source)
    plan = lower_and_synchronize_local(source)
    assert plan.source_ir == original and dump(source) == original
    assert isinstance(parse(plan.source_ir).body.block.first_op, ProgramOp)
    before = parse(plan.before_ir)
    assert sum(isinstance(op, TaskOp) for op in before.walk()) == 3
    assert not any(isinstance(op, (LoopChannelOp, LoopSyncOp)) for op in before.walk())
    assert any(isinstance(op, LoopSyncOp) for op in plan.module.walk())
    with pytest.raises(TypeError):
        plan.analysis_work["dependency_builds"] = 99


def test_external_physical_mutation_gets_a_fresh_independent_admission(monkeypatch):
    physical = lower_source(SCALE.to_ir())
    contracts, loops, scans = _count_admission(monkeypatch)
    first = synchronize_local(physical)
    first_asc, first_before = emit_local(first), first.before_ir
    compute = next(
        op for op in physical.walk() if isinstance(op, TaskOp) and op.native.data == "muls"
    )
    compute.attributes["scalar"] = FloatAttr(3.0, f32)
    second = synchronize_local(physical)
    assert len(contracts) == 6 and len(loops) == len(scans) == 2
    assert first.loop is not second.loop and first.dependencies is not second.dependencies
    assert first.source_ir is second.source_ir is None
    assert first.before_ir == first_before and emit_local(first) == first_asc
    assert first.before_ir != second.before_ir and emit_local(second) != first_asc
    assert not any(isinstance(op, LoopSyncOp) for op in physical.walk())
    assert first.analysis_work["loop_extractions"] == second.analysis_work["loop_extractions"] == 1


@pytest.mark.parametrize("invalid", ["params", "float_type"])
def test_verified_marker_cannot_skip_readmission_or_print_invalid_external_input(
    invalid, monkeypatch
):
    physical = lower_source(SCALE.to_ir())
    synchronize_local(physical)
    compute = next(
        op for op in physical.walk() if isinstance(op, TaskOp) and op.native.data == "muls"
    )
    physical.attributes["verified"] = UnitAttr()
    if invalid == "params":
        compute.attributes["params"] = ArrayAttr([IntAttr(7)])
        diagnostic = "Vector recipe has no unchecked native parameters"
    else:
        compute.attributes["scalar"] = FloatAttr(2.0, f64)
        diagnostic = "native Vector scalar literals must be f32"

    def no_snapshot(*args, **kwargs):
        raise AssertionError("invalid external program was serialized before native admission")

    monkeypatch.setattr(insert_sync, "dump", no_snapshot)
    with pytest.raises(CompileError, match=diagnostic):
        synchronize_local(physical)


def test_borrowed_tile_admission_remains_before_owned_synchronization(monkeypatch):
    stale = N.from_source("""
        def stale(X: N.Tensor((n, 64), N.f32), n: N.int64):
            with N.Kernel():
                x = N.alloc_ub((64,))
                y = N.alloc_ub((64,))
                for t in N.Pipelined(n, stages=1):
                    old = N.copy(X[t, :], x)
                    N.mul(old, 2.0, out=x)
                    N.add(old, 1.0, out=y)
    """)

    def no_handoff(*args, **kwargs):
        raise AssertionError("stale source reached synchronization before logical-value admission")

    monkeypatch.setattr(insert_sync, "_synchronize_admitted_local", no_handoff)
    with pytest.raises(CompileError, match="overwritten"):
        lower_and_synchronize_local(stale)


def test_complete_dependency_cache_keeps_tuple_identity_and_live_in_assumptions(monkeypatch):
    k = Kernel("live_in")
    source = k.buffer("source", "UB", 32)
    output = k.buffer("output", "UB", 32)
    k.loop(3)
    k.scale("consume", k.view(source, 32), k.view(output, 32), 2.0)
    facts = PhysicalFacts()
    loop = facts.extract_loop(k.finish())
    analysis = facts.dependency_analysis(loop, live_in={"aiv0/source"})

    def no_rebuild(*args, **kwargs):
        raise AssertionError("legacy dependency view rebuilt the complete analysis")

    with monkeypatch.context() as cached:
        cached.setattr(dependencies, "analyze_dependencies", no_rebuild)
        assert facts.loop_dependencies(loop, live_in=["aiv0/source"]) is analysis.obligations
        assert facts.dependency_analysis(loop, live_in=("aiv0/source",)) is analysis
    with pytest.raises(CompileError, match="undefined first read"):
        facts.dependency_analysis(loop)
    work = facts.statistics()
    assert work["dependency_builds"] == 1
    assert work["dependency_records"] == len(analysis.obligations)
    assert work["dependency_reuses"] == 2
