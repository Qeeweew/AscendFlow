"""Real capture snapshots carry upstream origins through queue-specific gates."""

import pytest

from ascendflow import language as N
from ascendflow.dialects.structured import NotificationOp
from ascendflow.passes.structured_sync import synchronize_structure
from ascendflow.target.ascend910b import Engine, SyncKind
from tests.models.structured_events import EventModelError, check_compiled_events


def chain(*, late_capture=False, runtime=False, branch=False):
    guard = "if t%2==0:\n                    N.copy(y,O0[t,:])" if branch else "N.copy(y,O0[t,:])"
    last = "N.copy(x.at(t),O1[t,:])"
    body = f"""N.copy(X[t,:],x.at(t))
                N.copy(x.at(t),y)
                {guard}
                {last}"""
    if late_capture:
        body = f"""N.copy(X[t,:],x.at(t))
                N.copy(x.at(t),O1[t,:])
                N.copy(x.at(t),y)
                {guard}"""
    return N.from_source(
        f"""
def chain(X:N.Tensor((n,64),N.bf16),O0:N.Tensor((n,64),N.bf16,access="write"),
          O1:N.Tensor((n,64),N.bf16,access="write"),n:N.i64):
    with N.Kernel():
        x=N.alloc_ub((64,),N.bf16,slots=3)
        y=N.alloc_ub((64,),N.bf16)
        for t in N.Pipelined({"n" if runtime else "1"}):
            with N.Stage(0):
                {body}
""",
        disjoint_external=True,
    )


def direct_load_to_store(plan):
    return [
        n
        for n in plan.selection.notifications
        if n.source_engine is Engine.MTE2
        and n.destination_engine is Engine.MTE3
        and any(
            plan.generation.domains.active[n, case] for case in plan.generation.domains.cases[n]
        )
    ]


@pytest.mark.parametrize("runtime", [False, True])
def test_composed_prefix_covers_direct_bf16_load_to_store(runtime):
    plan = synchronize_structure(chain(runtime=runtime))
    assert plan.generation.domains.work.completion_covered
    assert not direct_load_to_store(plan)
    assert any(item.completion is not None for item in plan.selection.coverage)
    for count in (1,) if not runtime else (0, 1, 2, 7):
        check_compiled_events(plan, {"n": count}, invocations=2)


def test_acquire_after_the_needed_capture_cannot_backdate_a_prefix():
    plan = synchronize_structure(chain(late_capture=True, runtime=True))
    assert direct_load_to_store(plan)
    for count in (0, 1, 3):
        check_compiled_events(plan, {"n": count}, invocations=2)


def test_a_relay_on_only_one_branch_does_not_cover_unconditional_readiness():
    plan = synchronize_structure(chain(branch=True, runtime=True))
    assert direct_load_to_store(plan)
    for count in (0, 1, 2, 7):
        check_compiled_events(plan, {"n": count}, invocations=2)


def test_deleting_a_required_gate_breaks_the_independent_byte_model():
    plan = synchronize_structure(chain(runtime=True))
    wait = next(
        op
        for op in plan.module.walk()
        if isinstance(op, NotificationOp)
        and op.kind.data == SyncKind.LOCAL_WAIT.value
        and op.source.data == Engine.V.value
        and op.destination.data == Engine.MTE3.value
    )
    wait.parent.erase_op(wait)
    with pytest.raises(EventModelError):
        check_compiled_events(plan, {"n": 3})


def test_completion_proof_keeps_original_demands_and_retained_support_actions():
    plan = synchronize_structure(chain(runtime=True))
    retained = set(plan.selection.notifications)
    for item in plan.selection.coverage:
        if item.completion is not None:
            proof = item.completion
            assert set(proof.supports()) <= retained
            assert proof.covers(item.binding)
            assert item.binding.requirement is item.requirement
            assert any(item.requirement is need for need in plan.dependencies.required)


def test_support_frontier_admits_all_proposed_real_actions(monkeypatch):
    from ascendflow.analysis.protocol_transfers import ProtocolTransfers

    completion, proposals = ProtocolTransfers.completion, []

    def checked(graph, *args, **kwargs):
        available = set(graph.domains.notifications) - kwargs["excluded"]
        proof = completion(graph, *args, **kwargs)
        pending, seen = [proof] if proof is not None else [], set()
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            if current.node.notification is not None:
                assert current.node.notification in available
            pending.extend(current.inputs)
        if proof is not None:
            proposals.append(proof)
        return proof

    monkeypatch.setattr(ProtocolTransfers, "completion", checked)
    plan = synchronize_structure(chain(runtime=True))
    assert proposals
    retained = set(plan.selection.notifications)
    for item in plan.selection.coverage:
        if item.completion is not None:
            assert set(item.completion.supports()) <= retained
    assert not direct_load_to_store(plan)
    for count in (0, 1, 3):
        check_compiled_events(plan, {"n": count}, invocations=2)


def test_completion_preserves_each_sibling_loop_writer_in_a_merged_capture():
    source = N.from_source(
        """
def siblings(O:N.Tensor((2,64),N.f32,access="write"),n:N.i64,m:N.i64):
    with N.Kernel():
        x=N.alloc_ub((2,64))
        N.fill(x,1.0)
        for a in N.serial(n):
            N.add(x[0,:],x[0,:],out=x[0,:])
        for b in N.serial(m):
            N.add(x[1,:],x[1,:],out=x[1,:])
        N.copy(x,O)
""",
        disjoint_external=True,
    )
    plan = synchronize_structure(source)
    for n, m in ((0, 0), (0, 1), (1, 0), (1, 1), (2, 3)):
        check_compiled_events(plan, {"n": n, "m": m}, invocations=2)


def test_completion_closure_rejects_a_proof_missing_one_original_binding():
    from dataclasses import replace

    from ascendflow.diagnostics import CompileError

    plan = synchronize_structure(chain(runtime=True))
    item = next(item for item in plan.selection.coverage if item.completion is not None)
    incomplete = replace(
        item.completion,
        obligations=tuple(
            replace(obligation, bindings=()) for obligation in item.completion.obligations
        ),
    )
    damaged = replace(
        plan.selection,
        coverage=tuple(
            replace(coverage, completion=incomplete) if coverage is item else coverage
            for coverage in plan.selection.coverage
        ),
    )
    with pytest.raises(CompileError, match="completion coverage belongs to another requirement"):
        damaged.require_closed()


def test_both_real_branch_transfers_cover_the_common_exit():
    original = chain(branch=True, runtime=True)
    source = original.source.replace(
        "N.copy(y,O0[t,:])",
        "N.copy(y,O0[t,:])\n                else:\n                    N.copy(y,O0[t,:])",
    )
    plan = synchronize_structure(N.from_source(source, disjoint_external=True))
    assert not direct_load_to_store(plan)
    for count in (0, 1, 2, 7):
        check_compiled_events(plan, {"n": count}, invocations=2)


@pytest.mark.parametrize("blocks", [8, 32, 64])
def test_regular_controller_stream_grows_by_real_frontiers_instead_of_all_pairs(blocks):
    declarations = "\n".join(f"        x{i}=N.alloc_ub((64,))" for i in range(blocks))
    body = "\n".join(
        f"        N.copy(X[{i},:],x{i})\n        N.load_scalar(x{i}[:1])" for i in range(blocks)
    )
    source = (
        f"def stream(X:N.Tensor(({blocks},64),N.f32)):\n    with N.Kernel():\n"
        + declarations
        + "\n"
        + body
        + "\n"
    )
    plan = synchronize_structure(N.from_source(source))
    work = plan.generation.domains.work
    assert work.transfer_nodes <= 6 * blocks
    assert work.completion_conclusions <= 2 * blocks
    assert work.transfer_order_bands <= 6 * blocks
    check_compiled_events(plan, invocations=2)


@pytest.mark.parametrize("blocks", [8, 32, 64])
def test_closed_sibling_regions_share_exits_without_rescanning_each_interior(blocks):
    declarations = "\n".join(f"        x{i}=N.alloc_ub((64,))" for i in range(blocks))
    regions = "\n".join(
        f"        for child{i} in N.serial(n):\n"
        f"            N.fill(x{i},1.0)\n"
        f"            N.load_scalar(x{i}[:1])"
        for i in range(blocks)
    )
    source = "def siblings(n:N.i64):\n    with N.Kernel():\n" + declarations + "\n" + regions + "\n"
    plan = synchronize_structure(N.from_source(source))
    work = plan.generation.domains.work
    assert work.transfer_order_bands <= 24 * blocks
    # Each query stores logarithmically many shared prefix ranges, never all
    # previous sibling interiors. The finite tree height is part of the bound.
    assert work.transfer_edges <= 4 * blocks * (blocks.bit_length() + 1)
    for count in (0, 1, 3):
        check_compiled_events(plan, {"n": count}, invocations=2)
