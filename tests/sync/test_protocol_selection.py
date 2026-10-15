"""Prospective instructions grant no completion until their protocol is selected."""

from types import SimpleNamespace

import pytest

from xdsl_npu.loop_planning import _DomainFlow


@pytest.mark.parametrize("order", [(1, 2), (2, 1)])
def test_a_route_requiring_two_protocols_cannot_borrow_an_inactive_effect(order):
    # Native 1 is free; 2 and 3 belong to separate protocols.
    successors = [
        [(1, (0, 1, 0)), (2, (0, 2, 0))],
        [(2, (1, 2, 0))],
        [(3, (2, 3, 0))],
        [(4, (3, 4, 0))],
        [],
    ]
    owners = [0, 0, 1, 2, 0]
    flow = _DomainFlow(SimpleNamespace(successors=successors), owners, 0)
    flow.seed(0, 1)
    flow.propagate()
    assert flow.facts[1] == 1 and flow.facts[4] == 0
    assert flow.route(0, 4, 0) == 3
    flow.activate(order[0], order[0])
    flow.propagate()
    assert flow.facts[4] == 0
    flow.activate(order[1], 3)
    flow.propagate()
    assert flow.facts[4] == 1


def test_only_selected_candidate_instructions_enter_the_ir(monkeypatch):
    import xdsl_npu.loop_sync as sync
    from examples.legacy.multistage import vector_source
    from xdsl_npu import autosync
    from xdsl_npu import language as N
    from xdsl_npu.ir import LoopChannelOp, LoopSyncOp
    from xdsl_npu.lowering import lower_source

    original = sync._materialize
    snapshots = []

    def inspect(loop, candidates, selected):
        assert not any(
            isinstance(op, (LoopSyncOp, LoopChannelOp)) for op in loop.protocol_scope.walk()
        )
        assert any(not selected & (1 << i) for i in range(len(candidates)))
        original(loop, candidates, selected)
        actual = set(loop.protocol_scope.walk())
        for i, candidate in enumerate(candidates):
            for _, _, op in candidate.placements:
                assert (op in actual) == bool(selected & (1 << i))
        snapshots.append(selected)

    monkeypatch.setattr(sync, "_materialize", inspect)
    plan = autosync(lower_source(vector_source(N.dynamic("tasks"))))
    operations = set(plan.walk())
    for op in operations:
        for region in op.regions:
            for block in region.blocks:
                for argument in block.args:
                    assert all(use.operation in operations for use in argument.uses)
    assert len(snapshots) == 1
