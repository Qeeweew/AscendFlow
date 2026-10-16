"""Child hardware event-width choices reuse one unchanged completion graph."""

from collections import defaultdict

import pytest

from tests.fixtures.guarded import native_guarded_recurrence
from tests.fixtures.nested import nested_copy, nested_cv
from tests.test_loop_sync import finite_hardware_dag
from xdsl_npu import compile
from xdsl_npu import language as N
from xdsl_npu.dialects.io import dump
from xdsl_npu.loop_graph import Graph
from xdsl_npu.loop_sync import _plan_leaf
from xdsl_npu.scope_access import analyze_scope_access
from xdsl_npu.scope_events import event_choices, resource_inputs, resource_sites
from xdsl_npu.scope_lowering import lower_scopes
from xdsl_npu.scope_model import extract_scopes
from xdsl_npu.scope_sync import (
    ParentCompletion,
    _allocate_child_resources,
    _analyses,
    _verify_scope_parent,
)


def planned_children(source):
    program = extract_scopes(lower_scopes(source.body.block.first_op))
    banks = defaultdict(lambda: defaultdict(int))
    for leaf in program.leaves.values():
        loop = leaf.loop
        roots = {
            loop.root(actor, ref)
            for actor, data in loop.actors.items()
            for task in data.program.tasks
            for ref in task.refs
        }
        _plan_leaf(loop, live_in=roots, banks=banks[leaf.actor])
        data = next(iter(loop.actors.values()))
        aliases = {leaf.value_map[captured.op.ref]: captured.op.ref for captured in leaf.captures}
        replacement = data.stream.clone(value_mapper=aliases)
        leaf.call.body.block.insert_op_before(replacement, leaf.stream)
        leaf.call.body.block.erase_op(leaf.stream)
    program = extract_scopes(program.module)
    return program, _analyses(program)[1]


@pytest.fixture(scope="module")
def copied():
    return planned_children(nested_copy.to_ir())


@pytest.fixture(
    scope="module", params=(nested_copy, native_guarded_recurrence), ids=("plain", "cases")
)
def resource_parent(request):
    plan = compile(request.param, timeout_s=40).plan
    _, checked, reference = _verify_scope_parent(extract_scopes(plan))

    def build():
        proof = ParentCompletion(
            checked.program,
            checked.analyses,
            checked.access,
            reference.period,
            extra_protocols=checked.channels.values(),
        )
        for protocol in checked.channels.values():
            proof.insert(protocol)
        return proof, checked

    return build


def test_all_widths_batch_on_existing_child_graphs_without_ir_changes(copied, monkeypatch):
    program, analyses = copied
    before = dump(program.module)
    calls = defaultdict(int)
    original = Graph.covered_many

    def counted(graph, queries):
        calls[id(graph)] += 1
        return original(graph, queries)

    def forbidden(*args, **kwargs):
        pytest.fail("event-width choices must not rebuild or instantiate child analyses")

    monkeypatch.setattr(Graph, "covered_many", counted)
    monkeypatch.setattr(Graph, "domain", forbidden)
    monkeypatch.setattr("xdsl_npu.scope_model.extract_scopes", forbidden)
    monkeypatch.setattr("xdsl_npu.loop_sync._plan_leaf", forbidden)
    monkeypatch.setattr("xdsl_npu.loop_sync._verify_leaf", forbidden)
    choices = event_choices(program, analyses)
    assert len(choices) == 4
    assert calls == {id(analysis[3]): 1 for analysis in analyses.values()}
    assert dump(program.module) == before
    assert all(
        tuple(option.width for option in choice.options) == tuple(range(2, 9))
        for choice in choices.values()
    )


def test_locally_legal_widths_match_independent_finite_hardware_reuse(copied):
    program, analyses = copied
    choices = event_choices(program, analyses)
    for uid, analysis in analyses.items():
        loop, channels, streams, _, _ = analysis
        ids, reach = finite_hardware_dag(loop, streams, channels, loop.constant_count)
        commands = {
            (command.channel, command.kind): command
            for sequence in streams.values()
            for command in sequence
            if command.kind in {"set", "wait"}
        }
        for name, channel in channels.items():
            active = [
                generation
                for generation in range(loop.constant_count - channel.distance)
                if generation % channel.period in channel.residues
            ]
            valid = []
            for width in range(1, 9):
                if all(
                    reach[ids[commands[name, "wait"].index, old + channel.distance, "begin"]]
                    & (1 << ids[commands[name, "set"].index, new, "end"])
                    for old, new in zip(active, active[width:])
                ):
                    valid.append(width)
            assert tuple(option.width for option in choices[uid, name].options) == tuple(valid)


def test_fixed_candidate_endpoints_are_exact_and_unused_ids_add_no_boundaries(copied):
    program, analyses = copied
    choices = event_choices(program, analyses)
    for choice in choices.values():
        if choice.bank[1:] == ("MTE2", "MTE3"):
            four = next(option for option in choice.options if option.width == 4)
            assert len(four.resources) == 4
            assert {
                (resource.republished.position.offset, resource.consumed.position.offset)
                for resource in four.resources
            } == {(0, 0), (1, 1), (2, 2), (3, 3)}
        else:
            assert choice.bank[1:] == ("MTE3", "MTE2")
            assert all(len(option.resources) == 2 for option in choice.options)
        for option in choice.options:
            assert all(resource.domain.exact == 4 for resource in option.resources)
    inputs = resource_inputs(choices)
    assert set(inputs) == set(program.leaves)
    assert all(len(resources) == len(set(resources)) for resources in inputs.values())


def test_normalized_aiv1_sites_and_banks_retain_the_actual_parent_actor():
    program, analyses = planned_children(nested_cv.to_ir())
    choices = event_choices(program, analyses)
    aiv1 = [choice for choice in choices.values() if choice.bank[0] == "aiv1"]
    assert aiv1
    for choice in aiv1:
        leaf = program.leaves[choice.uid]
        assert next(iter(analyses[choice.uid][0].actors)) == "aiv0"
        for option in choice.options:
            for resource in option.resources:
                consumed, republished = resource_sites(
                    choice.uid, leaf.actor, choice.count, resource
                )
                assert consumed.task[0] == republished.task[0] == "aiv1"
                assert consumed.region == republished.region == choice.uid
                assert consumed.action == "wait" and republished.action == "set"


def test_more_ids_cannot_repair_a_missing_one_notification_cross_call_path():
    source = N.from_source(
        """
        def one(X:N.Tensor((n,8),N.f32),n:N.int64):
            with N.Kernel():
                a=N.alloc_ub((8,))
                b=N.alloc_ub((8,))
                for t in N.Pipelined(n,stages=1):
                    with N.Stage(0,name="body"):
                        for j in N.Pipelined(1,stages=1):
                            z=N.copy(X[t,:],a)
                            N.mul(z,2.0,out=b)
        """,
        disjoint_external=True,
    ).to_ir()
    program, analyses = planned_children(source)
    choices = event_choices(program, analyses)
    assert len(choices) == 1
    choice = next(iter(choices.values()))
    assert tuple(option.width for option in choice.options) == tuple(range(1, 9))
    assert all(option.resources == choice.options[0].resources for option in choice.options)
    proof = ParentCompletion(
        program,
        analyses,
        analyze_scope_access(program),
        1,
        resource_inputs=resource_inputs(choices),
    )
    for option in choice.options:
        resource = option.resources[0]
        sites = resource_sites(choice.uid, choice.bank[0], choice.count, resource)
        assert not proof.resource_covered(choice.uid, *sites)
    resources = [
        (choice.uid, *resource_sites(choice.uid, choice.bank[0], choice.count, option.resources[0]))
        for option in choice.options
    ]
    assert proof.resource_covered_many(iter(resources)) == [False] * len(resources)


def test_all_real_resource_pairs_seed_one_batch_without_changing_ir_or_ids(
    resource_parent, monkeypatch
):
    proof, checked = resource_parent()
    resources = [(uid, consumed, send) for uid, consumed, send, _ in proof.resource_pairs]
    assert resources
    before = dump(checked.program.module)
    calls = {"graph": [], "guarded": [], "anchored": []}

    def record(owner, name):
        original = owner.covered_many

        def batch(queries):
            queries = tuple(queries)
            calls[name].append(queries)
            return original(queries)

        monkeypatch.setattr(owner, "covered_many", batch)

    record(proof.graph, "graph")
    if proof.node_activations:
        record(proof.guarded(), "guarded")
        record(proof.anchored(), "anchored")
    # Duplicates and order belong to the caller, even when solver origins deduplicate.
    requested = [resources[-1], *resources, resources[0]]
    assert proof.resource_covered_many(iter(requested)) == [True] * len(requested)
    if proof.node_activations:
        assert not calls["graph"] and len(calls["guarded"]) == 1
        assert len(calls["anchored"]) == 1
        assert len(calls["guarded"][0]) > len(requested)
    else:
        assert len(calls["graph"]) == 1 and not calls["guarded"] and not calls["anchored"]
        assert len(calls["graph"][0]) == len(requested) * proof.period
    assert dump(checked.program.module) == before


def test_resource_checks_and_initial_allocation_use_the_same_batch(resource_parent, monkeypatch):
    proof, checked = resource_parent()
    resources = [(uid, consumed, send) for uid, consumed, send, _ in proof.resource_pairs]
    before = dump(checked.program.module)
    calls = []
    original = proof.resource_covered_many

    def batch(requested):
        requested = tuple(requested)
        calls.append(requested)
        return original(requested)

    def scalar(*args, **kwargs):
        pytest.fail("resource checking and initial allocation must register every pair together")

    monkeypatch.setattr(proof, "resource_covered_many", batch)
    monkeypatch.setattr(proof, "resource_covered", scalar)
    assert proof.check_child_resources() == len(resources) * proof.period
    allocated, _ = _allocate_child_resources(
        checked.program, checked.analyses, checked.access, proof, checked.channels.values()
    )
    assert allocated is proof and calls == [tuple(resources), tuple(resources)]
    assert dump(checked.program.module) == before


def test_one_failing_case_in_each_solver_stays_with_its_actual_resource(
    resource_parent, monkeypatch
):
    proof, _ = resource_parent()
    resources = [(uid, consumed, send) for uid, consumed, send, _ in proof.resource_pairs]
    assert len(resources) >= 3
    if not proof.node_activations:
        original = proof.graph.covered_many
        calls = []

        def fail_residue(queries):
            queries = tuple(queries)
            answers = original(queries)
            assert all(answers)
            answers[0] = False
            calls.append(len(queries))
            return answers

        monkeypatch.setattr(proof.graph, "covered_many", fail_residue)
        assert proof.resource_covered_many(resources) == [False, *([True] * (len(resources) - 1))]
        assert calls == [len(resources) * proof.period]
        return
    assert all(proof.program.leaves[uid].cases for uid, _, _ in resources)
    guarded, anchored = proof.guarded(), proof.anchored()
    direct, boundary = guarded.covered_many, anchored.covered_many
    observed = []

    def fail_direct_case(queries):
        queries = tuple(queries)
        answers = direct(queries)
        assert all(answers)
        # Resource zero's other genuine native cases still have valid paths.
        answers[0] = False
        observed.append(("direct", len(queries)))
        return answers

    def fail_boundary_case(queries):
        queries = tuple(queries)
        answers = boundary(queries)
        assert all(answers)
        # Resource zero already failed. The first anchored group is resource one.
        answers[0] = False
        observed.append(("boundary", len(queries)))
        return answers

    monkeypatch.setattr(guarded, "covered_many", fail_direct_case)
    monkeypatch.setattr(anchored, "covered_many", fail_boundary_case)
    assert proof.resource_covered_many(resources) == [
        False,
        False,
        *([True] * (len(resources) - 2)),
    ]
    assert [name for name, _ in observed] == ["direct", "boundary"]


def test_empty_resource_batch_has_no_solver_work(resource_parent):
    proof, _ = resource_parent()
    assert proof._guarded is None and proof._anchored is None
    assert proof.resource_covered_many(iter(())) == []
    assert proof._guarded is None and proof._anchored is None
