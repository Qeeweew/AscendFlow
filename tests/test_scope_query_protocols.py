"""Actual local query instructions retain three clocks and independent KV domains."""

from dataclasses import replace

import pytest
from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, IntAttr, StringAttr

from tests.fixtures.scoped_queries import scoped_query_recurrence
from xdsl_npu.budget import limit
from xdsl_npu.completion import CountDomain, Position
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.dialects.io import dump, parse
from xdsl_npu.dialects.physical import (
    BufferOp,
    CaptureOp,
    LoopSyncOp,
    QueryChannelOp,
    QuerySyncOp,
    ScopeSyncOp,
    TaskOp,
    ViewOp,
)
from xdsl_npu.lowering import lower_source
from xdsl_npu.scope_access import analyze_scope_access
from xdsl_npu.scope_model import extract_scopes
from xdsl_npu.scope_query_completion import QuerySite
from xdsl_npu.scope_query_protocols import (
    QueryProtocol,
    install_query_protocols,
    read_query_protocols,
    validate_query_protocol,
)
from xdsl_npu.scope_sync import ScopeProtocol, _install, _plan_scope_parent

FIRST, LAST = Position(False, 0), Position(True, 0)
POSITIVE = CountDomain(1, 0, minimum=1)


@pytest.fixture(scope="module")
def template():
    with limit(60):
        module = lower_source(scoped_query_recurrence)
        _plan_scope_parent(extract_scopes(module))
    return module


@pytest.fixture
def actual(template):
    program = extract_scopes(template.clone())
    return program, analyze_scope_access(program)


def endpoint(actual, *, actor="aiv0", pipe=None, native=None, query=0, parent=LAST, source=True):
    program, access = actual
    for uid, site in access.endpoints.values():
        if site.task[0] != actor or site.position.offset:
            continue
        task = next(
            task
            for data in program.leaves[uid].loop.actors.values()
            for task in data.program.tasks
            if task.uid == site.task[1]
        )
        if (pipe is None or task.pipe == pipe) and (native is None or task.native == native):
            return QuerySite(query, uid, replace(site, point="end" if source else "begin"), parent)
    pytest.fail("fixture has no matching actual native")


def protocol(actual, *, name="query_release", distance=1, ids=(7,), domain=POSITIVE):
    return QueryProtocol(
        name,
        endpoint(actual, pipe="MTE3"),
        endpoint(actual, native="fill", query=distance, parent=FIRST, source=False),
        domain,
        ids,
    )


def installed(actual, **kwargs):
    value = protocol(actual, **kwargs)
    install_query_protocols(*actual, (value,))
    return value


def synchronization(program, action):
    return next(
        op
        for op in program.module.walk()
        if isinstance(op, QuerySyncOp) and op.action.data == action
    )


@pytest.mark.parametrize(
    "domain",
    [
        POSITIVE,
        CountDomain(2, 1, minimum=3, maximum=101),
        CountDomain(1, 0, exact=1),
    ],
)
def test_actual_ir_roundtrip_keeps_clocks_positions_count_and_native_storage(actual, domain):
    program, access = actual
    preserved = {
        op: (dict(op.attributes), tuple(op.operands))
        for op in program.module.walk()
        if isinstance(op, (BufferOp, CaptureOp, TaskOp, ViewOp, LoopSyncOp, ScopeSyncOp))
    }
    value = installed(actual, domain=domain)
    assert read_query_protocols(program, access) == (value,)
    assert all(
        op.attributes == attrs and tuple(op.operands) == operands
        for op, (attrs, operands) in preserved.items()
    )
    for action, point in (("set", value.source), ("wait", value.target)):
        op = synchronization(program, action)
        leaf = program.leaves[point.call]
        assert op.query_generation is program.invocations[leaf.actor].op.body.block.args[0]
        assert op.parent_generation is program.parents[leaf.actor].body.block.args[0]
        assert op.child_generation is leaf.stream.body.block.args[0]
        assert len({op.query_generation, op.parent_generation, op.child_generation}) == 3
    module = parse(dump(program.module))
    restored = extract_scopes(module)
    assert read_query_protocols(restored, analyze_scope_access(restored)) == (value,)
    assert all(
        not isinstance(op, QuerySyncOp)
        for leaf in restored.leaves.values()
        for op in leaf.module.walk()
    )


def test_query_distance_two_and_prefix_are_independent_of_kv_positions(actual):
    value = protocol(actual, distance=2, ids=(5, 6), domain=CountDomain(1, 0, exact=1))
    value = replace(
        value, source=replace(value.source, site=replace(value.source.site, point="prefix"))
    )
    install_query_protocols(*actual, (value,))
    assert read_query_protocols(*actual) == (value,)
    assert value.target.query == 2 and value.source.parent == LAST and value.target.parent == FIRST


def test_unallocated_candidate_is_valid_but_cannot_be_installed(actual):
    value = protocol(actual, ids=())
    validate_query_protocol(*actual, value)
    before = dump(actual[0].module)
    with pytest.raises(CompileError, match="installed events need an ID"):
        install_query_protocols(*actual, (value,))
    assert dump(actual[0].module) == before


@pytest.mark.parametrize("ids", [(8,), (-1,), (1, 1), (True,), ("1",)])
def test_invalid_ids_are_rejected_before_mutating(actual, ids):
    before = dump(actual[0].module)
    with pytest.raises(CompileError, match="distinct IDs"):
        install_query_protocols(*actual, (protocol(actual, ids=ids),))
    assert dump(actual[0].module) == before


@pytest.mark.parametrize("distance", [0, -1, True, 1 << 63])
def test_query_distance_has_its_own_positive_int64_domain(actual, distance):
    with pytest.raises(CompileError, match="positive query distance"):
        validate_query_protocol(*actual, protocol(actual, distance=distance))


@pytest.mark.parametrize(
    "kind", ["same_pipe", "cross_actor", "source_begin", "target_end", "source_query"]
)
def test_only_real_local_retirement_to_begin_protocols_are_accepted(actual, kind):
    value = protocol(actual)
    if kind == "same_pipe":
        value = replace(value, source=endpoint(actual, native="add"))
    elif kind == "cross_actor":
        value = replace(
            value,
            target=endpoint(
                actual, actor="aiv1", native="fill", query=1, parent=FIRST, source=False
            ),
        )
    elif kind == "source_begin":
        value = replace(
            value, source=replace(value.source, site=replace(value.source.site, point="begin"))
        )
    elif kind == "target_end":
        value = replace(
            value, target=replace(value.target, site=replace(value.target.site, point="end"))
        )
    else:
        value = replace(value, source=replace(value.source, query=1))
    with pytest.raises(
        CompileError, match="local events|retirement source|positive query distance"
    ):
        validate_query_protocol(*actual, value)


@pytest.mark.parametrize("mutation", ["source_first", "target_last", "inner", "region", "call"])
def test_endpoint_existence_never_borrows_inactive_native_cases(actual, mutation):
    value = protocol(actual, domain=CountDomain(1, 0, minimum=2))
    if mutation == "source_first":
        value = replace(value, source=replace(value.source, parent=FIRST))
    elif mutation == "target_last":
        value = replace(value, target=replace(value.target, parent=LAST))
    elif mutation == "inner":
        value = replace(
            value,
            source=replace(
                value.source, site=replace(value.source.site, position=Position(False, 99))
            ),
        )
    elif mutation == "region":
        value = replace(
            value,
            source=replace(
                value.source, site=replace(value.source.site, region=value.target.call + "_other")
            ),
        )
    else:
        value = replace(value, source=replace(value.source, call=value.source.call + "_other"))
    with pytest.raises(CompileError, match="activation|endpoint|identity|retirement source"):
        validate_query_protocol(*actual, value)


@pytest.mark.parametrize(
    "domain",
    [
        CountDomain(1, 0, minimum=0),
        CountDomain(0, 0),
        CountDomain(2, 2),
        CountDomain(1, 0, minimum=4, maximum=2),
        CountDomain(1, 0, minimum=1 << 63),
    ],
)
def test_count_domain_is_kv_and_all_declared_counts_need_real_endpoints(actual, domain):
    with pytest.raises(CompileError, match="domain|count|activation|CountDomain"):
        validate_query_protocol(*actual, protocol(actual, domain=domain))


@pytest.mark.parametrize("clock", [0, 1, 2])
def test_reader_rejects_each_substituted_actual_clock(actual, clock):
    installed(actual)
    op = synchronization(actual[0], "set")
    operands = list(op.operands)
    operands[clock] = operands[(clock + 1) % 3]
    op.operands = operands
    with pytest.raises(CompileError, match="query/KV/child clock"):
        read_query_protocols(*actual)


@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "position", "action", "escaped", "child_command"]
)
def test_reader_reconstructs_actual_native_placement(actual, mutation):
    if mutation == "child_command":
        value = QueryProtocol(
            "query_release",
            endpoint(actual, actor="aic", pipe="MTE2"),
            endpoint(actual, actor="aic", pipe="MTE1", query=1, parent=FIRST, source=False),
            POSITIVE,
            (7,),
        )
        install_query_protocols(*actual, (value,))
    else:
        installed(actual)
    program = actual[0]
    op = synchronization(program, "wait")
    if mutation == "missing":
        op.parent.erase_op(op)
    elif mutation == "duplicate":
        op.parent.insert_op_before(op.clone(), op)
    elif mutation == "position":
        op.attributes["position"] = IntAttr(1)
    elif mutation == "action":
        op.attributes["action"] = StringAttr("barrier")
    elif mutation == "escaped":
        op.parent.detach_op(op)
        program.root.body.block.insert_op_before(op, next(iter(program.kernels.values())))
    else:
        # Crossing a real child Wait must invalidate the native anchor.
        preceding = op.prev_op
        while isinstance(preceding, ScopeSyncOp):
            preceding = preceding.prev_op
        assert isinstance(preceding, LoopSyncOp)
        op.parent.detach_op(op)
        preceding.parent.insert_op_before(op, preceding)
    with pytest.raises(CompileError, match="placement|duplicate|local Set/Wait|escaped|adjacent"):
        read_query_protocols(*actual)


def test_same_action_scope_then_query_then_child_order_is_both_installed_and_checked(actual):
    program, access = actual
    source = endpoint(actual, actor="aic", pipe="MTE2")
    target = endpoint(actual, actor="aic", pipe="MTE1", query=1, parent=FIRST, source=False)
    owners = [
        next(
            owner
            for owner, (uid, site) in access.endpoints.items()
            if uid == point.call and replace(point.site, point=site.point) == site
        )
        for point in (source, target)
    ]
    _install(
        program,
        access,
        (ScopeProtocol("scope_order", "local", (owners[0],), (owners[1],), 0, 1, (0,), (7,)),),
    )
    value = QueryProtocol("query_order", source, target, POSITIVE, (6,))
    install_query_protocols(program, access, (value,))
    assert read_query_protocols(program, access) == (value,)
    for action in ("set", "wait"):
        query = synchronization(program, action)
        assert isinstance(query.prev_op, ScopeSyncOp)
        scope = query.prev_op
        query.parent.detach_op(query)
        scope.parent.insert_op_before(query, scope)
        with pytest.raises(CompileError, match="ScopeSync.*precede QuerySync"):
            read_query_protocols(program, access)
        query.parent.detach_op(query)
        scope.parent.insert_op_after(query, scope)
    assert read_query_protocols(program, access) == (value,)


@pytest.mark.parametrize(
    "field", ["kind", "count", "source_parent", "source_point", "distance", "event_ids"]
)
def test_declaration_cannot_hide_unsupported_forms_or_invalid_fields(actual, field):
    installed(actual)
    declaration = next(op for op in actual[0].module.walk() if isinstance(op, QueryChannelOp))
    values = {
        "kind": StringAttr("cross"),
        "count": DictionaryAttr(
            {"period": IntAttr(1), "residue": IntAttr(0), "query_count": IntAttr(2)}
        ),
        "source_parent": DictionaryAttr({"end": IntAttr(2), "offset": IntAttr(0)}),
        "source_point": StringAttr("query_exit"),
        "distance": IntAttr(0),
        "event_ids": ArrayAttr([]),
    }
    declaration.attributes[field] = values[field]
    with pytest.raises(
        CompileError,
        match="local event fields|fields|retirement source|positive query distance|installed events",
    ):
        read_query_protocols(*actual)


def test_native_mutation_invalidates_cached_extraction(actual):
    value = protocol(actual)
    program = actual[0]
    native = next(
        op
        for op in program.leaves[value.source.call].stream.walk()
        if isinstance(op, TaskOp) and op.uid.data == value.source.site.task[1]
    )
    native.attributes["scalar"] = next(
        op for op in program.module.walk() if isinstance(op, TaskOp) and op.native.data == "muls"
    ).scalar
    with pytest.raises(CompileError, match="stale native/view descriptor"):
        validate_query_protocol(*actual, value)


def test_duplicate_names_and_partial_batches_do_not_mutate_ir(actual):
    value = protocol(actual)
    before = dump(actual[0].module)
    with pytest.raises(CompileError, match="duplicate query/parent protocol name"):
        install_query_protocols(*actual, (value, value))
    assert dump(actual[0].module) == before


def test_one_actual_local_edge_does_not_cover_all_public_query_obligations(actual):
    from xdsl_npu.scope_sync import verify_scopes

    installed(actual)
    with pytest.raises(CompileError, match="missing cross-query memory and event lifetime"):
        verify_scopes(actual[0].module)
