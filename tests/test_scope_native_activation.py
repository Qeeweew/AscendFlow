"""Declared parent synchronization cannot outlive its real native endpoint."""

from dataclasses import replace
from random import Random

import pytest
from xdsl.dialects.builtin import ArrayAttr, IntAttr, StringAttr

from tests.fixtures.guarded import native_guarded_recurrence
from xdsl_npu import compile
from xdsl_npu.completion import CountDomain, Position
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.dialects.physical import ScopeChannelOp, ScopeSyncOp, TaskOp
from xdsl_npu.scope_domains import Activation, encode_activation, legacy_activation
from xdsl_npu.scope_model import extract_scopes
from xdsl_npu.scope_native_activation import _intersects, validate_scope_protocol
from xdsl_npu.scope_predicates import PARENT_CASES, case_activation
from xdsl_npu.scope_sync import ScopeProtocol, verify_scopes
from xdsl_npu.scope_variants import scope_operations

FIRST, LAST = Position(False, 0), Position(True, 0)


@pytest.fixture(scope="module")
def compiled():
    artifact = compile(native_guarded_recurrence, timeout_s=40)
    _, checked = verify_scopes(artifact.plan, _details=True)
    return artifact, checked


def owner(checked, kind):
    guards = {
        (uid, op.uid.data): (kinds, op.native.data)
        for uid, leaf in checked.program.leaves.items()
        for op, kinds in scope_operations(leaf.stream)
        if isinstance(op, TaskOp)
    }
    for key, (uid, site) in checked.access.endpoints.items():
        kinds, native = guards[uid, site.task[1]]
        if site.position.evaluate(checked.program.leaves[uid].stream.count.data) != 0:
            continue
        if kind == "always" and not kinds and native == "copy":
            return key
        if kinds == (kind,):
            return key
    pytest.fail(f"public source has no {kind} native endpoint")


def local(
    checked,
    *,
    source="first",
    destination="always",
    distance=0,
    period=1,
    residues=(0,),
    activation=None,
):
    return ScopeProtocol(
        "native_contract",
        "local",
        (owner(checked, source),),
        (owner(checked, destination),),
        distance,
        period,
        residues,
        (7,),
        activation=activation,
    )


def barrier(checked, *, destination="first", distance=0, period=1, residues=(0,), activation=None):
    return ScopeProtocol(
        "native_contract",
        "barrier",
        (),
        (owner(checked, destination),),
        distance,
        period,
        residues,
        activation=activation,
    )


def test_all_actual_public_protocols_respect_native_existence(compiled):
    _, checked = compiled
    for protocol in checked.channels.values():
        assert validate_scope_protocol(checked.program, checked.access, protocol) is None


def test_first_only_set_needs_explicit_source_epoch_activation(compiled):
    _, checked = compiled
    first = Activation(CountDomain(1, 0, minimum=1), FIRST, FIRST)
    protocol = local(checked, activation=first)
    validate_scope_protocol(checked.program, checked.access, protocol)
    for widened in (Activation(), None):
        with pytest.raises(CompileError, match="set activation exceeds actual native activation"):
            validate_scope_protocol(
                checked.program, checked.access, replace(protocol, activation=widened)
            )


def test_wait_epoch_and_residues_shift_while_legacy_barrier_residues_are_already_destination(
    compiled,
):
    _, checked = compiled
    two = replace(checked.program, count=2)
    # Set g=0 is legal, but the shifted Wait at g=1 cannot execute beside an
    # initializer that exists only at g=0. Omitting the residue shift hides it.
    protocol = local(checked, source="always", destination="first", distance=1, period=2)
    with pytest.raises(CompileError, match="wait activation exceeds actual native activation"):
        validate_scope_protocol(two, checked.access, protocol)
    # A legacy barrier declares destination residue 0 directly. [1,N-1]
    # has no even epoch at N=2; shifting those residues again would reject it.
    protocol = barrier(checked, distance=1, period=2)
    validate_scope_protocol(two, checked.access, protocol)
    with pytest.raises(CompileError, match="barrier activation exceeds actual native activation"):
        validate_scope_protocol(replace(checked.program, count=3), checked.access, protocol)


def test_explicit_barrier_activation_is_source_epoch_even_without_a_set(compiled):
    _, checked = compiled
    first = Activation(CountDomain(1, 0, minimum=2), FIRST, FIRST, 2, (0,))
    protocol = barrier(checked, distance=1, period=2, activation=first)
    with pytest.raises(CompileError, match="barrier activation exceeds actual native activation"):
        validate_scope_protocol(replace(checked.program, count=2), checked.access, protocol)
    validate_scope_protocol(
        replace(checked.program, count=2),
        checked.access,
        replace(protocol, destinations=(owner(checked, "last"),)),
    )


def test_legacy_set_window_and_actual_epoch_residues_can_make_missing_cases_empty(compiled):
    _, checked = compiled
    protocol = local(checked, distance=1, period=2)
    for count in (0, 1, 2, 3):
        validate_scope_protocol(replace(checked.program, count=count), checked.access, protocol)
    with pytest.raises(CompileError, match="set activation exceeds actual native activation"):
        validate_scope_protocol(replace(checked.program, count=4), checked.access, protocol)


def test_original_count_parity_can_disable_a_count_relative_protocol_without_enumerating_n(
    compiled,
):
    _, checked = compiled
    point = Position(True, 1)
    odd = CountDomain(2, 1, minimum=(1 << 50) + 1)
    protocol = local(checked, period=2, activation=Activation(odd, point, point, 2, (0,)))
    validate_scope_protocol(checked.program, checked.access, protocol)
    even = CountDomain(2, 0, minimum=1 << 50)
    with pytest.raises(CompileError, match="set activation exceeds actual native activation"):
        validate_scope_protocol(
            checked.program,
            checked.access,
            replace(protocol, activation=replace(protocol.activation, count=even)),
        )


def test_unknown_actual_endpoint_and_missing_case_metadata_are_rejected(compiled):
    _, checked = compiled
    protocol = barrier(checked, activation=Activation())
    with pytest.raises(CompileError, match="unknown actual native endpoint"):
        validate_scope_protocol(
            checked.program, checked.access, replace(protocol, destinations=(("aiv0", "unknown"),))
        )
    key = protocol.destinations[0]
    uid, _ = checked.access.endpoints[key]
    leaf = checked.program.leaves[uid]
    damaged = replace(
        leaf, cases={label: case for label, case in leaf.cases.items() if label != "middle"}
    )
    program = replace(checked.program, leaves=checked.program.leaves | {uid: damaged})
    with pytest.raises(CompileError, match="all four actual parent cases"):
        validate_scope_protocol(program, checked.access, protocol)
    # An unconditional actual leaf retains the established proof path.
    program = replace(
        checked.program, leaves=checked.program.leaves | {uid: replace(leaf, cases={})}
    )
    validate_scope_protocol(program, checked.access, protocol)


def test_symbolic_intersection_matches_independent_small_n_epoch_enumeration():
    random = Random(6933)
    for _ in range(160):
        period = random.choice((1, 2, 3, 5))
        residues = tuple(r for r in range(period) if random.random() < 0.6) or (0,)
        count_period = random.choice((1, 2, 3))
        count = CountDomain(count_period, random.randrange(count_period), minimum=2, maximum=12)
        if random.randrange(2):
            first, last = Position(False, random.randrange(2)), Position(True, random.randrange(2))
        else:
            first = last = Position(bool(random.randrange(2)), random.randrange(2))
        required = Activation(count, first, last, period, residues)
        for label in PARENT_CASES:
            missing = case_activation(label)
            expected = any(
                count.contains(n)
                and missing.count.contains(n)
                and any(
                    g % period in residues
                    for g in range(
                        max(first.evaluate(n), missing.first.evaluate(n)),
                        min(last.evaluate(n), missing.last.evaluate(n)) + 1,
                    )
                )
                for n in range(13)
            )
            assert _intersects(required, missing, None) == expected


@pytest.mark.parametrize("mutation", ["widen", "erase"])
def test_public_final_verifier_rejects_parent_barrier_broader_than_its_first_only_native(
    first_barrier_plan, mutation
):
    plan = first_barrier_plan.clone()
    declaration = next(
        op
        for op in plan.walk()
        if isinstance(op, ScopeChannelOp) and op.sym_name.data == "first_native_contract"
    )
    if mutation == "widen":
        declaration.attributes["activation"] = encode_activation(legacy_activation(0))
    else:
        del declaration.attributes["activation"]
    with pytest.raises(CompileError, match="activation exceeds actual native activation"):
        verify_scopes(plan)


@pytest.fixture(scope="module")
def first_barrier_plan(compiled):
    """A real first-only fence added to the cached public compiled native IR."""
    artifact, checked = compiled
    key = owner(checked, "first")
    uid, site = checked.access.endpoints[key]
    plan = artifact.plan.clone()
    program = extract_scopes(plan)
    leaf = program.leaves[uid]
    position = site.position.evaluate(leaf.stream.count.data)
    name = "first_native_contract"
    declaration = ScopeChannelOp.create(
        attributes={
            "sym_name": StringAttr(name),
            "kind": StringAttr("barrier"),
            "sources": ArrayAttr([]),
            "destinations": ArrayAttr([StringAttr(f"{key[0]}/{uid}/{site.task[1]}/{position}")]),
            "distance": IntAttr(0),
            "period": IntAttr(1),
            "residues": ArrayAttr([IntAttr(0)]),
            "event_ids": ArrayAttr([]),
            "activation": encode_activation(Activation(CountDomain(1, 0, minimum=1), FIRST, FIRST)),
        }
    )
    program.root.body.block.insert_op_before(declaration, program.parents[leaf.actor])
    native = next(
        op for op in leaf.stream.walk() if isinstance(op, TaskOp) and op.uid.data == site.task[1]
    )
    fence = ScopeSyncOp.create(
        operands=[leaf.call.generation, leaf.stream.body.block.args[0]],
        attributes={
            "action": StringAttr("barrier"),
            "channel": StringAttr(name),
            "position": IntAttr(position),
        },
    )
    native.parent.insert_op_before(fence, native)
    verify_scopes(plan)  # Establish a valid actual declaration/placement baseline.
    return plan
