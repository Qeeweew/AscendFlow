"""Small independent oracle for definition ownership and parent activation.

This interprets the QK/PV source policy directly. Its total native order is a
candidate order whose retirement edges still need compiler completion proofs;
dispatch alone is never a certificate. No production access/frontier or sync
planner contributes either the semantic bindings or the expected result.
"""

from collections import defaultdict
from dataclasses import dataclass
from itertools import pairwise

import pytest
from xdsl.dialects import arith
from xdsl.dialects.builtin import ArrayAttr, IntAttr, StringAttr

from tests.fixtures.nested import shared_l0c_qk_pv
from xdsl_npu.completion import CountDomain, Position, Site
from xdsl_npu.dialects.physical import (
    BufferOp,
    CallOp,
    CaptureOp,
    CoreIdOp,
    KernelOp,
    OnceOp,
    RuntimeSizeOp,
    StreamOp,
    TaskOp,
    ViewOp,
)
from xdsl_npu.scope_lowering import lower_scopes


@dataclass(frozen=True)
class Activation:
    count: CountDomain
    first: Position
    last: Position
    step: int = 1

    def anchors(self, count):
        if not self.count.contains(count):
            return ()
        return range(self.first.evaluate(count), self.last.evaluate(count) + 1, self.step)


@dataclass(frozen=True)
class Endpoint:
    call: str
    site: Site
    parent_shift: int = 0

    def at(self, anchor):
        # The two Position coordinates are intentionally evaluated separately.
        inner = self.site.position.evaluate(1 if self.call == "FIX" else 8)
        return self.call, anchor + self.parent_shift, self.site.task[1], inner


@dataclass(frozen=True)
class Relation:
    source: Endpoint
    destination: Endpoint
    activation: Activation

    def for_count(self, count):
        return {
            (self.source.at(anchor), self.destination.at(anchor))
            for anchor in self.activation.anchors(count)
        }


def _endpoint(call, native, inner, shift=0):
    point = "end" if native == "FIX" else "begin"
    return Endpoint(call, Site(("aic", native), Position(False, inner), point, region=call), shift)


def _activation(minimum, first, last, *, exact=None):
    return Activation(CountDomain(1, 0, exact=exact, minimum=minimum), first, last)


# These are source-policy deductions, not samples of a production frontier.
C0_RETIREMENTS = (
    Relation(
        _endpoint("QK", "FIX", 6),
        _endpoint("PV", "MAD", 0),
        _activation(1, Position(False, 0), Position(False, 0), exact=1),
    ),
    Relation(
        _endpoint("QK", "FIX", 6),
        _endpoint("QK", "MAD", 0, 1),
        _activation(2, Position(False, 0), Position(False, 0)),
    ),
    Relation(
        _endpoint("QK", "FIX", 6, 1),
        _endpoint("PV", "MAD", 0),
        _activation(2, Position(False, 0), Position(True, 1)),
    ),
    Relation(
        _endpoint("FIX", "FIX", 0),
        _endpoint("QK", "MAD", 0, 2),
        _activation(3, Position(False, 0), Position(True, 2)),
    ),
    Relation(
        _endpoint("FIX", "FIX", 0),
        _endpoint("PV", "MAD", 0, 1),
        _activation(2, Position(True, 1), Position(True, 1)),
    ),
)


@dataclass(frozen=True)
class Event:
    uid: tuple
    reads: tuple[object, ...] = ()
    writes: tuple[object, ...] = ()


def _qk(generation, *, physical, shared_operands):
    parts = []
    for inner in range(8):
        slot = inner % 2
        operands = (f"a{slot}", f"b{slot}") if shared_operands else ()
        if operands:
            parts.append((inner + 1, 1, Event(("QK", generation, "OPERANDS", inner), (), operands)))
        parts.append(
            (inner + 2, 0, Event(("QK", generation, "MAD", inner), operands, (f"c{slot}",)))
        )
        parts.append((inner + 3, 2, Event(("QK", generation, "FIX", inner), (f"c{slot}",))))
    logical_order = {"OPERANDS": 0, "MAD": 1, "FIX": 2}
    parts.sort(
        key=(lambda item: item[:2])
        if physical
        else (lambda item: (item[2].uid[3], logical_order[item[2].uid[2]]))
    )
    return tuple(event for _, _, event in parts)


def _pv(generation, *, shared_operands, initialize):
    operands = ("a0", "b0") if shared_operands else ()
    return tuple(
        Event(
            ("PV", generation, "MAD", inner),
            operands + (("c0",) if inner or not initialize else ()),
            ("c0",),
        )
        for inner in range(8)
    )


def _trace(count, *, physical, shared_operands=False, initialize=True, empty_qk=False):
    calls = []
    if physical:
        for step in range(count + 1):
            if step < count:
                calls.append(("QK", step))
            if 0 <= step - 1 < count:
                calls.extend((("PV", step - 1), ("FIX", step - 1)))
    else:
        calls = [(role, generation) for generation in range(count) for role in ("QK", "PV", "FIX")]
    result = []
    for role, generation in calls:
        if role == "QK":
            if not empty_qk:
                result.extend(_qk(generation, physical=physical, shared_operands=shared_operands))
        elif role == "PV":
            result.extend(_pv(generation, shared_operands=shared_operands, initialize=initialize))
        else:
            result.append(Event((role, generation, "FIX", 0), ("c0",)))
    return tuple(result)


def _logical_model(events):
    writers, owner, bindings, last_write, readers = {}, {}, {}, {}, {}
    for event in events:
        for cell in event.reads:
            if cell not in writers:
                raise ValueError(f"undefined {cell} at {event.uid}")
            writer = writers[cell]
            bindings[event.uid, cell] = writer
            readers[owner[writer, cell], cell].append(event.uid)
        # All reads precede all writes, including aliased R/W operands.
        for cell in event.writes:
            definition = owner[writers[cell], cell] if cell in event.reads else event.uid
            writers[cell] = event.uid
            owner[event.uid, cell] = definition
            last_write[definition, cell] = event.uid
            readers[definition, cell] = []
    return bindings, owner, last_write, readers


def _validate_bindings(events, bindings):
    current = {}
    for event in events:
        for cell in event.reads:
            if current.get(cell) != bindings[event.uid, cell]:
                raise ValueError(f"producer binding changed for {cell} at {event.uid}")
        for cell in event.writes:
            current[cell] = event.uid


def _c0_cross_call_retirements(events, model, cell="c0"):
    _, owner, _, readers = model
    definitions = [
        event.uid
        for event in events
        if cell in event.writes and owner[event.uid, cell] == event.uid
    ]
    return {
        (reader, following)
        for previous, following in pairwise(definitions)
        for reader in readers[previous, cell]
        if reader[:2] != following[:2]
    }


@pytest.mark.parametrize("count", range(7))
def test_c0_activation_families_equal_independent_definition_oracle(count):
    logical = _trace(count, physical=False)
    physical = _trace(count, physical=True)
    model = _logical_model(logical)
    _validate_bindings(physical, model[0])
    actual = set().union(*(relation.for_count(count) for relation in C0_RETIREMENTS))
    assert actual == _c0_cross_call_retirements(physical, model)
    for source, destination in actual:
        assert 0 <= source[1] < count and 0 <= destination[1] < count


def test_pv_update_chain_and_separate_fix_keep_logical_producers():
    bindings, owner, _, readers = _logical_model(_trace(3, physical=False))
    for generation in range(3):
        definition = "PV", generation, "MAD", 0
        for inner in range(1, 8):
            update = "PV", generation, "MAD", inner
            assert bindings[update, "c0"] == ("PV", generation, "MAD", inner - 1)
            assert owner[update, "c0"] == definition
        assert bindings[("FIX", generation, "FIX", 0), "c0"] == ("PV", generation, "MAD", 7)
        assert readers[definition, "c0"] == [("FIX", generation, "FIX", 0)]


def test_steady_domain_must_not_substitute_for_startup_or_drain():
    only_steady = C0_RETIREMENTS[2:4]
    for count in (1, 2, 3):
        model = _logical_model(_trace(count, physical=False))
        expected = _c0_cross_call_retirements(_trace(count, physical=True), model)
        assert set().union(*(relation.for_count(count) for relation in only_steady)) != expected
    assert C0_RETIREMENTS[2].source.parent_shift > C0_RETIREMENTS[2].destination.parent_shift


def test_shared_qk_operands_cannot_be_rebound_to_next_parent():
    bindings = _logical_model(_trace(2, physical=False, shared_operands=True))[0]
    with pytest.raises(ValueError, match="producer binding changed for a0.*PV.*0"):
        _validate_bindings(_trace(2, physical=True, shared_operands=True), bindings)


def test_first_pv_rw_is_a_recurrence_not_an_independent_definition():
    bindings = _logical_model(_trace(2, physical=False, initialize=False))[0]
    assert bindings[("PV", 0, "MAD", 0), "c0"] == ("QK", 0, "MAD", 6)
    with pytest.raises(ValueError, match="producer binding changed for c0.*PV.*0"):
        _validate_bindings(_trace(2, physical=True, initialize=False), bindings)


def test_empty_child_is_not_an_initialized_incoming_definition():
    assert _logical_model(_trace(0, physical=False, empty_qk=True)) == ({}, {}, {}, {})
    with pytest.raises(ValueError, match="undefined c0.*PV.*0"):
        _logical_model(_trace(1, physical=False, empty_qk=True, initialize=False))


def test_final_external_observer_preserves_last_logical_output_version():
    first = Event(("write", 0), (), ("external",))
    last = Event(("write", 1), (), ("external",))
    observe = Event(("external_observer",), ("external",))
    bindings = _logical_model((first, last, observe))[0]
    with pytest.raises(ValueError, match="producer binding changed for external"):
        _validate_bindings((last, first, observe), bindings)


def test_every_pending_reader_needs_retirement_before_reuse():
    define = Event(("write",), (), ("c0",))
    dma = Event(("MTE2",), ("c0",))
    fix = Event(("FIX",), ("c0",))
    _, _, _, readers = _logical_model((define, dma, fix))
    assert readers[define.uid, "c0"] == [dma.uid, fix.uid]


# Actual physical-IR interpretation, independent of the handwritten trace above.
# Only 16x16 whole-fractal Cube recipes in shared_l0c_qk_pv are supported. A
# total candidate retirement order selects versions; these tests do not claim
# that dispatch establishes those retirement edges on a device.


@dataclass(frozen=True)
class Footprint:
    root: BufferOp
    rows: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class Native:
    uid: tuple
    task: TaskOp
    refs: tuple[Footprint, ...]
    modes: tuple[str, ...]


def _int(attribute, default=0):
    return default if attribute is None else attribute.data


def _index(value, environment, count):
    if value in environment:
        return environment[value]
    op = value.owner
    if isinstance(op, RuntimeSizeOp):
        assert op.sym_name.data == "n"
        return count
    if isinstance(op, CoreIdOp):
        return 0
    if isinstance(op, arith.ConstantOp):
        return op.value.value.data
    functions = {
        arith.AddiOp: lambda a, b: a + b,
        arith.MuliOp: lambda a, b: a * b,
        arith.DivUIOp: lambda a, b: a // b,
        arith.RemUIOp: lambda a, b: a % b,
    }
    assert type(op) in functions, "unsupported actual fixture index arithmetic"
    return functions[type(op)](*(_index(v, environment, count) for v in (op.lhs, op.rhs)))


def _root_extent(root, count):
    return (
        root.nbytes.data
        if root.extent_count is None
        else _index(root.extent_count, {}, count) * root.count_scale.data
    )


def _footprint(value, environment, count, child_count):
    view = value.owner
    assert isinstance(view, ViewOp)
    capture = view.source.owner
    assert isinstance(capture, CaptureOp) and isinstance(capture.source.owner, BufferOp)
    root = capture.source.owner
    base = _index(capture.offset, environment, count)
    assert 0 <= base and base + capture.nbytes.data <= _root_extent(root, count)
    generation = _index(view.generation, environment, count)
    coordinate = generation // _int(view.divisor, 1)
    modulus = view.modulus.data
    offset = view.offset.data + view.stride.data * (coordinate % modulus if modulus else coordinate)
    for term in view.index_terms or ():
        coefficient, divisor, modulus = (v.data for v in term)
        offset += coefficient * (
            generation // divisor % modulus if modulus else generation // divisor
        )
    assert not _int(view.core_stride) and not view.core_terms
    assert not _int(view.prefix_elements) and not view.tail_nbytes.data
    offset += _int(view.count_offset) * child_count
    assert not _int(view.count_core_stride)
    rows = tuple(
        (base + offset + row * _int(view.row_stride), view.nbytes.data)
        for row in range(_int(view.rows, 1))
    )
    for start, width in rows:
        assert base <= start < start + width <= base + capture.nbytes.data
        assert start + width <= _root_extent(root, count)
    return Footprint(root, rows)


def _native_modes(task, generation):
    if task.native.data in {"nd2nz", "load2d_a", "load2d_b_nt", "fix"}:
        return "R", "W"
    if task.native.data == "mad_first":
        return "R", "R", "W" if generation == 0 else "RW"
    assert task.native.data == "mad", "unsupported actual fixture native"
    assert len(task.params) == 4 and task.params.data[3].data in (0, 1)
    return "R", "R", "W" if task.params.data[3].data else "RW"


def _actual_calls(plan):
    kernel = plan.body.block.first_op
    assert isinstance(kernel, KernelOp) and kernel.core.data == "aic"
    parent = next(op for op in kernel.body.block.ops if isinstance(op, StreamOp))
    calls = [op for op in parent.body.block.ops if isinstance(op, CallOp)]
    assert len(calls) == 3
    logical = sorted(enumerate(calls), key=lambda item: (item[1].logical_phase.data, item[0]))
    roles = {call: role for (_, call), role in zip(logical, ("QK", "PV", "FIX"), strict=True)}
    return parent, calls, roles


def _actual_natives(plan, count, *, physical):
    parent, calls, roles = _actual_calls(plan)
    assert _index(parent.limit, {}, count) == count
    if physical:
        call_order = sorted(enumerate(calls), key=lambda item: (item[1].issue_order.data, item[0]))
        invocations = (
            (call, step - call.stage.data)
            for step in range(count + max(call.stage.data for call in calls))
            for _, call in call_order
            if 0 <= step - call.stage.data < count
        )
    else:
        call_order = sorted(
            enumerate(calls), key=lambda item: (item[1].logical_phase.data, item[0])
        )
        invocations = ((call, pg) for pg in range(count) for _, call in call_order)
    result = []
    for call, pg in invocations:
        child = next(op for op in call.body.block.ops if isinstance(op, (StreamOp, OnceOp)))
        tasks = [op for op in child.body.block.ops if isinstance(op, TaskOp)]
        child_count = child.count.data
        if physical:
            task_order = sorted(
                enumerate(tasks), key=lambda item: (_int(item[1].issue_order), item[0])
            )
            instances = (
                (task, step - task.stage.data)
                for step in range(child_count + max(task.stage.data for task in tasks))
                for _, task in task_order
                if 0 <= step - task.stage.data < child_count
            )
        else:
            task_order = sorted(
                enumerate(tasks), key=lambda item: (_int(item[1].logical_phase), item[0])
            )
            instances = ((task, g) for g in range(child_count) for _, task in task_order)
        for task, g in instances:
            upper = _int(task.upper, -1)
            if not _int(task.lower) <= g < (child_count if upper == -1 else upper):
                continue
            if (
                not _int(task.phase)
                <= g % _int(task.period, 1)
                < (_int(task.phase) + _int(task.phase_width, 1))
            ):
                continue
            environment = {parent.body.block.args[0]: pg, child.body.block.args[0]: g}
            refs = tuple(_footprint(ref, environment, count, child_count) for ref in task.refs)
            modes = _native_modes(task, g)
            assert len(refs) == len(modes)
            result.append(
                Native((roles[call], pg, task.native.data, g, task.uid.data), task, refs, modes)
            )
    return tuple(result)


def _actual_events(commands):
    """Partition at every actual row boundary; every byte in an atom has one history.

    This is exact interval partitioning of instantiated references, rather than
    a tile or enclosing-rectangle approximation. Pitched GM rows retain holes.
    No production region summary supplies any interval or producer binding.
    """
    boundaries = defaultdict(set)
    for command in commands:
        for ref in command.refs:
            for start, width in ref.rows:
                boundaries[ref.root].update((start, start + width))
    atoms = {
        root: tuple((root, first, last) for first, last in pairwise(sorted(points)))
        for root, points in boundaries.items()
    }
    events = []
    initial = tuple(
        cell for root, cells in atoms.items() if root.initialized.data for cell in cells
    )
    if initial:
        events.append(Event(("INITIAL",), (), initial))
    observed = set()
    for command in commands:
        reads, writes = set(), set()
        for ref, mode in zip(command.refs, command.modes, strict=True):
            cells = {
                cell
                for start, width in ref.rows
                for cell in atoms[ref.root]
                if start <= cell[1] and cell[2] <= start + width
            }
            if "R" in mode:
                reads.update(cells)
            if "W" in mode:
                writes.update(cells)
                if ref.root.external.data:
                    observed.update(cells)
        key = lambda cell: (cell[0].sym_name.data, cell[1], cell[2])
        events.append(
            Event(command.uid, tuple(sorted(reads, key=key)), tuple(sorted(writes, key=key)))
        )
    if observed:
        events.append(Event(("OBSERVE",), tuple(sorted(observed, key=key))))
    return tuple(events)


def _project_c(events):
    result = []
    for event in events:
        reads, writes = [], []
        for target, cells in ((reads, event.reads), (writes, event.writes)):
            for root, first, last in cells:
                if root.ref.type.bank.data == "L0C":
                    assert first in (0, 1024) and last == first + 1024
                    target.append("c" + str(first // 1024))
        if reads or writes:
            role, pg, native, g, _ = event.uid
            uid = role, pg, "MAD" if native in {"mad", "mad_first"} else "FIX", g
            result.append(Event(uid, tuple(reads), tuple(writes)))
    return tuple(result)


@pytest.fixture(scope="module")
def actual_shared_l0c():
    # Retain the real fixture and its currently rejected schedule unchanged.
    # Lowering supplies only the physical instructions, never a sync proof.
    return lower_scopes(shared_l0c_qk_pv.to_ir().body.block.first_op)


@pytest.mark.parametrize("count", (0, 1, 2, 3, 7))
def test_actual_ref_interpreter_agrees_with_handwritten_c_policy(actual_shared_l0c, count):
    for physical in (False, True):
        actual = _actual_events(_actual_natives(actual_shared_l0c, count, physical=physical))
        assert _project_c(actual) == _trace(count, physical=physical)
    logical = _actual_events(_actual_natives(actual_shared_l0c, count, physical=False))
    candidate = _actual_events(_actual_natives(actual_shared_l0c, count, physical=True))
    _validate_bindings(candidate, _logical_model(logical)[0])


@pytest.mark.parametrize("count", (0, 1, 2, 3, 7))
def test_actual_c0_and_c1_retirement_boundaries_include_startup_and_drain(actual_shared_l0c, count):
    logical = _project_c(_actual_events(_actual_natives(actual_shared_l0c, count, physical=False)))
    candidate = _project_c(_actual_events(_actual_natives(actual_shared_l0c, count, physical=True)))
    model = _logical_model(logical)
    expected = set().union(*(relation.for_count(count) for relation in C0_RETIREMENTS))
    assert _c0_cross_call_retirements(candidate, model) == expected
    assert _c0_cross_call_retirements(candidate, model, "c1") == {
        (("QK", pg, "FIX", 7), ("QK", pg + 1, "MAD", 1)) for pg in range(count - 1)
    }


def test_actual_private_operand_raws_stay_bound_to_their_own_copy(actual_shared_l0c):
    commands = _actual_natives(actual_shared_l0c, 3, physical=True)
    bindings = _logical_model(_actual_events(commands))[0]
    roots = defaultdict(set)
    for command in commands:
        roots[command.uid[0]].update(ref.root for ref in command.refs if not ref.root.external.data)
    assert {root.sym_name.data for root in roots["QK"] & roots["PV"]} == {"c"}
    for command in commands:
        if command.task.native.data not in {"mad", "mad_first"}:
            continue
        for cell, writer in (
            (cell, writer)
            for (reader, cell), writer in bindings.items()
            if reader == command.uid and cell[0].ref.type.bank.data in {"L0A", "L0B"}
        ):
            assert writer[:2] == command.uid[:2] and writer[3] == command.uid[3]
            assert writer[2] == (
                "load2d_a" if cell[0].ref.type.bank.data == "L0A" else "load2d_b_nt"
            )


def _read_qk_operand_zero(plan):
    _, calls, _ = _actual_calls(plan)
    pv = calls[1]
    child = next(op for op in pv.body.block.ops if isinstance(op, StreamOp))
    mad = next(
        op
        for op in child.body.block.ops
        if isinstance(op, TaskOp) and op.native.data == "mad_first"
    )
    for ref, name in zip(mad.refs[:2], ("qk_a", "qk_b"), strict=True):
        view = ref.owner
        old = view.source.owner
        root = next(
            op for op in plan.walk() if isinstance(op, BufferOp) and op.sym_name.data == name
        )
        capture = CaptureOp.create(
            operands=[root.ref, old.offset],
            result_types=[root.ref.type],
            attributes={"sym_name": StringAttr("shared_" + name), "nbytes": IntAttr(512)},
        )
        pv.body.block.insert_op_before(capture, child)
        view.operands = (capture.ref, view.generation)
        view.attributes.pop("index_terms", None)
    plan.verify()


def _pv_recurrence(plan):
    _, calls, _ = _actual_calls(plan)
    task = next(
        op for op in calls[1].walk() if isinstance(op, TaskOp) and op.native.data == "mad_first"
    )
    task.attributes["native"] = StringAttr("mad")
    task.attributes["params"] = ArrayAttr((*task.params, IntAttr(0)))
    plan.verify()


def test_actual_shared_operand_ref_reads_a_different_parent_version(actual_shared_l0c):
    plan = actual_shared_l0c.clone()
    _read_qk_operand_zero(plan)
    logical = _actual_events(_actual_natives(plan, 2, physical=False))
    candidate = _actual_events(_actual_natives(plan, 2, physical=True))
    bindings = _logical_model(logical)[0]
    actual_bindings = _logical_model(candidate)[0]
    pv_reads = {
        key: writer
        for key, writer in bindings.items()
        if key[0][:2] == ("PV", 0) and key[1][0].sym_name.data in {"qk_a", "qk_b"}
    }
    assert len(pv_reads) == 16
    assert all(writer[:4] == ("QK", 0, writer[2], 6) for writer in pv_reads.values())
    assert all(
        actual_bindings[key][:4] == ("QK", 1, writer[2], 6) for key, writer in pv_reads.items()
    )
    # This exactly realizes the handwritten shared_operands assumption with
    # actual Capture/View/Task references; merely aliasing independently
    # rewritten PV scratch would introduce occupancy hazards, not this RAW.
    reference = _logical_model(_trace(2, physical=False, shared_operands=True))[0]
    for (reader, cell), writer in pv_reads.items():
        label = "a0" if cell[0].ref.type.bank.data == "L0A" else "b0"
        assert reference[("PV", 0, "MAD", reader[3]), label] == ("QK", 0, "OPERANDS", writer[3])
    with pytest.raises(ValueError, match="producer binding changed"):
        _validate_bindings(candidate, bindings)


def test_actual_first_pv_rw_cannot_move_past_next_qk_definition(actual_shared_l0c):
    plan = actual_shared_l0c.clone()
    _pv_recurrence(plan)
    logical = _project_c(_actual_events(_actual_natives(plan, 2, physical=False)))
    candidate = _project_c(_actual_events(_actual_natives(plan, 2, physical=True)))
    assert logical == _trace(2, physical=False, initialize=False)
    assert candidate == _trace(2, physical=True, initialize=False)
    bindings = _logical_model(logical)[0]
    assert bindings[("PV", 0, "MAD", 0), "c0"] == ("QK", 0, "MAD", 6)
    with pytest.raises(ValueError, match="producer binding changed for c0.*PV.*0"):
        _validate_bindings(candidate, bindings)


def test_actual_empty_qk_child_cannot_create_an_incoming_c_definition(actual_shared_l0c):
    plan = actual_shared_l0c.clone()
    _pv_recurrence(plan)
    _, calls, _ = _actual_calls(plan)
    qk = next(op for op in calls[0].body.block.ops if isinstance(op, StreamOp))
    qk.attributes["count"] = IntAttr(0)
    assert _actual_natives(plan, 0, physical=True) == ()
    events = _project_c(_actual_events(_actual_natives(plan, 1, physical=False)))
    assert events == _trace(1, physical=False, initialize=False, empty_qk=True)
    with pytest.raises(ValueError, match="undefined c0.*PV.*0"):
        _logical_model(events)


def test_actual_empty_pv_child_does_not_execute_its_syntactic_first_clear(actual_shared_l0c):
    plan = actual_shared_l0c.clone()
    _, calls, _ = _actual_calls(plan)
    pv = next(op for op in calls[1].body.block.ops if isinstance(op, StreamOp))
    pv.attributes["count"] = IntAttr(0)
    events = _project_c(_actual_events(_actual_natives(plan, 1, physical=False)))
    bindings = _logical_model(events)[0]
    assert not any(event.uid[0] == "PV" for event in events)
    assert bindings[("FIX", 0, "FIX", 0), "c0"] == ("QK", 0, "MAD", 6)
    qk = next(op for op in calls[0].body.block.ops if isinstance(op, StreamOp))
    qk.attributes["count"] = IntAttr(0)
    events = _project_c(_actual_events(_actual_natives(plan, 1, physical=False)))
    with pytest.raises(ValueError, match="undefined c0.*FIX"):
        _logical_model(events)


def shared_l0c_device_input(count):
    """Small dyadic f16 values distinguish parent, part, row, and reduction index.

    For counts <=7 all 128 products per output sum exactly in f32. This avoids
    using a numerical tolerance to hide a wrong parent, ring slot, or clear.
    """
    np = pytest.importorskip("numpy")
    parent = np.arange(count)[:, None, None, None]
    part = np.arange(8)[None, :, None, None]
    row, column = np.indices((16, 16))
    values = ((3 * row + 5 * column + 7 * (parent + 1) + 11 * (part + 1)) % 17 - 8) / 8
    return (values + parent / 32 + part / 64)[None].astype(np.float16)


def shared_l0c_device_oracle(x, *, score_before=None):
    """Device expectations for the unchanged shared_l0c_qk_pv source.

    Y[t] = sum_j X[t,j] @ X[t,j].T, in j order with the first MAD clear.
    Score[slot,:,16*j:16*(j+1)] keeps the largest t with t%2==slot.
    Unwritten Score slots retain host initialization, including all of N=0.
    The returned mask identifies bytes/elements the source promises to write.
    """
    np = pytest.importorskip("numpy")
    assert x.dtype == np.float16 and x.shape[:1] == (1,) and x.shape[2:] == (8, 16, 16)
    count = x.shape[1]
    score = (
        np.full((1, 2, 16, 128), np.nan, dtype=np.float32)
        if score_before is None
        else np.array(score_before, dtype=np.float32, copy=True)
    )
    assert score.shape == (1, 2, 16, 128)
    written = np.zeros(score.shape, dtype=np.bool_)
    y = np.empty((1, count, 16, 16), dtype=np.float32)
    values = x.astype(np.float32)
    products = values @ values.swapaxes(-1, -2)
    for parent in range(count):
        for part in range(8):
            value = products[0, parent, part]
            if part == 0:
                y[0, parent] = value
            else:
                y[0, parent] += value
            columns = slice(16 * part, 16 * (part + 1))
            score[0, parent % 2, :, columns] = value
            written[0, parent % 2, :, columns] = True
    return score, y, written


def _numeric_candidate(plan, x, score_before, *, physical):
    """Execute actual native refs in one candidate total retirement order.

    Internal whole 16x16 fractals are represented as logical matrices. Their
    byte slot addresses remain actual Ref addresses; the model does not claim
    a general NZ/ZZ/ZN packing implementation or asynchronous device proof.
    """
    np = pytest.importorskip("numpy")
    count = x.shape[1]
    memory = {}
    roots = {op.sym_name.data: op for op in plan.walk() if isinstance(op, BufferOp)}
    for name, root in roots.items():
        dtype = {"f16": np.float16, "f32": np.float32}[root.ref.type.dtype.data]
        memory[root] = np.full(
            _root_extent(root, count) // np.dtype(dtype).itemsize, np.nan, dtype=dtype
        )
        if name == "X":
            memory[root][:] = x.ravel()
        elif name == "Score":
            memory[root][:] = score_before.ravel()

    def read(ref):
        values = memory[ref.root]
        itemsize = values.dtype.itemsize
        for start, width in ref.rows:
            assert start % itemsize == width % itemsize == 0
        return np.concatenate(
            [values[start // itemsize : (start + width) // itemsize] for start, width in ref.rows]
        ).reshape(16, 16)

    def write(ref, value):
        values, flat = memory[ref.root], value.ravel()
        itemsize, position = values.dtype.itemsize, 0
        for start, width in ref.rows:
            size = width // itemsize
            values[start // itemsize : start // itemsize + size] = flat[position : position + size]
            position += size
        assert position == len(flat)

    for command in _actual_natives(plan, count, physical=physical):
        native = command.task.native.data
        assert tuple(v.data for v in command.task.params)[:2] == (16, 16)
        if native in {"nd2nz", "load2d_a", "load2d_b_nt", "fix"}:
            value = read(command.refs[0])
            write(command.refs[1], value.T if native == "load2d_b_nt" else value)
        else:
            a, b = (read(ref).astype(np.float32) for ref in command.refs[:2])
            value = a @ b
            if command.modes[2] == "RW":
                value += read(command.refs[2])
            write(command.refs[2], value)
    return memory[roots["Score"]].reshape(score_before.shape), memory[roots["Y"]].reshape(
        1, count, 16, 16
    )


@pytest.mark.parametrize("count", (0, 1, 2, 3, 7))
def test_actual_native_numbers_match_the_same_fixture_device_oracle(actual_shared_l0c, count):
    np = pytest.importorskip("numpy")
    x = shared_l0c_device_input(count)
    before = (np.arange(4096, dtype=np.float32) / 16 - 2048).reshape(1, 2, 16, 128)
    score, y, written = shared_l0c_device_oracle(x, score_before=before)
    assert written.sum() == min(count, 2) * 2048
    for physical in (False, True):
        actual_score, actual_y = _numeric_candidate(actual_shared_l0c, x, before, physical=physical)
        np.testing.assert_array_equal(actual_score, score)
        np.testing.assert_array_equal(actual_y, y)
    for slot in range(2):
        parents = list(range(slot, count, 2))
        if parents:
            last = x[0, parents[-1]].astype(np.float32)
            expected = np.concatenate([part @ part.T for part in last], axis=1)
            np.testing.assert_array_equal(score[0, slot], expected)
        else:
            np.testing.assert_array_equal(score[0, slot], before[0, slot])


@pytest.mark.parametrize("mutation", (_read_qk_operand_zero, _pv_recurrence))
def test_device_inputs_expose_real_raw_version_changes_numerically(actual_shared_l0c, mutation):
    np = pytest.importorskip("numpy")
    plan = actual_shared_l0c.clone()
    mutation(plan)
    x = shared_l0c_device_input(2)
    before = np.full((1, 2, 16, 128), -7777.25, dtype=np.float32)
    _, logical_y = _numeric_candidate(plan, x, before, physical=False)
    _, candidate_y = _numeric_candidate(plan, x, before, physical=True)
    assert np.any(candidate_y[0, 0] != logical_y[0, 0])
    if mutation is _pv_recurrence:
        # With no first clear, the physical drain continues PV0's accumulator;
        # source order instead recreates c0 at QK1 before PV1's first read.
        _, ordinary_y, _ = shared_l0c_device_oracle(x, score_before=before)
        np.testing.assert_array_equal(candidate_y[0, 1], logical_y[0, 1] + ordinary_y[0, 0])
    else:
        np.testing.assert_array_equal(candidate_y[0, 1], logical_y[0, 1])
