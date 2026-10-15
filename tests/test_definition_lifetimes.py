"""Small independent oracle for definition ownership and parent activation.

This interprets the QK/PV source policy directly. Its total native order is a
candidate order whose retirement edges still need compiler completion proofs;
dispatch alone is never a certificate. No production access/frontier or sync
planner contributes either the semantic bindings or the expected result.
"""

from dataclasses import dataclass
from itertools import pairwise

import pytest

from xdsl_npu.completion import CountDomain, Position, Site


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
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()


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


def _c0_cross_call_retirements(events, model):
    _, owner, _, readers = model
    definitions = [
        event.uid
        for event in events
        if "c0" in event.writes and owner[event.uid, "c0"] == event.uid
    ]
    return {
        (reader, following)
        for previous, following in pairwise(definitions)
        for reader in readers[previous, "c0"]
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
