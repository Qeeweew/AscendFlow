"""Selected event domains, pairing and actual queue/controller order facts.

This stage owns one selected dispatch and its activation domains. It performs
no event allocation, lifetime search, progress construction or materialization.
Each clock has an exact membership window. Child existence is eliminated
from the innermost clock outward using that window's explicit projection;
unknown sparse or correlated activation is diagnosed at its actual boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

from xdsl.dialects import arith

from ..diagnostics import CompileError, ErrorKind, require
from ..indexing import arguments, normalize
from ..scalar import Expr, Fact, expression, infer, literal
from ..target.ascend910b import Actor, Engine
from .event_contracts import ResourceWork, _Order
from .event_windows import EventWindow, WindowExistence, build_event_window
from .structured_control import ControlFacts
from .structured_dispatch import StructuredDispatch
from .structured_memory import NativeMemoryFacts
from .structured_protocols import (
    LogicalNotification,
    NotificationKind,
    SynchronizationSelection,
    substitute,
)
from .structured_schedule import ActionGuard, Comparison


def _normalize(value, facts):
    try:
        return normalize(value, facts)
    except CompileError as error:
        if error.message != "runtime address retains a fractional coefficient":
            raise
        return value


def _bounded(value, facts=None):
    """Normalize integer ring arithmetic while retaining division atoms.

    Quotient expansion is useful for equality, but loses the actual remainder
    interval needed by inequalities. These temporary names are local algebra
    variables only; none enters an event expression or emitted program.
    """
    facts = {} if facts is None else facts
    atoms, replacements = {}, {}

    def encode(current):
        if current.kind in {"constant", "argument"}:
            return current
        if current.kind in {"add", "mul"}:
            return expression(current.kind, *(encode(child) for child in current.args))
        atom = expression(current.kind, *(_bounded(child, facts) for child in current.args))
        if atom.kind in {"floordiv", "ceildiv", "rem"} and atom.args[1].constant == 1:
            return encode(literal(0) if atom.kind == "rem" else atom.args[0])
        if atom.kind in {"min", "max"}:
            a, b = (infer(arg, facts) for arg in atom.args)
            if a is not None and b is not None:
                if a.maximum is not None and b.minimum is not None and a.maximum <= b.minimum:
                    return encode(atom.args[0 if atom.kind == "min" else 1])
                if b.maximum is not None and a.minimum is not None and b.maximum <= a.minimum:
                    return encode(atom.args[1 if atom.kind == "min" else 0])
        if atom not in atoms:
            symbol = Expr("argument", (f"$resource_atom_{len(atoms)}",))
            atoms[atom], replacements[symbol] = symbol, atom
        return atoms[atom]

    encoded = encode(value)
    return substitute(normalize(encoded, {}), replacements)


def _term(test):
    if test.predicate in {">", ">="}:
        value = expression("sub", test.lhs, test.rhs)
        return expression("sub", value, literal(int(test.predicate == ">")))
    if test.predicate in {"<", "<="}:
        value = expression("sub", test.rhs, test.lhs)
        return expression("sub", value, literal(int(test.predicate == "<")))
    return None


def _loop_ancestors(control, operation):
    result, frame = [], control.frames[operation]
    while frame is not None:
        if frame.loop is not None:
            result.append(frame.loop.operation)
        frame = frame.parent
    return tuple(result)


@dataclass(frozen=True, init=False)
class EventDomains:
    """One owned collection of event-domain and physical ordering facts.

    Endpoint activation/ID checks may reject an unsupported proof. ``tests``
    and ``windows`` are built once and exposed through read-only mappings.
    Query counters in ``work`` are the only mutation after construction.
    """

    selection: SynchronizationSelection
    memory: NativeMemoryFacts
    control: ControlFacts
    facts: Mapping[str, Fact]
    dispatch: StructuredDispatch
    notifications: tuple[LogicalNotification, ...]
    work: ResourceWork
    tests: Mapping[LogicalNotification, tuple[Comparison, ...]]
    windows: Mapping[LogicalNotification, tuple[EventWindow, ...]]
    projections: Mapping[LogicalNotification, tuple[WindowExistence, ...]]

    def __init__(self, selection: SynchronizationSelection, work: ResourceWork):
        memory = selection.dependencies.memory
        for name, value in (
            ("selection", selection),
            ("memory", memory),
            ("control", memory.control),
            ("facts", memory.control.environment.facts),
            ("dispatch", selection.dispatch),
            ("notifications", selection.notifications),
            ("work", work),
            ("tests", {}),
            ("windows", {}),
            ("projections", {}),
        ):
            object.__setattr__(self, name, value)
        self.work.notifications = len(self.notifications)
        self._validate()
        for name in ("tests", "windows", "projections"):
            object.__setattr__(self, name, MappingProxyType(getattr(self, name)))

    def _validate(self):
        self.selection.require_closed()
        require(self.dispatch.memory is self.memory, "event resources belong to another dispatch")
        require(
            all(
                item.notification is None
                or any(item.notification is notification for notification in self.notifications)
                for item in self.selection.coverage
            ),
            "coverage refers to an absent selected notification",
        )
        for notification in self.notifications:
            require(
                notification.sources and notification.consumers,
                "selected notification has no endpoints",
            )
            require(
                all(
                    (bound.operation, bound.actor) in self.dispatch.tasks
                    for bound in notification.sources + notification.consumers
                ),
                "selected endpoint belongs to another physical dispatch",
            )
            require(
                all(
                    self.dispatch.tasks[bound.operation, bound.actor].engine
                    is notification.source_engine
                    for bound in notification.sources
                )
                and all(
                    self.dispatch.tasks[bound.operation, bound.actor].engine
                    is notification.destination_engine
                    for bound in notification.consumers
                ),
                "selected route differs from its actual native queues",
            )
            if notification.mechanism is NotificationKind.CROSS_EVENT:
                sources = tuple(bound.actor for bound in notification.sources)
                consumers = tuple(bound.actor for bound in notification.consumers)
                require(
                    (
                        sources == (Actor.AIC,)
                        and set(consumers) == {Actor.AIV0, Actor.AIV1}
                        and len(consumers) == 2
                    )
                    or (
                        consumers == (Actor.AIC,)
                        and set(sources) == {Actor.AIV0, Actor.AIV1}
                        and len(sources) == 2
                    ),
                    "mode-2 notification needs the actual complete 1 AIC : 2 AIV quorum",
                )
            self.tests[notification] = tuple(
                dict.fromkeys(
                    test
                    for bound in notification.sources + notification.consumers
                    for test in self._ambient(bound)
                )
            )
            self._pair(notification)
            clocks = self._clocks(notification)
            projected = self.tests[notification]
            windows, projections = {}, []
            for loop in reversed(clocks):
                logical = self.control.loops[loop]
                count = substitute(
                    logical.count,
                    {self.control.sid: literal(int(notification.sources[0].actor is Actor.AIV1))},
                )
                window = build_event_window(
                    notification,
                    loop,
                    ordinal=logical.ordinal,
                    count=count,
                    guards=projected,
                    facts=self.facts,
                    proves=self._prove,
                    bounds=self._range,
                    invariant=lambda value, loop=loop: self.control.invariant_over(value, loop),
                )
                windows[loop] = window
                existence = window.project_exists()
                projections.append(existence)
                projected = existence.guard.comparisons
            self.windows[notification] = tuple(windows[loop] for loop in clocks)
            self.projections[notification] = tuple(projections)

    def _pair(self, notification):
        """Every real endpoint exists in exactly the same canonical domain."""
        if notification.mechanism is NotificationKind.PIPE_FENCE:
            return
        reference = self._ambient(notification.sources[0])
        for bound in notification.sources + notification.consumers:
            tests = self._ambient(bound)
            if tests == reference:
                # Equality of canonical domains includes the empty domain.
                continue
            if not all(self._prove(test, reference) for test in tests) or not all(
                self._prove(test, tests) for test in reference
            ):
                self.fail(
                    notification,
                    "selected Set/Wait endpoint activation domains are not proved equivalent",
                    phase="structured-event-pairing",
                )

    def _paired_id(self, assignment, bound, event):
        bindings = {self.control.loops[loop].ordinal: value for loop, value in bound.coordinates}
        restored = substitute(bound.expression(event), bindings)
        if _normalize(expression("sub", restored, event), self.facts).constant != 0:
            self.fail(
                assignment.notification,
                "event ID does not round-trip through the actual endpoint coordinates",
                phase="structured-event-pairing",
            )

    def _exclusive(self, left, right):
        # Only predicates uniform for a complete invocation establish that two
        # channels never coexist. Disjoint loop phases alone do not do so.
        def uniform(notification):
            return tuple(
                test
                for test in self.tests[notification]
                if self.control.invariant_over(test.lhs, self.memory.module.body.block.first_op)
                and self.control.invariant_over(test.rhs, self.memory.module.body.block.first_op)
            )

        a, b = uniform(left), uniform(right)
        opposite = {"==": "!=", "!=": "==", ">": "<=", ">=": "<", "<": ">=", "<=": ">"}
        return any(
            self._prove(Comparison(test.lhs, opposite[test.predicate], test.rhs), b) for test in a
        ) or any(
            self._prove(Comparison(test.lhs, opposite[test.predicate], test.rhs), a) for test in b
        )

    def fail(self, notification, message, *, phase="structured-event-resources"):
        raise CompileError(
            f"{notification.name}: {message}",
            kind=ErrorKind.UNSUPPORTED,
            phase=phase,
            location=str(notification.sources[0].operation.location),
            witness=(
                f"mechanism={notification.mechanism.value}",
                "no schedule change, hidden controller gate, or barrier fallback",
            ),
        )

    def _ambient(self, placement):
        bindings = {
            self.control.loops[loop].ordinal: value for loop, value in placement.coordinates
        }
        bindings[self.control.sid] = literal(1 if placement.actor is Actor.AIV1 else 0)
        tests = []
        # These already hold before the placed action is reached. Keep their
        # lexical order ahead of the action guard so later conjuncts cannot
        # authorize an earlier undefined divide/remainder.
        # These are actual lexical activation conditions, including empty
        # enclosing loops. A selected coordinate does not remove its scope.
        for frame in self.control.frames[placement.operation].ancestors():
            if frame.loop is not None:
                tests.extend(
                    (
                        Comparison(frame.loop.ordinal, ">=", literal(0)),
                        Comparison(frame.loop.count, ">", frame.loop.ordinal),
                    )
                )
            if frame.branch is not None:
                compare = frame.operation.cond.owner
                require(
                    isinstance(compare, arith.CmpiOp),
                    "event scope needs its actual integer branch predicate",
                )
                predicates = ("==", "!=", "<", "<=", ">", ">=")
                require(
                    compare.predicate.value.data < len(predicates),
                    "event scope unsigned predicate is unsupported",
                    kind=ErrorKind.UNSUPPORTED,
                    phase="structured-event-pairing",
                )
                predicate = predicates[compare.predicate.value.data]
                if frame.branch:
                    predicate = {
                        "==": "!=",
                        "!=": "==",
                        ">": "<=",
                        ">=": "<",
                        "<": ">=",
                        "<=": ">",
                    }[predicate]
                tests.append(
                    Comparison(
                        self.control.environment.expression(compare.lhs),
                        predicate,
                        self.control.environment.expression(compare.rhs),
                    )
                )
        tests.extend(placement.guard.comparisons)
        return tuple(
            dict.fromkeys(
                Comparison(
                    substitute(test.lhs, bindings), test.predicate, substitute(test.rhs, bindings)
                )
                for test in tests
            )
        )

    def _instance(self, placement, bindings):
        tests = tuple(
            Comparison(
                substitute(test.lhs, bindings), test.predicate, substitute(test.rhs, bindings)
            )
            for test in self._ambient(placement)
        )
        return replace(
            placement,
            coordinates=tuple(
                (loop, substitute(value, bindings)) for loop, value in placement.coordinates
            ),
            guard=ActionGuard(tests),
            substitutions=(),
        )

    def _shift_instance(self, placement, bindings):
        return replace(
            placement,
            coordinates=tuple(
                (loop, substitute(value, bindings)) for loop, value in placement.coordinates
            ),
            guard=ActionGuard(
                tuple(
                    Comparison(
                        substitute(test.lhs, bindings),
                        test.predicate,
                        substitute(test.rhs, bindings),
                    )
                    for test in placement.guard.comparisons
                )
            ),
        )

    def _range(self, value, assumptions=()):
        value = _bounded(value, self.facts)
        fact = infer(value, self.facts)
        lower, upper = (None, None) if fact is None else (fact.minimum, fact.maximum)
        for test in assumptions:
            if test.rhs.constant is not None and (
                value.kind == "argument" or _bounded(test.lhs, self.facts) == value
            ):
                offset = _normalize(
                    _bounded(expression("sub", test.lhs, value), self.facts), self.facts
                ).constant
                if offset is None:
                    continue
                bound = test.rhs.constant - offset
                if test.predicate in {">", ">=", "=="}:
                    candidate = bound + int(test.predicate == ">")
                    lower = candidate if lower is None else max(lower, candidate)
                if test.predicate in {"<", "<=", "=="}:
                    candidate = bound - int(test.predicate == "<")
                    upper = candidate if upper is None else min(upper, candidate)
        if value.kind == "constant":
            return value.constant, value.constant
        if value.kind == "rem" and value.args[1].constant is not None:
            return 0, value.args[1].constant - 1
        if lower is not None and upper is not None:
            return lower, upper
        if value.kind in {"add", "mul"}:
            (al, au), (bl, bu) = (self._range(arg, assumptions) for arg in value.args)
            if value.kind == "add":
                calculated_lower = None if al is None or bl is None else al + bl
                calculated_upper = None if au is None or bu is None else au + bu
            elif all(bound is not None for bound in (al, au, bl, bu)):
                products = (al * bl, al * bu, au * bl, au * bu)
                calculated_lower, calculated_upper = min(products), max(products)
            else:
                calculated_lower = (
                    0 if al is not None and bl is not None and al >= 0 and bl >= 0 else None
                )
                calculated_upper = None
            lower = calculated_lower if lower is None else lower
            upper = calculated_upper if upper is None else upper
        return lower, upper

    def _prove(self, test, assumptions):
        difference = _normalize(
            _bounded(expression("sub", test.lhs, test.rhs), self.facts), self.facts
        )
        if difference.constant is not None:
            return Comparison(difference, test.predicate, literal(0)).evaluate({})
        if test in assumptions:
            return True
        if test.predicate in {"==", "!="}:
            for prior in assumptions:
                if (
                    prior.predicate == test.predicate
                    and _normalize(
                        expression("sub", expression("sub", prior.lhs, prior.rhs), difference),
                        self.facts,
                    ).constant
                    == 0
                ):
                    return True
            if test.predicate == "!=":
                lower, upper = self._range(difference, assumptions)
                return lower is not None and lower > 0 or upper is not None and upper < 0
            return self._prove(Comparison(test.lhs, ">=", test.rhs), assumptions) and self._prove(
                Comparison(test.lhs, "<=", test.rhs), assumptions
            )
        term = _bounded(_term(test), self.facts)
        for prior in assumptions:
            constraint = _term(prior)
            if constraint is not None:
                remainder = _bounded(expression("sub", term, constraint), self.facts)
                if remainder.constant is not None:
                    if remainder.constant >= 0:
                        return True
                    continue
                bound, _ = self._range(remainder)
                if bound is not None and bound >= 0:
                    return True
        lower, _ = self._range(term, assumptions)
        return lower is not None and lower >= 0

    def _exists(self, placements, assumptions):
        return all(
            self._prove(test, assumptions)
            for placement in placements
            for test in placement.guard.comparisons
        )

    def _window(self, notification, loop):
        for window in self.windows[notification]:
            if window.loop is loop:
                return window
        self.fail(notification, "endpoint projection has no established window for this clock")

    def _clocks(self, notification):
        placements = notification.sources + notification.consumers
        common = None
        for placement in placements:
            free = {
                loop
                for loop, value in placement.coordinates
                if self.control.loops[loop].ordinal.args[0] in arguments(value)
            }
            common = free if common is None else common & free
        return tuple(loop for loop in self.control.loops if loop in (common or ()))

    def _path(self, task):
        result, scope = [], task.scope
        while scope is not None:
            result.append(scope)
            scope = scope.parent
        return tuple(reversed(result))

    def _order(self, before, after, assumptions):
        return self._order_relation(before, after, assumptions) is _Order.PROVED

    def _order_relation(self, before, after, assumptions):
        if before.actor is not after.actor:
            return _Order.IMPOSSIBLE
        a = self.dispatch.tasks[before.operation, before.actor]
        b = self.dispatch.tasks[after.operation, after.actor]
        ap, bp = self._path(a), self._path(b)
        common = 0
        while common < min(len(ap), len(bp)) and ap[common] is bp[common]:
            common += 1

        def position(task, path, bound, native, index):
            owner = path[index]
            coordinate = dict(bound.coordinates).get(owner.operation, literal(0))
            if task.scope is owner:
                stage, lexical = task.stage, task.lexical
                phase = 1 if native else 0
            else:
                child = path[index + 1]
                stage, lexical, phase = child.parent_stage, child.parent_lexical, 1
            return expression("add", coordinate, literal(stage)), (lexical, phase)

        # A later outer invocation contains every instruction in its child,
        # even when the child's coordinate resets to FIRST. Compare those
        # real common ancestors before comparing the deepest native clock.
        for index in range(common):
            ag, ar = position(a, ap, before, False, index)
            bg, br = position(b, bp, after, True, index)
            difference = _normalize(expression("sub", bg, ag), self.facts)
            if difference.constant == 0:
                if index + 1 < common:
                    continue
                return _Order.PROVED if br >= ar else _Order.IMPOSSIBLE
            if difference.constant is not None:
                return _Order.PROVED if difference.constant > 0 else _Order.IMPOSSIBLE
            if self._prove(Comparison(difference, ">", literal(0)), assumptions):
                return _Order.PROVED
            if self._prove(Comparison(difference, "<", literal(0)), assumptions):
                return _Order.IMPOSSIBLE
            if index + 1 == common:
                if br >= ar and self._prove(Comparison(difference, ">=", literal(0)), assumptions):
                    return _Order.PROVED
                if br < ar and self._prove(Comparison(difference, "<=", literal(0)), assumptions):
                    return _Order.IMPOSSIBLE
                return _Order.POSSIBLE
            if not self._prove(Comparison(difference, "==", literal(0)), assumptions):
                return _Order.POSSIBLE
        return _Order.IMPOSSIBLE

    def _gate(self, notification):
        return (
            notification.mechanism is NotificationKind.CROSS_EVENT
            or notification.destination_engine is Engine.S
        )

    def _follows(self, consume_notification, consume, source_notification, source, assumptions):
        self.work.lifetime_interfaces += 1
        if consume.actor is not source.actor:
            return False
        if (
            not self._gate(consume_notification)
            and consume_notification.destination_engine is not source_notification.source_engine
        ):
            return False
        return self._order(consume, source, assumptions)

    def _may_follow(self, consume_notification, consume, source_notification, source, assumptions):
        if consume.actor is not source.actor:
            return _Order.IMPOSSIBLE
        if (
            not self._gate(consume_notification)
            and consume_notification.destination_engine is not source_notification.source_engine
        ):
            return _Order.IMPOSSIBLE
        return self._order_relation(consume, source, assumptions)

    def _first_last(self, window):
        return window.first_last()

    def _instances(self, notification, loop, offset, *, terminal=False):
        logical = self.control.loops[loop]
        for last in (False, True):
            bindings = {logical.ordinal: expression("add", logical.ordinal, literal(offset))}
            for child in self.windows[notification]:
                if child.loop is loop:
                    continue
                paths = _loop_ancestors(self.control, child.loop)
                if loop in paths:
                    first, end = self._first_last(child)
                    bindings[self.control.loops[child.loop].ordinal] = substitute(
                        end if last else first, bindings
                    )
            sources = tuple(self._instance(p, bindings) for p in notification.sources)
            consumers = tuple(self._instance(p, bindings) for p in notification.consumers)
            yield sources, consumers
            if terminal:
                break

    def _project(self, notification, clock, last):
        bindings = {}
        for loop in self.control.loops:
            if clock is not None and (
                loop is clock or clock not in _loop_ancestors(self.control, loop)
            ):
                continue
            if not any(
                self.control.loops[loop].ordinal.args[0] in arguments(value)
                for bound in notification.sources + notification.consumers
                for owner, value in bound.coordinates
                if owner is loop
            ):
                continue
            window = self._window(notification, loop)
            first, end = self._first_last(window)
            bindings[self.control.loops[loop].ordinal] = substitute(
                end if last else first, bindings
            )
        return tuple(self._instance(bound, bindings) for bound in notification.sources), tuple(
            self._instance(bound, bindings) for bound in notification.consumers
        )

    def _possible(self, bounds):
        tests = tuple(test for bound in bounds for test in bound.guard.comparisons)
        opposite = {"==": "!=", "!=": "==", ">": "<=", ">=": "<", "<": ">=", "<=": ">"}
        return not any(
            self._prove(
                Comparison(test.lhs, opposite[test.predicate], test.rhs),
                tests[:index] + tests[index + 1 :],
            )
            for index, test in enumerate(tests)
        )
