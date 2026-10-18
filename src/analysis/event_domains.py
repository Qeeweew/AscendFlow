"""Selected event domains, pairing and actual queue/controller order facts.

This stage owns one selected dispatch and its activation domains. It performs
no event allocation, lifetime search, progress construction or materialization.
Each clock has an exact membership window. Child existence is eliminated
from the innermost clock outward using that window's explicit projection;
unknown sparse or correlated activation is diagnosed at its actual boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType

from xdsl.dialects import arith

from ..diagnostics import CompileError, ErrorKind, require
from ..indexing import arguments
from ..scalar import Expr, Fact, expression, literal
from ..target.ascend910b import Actor, Engine
from .access_frontier import QueueEndpoint
from .event_contracts import ResourceWork
from .event_windows import EventWindow, WindowExistence
from .integer_facts import IntegerFacts
from .structured_control import ControlFacts
from .structured_dispatch import StructuredDispatch
from .structured_memory import NativeMemoryFacts
from .structured_protocols import (
    BoundPlacement,
    LogicalNotification,
    NotificationKind,
    SynchronizationSelection,
    substitute,
)
from .structured_schedule import ActionGuard, Comparison


def _loop_ancestors(control, operation):
    result, frame = [], control.frames[operation]
    while frame is not None:
        if frame.loop is not None:
            result.append(frame.loop.operation)
        frame = frame.parent
    return tuple(result)


@dataclass(frozen=True, eq=False, slots=True)
class EventCase:
    """One group-invocation-uniform alternative for a notification stream.

    Identity is intentional: ports of distinct alternatives never pair. The
    condition authorizes parameter substitution, not a change of loop clock.
    Empty alternatives carry no selected FIRST/LAST expression.
    """

    condition: ActionGuard
    substitutions: tuple[tuple[Expr, Expr], ...]
    windows: tuple[EventWindow, ...]
    projections: tuple[WindowExistence, ...]
    empty: bool = False

    def window(self, loop):
        for window in self.windows:
            if window.loop is loop:
                return window
        require(False, "event case has no established window for this clock")

    def specialize(self, value):
        return substitute(value, dict(self.substitutions))


@dataclass(frozen=True, init=False)
class EventDomains:
    """One owned collection of event-domain and physical ordering facts.

    Endpoint activation/ID checks may reject an unsupported proof. ``tests``
    and ``cases`` are built once and exposed through read-only mappings.
    Integer queries belong to one immutable input-fact environment; their
    answers and query counters may be filled without altering established domains.
    """

    selection: SynchronizationSelection
    memory: NativeMemoryFacts
    control: ControlFacts
    facts: Mapping[str, Fact]
    dispatch: StructuredDispatch
    arithmetic: IntegerFacts
    notifications: tuple[LogicalNotification, ...]
    work: ResourceWork
    tests: Mapping[LogicalNotification, tuple[Comparison, ...]]
    cases: Mapping[LogicalNotification, tuple[EventCase, ...]]
    clocks: Mapping[LogicalNotification, tuple]
    endpoints: Mapping[tuple[BoundPlacement, EventCase], BoundPlacement]
    active: Mapping[tuple[LogicalNotification, EventCase], bool]
    _order_facts: dict[tuple, tuple[ActionGuard, ...]] = field(compare=False, repr=False)

    def __init__(self, selection: SynchronizationSelection, work: ResourceWork):
        memory = selection.dependencies.memory
        for name, value in (
            ("selection", selection),
            ("memory", memory),
            ("control", memory.control),
            ("facts", memory.control.environment.facts),
            ("dispatch", selection.dispatch),
            ("arithmetic", IntegerFacts(memory.control.environment.facts)),
            ("notifications", selection.notifications),
            ("work", work),
            ("tests", {}),
            ("cases", {}),
            ("clocks", {}),
            ("endpoints", {}),
            ("active", {}),
            ("_order_facts", {}),
        ):
            object.__setattr__(self, name, value)
        self.work.notifications = len(self.notifications)
        self._validate()
        for name in ("tests", "cases", "clocks", "endpoints", "active"):
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
            self.clocks[notification] = self._common_clocks(notification)
            require(
                notification.sources and notification.consumers,
                "selected notification has no endpoints",
            )
            require(
                all(
                    (bound.operation, bound.actor)
                    in (
                        self.dispatch.boundaries
                        if isinstance(bound.occurrence.endpoint, QueueEndpoint)
                        else self.dispatch.tasks
                    )
                    for bound in notification.sources + notification.consumers
                ),
                "selected endpoint belongs to another physical dispatch",
            )
            require(
                all(
                    self.dispatch.site(bound).engine is notification.source_engine
                    for bound in notification.sources
                )
                and all(
                    self.dispatch.site(bound).engine is notification.destination_engine
                    for bound in notification.consumers
                ),
                "selected route differs from its actual native queues",
            )
            from .structured_schedule import Placement

            require(
                all(
                    p.placement is p.occurrence.endpoint.placement
                    if isinstance(p.occurrence.endpoint, QueueEndpoint)
                    else p.placement is expected
                    for points, expected in (
                        (notification.sources, Placement.AFTER),
                        (notification.consumers, Placement.BEFORE),
                    )
                    for p in points
                ),
                "a publication must follow its native capture and an acquire must precede its native",
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
            endpoints = tuple(
                dict.fromkeys(
                    self._ambient(bound) for bound in notification.sources + notification.consumers
                )
            )
            for tests in endpoints:
                self.work.endpoint_guard_sets += 1
                missing = self.arithmetic.ordered_definedness(tests)
                if missing is not None:
                    required, prefix = missing
                    self.fail(
                        notification,
                        "unsigned arithmetic lacks an ordered activation prefix proof",
                        phase="structured-event-pairing",
                        witness=(
                            f"required: {required.lhs.text()} {required.predicate} {required.rhs.text()}",
                            "endpoint prefix: "
                            + "; ".join(
                                f"{test.lhs.text()} {test.predicate} {test.rhs.text()}"
                                for test in prefix
                            ),
                        ),
                    )
            self.tests[notification] = tuple(
                dict.fromkeys(test for tests in endpoints for test in tests)
            )
            for bound in notification.sources + notification.consumers:
                bindings = {
                    self.control.loops[loop].ordinal: value for loop, value in bound.coordinates
                }
                for loop in self.clocks[notification]:
                    self.work.binding_roundtrips += 1
                    ordinal = self.control.loops[loop].ordinal
                    restored = substitute(bound.expression(ordinal), bindings)
                    if (
                        self.arithmetic.normalize(expression("sub", restored, ordinal)).constant
                        != 0
                    ):
                        self.fail(
                            notification,
                            "endpoint coordinates and inverse binding do not round-trip",
                            phase="structured-event-pairing",
                        )
            # A selected same-queue fence has one execution point and no
            # Set/Wait identity or reuse ordinal. Selection owns its real
            # capture coverage; retain endpoint admission without inventing
            # a binary event domain that no resource consumer needs.
            if notification.mechanism is NotificationKind.PIPE_FENCE:
                continue
            from .cell_channels import owns_stable_cell

            if owns_stable_cell(notification, self.control, self.clocks[notification]):
                # A cell credit is one physical resource across the complete
                # invocation. It has no dense rank, mixed radix or event
                # width to derive. Check actual pairing once; only unresolved
                # periodic parameter premises need exact family refinement.
                self.cases[notification] = (EventCase(ActionGuard(), (), (), ()),)
                if self._paired(notification):
                    self.work.parameter_cases += 1
                    continue
                self.work.pairing_refinements += 1
            self.cases[notification] = self._family(notification)
            self._pair(notification)

    def _family(self, notification):
        from .event_families import build_event_family

        return build_event_family(self, notification)

    def _pair(self, notification):
        """Every real endpoint exists in exactly the same canonical domain."""
        if not self._paired(notification):
            self.fail(
                notification,
                "selected Set/Wait endpoint activation domains are not proved equivalent",
                phase="structured-event-pairing",
            )

    def _paired(self, notification):
        def implies(tests, premises):
            for test in tests:
                self.work.pairing_implications += 1
                if not self.arithmetic.prove(test, premises):
                    return False
            return True

        for case in self.cases[notification]:
            points = notification.sources + notification.consumers
            instances = tuple(self._instance(p, {}, case=case) for p in points)
            first = instances[0]
            reference = first.guard.comparisons
            for endpoint in instances[1:]:
                tests = endpoint.guard.comparisons
                if tests == reference:
                    continue
                if not self._possible((first,)) and not self._possible((endpoint,)):
                    continue
                if not implies(tests, reference) or not implies(reference, tests):
                    return False
            self.endpoints.update(zip(((p, case) for p in points), instances, strict=True))
            # Pairing makes this one common activation fact for every port.
            # Unknown remains possible; consumers must never drop it.
            self.active[notification, case] = not case.empty and self._possible((first,))
        return True

    def _paired_id(self, assignment, bound, event, *, case):
        bindings = {self.control.loops[loop].ordinal: value for loop, value in bound.coordinates}
        restored = case.specialize(substitute(bound.expression(event), bindings))
        event = case.specialize(event)
        if self.arithmetic.normalize(expression("sub", restored, event)).constant != 0:
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
            self.arithmetic.prove(Comparison(test.lhs, opposite[test.predicate], test.rhs), b)
            for test in a
        ) or any(
            self.arithmetic.prove(Comparison(test.lhs, opposite[test.predicate], test.rhs), a)
            for test in b
        )

    def fail(self, notification, message, *, phase="structured-event-resources", witness=()):
        raise CompileError(
            f"{notification.name}: {message}",
            kind=ErrorKind.UNSUPPORTED,
            phase=phase,
            location=str(notification.sources[0].operation.location),
            witness=(
                f"mechanism={notification.mechanism.value}",
                "no schedule change, hidden controller gate, or barrier fallback",
                *witness,
            ),
        )

    def _ambient(self, placement, *, extra_coordinate_guards=()):
        bindings = {
            self.control.loops[loop].ordinal: value for loop, value in placement.coordinates
        }
        bindings[self.control.sid] = literal(1 if placement.actor is Actor.AIV1 else 0)
        coordinate_guards = dict(placement.coordinate_guards)
        for loop, guard in extra_coordinate_guards:
            prior = coordinate_guards.get(loop, ActionGuard())
            coordinate_guards[loop] = ActionGuard(prior.comparisons + guard.comparisons)
        tests = []

        def bind(test, substitutions):
            return Comparison(
                substitute(test.lhs, substitutions),
                test.predicate,
                substitute(test.rhs, substitutions),
            )

        # Retain lexical activation order, including empty enclosing loops.
        # A selected ordinal is evaluated only after its own nonempty fiber
        # prerequisite, and cannot borrow a later action-guard conjunct.
        # Coordinate prerequisites are already canonical; applying the actual
        # coordinate map again would translate a previous invocation twice.
        canonical = {self.control.sid: bindings[self.control.sid]}
        for frame in placement.occurrence.frame.ancestors():
            if frame.loop is not None:
                prerequisite = coordinate_guards.get(frame.loop.operation)
                if prerequisite is not None:
                    tests.extend(bind(test, canonical) for test in prerequisite.comparisons)
                selected = next(
                    (
                        item
                        for item in placement.selected
                        if item.loop is frame.loop.operation
                        and bindings.get(frame.loop.ordinal) == item.value
                    ),
                    None,
                )
                # The finite-selection constructor already establishes this
                # scope membership under the exact nonempty prerequisite.
                # Keep raw LAST arithmetic in the original selector guard so
                # ordered definedness is never replaced by this theorem.
                if (
                    selected is not None
                    and prerequisite is not None
                    and Comparison(selected.count, ">", selected.first) in prerequisite.comparisons
                ):
                    continue
                tests.extend(
                    bind(test, bindings)
                    for test in (
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
                    bind(
                        Comparison(
                            self.control.environment.expression(compare.lhs),
                            predicate,
                            self.control.environment.expression(compare.rhs),
                        ),
                        bindings,
                    )
                )
        tests.extend(bind(test, bindings) for test in placement.guard.comparisons)
        return tuple(dict.fromkeys(tests))

    def _instance(self, placement, bindings, *, case, extra_coordinate_guards=()):
        if (
            not extra_coordinate_guards
            and (known := self.endpoints.get((placement, case))) is not None
        ):
            return self._shift_instance(known, bindings) if bindings else known
        # Parameter specialization is separate from a coordinate translation:
        # a replacement coordinate must not itself be shifted a second time.
        tests = list(case.condition.comparisons)
        inactive = False
        for test in self._ambient(placement, extra_coordinate_guards=extra_coordinate_guards):
            specialized = Comparison(
                substitute(case.specialize(test.lhs), bindings),
                test.predicate,
                substitute(case.specialize(test.rhs), bindings),
            )
            tests.append(specialized)
            if (
                specialized.lhs.constant is not None
                and specialized.rhs.constant is not None
                and not specialized.evaluate({})
            ):
                inactive = True
                break
        return replace(
            placement,
            # There is no selected FIRST/LAST coordinate on an inactive fiber.
            coordinates=()
            if inactive
            else tuple(
                (loop, substitute(case.specialize(value), bindings))
                for loop, value in placement.coordinates
            ),
            guard=ActionGuard(tuple(dict.fromkeys(tests))),
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

    def _common_clocks(self, notification):
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

    def _order_conditions(self, before, after):
        """Exact queue order under both endpoints' established activation.

        The first endpoint is a Wait admission, and the second is the native
        position after a Set. These roles retain their phase even if the two
        endpoints name the same native operation. Conditions prove ordering,
        never the existence or definedness of a selected coordinate.
        """
        if before.actor is not after.actor:
            return ()
        if any(
            test.lhs.constant is not None
            and test.rhs.constant is not None
            and not test.evaluate({})
            for bound in (before, after)
            for test in bound.guard.comparisons
        ):
            return ()
        key = (
            before.operation,
            before.actor,
            before.placement,
            before.coordinates,
            after.operation,
            after.actor,
            after.placement,
            after.coordinates,
        )
        if key in self._order_facts:
            return self._order_facts[key]
        a, b = self.dispatch.site(before), self.dispatch.site(after)
        ap, bp = self._path(a), self._path(b)
        ac, bc = dict(before.coordinates), dict(after.coordinates)
        if any(
            scope.operation in self.control.loops and scope.operation not in coordinates
            for path, coordinates in ((ap, ac), (bp, bc))
            for scope in path
        ):
            return ()
        common = 0
        while common < min(len(ap), len(bp)) and ap[common] is bp[common]:
            common += 1

        def position(task, path, coordinates, native, index):
            owner = path[index]
            coordinate = coordinates.get(owner.operation, literal(0))
            if task.scope is owner:
                stage, lexical = task.stage, task.lexical
                phase = 0
            else:
                child = path[index + 1]
                stage, lexical, phase = child.parent_stage, child.parent_lexical * 3 + 1, 0
            return expression("add", coordinate, literal(stage)), (lexical, phase)

        conditions, prefix = [], ()

        def append(tests, test):
            if test.lhs.constant is not None and test.rhs.constant is not None:
                return tests if test.evaluate({}) else None
            return tests + (test,)

        # Earlier clocks dominate every descendant. At equal outer clocks
        # both endpoints enter the same child, so compare the next clock.
        # Only the final common scope can have distinct lexical/phase ranks.
        for index in range(common):
            ag, ar = position(a, ap, ac, False, index)
            bg, br = position(b, bp, bc, True, index)
            difference = self.arithmetic.bounded(expression("sub", bg, ag))
            final = index + 1 == common
            clause = append(
                prefix,
                Comparison(difference, ">=" if final and br >= ar else ">", literal(0)),
            )
            if clause is not None:
                conditions.append(ActionGuard(clause))
            if final:
                break
            prefix = append(prefix, Comparison(difference, "==", literal(0)))
            if prefix is None:
                break
        result = tuple(conditions)
        self._order_facts[key] = result
        return result

    def _gate(self, notification):
        return (
            notification.mechanism is NotificationKind.CROSS_EVENT
            or notification.destination_engine is Engine.S
        )

    def _following_conditions(self, consume_notification, consume, source_notification, source):
        self.work.lifetime_interfaces += 1
        if consume.actor is not source.actor:
            return ()
        if (
            not self._gate(consume_notification)
            and consume_notification.destination_engine is not source_notification.source_engine
        ):
            return ()
        return self._order_conditions(consume, source)

    def _first_last(self, window):
        return window.first_last()

    def _instances(self, notification, case, loop, offset, *, terminal=False):
        # A parent consumes exact first/last ports of the child invocation.
        # Window projection has already established their membership; do not
        # solve the selected LAST range again from its expanded arithmetic.
        ancestors = () if loop is None else _loop_ancestors(self.control, loop)
        kept = tuple(w for w in case.windows if w.loop is loop or w.loop in ancestors)
        for last in (False, True):
            bindings = {}
            if loop is not None:
                ordinal = self.control.loops[loop].ordinal
                bindings[ordinal] = expression("add", ordinal, literal(offset))
            prerequisites = []
            for child in case.windows:
                if child in kept:
                    continue
                # These guards are canonical. _instance translates them once,
                # before evaluating the selected coordinate's lexical tests.
                prerequisites.append((child.loop, child.nonempty))
                first, end = self._first_last(child)
                bindings[self.control.loops[child.loop].ordinal] = substitute(
                    end if last else first, bindings
                )

            if kept:
                membership = kept[-1].membership.comparisons
            elif case.windows:
                membership = case.windows[0].project_exists().guard.comparisons
            else:
                membership = self.tests[notification]
            guard = ActionGuard(
                tuple(
                    dict.fromkeys(
                        case.condition.comparisons
                        + tuple(
                            Comparison(
                                substitute(case.specialize(test.lhs), bindings),
                                test.predicate,
                                substitute(case.specialize(test.rhs), bindings),
                            )
                            for test in membership
                        )
                    )
                )
            )

            def instance(
                placement, bindings=bindings, prerequisites=tuple(prerequisites), guard=guard
            ):
                point = self._instance(
                    placement,
                    bindings,
                    case=case,
                    extra_coordinate_guards=prerequisites,
                )
                return replace(point, guard=guard)

            yield (
                tuple(instance(p) for p in notification.sources),
                tuple(instance(p) for p in notification.consumers),
            )
            if terminal:
                break

    def _possible(self, bounds):
        tests = tuple(test for bound in bounds for test in bound.guard.comparisons)
        opposite = {"==": "!=", "!=": "==", ">": "<=", ">=": "<", "<": ">=", "<=": ">"}
        return not any(
            self.arithmetic.prove(
                Comparison(test.lhs, opposite[test.predicate], test.rhs),
                tests[:index] + tests[index + 1 :],
            )
            for index, test in enumerate(tests)
        )
