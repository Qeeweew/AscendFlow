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

from ..diagnostics import CompileError, ErrorKind, require
from ..dialects.physical import TaskOp
from ..indexing import arguments
from ..memo import memoized_method
from ..scalar import Expr, Fact, expression, literal, substitution
from ..symbols import Symbol, SymbolKind
from ..target.ascend910b import Actor, Engine
from .access_frontier import QueueEndpoint
from .event_contracts import ResourceWork
from .event_windows import EventWindow, WindowExistence
from .integer_facts import IntegerFacts
from .smt import all_of
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
from .structured_schedule import ActionGuard, Comparison, Placement


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


@dataclass(frozen=True, slots=True)
class _InputActivation:
    """Activation simplified only by the real dominating input loads.

    Keep each original ordered guard and the loads preceding its evaluation.
    These are per-occurrence input contracts, never global assumptions or
    facts borrowed from a load inside the guarded region.
    Loop/branch activation is not an input contract and cannot discharge
    itself. In particular a child's body cannot establish its own existence.
    """

    original: tuple[Comparison, ...]
    contracts: tuple[tuple[Comparison, tuple[tuple[TaskOp, tuple[Comparison, ...]], ...]], ...]
    discharged: tuple[Comparison, ...]
    tests: tuple[Comparison, ...]


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
    _activations: dict[tuple, _InputActivation] = field(compare=False, repr=False)
    _guard_inputs: dict[tuple, tuple] = field(compare=False, repr=False)
    _input_ranges_cache: dict[BoundPlacement, tuple] = field(compare=False, repr=False)
    _boundary_instances: dict[tuple, tuple] = field(compare=False, repr=False)

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
            ("_activations", {}),
            ("_guard_inputs", {}),
            ("_input_ranges_cache", {}),
            ("_boundary_instances", {}),
        ):
            object.__setattr__(self, name, value)
        self.work.notifications = len(self.notifications)
        self._validate()
        for name in ("tests", "cases", "clocks", "endpoints", "active"):
            object.__setattr__(self, name, MappingProxyType(getattr(self, name)))

    def _validate(self):
        admitted_endpoints = set()
        evaluated_at = {}
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
            for bound in notification.sources + notification.consumers:
                if bound in admitted_endpoints:
                    continue
                tests = self._ambient(bound)
                before = (
                    bound.operation
                    if bound.placement in {Placement.BEFORE, Placement.ENTRY}
                    else bound.operation.next_op
                )
                bindings = {
                    self.control.loops[loop].ordinal: value for loop, value in bound.coordinates
                }
                bindings[self.control.sid] = literal(1 if bound.actor is Actor.AIV1 else 0)
                site = bound.operation, bound.placement, bound.actor, bound.coordinates
                if site not in evaluated_at:
                    translate = substitution(bindings)
                    evaluated_at[site] = frozenset(
                        translate(self.control.environment.expression(value))
                        for value in self.control.dominating_scalars(bound.operation.parent, before)
                    )
                evaluated = evaluated_at[site]
                missing = self.arithmetic.ordered_definedness(tests, evaluated)
                if missing is not None:
                    self.fail(
                        notification,
                        "new notification arithmetic lacks an ordered activation prefix proof",
                        phase="structured-event-pairing",
                    )
                admitted_endpoints.add(bound)
            self.tests[notification] = tuple(
                dict.fromkeys(
                    test
                    for bound in notification.sources + notification.consumers
                    for test in self._activation(bound)
                )
            )
            for bound in notification.sources + notification.consumers:
                bindings = {
                    self.control.loops[loop].ordinal: value for loop, value in bound.coordinates
                }
                for loop in self.clocks[notification]:
                    self.work.binding_roundtrips += 1
                    ordinal = self.control.loops[loop].ordinal
                    restored = substitute(bound.expression(ordinal), bindings)
                    if self.arithmetic.exact_constant(expression("sub", restored, ordinal)) != 0:
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
            self.work.pairing_implications += len(tests)
            remaining = tuple(test for test in tests if test not in premises)
            return (
                not remaining
                or self.arithmetic.queries.implies(all_of(*remaining), premises).proved
            )

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
        if self.arithmetic.exact_constant(expression("sub", restored, event)) != 0:
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

        return self.arithmetic.queries.satisfies(uniform(left) + uniform(right)).impossible

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
        return tuple(
            test
            for test, _ in self._guarded_inputs(
                placement, extra_coordinate_guards=extra_coordinate_guards
            )
        )

    def _input_ranges(self, placement):
        """Load contracts known when this actual endpoint executes.

        These are local premises for an already active occurrence, not facts
        establishing its activation or authorizing an earlier guard. Bind the
        physical load's original scope directly to actual coordinates once;
        an instantiated placement's guard must not be translated again.
        """
        if placement in self._input_ranges_cache:
            return self._input_ranges_cache[placement]
        bindings = {
            self.control.loops[loop].ordinal: value for loop, value in placement.coordinates
        }
        bindings[self.control.sid] = literal(1 if placement.actor is Actor.AIV1 else 0)
        translate = substitution(bindings)
        tests = []
        for frame in placement.occurrence.frame.ancestors():
            operation = frame.operation
            if not isinstance(operation, TaskOp) or operation.native.data != "index_load":
                continue
            if operation is placement.operation and placement.placement is Placement.BEFORE:
                continue
            for value, lower, upper in frame.ranges:
                tests.extend(
                    Comparison(translate(value), predicate, translate(bound))
                    for predicate, bound in ((">=", lower), ("<=", upper))
                )
        result = tuple(dict.fromkeys(tests))
        self._input_ranges_cache[placement] = result
        return result

    def _guarded_inputs(self, placement, *, extra_coordinate_guards=()):
        """Bind each real guard with only its dominating input-load prefix."""
        key = placement, extra_coordinate_guards
        if key in self._guard_inputs:
            return self._guard_inputs[key]
        bindings = {
            self.control.loops[loop].ordinal: value for loop, value in placement.coordinates
        }
        bindings[self.control.sid] = literal(1 if placement.actor is Actor.AIV1 else 0)
        translate = substitution(bindings)
        coordinate_guards = dict(placement.coordinate_guards)
        for loop, guard in extra_coordinate_guards:
            prior = coordinate_guards.get(loop, ActionGuard())
            coordinate_guards[loop] = ActionGuard(prior.comparisons + guard.comparisons)
        tests, contracts = {}, []

        def append(values):
            for test in values:
                # A later repeated predicate cannot borrow a larger input
                # prefix to erase its earlier activation obligation.
                tests.setdefault(test, tuple(contracts))

        def bind(test, translate):
            return Comparison(translate(test.lhs), test.predicate, translate(test.rhs))

        # Retain lexical activation order, including empty enclosing loops.
        # A selected ordinal is evaluated only after its own nonempty fiber
        # prerequisite, and cannot borrow a later action-guard conjunct.
        # Coordinate prerequisites are already canonical; applying the actual
        # coordinate map again would translate a previous invocation twice.
        canonical = substitution({self.control.sid: bindings[self.control.sid]})
        for frame in placement.occurrence.frame.ancestors():
            if isinstance(frame.operation, TaskOp) and frame.operation.native.data == "index_load":
                contracts.append(
                    (
                        frame.operation,
                        tuple(
                            Comparison(translate(value), predicate, translate(bound))
                            for value, lower, upper in frame.ranges
                            for predicate, bound in ((">=", lower), ("<=", upper))
                        ),
                    )
                )
            if frame.loop is not None:
                prerequisite = coordinate_guards.get(frame.loop.operation)
                if prerequisite is not None:
                    append(bind(test, canonical) for test in prerequisite.comparisons)
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
                append(
                    bind(test, translate)
                    for test in (
                        Comparison(frame.loop.ordinal, ">=", literal(0)),
                        Comparison(frame.loop.count, ">", frame.loop.ordinal),
                    )
                )
            if frame.branch is not None:
                require(
                    len(frame.comparisons) == 1,
                    "event scope needs an admitted signed integer comparison",
                    kind=ErrorKind.UNSUPPORTED,
                    phase="structured-event-pairing",
                )
                append(bind(test, translate) for test in frame.comparisons)
        append(bind(test, translate) for test in placement.guard.comparisons)
        result = tuple(tests.items())
        self._guard_inputs[key] = result
        return result

    def _activation(self, placement, *, extra_coordinate_guards=()):
        key = placement, extra_coordinate_guards
        if key in self._activations:
            return self._activations[key].tests
        contexts = self._guarded_inputs(placement, extra_coordinate_guards=extra_coordinate_guards)
        original = tuple(test for test, _ in contexts)

        def input_expression(value):
            # Captures identify the load's invocation; they are not a bare
            # loop variable in this expression. Never erase an actual clock
            # predicate, even if a conditional load's range could imply it.
            if value.kind == "argument":
                symbol = value.args[0]
                return not isinstance(symbol, Symbol) or symbol.kind in {
                    SymbolKind.METADATA,
                    SymbolKind.PARAMETER,
                    SymbolKind.GROUP,
                    SymbolKind.PARTICIPANT,
                }
            return all(input_expression(child) for child in value.args if isinstance(child, Expr))

        discharged = tuple(
            test
            for test, contracts in contexts
            if contracts
            and any(
                isinstance(symbol, Symbol) and symbol.kind is SymbolKind.METADATA
                for symbol in arguments(test.lhs) | arguments(test.rhs)
            )
            and input_expression(test.lhs)
            and input_expression(test.rhs)
            and self.arithmetic.prove(test, tuple(t for _, tests in contracts for t in tests))
        )
        tests = tuple(test for test in original if test not in discharged)
        self._activations[key] = _InputActivation(original, contexts, discharged, tests)
        return tests

    def _instance(self, placement, bindings, *, case, extra_coordinate_guards=()):
        if (
            not extra_coordinate_guards
            and (known := self.endpoints.get((placement, case))) is not None
        ):
            return self._shift_instance(known, bindings) if bindings else known
        # Parameter specialization is separate from a coordinate translation:
        # a replacement coordinate must not itself be shifted a second time.
        translate = substitution(bindings)
        tests = list(case.condition.comparisons)
        inactive = False
        for test in self._activation(placement, extra_coordinate_guards=extra_coordinate_guards):
            specialized = Comparison(
                translate(case.specialize(test.lhs)),
                test.predicate,
                translate(case.specialize(test.rhs)),
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
                (loop, translate(case.specialize(value))) for loop, value in placement.coordinates
            ),
            guard=ActionGuard(tuple(dict.fromkeys(tests))),
            substitutions=(),
        )

    def _shift_instance(self, placement, bindings):
        translate = substitution(bindings)
        return replace(
            placement,
            coordinates=tuple((loop, translate(value)) for loop, value in placement.coordinates),
            guard=ActionGuard(
                tuple(
                    Comparison(
                        translate(test.lhs),
                        test.predicate,
                        translate(test.rhs),
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

    @memoized_method
    def _inactive(self, guard):
        return any(
            test.lhs.constant is not None
            and test.rhs.constant is not None
            and not test.evaluate({})
            for test in guard.comparisons
        )

    @memoized_method
    def _positions(self, site, coordinates):
        path = self.dispatch._position_path(site, coordinates)
        present = dict(coordinates)
        if any(
            scope.operation in self.control.loops and scope.operation not in present
            for scope, _, _ in path
        ):
            return None
        return path

    def _order_conditions(self, before, after):
        """Exact queue order under both endpoints' established activation.

        Endpoint placements retain their actual issue phases. Consume-to-Set
        handbacks additionally check the queue/controller gate; counter FIFO
        packing uses this same order for publications and receiver admissions.
        Conditions prove ordering, never completion, existence or definedness.
        """
        if before.actor is not after.actor:
            return ()
        if self._inactive(before.guard) or self._inactive(after.guard):
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
        ap = self._positions(self.dispatch.site(before), before.coordinates)
        bp = self._positions(self.dispatch.site(after), after.coordinates)
        if ap is None or bp is None:
            return ()
        common = 0
        while common < min(len(ap), len(bp)) and ap[common][0] is bp[common][0]:
            common += 1
        # Native identities choose the real common invocation and lexical
        # tie. The order formula then depends only on these actual clock
        # expressions and the final tie direction, not every native pair.
        word = tuple(
            (ap[index][1], bp[index][1], index + 1 == common and bp[index][2] >= ap[index][2])
            for index in range(common)
        )
        result = self._order_word(word)
        self._order_facts[key] = result
        return result

    @memoized_method
    def _order_word(self, word):
        self.work.issue_order_words += 1
        conditions, prefix = [], ()

        def append(tests, test):
            if test.lhs.constant is not None and test.rhs.constant is not None:
                return tests if test.evaluate({}) else None
            return tests + (test,)

        # Earlier clocks dominate every descendant. At equal outer clocks
        # both endpoints enter the same child, so compare the next clock.
        # Only the final common scope can have distinct lexical/phase ranks.
        for index, (ag, bg, inclusive) in enumerate(word):
            difference = (
                literal(0) if ag == bg else self.arithmetic.bounded(expression("sub", bg, ag))
            )
            final = index + 1 == len(word)
            clause = append(
                prefix,
                Comparison(difference, ">=" if inclusive else ">", literal(0)),
            )
            if clause is not None:
                conditions.append(ActionGuard(clause))
            if final:
                break
            prefix = append(prefix, Comparison(difference, "==", literal(0)))
            if prefix is None:
                break
        return tuple(conditions)

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
            # A family port is an owned fact for these exact coordinates and
            # active case. Keep FIRST/LAST independent and construct only the
            # requested port: asking for FIRST must never force LAST on an
            # empty or otherwise unavailable child.
            key = notification, case, loop, offset, last
            if key in self._boundary_instances:
                self.work.boundary_instance_reuses += 1
                yield self._boundary_instances[key]
                if terminal:
                    break
                continue
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

            result = (
                tuple(instance(p) for p in notification.sources),
                tuple(instance(p) for p in notification.consumers),
            )
            self._boundary_instances[key] = result
            self.work.boundary_instances += 1
            yield result
            if terminal:
                break

    def _possible(self, bounds):
        # _validate admitted each original endpoint's ordered evaluation before
        # pairing asks this purely mathematical coexistence question.
        tests = tuple(test for bound in bounds for test in bound.guard.comparisons)
        return not self.arithmetic.queries.satisfies(tests).impossible
