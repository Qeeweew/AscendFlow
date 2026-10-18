"""Select one notification contract for every retained memory requirement.

Selectors choose an instruction occurrence, never a completion fact. A family
capture is a real Set after its last relevant issue on the source engine; its
matching acquire precedes the first relevant consumer. No child-exit drain or
iteration expansion participates in this binding.
"""

from __future__ import annotations

from dataclasses import replace
from math import gcd, lcm

from xdsl.dialects import arith

from ..analysis.access_frontier import LoopResidueSelection, ResiduePosition
from ..analysis.integer_facts import IntegerFacts
from ..analysis.structured_dependencies import StructuredDependencies
from ..analysis.structured_dispatch import StructuredDispatch
from ..analysis.structured_protocols import (
    BoundPlacement,
    FamilyCapture,
    LogicalNotification,
    NotificationCoverage,
    NotificationKind,
    OrderingWitness,
    SelectedOrdinal,
    SynchronizationSelection,
    UnresolvedSynchronization,
    substitute,
)
from ..analysis.structured_regions import (
    AccessOccurrence,
    BranchGuard,
    LoopCountGuard,
    LoopOrdinalGuard,
)
from ..analysis.structured_schedule import ActionGuard, Comparison, Placement
from ..diagnostics import CompileError, ErrorKind, require
from ..indexing import arguments, equivalent, nonnegative, normalize
from ..scalar import Expr, expression, literal
from ..target.ascend910b import Actor, Engine


def _integer(value):
    return literal(value) if type(value) is int else value


def _comparison(test, substitutions, facts):
    lhs, rhs = substitute(test.lhs, substitutions), substitute(test.rhs, substitutions)
    if lhs.kind == "add" and nonnegative(rhs, facts):
        for index in (0, 1):
            shift = lhs.args[index].constant
            if shift is not None and shift > 0:
                # A source's future consumer may not exist. Test g < N-d
                # before evaluating its guarded g+d event arithmetic.
                lhs, rhs = lhs.args[1 - index], expression("sub", rhs, literal(shift))
                break
    return Comparison(lhs, test.predicate, rhs)


def _branch_test(operation, branch, environment):
    comparison = operation.cond.owner
    require(
        isinstance(comparison, arith.CmpiOp),
        "notification branch guard needs an actual integer comparison",
    )
    predicates = ("==", "!=", "<", "<=", ">", ">=")
    predicate = comparison.predicate.value.data
    require(predicate < len(predicates), "unsigned notification branch guard is unsupported")
    op = predicates[predicate]
    if branch:
        op = {"==": "!=", "!=": "==", "<": ">=", "<=": ">", ">": "<=", ">=": "<"}[op]
    return Comparison(
        environment.expression(comparison.lhs), op, environment.expression(comparison.rhs)
    )


def _case_tests(guards, environment):
    """Retain effect predicates in the requirement's consumer coordinates."""
    for guard in guards:
        if isinstance(guard, LoopCountGuard):
            yield Comparison(guard.count, guard.relation.value, _integer(guard.threshold))
        elif isinstance(guard, LoopOrdinalGuard):
            yield guard.comparison
        else:
            assert isinstance(guard, BranchGuard)
            yield _branch_test(guard.operation, guard.branch, environment)


def _guard(tests):
    """Keep short-circuit order and reject no valid inactive alternatives."""
    result = []
    for test in tests:
        if test.lhs.constant is not None and test.rhs.constant is not None and test.evaluate({}):
            continue
        if test not in result:
            result.append(test)
    return ActionGuard(tuple(result))


def _selected(count, period, residue, first=0):
    lower = literal(first)
    residue = _integer(residue)
    start = expression(
        "add",
        lower,
        expression(
            "rem",
            expression(
                "sub",
                expression("add", residue, literal(period)),
                expression("rem", lower, literal(period)),
            ),
            literal(period),
        ),
    )
    if (
        count.constant is not None
        and start.constant is not None
        and count.constant <= start.constant
    ):
        return start, start
    # The count > start test must precede this expression's unsigned remainder.
    last = expression(
        "sub",
        expression("sub", count, literal(1)),
        expression(
            "rem", expression("sub", expression("sub", count, literal(1)), residue), literal(period)
        ),
    )
    return start, last


def _restricted_family(family, need, memory):
    """Intersect an all-member family with this requirement's finite domain."""
    facts = memory.control.environment.facts
    for domain in need.domains:
        if domain.loop is not family.loop:
            continue
        a, b = _integer(family.residue), _integer(domain.residue)
        if family.period % domain.period == 0 and equivalent(
            expression("rem", a, literal(domain.period)), b, facts
        ):
            period, residue = family.period, family.residue
        elif domain.period % family.period == 0 and equivalent(
            expression("rem", b, literal(family.period)), a, facts
        ):
            period, residue = domain.period, domain.residue
        else:
            require(
                a.constant is not None and b.constant is not None,
                "family and requirement residues lack a proved exact intersection",
                kind=ErrorKind.UNSUPPORTED,
                phase="notification-selection",
            )
            period = lcm(family.period, domain.period)
            divisor = gcd(family.period, domain.period)
            difference = b.constant - a.constant
            require(
                difference % divisor == 0,
                "family and requirement residue intersection is empty",
                kind=ErrorKind.UNSUPPORTED,
                phase="notification-selection",
            )
            modulus = domain.period // divisor
            residue = (
                a.constant
                + family.period
                * (difference // divisor * pow(family.period // divisor, -1, modulus) % modulus)
            ) % period
        family = replace(
            family,
            count=expression("min", family.count, domain.count),
            period=period,
            residue=residue,
            first=max(family.first, domain.first),
            distance=max(family.distance, domain.distance),
        )
    return family


def _bind_endpoint(
    need, occurrence, memory, *, source, consumer_coordinates=(), effective_guards=None
):
    control, environment = memory.control, memory.control.environment
    template = next(item for item in memory.templates if item.operation is occurrence.operation)
    engine = template.contract.engine
    substitutions, coordinate_guards, tests, families, selector_tests, coordinates = (
        {},
        [],
        [],
        [],
        [],
        [],
    )
    own_bindings = {binding.loop: binding.ordinal for binding in occurrence.bindings}
    selections = []
    family_map = {domain.loop: domain for domain in occurrence.families}
    available = {control.group.args[0], control.sid.args[0]}
    available.update(name for name in environment.facts if not name.startswith("$"))
    for binding in occurrence.bindings:
        logical = control.loops[binding.loop]
        available.add(logical.ordinal.args[0])
        value = binding.ordinal
        selector = isinstance(value, LoopResidueSelection)
        # Ancestors are resolved in lexical order, including FIRST/LAST.
        # Raw headers use their actual parent coordinates; occurrence Exprs
        # already carry ordinary parent translations from region binding.
        outer = {control.loops[loop].ordinal: coordinate for loop, coordinate in coordinates}
        selected_outer = {
            ordinal: coordinate
            for ordinal, coordinate in outer.items()
            if ordinal.args[0] not in arguments(coordinate)
        }
        if isinstance(value, Expr):
            # A computed LAST can already be an Expr, rather than a residue
            # selector. Its selected ancestors must resolve before recording
            # coordinates used by ordered endpoint activation.
            value = substitute(value, selected_outer)
        if binding.loop in family_map:
            require(
                isinstance(value, Expr) and equivalent(value, logical.ordinal, environment.facts),
                "all-member family requires its actual unselected loop coordinate",
                kind=ErrorKind.UNSUPPORTED,
                phase="notification-selection",
            )
            family = _restricted_family(family_map[binding.loop], need, memory)
            first, last = _selected(
                family.count, family.period, family.residue, max(family.distance, family.first)
            )
            activation = Comparison(family.count, ">", first)
            tests.append(activation)
            coordinate_guards.append((binding.loop, ActionGuard((activation,))))
            families.append(
                FamilyCapture(
                    binding.loop,
                    family.period,
                    _integer(family.residue),
                    first,
                    last,
                    engine,
                )
            )
            value = last if source else first
            selector = True
        elif selector:
            assert isinstance(value, LoopResidueSelection)
            count = substitute(logical.count, outer)
            first, last = _selected(count, value.period, value.residue)
            activation = Comparison(count, ">", first)
            tests.append(activation)
            coordinate_guards.append((binding.loop, ActionGuard((activation,))))
            value = first if value.position is ResiduePosition.FIRST else last
        assert isinstance(value, Expr)
        if selector:
            scope_count = substitute(logical.count, outer)

            def contained(count, scope_count=scope_count):
                if count == scope_count:
                    return True
                if count.kind == "min" and any(contained(part) for part in count.args):
                    return True
                return nonnegative(expression("sub", scope_count, count), environment.facts)

            selection_count = family.count if binding.loop in family_map else count
            require(
                contained(selection_count),
                "selected ordinal family exceeds the actual native loop count",
                kind=ErrorKind.UNSUPPORTED,
                phase="notification-binding",
            )
            selections.append(SelectedOrdinal(binding.loop, selection_count, first, value))
        if not selector and logical.ordinal.args[0] in arguments(value):
            difference = normalize(expression("sub", value, logical.ordinal), environment.facts)
            if difference.constant is not None:
                # Invert an occurrence binding j-d so both action ends name
                # the same logical consumer j in their real native scope.
                substitutions[logical.ordinal] = expression("sub", logical.ordinal, difference)
                coordinates.append((binding.loop, value))
                continue
            require(
                logical.ordinal.args[0] not in arguments(value),
                "recursive notification needs a translation or explicit selected occurrence",
                kind=ErrorKind.UNSUPPORTED,
                phase="notification-selection",
            )
        selected = substitute(value, substitutions)
        coordinates.append((binding.loop, value))
        substitutions[logical.ordinal] = selected
        selector_tests.append(Comparison(logical.ordinal, "==", value))

    # A finite selected source can be coordinate 0 while the need's canonical
    # consumer coordinate is 1. Domain predicates still refer to that consumer;
    # the source's actual selector equality keeps its own physical coordinate.
    actual_substitutions = dict(substitutions)
    if source:
        consumer_coordinates = dict(consumer_coordinates)
        for domain in need.domains:
            ordinal = control.loops[domain.loop].ordinal
            selected_consumer = consumer_coordinates.get(domain.loop)
            if (
                selected_consumer is not None
                and ordinal in substitutions
                and ordinal.args[0] not in arguments(selected_consumer)
            ):
                substitutions[ordinal] = substitute(selected_consumer, actual_substitutions)

    for domain in need.domains:
        logical = control.loops[domain.loop]
        lower = max(domain.distance, domain.first)
        if domain.loop in own_bindings:
            tests += [
                Comparison(logical.ordinal, ">=", literal(lower)),
                Comparison(logical.ordinal, "<", domain.count),
            ]
            if domain.period != 1:
                tests.append(
                    Comparison(
                        expression("rem", logical.ordinal, literal(domain.period)),
                        "==",
                        _integer(domain.residue),
                    )
                )
        else:
            first, _ = _selected(domain.count, domain.period, domain.residue, lower)
            tests.append(Comparison(domain.count, ">", first))
    tests.extend(
        _case_tests(
            _effective_guards(need, memory) if effective_guards is None else effective_guards,
            environment,
        )
    )

    # Selector equalities compare the current actual ordinal, whereas every
    # other test is phrased in the requirement's logical consumer coordinates.
    rendered = [
        _branch_test(frame.operation, frame.branch, environment)
        for frame in occurrence.frame.ancestors()
        if frame.branch is not None
    ]
    rendered.extend(_comparison(test, substitutions, environment.facts) for test in tests)
    rendered.extend(
        Comparison(test.lhs, test.predicate, substitute(test.rhs, actual_substitutions))
        for test in selector_tests
    )
    guard = _guard(rendered)
    require(
        all(arguments(test.lhs) | arguments(test.rhs) <= available for test in guard.comparisons),
        "notification retains another invocation's unbound coordinate",
        kind=ErrorKind.UNSUPPORTED,
        phase="notification-selection",
    )
    return BoundPlacement(
        occurrence,
        Placement.AFTER if source else Placement.BEFORE,
        guard,
        tuple(substitutions.items()),
        tuple(coordinate_guards),
        tuple(families),
        tuple(coordinates),
        tuple(selections),
    )


def _dispatch_path(placement, dispatch):
    """Actual hierarchical controller issue positions, in canonical coordinates."""
    task = dispatch.tasks[placement.operation, placement.actor]
    coordinates = dict(placement.coordinates)
    scopes, scope = [], task.scope
    while scope.parent is not None:
        scopes.append(scope)
        scope = scope.parent
    result = []
    for child in reversed(scopes):
        parent = child.parent
        ordinal = coordinates.get(parent.operation, literal(0))
        result.append(
            (
                parent.operation,
                expression("add", ordinal, literal(child.parent_stage)),
                child.parent_lexical,
            )
        )
    ordinal = coordinates.get(task.scope.operation, literal(0))
    result.append(
        (task.scope.operation, expression("add", ordinal, literal(task.stage)), task.lexical)
    )
    return tuple(result)


class _IssueOrder:
    """Owned signed comparisons of actual hierarchical insertion positions.

    A need's mathematical activation premises are established once. They do
    not authorize unsigned guard evaluation; the event-domain stage retains
    that ordered definedness obligation. Unknown order retains the required
    notification, without supplying completion, retirement or visibility.
    """

    def __init__(self, dispatch):
        self.dispatch = dispatch
        self.arithmetic = IntegerFacts(dispatch.memory.control.environment.facts)
        self._loop_order = {loop: index for index, loop in enumerate(dispatch.memory.control.loops)}
        self._premises = {}
        self._guards = {}
        self._positions = {}

    def guards(self, need):
        if need not in self._guards:
            self._guards[need] = _effective_guards(need, self.dispatch.memory)
        return self._guards[need]

    def position(self, point):
        key = point.operation, point.actor, point.coordinates
        if key not in self._positions:
            self._positions[key] = _dispatch_path(point, self.dispatch)
        return self._positions[key]

    def premises(self, need):
        if need not in self._premises:
            control = self.dispatch.memory.control
            tests = list(_case_tests(need.guards, control.environment))
            for domain in sorted(need.domains, key=lambda item: self._loop_order[item.loop]):
                ordinal = control.loops[domain.loop].ordinal
                tests.extend(
                    (
                        Comparison(ordinal, ">=", literal(max(domain.first, domain.distance))),
                        Comparison(ordinal, "<", domain.count),
                    )
                )
                if domain.period != 1:
                    tests.append(
                        Comparison(
                            expression("rem", ordinal, literal(domain.period)),
                            "==",
                            _integer(domain.residue),
                        )
                    )
            self._premises[need] = tuple(dict.fromkeys(tests))
        return self._premises[need]

    def compare(self, before, after, need):
        """Return -1/0/1 only for proved order on one actual controller."""
        if before.actor is not after.actor:
            return None
        premises = self.premises(need)
        for a, b in zip(self.position(before), self.position(after)):
            if a[0] is not b[0]:
                return None
            if a[1] == b[1] or self.arithmetic.prove(Comparison(a[1], "==", b[1]), premises):
                if a[2] != b[2]:
                    return -1 if a[2] < b[2] else 1
                continue
            if self.arithmetic.prove(Comparison(b[1], ">", a[1]), premises):
                return -1
            if self.arithmetic.prove(Comparison(a[1], ">", b[1]), premises):
                return 1
            return None
        if before.operation is not after.operation:
            return None
        if before.placement is after.placement:
            return 0
        return -1 if before.placement is Placement.BEFORE else 1


def _point_key(point, *, actor=True):
    return (
        point.operation,
        point.actor if actor else None,
        point.placement,
        point.guard,
        point.coordinates,
        point.coordinate_guards,
    )


def _domain_key(notification):
    return (
        notification.clock,
        notification.ordinal,
        notification.period,
        notification.distance,
        notification.domains,
    )


def _effective_guards(need, memory):
    """Remove only count tests implied by the exact active domain itself."""
    result = []
    for guard in need.guards:
        implied = False
        if isinstance(guard, LoopCountGuard) and guard.relation.value == ">":
            for domain in need.domains:
                if domain.loop is not guard.loop or domain.count != guard.count:
                    continue
                first, _ = _selected(
                    domain.count, domain.period, domain.residue, max(domain.first, domain.distance)
                )
                if nonnegative(
                    expression("sub", first, _integer(guard.threshold)),
                    memory.control.environment.facts,
                ):
                    implied = True
                    break
        if not implied:
            result.append(guard)
    return tuple(result)


def _clock(need, source, consumer, memory):
    if need.loop is not None:
        return need.loop
    shared = set(dict(source.substitutions)) & set(dict(consumer.substitutions))
    for loop, logical in reversed(tuple(memory.control.loops.items())):
        if logical.ordinal not in shared:
            continue
        a = dict(source.substitutions)[logical.ordinal]
        b = dict(consumer.substitutions)[logical.ordinal]
        if logical.ordinal.args[0] in arguments(a) & arguments(b):
            return loop
    return None


def _bind_notification(need, dispatch, effective_guards):
    memory = dispatch.memory
    consumer = _bind_endpoint(
        need, need.consumer, memory, source=False, effective_guards=effective_guards
    )
    source = _bind_endpoint(
        need,
        need.source,
        memory,
        source=True,
        consumer_coordinates=consumer.coordinates,
        effective_guards=effective_guards,
    )
    a, b = (dispatch.tasks[p.operation, p.actor] for p in (source, consumer))
    cross = source.actor is not consumer.actor
    require(
        not cross or need.cell.root.type.bank.data == "GM",
        "cross-core notification cannot transfer private actor storage",
    )
    if cross:
        require(
            (source.actor is Actor.AIC) != (consumer.actor is Actor.AIC),
            "mode-2 requires a Cube/Vector handoff, not AIV-to-AIV visibility",
        )
    kind = (
        NotificationKind.CROSS_EVENT
        if cross
        else (NotificationKind.PIPE_FENCE if a.engine is b.engine else NotificationKind.LOCAL_EVENT)
    )
    clock = _clock(need, source, consumer, memory)
    return LogicalNotification(
        "",
        (source,),
        (consumer,),
        a.engine,
        b.engine,
        kind,
        (need,),
        clock,
        literal(0) if clock is None else memory.control.loops[clock].ordinal,
        need.period,
        need.distance,
        need.domains,
        (need,),
    )


def _implicit(notification, order):
    source, consumer = notification.sources[0], notification.consumers[0]
    if (
        source.actor is not consumer.actor
        or order.compare(source, consumer, notification.requirements[0]) != -1
    ):
        return None
    native = order.dispatch.tasks[source.operation, source.actor]
    if native.engine is Engine.S:
        return "synchronous Scalar native end before controller dispatch"
    if native.engine is notification.destination_engine and native.implicit_completion:
        return "qualified native completion before later same-engine admission"
    return None


def _group_captures(notifications, order):
    """One exact acquire may use its latest already-required engine capture."""
    grouped = {}
    for item in notifications:
        key = (
            _point_key(item.consumers[0]),
            item.source_engine,
            item.destination_engine,
            item.mechanism,
            item.sources[0].actor,
            _domain_key(item),
        )
        previous = grouped.get(key)
        if previous is None:
            grouped[key] = item
            continue
        orders = {
            order.compare(previous.sources[0], item.sources[0], need)
            for need in previous.requirements + item.requirements
        }
        if None in orders or len(orders) != 1:
            grouped[key + (len(grouped),)] = item
            continue
        relative = next(iter(orders))
        selected = item if relative <= 0 else previous
        grouped[key] = replace(selected, requirements=previous.requirements + item.requirements)
    return tuple(grouped.values())


def _contains_activation(wide, narrow, order):
    if wide.clock is not narrow.clock:
        return False
    facts = order.dispatch.memory.control.environment.facts
    needed = narrow.requirements[0]
    if not set(order.guards(wide.requirements[0])) <= set(order.guards(needed)):
        return False
    domains = {domain.loop: domain for domain in narrow.domains}
    selected_families = {f.loop: f for point in wide.consumers for f in point.families}
    for domain in wide.domains:
        other = domains.get(domain.loop)
        if other is None:
            family = selected_families.get(domain.loop)
            if family is not None and nonnegative(
                expression("sub", domain.count, expression("add", family.first, literal(1))),
                facts,
            ):
                continue
            return False
        if (
            domain.count != other.count
            or max(domain.first, domain.distance) > max(other.first, other.distance)
            or other.period % domain.period
        ):
            return False
        if domain.period != 1:
            difference = normalize(
                expression("sub", _integer(other.residue), _integer(domain.residue)), facts
            ).constant
            if difference is None or difference % domain.period:
                return False
    return True


def _reuse_acquisitions(notifications, order):
    """Reuse an already-selected real acquire on the same queue and domain.

    Coverage needs two concrete comparisons: the capture is after the needed
    source issue, and its queue acquire precedes the needed consumer issue.
    A pipe fence additionally captures every already-issued same-pipe task.
    There is no native-end FIFO or origin/reachability closure here.
    """

    def key(item):
        task = order.dispatch.tasks[item.consumers[0].operation, item.consumers[0].actor]
        steps, scope = [(task.stage, task.lexical)], task.scope
        while scope.parent is not None:
            steps.append((scope.parent_stage, scope.parent_lexical))
            scope = scope.parent
        return task.actor.value, tuple(reversed(steps)), item.source_engine.value

    selected = []
    for item in sorted(notifications, key=key):
        if item.mechanism is NotificationKind.CROSS_EVENT:
            selected.append(item)
            continue
        for i, previous in enumerate(selected):
            if previous.mechanism is NotificationKind.CROSS_EVENT or (
                previous.source_engine is not item.source_engine
                or previous.destination_engine is not item.destination_engine
                or previous.sources[0].actor is not item.sources[0].actor
                or not _contains_activation(previous, item, order)
            ):
                continue
            acquire = {
                order.compare(previous.consumers[0], item.consumers[0], need)
                for need in item.requirements
            }
            if not acquire <= {-1, 0}:
                continue
            capture_point = (
                previous.consumers[0]
                if (previous.mechanism is NotificationKind.PIPE_FENCE)
                else previous.sources[0]
            )
            capture = {
                order.compare(item.sources[0], capture_point, need) for need in item.requirements
            }
            if not capture <= {-1, 0}:
                continue
            selected[i] = replace(previous, requirements=previous.requirements + item.requirements)
            break
        else:
            selected.append(item)
    return tuple(selected)


def _unique_points(points):
    result = {}
    for point in points:
        key = _point_key(point)
        if key not in result:
            result[key] = point
        else:
            old = result[key]
            result[key] = replace(old, families=tuple(dict.fromkeys(old.families + point.families)))
    return tuple(result.values())


def _quorum_cube_point(points, requirements, order, *, source):
    """Capture the latest Cube prefix, or acquire before its first reader.

    The caller groups only identical activation/coordinate domains. Every
    moved endpoint must have a proved controller order under every original
    requirement; joining the two Vector participants alone supplies no order.
    """
    selected = points[0]
    for point in points[1:]:
        relative = {order.compare(selected, point, need) for need in requirements}
        if relative <= {-1, 0}:
            if source:
                selected = point
        elif relative <= {0, 1}:
            if not source:
                selected = point
        else:
            return None
    return selected


def _quorums(notifications, order):
    local, cross = [], {}
    for item in notifications:
        if item.mechanism is not NotificationKind.CROSS_EVENT:
            local.append(item)
            continue
        cube_source = item.sources[0].actor is Actor.AIC
        cube = item.sources[0] if cube_source else item.consumers[0]
        vector = item.consumers[0] if cube_source else item.sources[0]
        key = (
            cube_source,
            _point_key(vector, actor=False),
            # Different Cube operations may produce/consume the two AIV
            # partitions. They must still share the exact activation and
            # invocation; cumulative capture/acquire order is checked below.
            _point_key(cube, actor=False)[1:],
            item.source_engine,
            item.destination_engine,
            _domain_key(item),
        )
        cross.setdefault(key, []).append(item)
    unresolved = []
    for items in cross.values():
        first = items[0]
        sources = _unique_points(p for item in items for p in item.sources)
        consumers = _unique_points(p for item in items for p in item.consumers)
        a, b = {p.actor for p in sources}, {p.actor for p in consumers}
        complete = (a == {Actor.AIC} and b == {Actor.AIV0, Actor.AIV1}) or (
            b == {Actor.AIC} and a == {Actor.AIV0, Actor.AIV1}
        )
        requirements = tuple(n for item in items for n in item.requirements)
        if complete:
            cube_source = a == {Actor.AIC}
            cube = sources if cube_source else consumers
            vector = consumers if cube_source else sources
            selected = _quorum_cube_point(cube, requirements, order, source=cube_source)
            complete = selected is not None and len(vector) == 2
            if complete:
                if cube_source:
                    sources = (selected,)
                else:
                    consumers = (selected,)
        if not complete:
            unresolved.extend(
                UnresolvedSynchronization(
                    n, "mode-2 notification lacks the two actual AIV quorum members"
                )
                for n in requirements
            )
            continue
        local.append(
            replace(
                first,
                sources=sources,
                consumers=consumers,
                requirements=requirements,
                origins=tuple(n for item in items for n in item.origins),
            )
        )
    return tuple(local), tuple(unresolved)


def bind_notifications(dependencies, dispatch) -> SynchronizationSelection:
    """Consume all finite recursive needs through one capture/acquire contract.

    This internal step establishes coverage and preserves its original
    ownership edges. generate_notifications then constructs pairing, phase
    order and complete lifetimes. Coverage alone cannot authorize allocation.
    """
    require(isinstance(dependencies, StructuredDependencies), "selection needs actual dependencies")
    dependencies.require_closed()
    require(
        isinstance(dispatch, StructuredDispatch) and dispatch.memory is dependencies.memory,
        "selection and dispatch belong to different owned memory facts",
    )
    order = _IssueOrder(dispatch)
    notifications, coverage, unresolved = [], [], []
    bundles = {}
    for need in dependencies.required:
        key = (
            need.source,
            need.consumer,
            need.loop,
            need.distance,
            need.period,
            need.domains,
            order.guards(need),
            need.cell.root.type.bank,
        )
        bundles.setdefault(key, []).append(need)
    for bundle in bundles.values():
        need = bundle[0]
        if not isinstance(need.source, AccessOccurrence):
            unresolved.extend(
                UnresolvedSynchronization(n, "incoming memory port is not bound") for n in bundle
            )
            continue
        try:
            item = replace(
                _bind_notification(need, dispatch, order.guards(need)), requirements=tuple(bundle)
            )
            if rule := _implicit(item, order):
                coverage.extend(
                    NotificationCoverage(
                        n,
                        rule,
                        ordering=OrderingWitness(
                            rule,
                            item.sources[0],
                            item.consumers[0],
                            item.source_engine,
                            item.destination_engine,
                            ("same actor", "proved hierarchical issue order"),
                        ),
                    )
                    for n in bundle
                )
            elif (
                item.mechanism is NotificationKind.PIPE_FENCE
                and order.compare(item.sources[0], item.consumers[0], need) != -1
            ):
                unresolved.append(
                    UnresolvedSynchronization(
                        need, "same-engine source is not proved issued before its consumer fence"
                    )
                )
            else:
                notifications.append(item)
        except CompileError as error:
            unresolved.append(UnresolvedSynchronization(need, error.message))
    # Physical cells own binary credits. Keep disjoint slot residues here;
    # allocation may unite their instruction spelling after assigning IDs.
    grouped = _group_captures(notifications, order)
    grouped = _reuse_acquisitions(grouped, order)
    grouped, quorum_issues = _quorums(grouped, order)
    unresolved.extend(quorum_issues)
    selected = tuple(replace(item, name=f"notification_{i}") for i, item in enumerate(grouped))
    for item in selected:
        for need in item.requirements:
            coverage.append(
                NotificationCoverage(
                    need,
                    "actual same-engine pipe fence"
                    if item.mechanism is NotificationKind.PIPE_FENCE
                    else "actual mode-2 quorum capture/acquire"
                    if item.mechanism is NotificationKind.CROSS_EVENT
                    else "actual cumulative source capture and consumer-queue acquire",
                    item,
                )
            )
    return SynchronizationSelection(
        dependencies, dispatch, selected, tuple(coverage), tuple(unresolved)
    )
