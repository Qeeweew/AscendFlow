"""Owned endpoint and invocation facts for actual progress queue relations.

The caller supplies admitted domains and one shared condition owner. Complete
case, placement and participant identities remain attached to endpoint facts;
queue envelopes retain their original exactness and lower-bound contracts.
"""

from __future__ import annotations

from dataclasses import replace

from ..diagnostics import CompileError, ErrorKind
from ..scalar import expression, literal, substitute
from . import progress_regions as regions
from .event_contracts import _Order
from .progress_activations import ProgressActivations
from .progress_regions import binding_key
from .structured_schedule import ActionGuard, Comparison


def _guard(tests):
    return ActionGuard(tuple(dict.fromkeys(tests)))


_UNESTABLISHED = object()


class ProgressEndpoints:
    def __init__(self, domains, activations: ProgressActivations):
        self.domains = domains
        self.activations = activations
        self.conditions = activations.conditions
        self.root = domains.memory.module.body.block.first_op
        self.control = domains.control
        self.dispatch = domains.dispatch
        # These facts belong to this immutable domain/dispatch invocation.
        # Equal own/imported placements share one established bound instance;
        # no notification-only or cross-compilation result is reused.
        self._bound_facts = {}
        self._invocation_facts = {}

    def fail(self, region, message, witness=(), *, kind=ErrorKind.UNSUPPORTED):
        raise CompileError(
            message,
            kind=kind,
            phase="structured-event-progress",
            location=str(region.location),
            witness=witness,
        )

    def ancestors(self, operation):
        return (self.root,) + tuple(
            frame.loop.operation
            for frame in self.control.frames[operation].ancestors()
            if frame.loop is not None
        )

    def inside(self, bound, region):
        return region in self.ancestors(bound.operation)

    def ambient(self, port):
        if isinstance(port, regions.QueueCut):
            return ()
        return self.instance(port).guard.comparisons

    def instance(self, port):
        key = binding_key(port)
        instance = self._bound_facts.get(key)
        if instance is None:
            instance = self.domains._instance(port.bound, dict(port.bindings), case=port.case)
            self._bound_facts[key] = instance
        return instance

    def coordinates(self, port):
        return (
            port.coordinates
            if isinstance(port, regions.QueueCut)
            else self.instance(port).coordinates
        )

    def bind_port(self, port, bindings):
        combined = dict(bindings)
        combined.update((symbol, substitute(value, bindings)) for symbol, value in port.bindings)
        return replace(
            port,
            bindings=tuple(
                (symbol, value) for symbol, value in combined.items() if symbol != value
            ),
        )

    def stream_port(self, port, region):
        """Rebase logical ancestor translations onto one actual invocation."""
        bindings = {}
        instance = self.instance(port)
        if self.conditions.impossible(instance.guard.comparisons):
            return None
        coordinates = dict(instance.coordinates)
        for loop in self.ancestors(region)[:-1]:
            if loop is self.root:
                continue
            ordinal = self.control.loops[loop].ordinal
            if loop not in coordinates:
                self.fail(region, "active stream port has no established ancestor coordinate")
            offset = self.domains.arithmetic.normalize(
                expression("sub", coordinates[loop], ordinal)
            ).constant
            if offset is None:
                self.fail(region, "stream ancestor binding is not an actual unit-rate invocation")
            bindings[ordinal] = expression("sub", ordinal, literal(offset))
        port = self.bind_port(port, bindings)
        return None if self.conditions.impossible(self.ambient(port)) else port

    def invocation_guard(self, region, *ports):
        if region is self.root or len(ports) < 2:
            return ActionGuard()
        ambient = tuple(self.ambient(port) for port in ports)
        key = (region, tuple(binding_key(port) for port in ports), ambient)
        established = self._invocation_facts.get(key, _UNESTABLISHED)
        if established is not _UNESTABLISHED:
            return established
        established = self._establish_invocation(region, ports, ambient)
        self._invocation_facts[key] = established
        return established

    def _establish_invocation(self, region, ports, ambient):
        coordinates = tuple(dict(self.coordinates(port)) for port in ports)
        # Queue endpoints may be active at different iterations of this loop.
        # Their local membership cannot disprove a shared ancestor invocation.
        assumptions = self.export_guard(
            _guard(tuple(test for tests in ambient for test in tests)), region
        ).comparisons
        reference = coordinates[0]
        tests = []
        for loop in self.ancestors(region)[:-1]:
            if loop is self.root:
                continue
            if loop not in reference or any(loop not in item for item in coordinates):
                return None
            for item in coordinates[1:]:
                test = Comparison(reference[loop], "==", item[loop])
                if not self.domains.arithmetic.prove(test, assumptions):
                    tests.append(test)
        if self.conditions.impossible(assumptions + tuple(tests)):
            return None
        return _guard(tuple(tests))

    def window_guard(self, port, region):
        guard = _guard(
            port.case.condition.comparisons + port.case.window(region).nonempty.comparisons
        )
        bindings = dict(port.bindings)
        return _guard(
            tuple(
                Comparison(
                    substitute(test.lhs, bindings),
                    test.predicate,
                    substitute(test.rhs, bindings),
                )
                for test in guard.comparisons
            )
        )

    def _transfer_binding(self, region, relation, streams):
        """Establish one common substitution for the entire child relation."""
        endpoints = tuple(
            self.stream_port(node, region)
            if isinstance(node, regions.ProgressPort) and node.notification in streams
            else node
            for node in (relation.source, relation.destination)
        )
        if any(node is None for node in endpoints):
            return None
        first, last = (
            node.bindings if isinstance(node, regions.ProgressPort) else () for node in endpoints
        )
        if first != last:
            self.fail(
                region,
                "child interface transfer needs a shared actual ancestor parameter binding",
            )
        return replace(relation, source=endpoints[0], destination=endpoints[1]), first

    def import_interface(self, region, interface, streams):
        """Rebind each shared subformula once per established ancestor binding."""
        relations, groups = [], {}
        for relation in interface.relations:
            established = self._transfer_binding(region, relation, streams)
            if established is None:
                continue
            imported, bindings = established
            index = len(relations)
            relations.append(imported)
            if bindings:
                groups.setdefault(bindings, []).append(index)
        scope = self.activations.region(region)
        for bindings, indices in groups.items():
            replacements = dict(bindings)

            def transform(guard, replacements=replacements):
                return _guard(
                    tuple(
                        Comparison(
                            substitute(test.lhs, replacements),
                            test.predicate,
                            substitute(test.rhs, replacements),
                        )
                        for test in guard.comparisons
                    )
                )

            mapped = scope.map_guards(
                tuple(relations[index].activation for index in indices), transform
            )
            for index, activation in zip(indices, mapped, strict=True):
                relations[index] = replace(relations[index], activation=activation)
        return regions.RegionInterface(
            interface.operation,
            tuple(
                imported
                for port in interface.ports
                if (
                    imported := self.stream_port(port, region)
                    if port.notification in streams
                    else port
                )
                is not None
            ),
            interface.cuts,
            tuple(relations),
        )

    def route(self, wait, source):
        return wait.actor is source.actor and (
            self.controller(wait) or wait.engine is source.engine
        )

    def controller(self, node):
        return (
            node.engine is None
            if isinstance(node, regions.QueueCut)
            else self.domains._gate(node.notification)
        )

    def _node_path(self, node):
        if isinstance(node, regions.ProgressPort):
            return self.domains._path(self.dispatch.tasks[node.bound.operation, node.actor])
        result, scope = [], self.dispatch.scopes[node.operation, node.actor]
        while scope is not None:
            result.append(scope)
            scope = scope.parent
        return tuple(reversed(result))

    def _position(self, port, region):
        if isinstance(port, regions.QueueCut):
            path = self._node_path(port)
            owner = next(index for index, scope in enumerate(path) if scope.operation is region)
            child = path[owner + 1]
            return child.parent_stage, (child.parent_lexical, 1)
        task = self.dispatch.tasks[port.bound.operation, port.actor]
        path = self.domains._path(task)
        owner = next(index for index, scope in enumerate(path) if scope.operation is region)
        if task.scope.operation is region:
            return task.stage, (task.lexical, 1 if port.phase == "set" else 0)
        child = path[owner + 1]
        return child.parent_stage, (child.parent_lexical, 1)

    def order_conditions(self, wait, source):
        """Actual native order alternatives, or one opaque call envelope."""
        if isinstance(wait, regions.QueueCut) or isinstance(source, regions.QueueCut):
            condition = self._cut_order(wait, source)
            return () if condition is None else (condition,)
        before, after = self.instance(wait), self.instance(source)
        assumptions = before.guard.comparisons + after.guard.comparisons
        conditions = []
        for clause in self.domains._following_conditions(
            wait.notification, before, source.notification, after
        ):
            guard = _guard(assumptions + clause.comparisons)
            # Unknown alternatives remain blocking possibilities. Only a
            # refuted conjunction can be removed from the exact order union.
            if not self.conditions.impossible(guard.comparisons):
                conditions.append((guard, True))
        return tuple(conditions)

    def _cut_order(self, wait, source):
        """Call-level envelope; it makes no claim about hidden native order."""
        assumptions = self.ambient(wait) + self.ambient(source)
        ap, bp = self._node_path(wait), self._node_path(source)
        common = 0
        while common < min(len(ap), len(bp)) and ap[common] is bp[common]:
            common += 1
        for index in range(common):
            region = ap[index].operation
            if any(
                isinstance(node, regions.QueueCut) and node.operation is region
                for node in (wait, source)
            ):
                if isinstance(wait, regions.QueueCut) and isinstance(source, regions.QueueCut):
                    return None
                return _guard(assumptions), False
            ag, ar = self._position(wait, region)
            bg, br = self._position(source, region)
            av = dict(self.coordinates(wait)).get(region, literal(0))
            bv = dict(self.coordinates(source)).get(region, literal(0))
            delta = self.domains.arithmetic.normalize(
                expression(
                    "sub", expression("add", bv, literal(bg)), expression("add", av, literal(ag))
                ),
            )
            if delta.constant == 0:
                if index + 1 < common:
                    continue
                return (_guard(assumptions), False) if br >= ar else None
            predicate = ">=" if index + 1 < common or br >= ar else ">"
            test = Comparison(delta, predicate, literal(0))
            guard = _guard(assumptions + (test,))
            return None if self.conditions.impossible(guard.comparisons) else (guard, False)
        return _guard(assumptions), False

    def stream_offset(self, port, loop):
        value = dict(self.coordinates(port)).get(loop)
        if value is None:
            return None
        ordinal = self.control.loops[loop].ordinal
        return self.domains.arithmetic.normalize(expression("sub", value, ordinal)).constant

    def gate_step(self, wait, source, loop):
        wo, so = self.stream_offset(wait, loop), self.stream_offset(source, loop)
        if wo is None or so is None:
            self.fail(loop, "current-loop port is not a unit-rate translation of its actual clock")
        ws, wr = self._position(wait, loop)
        ss, sr = self._position(source, loop)
        shift = wo + ws - so - ss + int(sr < wr)
        if isinstance(wait, regions.QueueCut) or isinstance(source, regions.QueueCut):
            # A child's own internal gates were already proved there. Its
            # output can gate the same child's next parent invocation.
            if (
                isinstance(wait, regions.QueueCut)
                and isinstance(source, regions.QueueCut)
                and wait.operation is source.operation
            ):
                shift += 1
            guard = _guard(
                tuple(
                    test
                    for node in (wait, source)
                    if isinstance(node, regions.ProgressPort)
                    for test in self.window_guard(node, loop).comparisons
                )
            )
            return shift, guard, False
        ordinal = self.control.loops[loop].ordinal
        base = self.instance(source)
        shifted = self.domains._shift_instance(
            base, {ordinal: expression("add", ordinal, literal(shift))}
        )
        old = self.instance(wait)
        tests = old.guard.comparisons + shifted.guard.comparisons
        relation = self.domains._may_follow(
            wait.notification, old, source.notification, shifted, tests
        )
        if relation is _Order.IMPOSSIBLE:
            shift += 1
            shifted = self.domains._shift_instance(
                base, {ordinal: expression("add", ordinal, literal(shift))}
            )
            tests = old.guard.comparisons + shifted.guard.comparisons
        # The physical minimum is a lower bound for *every* later active
        # gate. It need not itself have two active endpoints (for example,
        # even/odd notification streams). Keep the envelope with invocation
        # invariants, never a false guard sampled at that minimum shift.
        invariant = (
            self.window_guard(wait, loop).comparisons + self.window_guard(source, loop).comparisons
        )
        return shift, _guard(invariant), False

    def export_guard(self, guard, region):
        return _guard(
            tuple(
                test
                for test in guard.comparisons
                if self.control.invariant_over(test.lhs, region)
                and self.control.invariant_over(test.rhs, region)
            )
        )
