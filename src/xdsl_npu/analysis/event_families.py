"""Finite invocation-uniform alternatives for one notification's window chain.

Partition only an unresolved bounded periodic comparison. Each split is a
mathematical integer trichotomy around its finite range, so the leaves cover
the admitted inputs and are disjoint by construction. No runtime iterations,
other notifications, or native bodies are duplicated. The concrete window
builder remains responsible for exact membership and definedness.
"""

from dataclasses import dataclass, field

from ..diagnostics import CompileError, ErrorKind
from ..scalar import literal, substitute
from ..target.ascend910b import Actor
from .event_windows import MAX_ENUMERATED_PERIOD, WindowPartitionRequired, build_event_window
from .periodic import period_of
from .structured_schedule import ActionGuard, Comparison

MAX_PARAMETER_CASES = 128
MAX_WINDOW_CONSTRUCTIONS = 512
_OPPOSITE = {"==": "!=", "!=": "==", ">": "<=", ">=": "<", "<": ">=", "<=": ">"}


@dataclass(frozen=True)
class _Branch:
    condition: ActionGuard = field(default_factory=ActionGuard)
    substitutions: tuple = ()


def _negative(test):
    return Comparison(test.lhs, _OPPOSITE[test.predicate], test.rhs)


def _impossible(arithmetic, tests):
    # This is sufficient infeasibility, not a satisfiability oracle. Unknown
    # leaves remain in the partition and must pass the concrete proof.
    return any(arithmetic.prove(_negative(test), tests[:i]) for i, test in enumerate(tests))


def _specialize(tests, bindings, arithmetic):
    result = []
    for test in tests:
        test = Comparison(
            substitute(test.lhs, bindings), test.predicate, substitute(test.rhs, bindings)
        )
        if arithmetic.prove(_negative(test), tuple(result)):
            return tuple(result), True
        result.append(test)
    return tuple(dict.fromkeys(result)), False


def _periodic_range(value, arithmetic, assumptions):
    """Mathematical range after the window established ordered definedness."""
    known = arithmetic.bounds(value, assumptions)
    if known[0] is not None and known[1] is not None:
        return known
    if value.kind == "rem" and value.args[1].constant is not None:
        modulus = value.args[1].constant
        if modulus > 0:
            return 0, modulus - 1
    if value.kind not in {"add", "mul", "min", "max"}:
        return known
    (al, au), (bl, bu) = (_periodic_range(v, arithmetic, assumptions) for v in value.args)
    if value.kind == "add":
        return (
            None if al is None or bl is None else al + bl,
            None if au is None or bu is None else au + bu,
        )
    if value.kind == "mul":
        if any(v is None for v in (al, au, bl, bu)):
            return known
        products = (al * bl, al * bu, au * bl, au * bu)
        return min(products), max(products)
    operation = min if value.kind == "min" else max
    return (
        None if al is None or bl is None else operation(al, bl),
        None if au is None or bu is None else operation(au, bu),
    )


def _partition(domains, notification, loop, branch, unresolved):
    arithmetic, control = domains.arithmetic, domains.control
    ordinal = control.loops[loop].ordinal
    root = domains.memory.module.body.block.first_op
    assumptions = branch.condition.comparisons
    for test in unresolved:
        if arithmetic.prove(test, assumptions) or arithmetic.prove(_negative(test), assumptions):
            continue
        for parameter, periodic in ((test.lhs, test.rhs), (test.rhs, test.lhs)):
            parameter = arithmetic.bounded(parameter)
            if parameter.constant is not None or not control.invariant_over(parameter, root):
                continue
            # A case condition cannot borrow a loop's activation prefix.
            if (
                arithmetic.ordered_definedness(
                    assumptions + (Comparison(parameter, "==", parameter),)
                )
                is not None
            ):
                continue
            if control.invariant_over(periodic, loop):
                continue
            period = period_of(periodic, ordinal, lambda value: control.invariant_over(value, loop))
            if period is None:
                continue
            lower, upper = _periodic_range(periodic, arithmetic, assumptions)
            if lower is None or upper is None:
                continue
            if upper - lower + 1 > MAX_ENUMERATED_PERIOD:
                raise CompileError(
                    f"{notification.name}: parameter partition range exceeds analysis budget "
                    f"{MAX_ENUMERATED_PERIOD}",
                    kind=ErrorKind.ANALYSIS_LIMIT,
                    phase="event-parameter-cases",
                )
            # The two tails retain inequalities; exact points alone authorize
            # substituting the parameter. All comparisons are kernel-uniform.
            alternatives = [(Comparison(parameter, "<", literal(lower)), None)]
            alternatives.extend(
                (Comparison(parameter, "==", literal(value)), literal(value))
                for value in range(lower, upper + 1)
            )
            alternatives.append((Comparison(parameter, ">", literal(upper)), None))
            result = []
            for condition, replacement in alternatives:
                tests = tuple(dict.fromkeys(assumptions + (condition,)))
                if _impossible(arithmetic, tests):
                    continue
                if condition in assumptions:
                    continue
                bindings = branch.substitutions
                if replacement is not None:
                    bindings += ((parameter, replacement),)
                result.append(_Branch(ActionGuard(tests), bindings))
            if result:
                return tuple(result)
    domains.fail(
        notification,
        "periodic membership needs a kernel-uniform bounded parameter partition; "
        "no supported exact split was established",
        phase="event-parameter-cases",
        witness=tuple(
            f"case: {test.lhs.text()} {test.predicate} {test.rhs.text()}" for test in assumptions
        )
        + tuple(
            f"unresolved: {test.lhs.text()} {test.predicate} {test.rhs.text()}"
            for test in unresolved
        ),
    )


def build_event_family(domains, notification):
    from .event_domains import EventCase

    clocks = domains._clocks(notification)
    pending, result = [_Branch()], []
    constructions = 0
    # A fact is reusable only with the same complete premises and specialized
    # metadata. Refinement invalidates exactly the affected window proof.
    established = {}
    while pending:
        branch = pending.pop()
        bindings = dict(branch.substitutions)
        projected, empty = _specialize(
            branch.condition.comparisons + domains.tests[notification],
            bindings,
            domains.arithmetic,
        )
        windows, projections = {}, []
        split = None
        if not empty:
            for loop in reversed(clocks):
                logical = domains.control.loops[loop]
                count = substitute(
                    logical.count,
                    {
                        domains.control.sid: literal(
                            int(notification.sources[0].actor is Actor.AIV1)
                        )
                    },
                )
                key = (loop, substitute(count, bindings), projected, branch.substitutions)
                window = established.get(key)
                if window is None:
                    constructions += 1
                    domains.work.window_constructions += 1
                    if constructions > MAX_WINDOW_CONSTRUCTIONS:
                        raise CompileError(
                            f"{notification.name}: window construction budget "
                            f"{MAX_WINDOW_CONSTRUCTIONS} exceeded",
                            kind=ErrorKind.ANALYSIS_LIMIT,
                            phase="event-parameter-cases",
                        )
                    try:
                        window = build_event_window(
                            notification,
                            loop,
                            ordinal=logical.ordinal,
                            count=count,
                            guards=projected,
                            facts=domains.facts,
                            proves=domains.arithmetic.prove,
                            bounds=domains.arithmetic.bounds,
                            ordered_definedness=domains.arithmetic.ordered_definedness,
                            invariant=lambda value, loop=loop: domains.control.invariant_over(
                                value, loop
                            ),
                            parameter_bindings=bindings,
                        )
                    except WindowPartitionRequired as request:
                        split = _partition(domains, notification, loop, branch, request.tests)
                        break
                    established[key] = window
                windows[loop] = window
                existence = window.project_exists()
                projections.append(existence)
                projected, empty = _specialize(
                    branch.condition.comparisons + existence.guard.comparisons,
                    bindings,
                    domains.arithmetic,
                )
                if empty:
                    break
        if split is not None:
            if len(result) + len(pending) + len(split) > MAX_PARAMETER_CASES:
                raise CompileError(
                    f"{notification.name}: parameter case budget {MAX_PARAMETER_CASES} exceeded",
                    kind=ErrorKind.ANALYSIS_LIMIT,
                    phase="event-parameter-cases",
                )
            domains.work.parameter_splits += 1
            pending.extend(reversed(split))
            continue
        result.append(
            EventCase(
                branch.condition,
                branch.substitutions,
                () if empty else tuple(windows[loop] for loop in clocks),
                () if empty else tuple(projections),
                empty,
            )
        )
    domains.work.parameter_cases += len(result)
    return tuple(result)
