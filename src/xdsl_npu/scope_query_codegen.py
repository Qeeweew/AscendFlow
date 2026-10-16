"""Render actual serial-query clocks and local event activation at native cuts."""

from .asc_scalar import ScalarScope
from .diagnostics import require
from .scalar import INT64_MAX


def _expression(build):
    scalar = ScalarScope()
    lines = scalar.finish([scalar.use(build(scalar))])
    require(len(lines) == 1, "query guard needs an unguarded scalar temporary")
    return lines[0]


def _count_guard(count):
    if count.exact is not None:
        return [f"npu_parent_n == {count.exact}"]
    terms = []
    minimum = count.period if count.minimum is None else count.minimum
    if minimum:
        terms.append(f"npu_parent_n >= {minimum}")
    if count.maximum is not None:
        terms.append(f"npu_parent_n <= {count.maximum}")
    if count.period != 1:
        remainder = _expression(lambda scalar: scalar.rem("npu_parent_n", count.period))
        terms.append(f"{remainder} == {count.residue}")
    return terms


def render_query_sync(checked, leaf, op, task, protocol):
    """One actual QuerySync, emitted in its surrounding mixed IR order."""
    from .loop_codegen import _event

    action = op.action.data
    require(action in {"set", "wait"}, "query ASC supports only local Set/Wait")
    endpoint = protocol.source if action == "set" else protocol.target
    require(
        endpoint.call == leaf.call.uid.data
        and endpoint.site.task == (leaf.actor, task.uid)
        and endpoint.site.position.evaluate(leaf.stream.count.data) == op.position.data,
        "query synchronization is not beside its checked native endpoint",
    )
    position, distance = endpoint.parent, protocol.target.query
    require(
        0 <= position.offset < INT64_MAX and 0 < distance <= INT64_MAX,
        "query ASC positions and distance must fit int64 arithmetic",
    )
    parent = (
        _expression(lambda scalar: scalar.sub("npu_parent_n", position.offset + 1))
        if position.end
        else str(position.offset)
    )
    conditions = [f"g == {op.position.data}", *_count_guard(protocol.domain)]
    conditions.append(f"npu_parent_g == {parent}")
    if action == "set":
        conditions += [
            f"npu_query_n > {distance}",
            f"npu_query_g < {_expression(lambda scalar: scalar.sub('npu_query_n', distance))}",
        ]
    else:
        conditions.append(f"npu_query_g >= {distance}")
    pipes = []
    for point in (protocol.source, protocol.target):
        source = checked.program.leaves[point.call]
        pipes.append(
            next(
                native.pipe
                for data in source.loop.actors.values()
                for native in data.program.tasks
                if native.uid == point.site.task[1]
            )
        )
    # _event's period-one source ordinal is exactly the source query. Use the
    # same scalar canonicalization/ID tuple selection as ordinary events.
    from types import SimpleNamespace

    scalar = ScalarScope()
    generation = (
        scalar.value("npu_query_g") if action == "set" else scalar.sub("npu_query_g", distance)
    )
    event = scalar.use(_event(SimpleNamespace(ids=protocol.ids, period=1), generation, scalar))
    function = "asc_sync_notify" if action == "set" else "asc_sync_wait"
    code = " ".join(
        scalar.finish([f"{function}(PIPE_{pipes[0]}, PIPE_{pipes[1]}, event_t({event}));"])
    )
    return f"if ({' && '.join(conditions)}) {{ {code} }} /* {protocol.name} */"


def query_loop_lines(invocation, uniform_names):
    """Evaluate the actual owner-local bound after the physical core ID exists."""
    if invocation.cyclic is None:
        return [f"  const int64_t npu_query_n = {invocation.bound.c(uniform_names)};"]
    domain = invocation.cyclic
    scalar = ScalarScope()
    total = (
        domain.total.constant
        if domain.total.constant is not None
        else domain.total.c(uniform_names)
    )
    remaining = scalar.sub(total, "core_id")
    owned = scalar.select_lt(remaining, 0, 0, remaining)
    quotient = scalar.div(owned, domain.groups)
    tail = scalar.select_eq(scalar.rem(owned, domain.groups), 0, 0, 1)
    count = scalar.use(scalar.add(quotient, tail))
    return ["  " + line for line in scalar.finish([f"const int64_t npu_query_n = {count};"])]
