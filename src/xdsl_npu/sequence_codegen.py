"""Format the checked region endpoints in their actual physical IR order."""

from .ir import RegionSyncOp, TaskOp
from .loop_boundary import stream_operations


def region_boundary(checked, region):
    from .sequence_sync import _data, _task

    before, after, pending, previous = {}, {}, [], None
    data = _data(checked.sequence.regions, region)
    bridges = {b.name: b for b in checked.bridges}
    for op, _ in stream_operations(data.stream):
        if isinstance(op, TaskOp):
            before[op.uid.data], pending, previous = pending, [], op.uid.data
        elif isinstance(op, RegionSyncOp):
            if op.action.data == "set":
                after.setdefault(previous, []).append(op)
            else:
                pending.append(op)

    def render(operation):
        bridge = bridges[operation.channel.data]
        source, destination = bridge.source, bridge.destination
        action = operation.action.data
        point = source if action == "set" else destination
        conditions = []
        for domain in bridge.domains:
            if domain.exact is not None:
                conditions.append(f"npu_sequence_count == {domain.exact}")
            else:
                terms = [f"npu_sequence_count >= {domain.minimum}"]
                if domain.maximum is not None:
                    terms.append(f"npu_sequence_count <= {domain.maximum}")
                if domain.period != 1:
                    terms.append(f"npu_sequence_count % {domain.period} == {domain.residue}")
                conditions.append(" && ".join(terms))
        position = (
            f"n - {point.position.offset + 1}" if point.position.end else str(point.position.offset)
        )
        guard = "(" + " || ".join(f"({c})" for c in conditions) + f") && g == {position}"
        if bridge.carried:
            guard += (
                " && npu_invocation + 1 < npu_invocations"
                if action == "set"
                else " && npu_invocation > 0"
            )
        a, b = _task(checked.sequence.regions, source).pipe, _task(checked.sequence.regions, destination).pipe
        if bridge.event_id == -1:
            sync = f"asc_sync_pipe(PIPE_{b});"
        else:
            call = "asc_sync_wait" if action == "wait" else "asc_sync_notify"
            sync = f"{call}(PIPE_{a}, PIPE_{b}, event_t({bridge.event_id}));"
        return f"if ({guard}) {{ {sync} }} /* {bridge.name} */"

    def emit(task, code):
        return " ".join(
            [
                *(render(op) for op in before[task.uid]),
                code,
                *(render(op) for op in after.get(task.uid, ())),
            ]
        )

    return emit


def emit_sequence(checked, *, host_launcher=True):
    from .loop_codegen import emit_loop

    primary = next(
        (a for a in checked.analyses.values() if a[0].bound is not None),
        next(iter(checked.analyses.values())),
    )
    return emit_loop(primary, host_launcher=host_launcher, sequence=checked)
