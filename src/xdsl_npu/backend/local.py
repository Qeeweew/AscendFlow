"""Transitional local ASC printing from an owned, already selected schedule.

Reuses native formatting and launch ABI while the unified region emitter is
being implemented. No completion planner, graph construction, or validation
entry is called. The public compiler has not switched to this M2 entry yet.
"""

from dataclasses import dataclass

from ..asc_scalar import ScalarScope
from ..codegen import native_line
from ..diagnostics import require
from ..loop_codegen import _event, emit_loop


@dataclass(frozen=True, slots=True)
class _Channel:
    name: str
    sources: tuple
    destinations: tuple
    distance: int
    period: int
    residues: tuple[int, ...]
    ids: tuple[int, ...]
    kind: str = "local"
    last: bool = False
    carry: bool = False


@dataclass(frozen=True, slots=True)
class _Command:
    task: object
    kind: str
    channel: str = ""
    lower: int = 0
    upper: int = 0
    period: int = 1
    residues: tuple[int, ...] = (0,)
    last_gap: int | None = None


def _body(actor, data, sequences, channels, tasks, invocation, invocation_channels, uniform_names):
    from .loops import emit_pipeline_body

    require(
        invocation is None and not invocation_channels,
        "local PLE output requires a directly owned loop",
    )

    def render_group(group, domain):
        scalar, statements = ScalarScope(), []
        for command in group:
            if not domain.permits(lower=command.lower, upper_gap=command.upper):
                continue
            task = command.task
            if command.kind == "task":
                code = native_line(task, "n", "core_id", uniform_names=uniform_names, scalar=scalar)
            elif command.kind == "barrier":
                code = f"asc_sync_pipe(PIPE_{task.pipe});"
            else:
                channel = channels[command.channel]
                epoch = (
                    scalar.value("g")
                    if command.kind == "set"
                    else scalar.sub("g", channel.distance)
                )
                source, destination = (
                    tasks[channel.sources[0]].pipe,
                    tasks[channel.destinations[0]].pipe,
                )
                function = "asc_sync_notify" if command.kind == "set" else "asc_sync_wait"
                code = (
                    f"{function}(PIPE_{source}, PIPE_{destination}, "
                    f"event_t({scalar.use(_event(channel, epoch, scalar))}));"
                )
            guard = []
            if command.lower and not domain.implies_lower(command.lower):
                guard.append(f"g >= {command.lower}")
            if command.upper and not domain.implies_upper(command.upper):
                guard.append(f"g < n - {command.upper}")
            if command.period != 1:
                guard.append(
                    "("
                    + " || ".join(f"g % {command.period} == {r}" for r in command.residues)
                    + ")"
                )
            if guard:
                body, marker, comment = code.partition("//")
                code = f"if ({' && '.join(guard)}) {{ {body.strip()} }}" + (
                    f" //{comment}" if marker else ""
                )
            statements.extend((f"// {task.uid}: {command.kind}", code))
        return [line for statement in scalar.finish(statements) for line in statement.splitlines()]

    lines = [
        f"float npu_value_{task.uid} = 0.0f;"
        for task in data.program.tasks
        if task.native == "scalar_load"
    ]
    lines += emit_pipeline_body(
        sequences[actor], render_group, schedule=data.stream.attributes["npu.loop_schedule"]
    )
    return ["  " + line for line in lines]


def emit_local(plan, *, host_launcher=True):
    """Only adapt final instructions to the existing ASC formatting helpers."""
    channels = {}
    for assignment in plan.resources.assignments:
        protocol = assignment.protocol
        channels[protocol.name] = _Channel(
            protocol.name,
            ((protocol.source.actor.value, protocol.source.operation.uid.data),),
            ((protocol.destination.actor.value, protocol.destination.operation.uid.data),),
            protocol.distance,
            protocol.activation.period,
            tuple(
                sorted(
                    (r - protocol.distance) % protocol.activation.period
                    for r in protocol.activation.residues
                )
            ),
            assignment.ids,
        )
    actor, data = next(iter(plan.loop.actors.items()))
    natives = {task.uid: task for task in data.program.tasks}
    commands = []
    for command in plan.commands:
        task = natives[command.task.operation.uid.data]
        if command.kind == "task":
            commands.append(_Command(task, "task"))
            continue
        protocol = command.protocol
        is_set = command.kind == "set"
        residues = (
            channels[protocol.name].residues
            if is_set
            else tuple(sorted(protocol.activation.residues))
        )
        commands.append(
            _Command(
                task,
                command.kind,
                "" if command.kind == "barrier" else protocol.name,
                0 if is_set else protocol.distance,
                protocol.distance if is_set else 0,
                protocol.activation.period,
                residues,
            )
        )
    return emit_loop(
        (plan.loop, channels, {actor: tuple(commands)}, None, None),
        host_launcher=host_launcher,
        body_emitter=_body,
    )
