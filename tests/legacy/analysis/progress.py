"""Historical M2 local ordering certificate used by the test oracle.

This checks progress, not memory coverage or event reuse. One physical-step
template supplies V milestone variables and E ordering inequalities. With
stride V+1, at most V relaxation rounds either establish an integer rank
strictly increasing along every edge or reject the cyclic schedule. Cost is
O(V*E), storage O(V+E); no origin sets, coverage queries, runtime iterations,
or repeated checks after each inserted event are present.

Inactive commands can only remove constraints or contract queue/prefix paths.
Thus a rank for the full repeated template also orders each actual finite
startup/steady/exit execution. Pair existence must be established separately
by each protocol's activation. This sufficient check can reject a schedule
whose mutually exclusive guards are necessary to break its cycle.
"""

from bisect import bisect_left
from dataclasses import dataclass

from tests.legacy.analysis.protocols import LocalCommand
from xdsl_npu.diagnostics import CompileError, ErrorKind, require
from xdsl_npu.target.ascend910b import Engine


@dataclass(frozen=True, slots=True)
class OrderPoint:
    command: LocalCommand
    point: str


@dataclass(frozen=True, slots=True)
class OrderArc:
    source: int
    destination: int
    steps: int
    reason: str


@dataclass(frozen=True, slots=True)
class ProgressCertificate:
    points: tuple[OrderPoint, ...]
    arcs: tuple[OrderArc, ...]
    stride: int
    offsets: tuple[int, ...]
    rounds: int
    edge_visits: int


def establish_local_progress(commands: tuple[LocalCommand, ...]) -> ProgressCertificate:
    """Prove fixed pipe/controller dispatch and selected handoffs can progress."""
    if not commands:
        return ProgressCertificate((), (), 1, (), 0, 0)
    require(
        tuple(c.index for c in commands) == tuple(range(len(commands))),
        "local progress needs the actual contiguous command order",
    )
    actor, clock = commands[0].task.actor, commands[0].task.clock
    require(
        all(c.task.actor is actor and c.task.clock is clock for c in commands),
        "local progress requires one actor and bound clock",
    )
    points, ids, arcs = [], {}, []

    def node(command, point):
        index = len(points)
        points.append(OrderPoint(command, point))
        ids[command.index, point] = index
        return index

    def edge(source, destination, steps, reason):
        arcs.append(OrderArc(source, destination, steps, reason))

    queues, natives, qualified = {}, {}, {}
    gates, sends, receives = [], {}, {}
    for command in commands:
        require(
            command.kind in {"task", "set", "wait", "barrier"}, "unknown selected local command"
        )
        node(command, "begin")
        if command.kind in {"task", "set"}:
            node(command, "end")
            edge(
                ids[command.index, "begin"],
                ids[command.index, "end"],
                0,
                "instruction admission before its own end",
            )
        if command.kind == "task":
            node(command, "retire")
            edge(
                ids[command.index, "end"],
                ids[command.index, "retire"],
                0,
                "native end contributes to retirement capture",
            )
            natives.setdefault(command.task.engine, []).append(command.index)
            if command.task.implicit_completion:
                qualified.setdefault(command.task.engine, []).append(command.index)
        queues.setdefault(command.task.engine, []).append(command.index)
        if command.task.engine is Engine.S and command.kind in {"task", "wait"}:
            gates.append(command.index)
        if command.kind in {"set", "wait"}:
            mapping = sends if command.kind == "set" else receives
            require(command.protocol.name not in mapping, "duplicate local notification endpoint")
            mapping[command.protocol.name] = command.index
    require(sends.keys() == receives.keys(), "local notifications are not paired")

    def previous(sequence, position):
        selected = bisect_left(sequence, position) - 1
        index = sequence[selected]
        return index, int(index >= position)

    for command in commands:
        index, engine = command.index, command.task.engine
        prior, shift = previous(queues[engine], index)
        edge(ids[prior, "begin"], ids[index, "begin"], shift, "pipeline admission order")
        if gates:
            prior, shift = previous(gates, index)
            point = "end" if commands[prior].kind == "task" else "begin"
            edge(
                ids[prior, point],
                ids[index, "begin"],
                shift,
                "blocking Scalar completion gates controller dispatch",
            )
        if command.kind == "task":
            prior, shift = previous(natives[engine], index)
            edge(
                ids[prior, "retire"],
                ids[index, "retire"],
                shift,
                "cumulative native retirement capture",
            )
            if qualified.get(engine):
                prior, shift = previous(qualified[engine], index)
                edge(
                    ids[prior, "end"],
                    ids[index, "begin"],
                    shift,
                    "qualified implicit end before next native admission",
                )
        elif command.kind in {"set", "barrier"}:
            require(engine in natives, "capture has no native source engine")
            prior, shift = previous(natives[engine], index)
            point = "end" if command.kind == "set" else "begin"
            edge(ids[prior, "retire"], ids[index, point], shift, "actual source retirement capture")
    for name, source in sends.items():
        destination = receives[name]
        protocol = commands[source].protocol
        require(
            commands[destination].protocol is protocol,
            "local endpoints lost their common selected protocol",
        )
        delta = protocol.destination.stage + protocol.distance - protocol.source.stage
        edge(
            ids[source, "end"],
            ids[destination, "begin"],
            delta,
            "selected notification effect before matching Wait",
        )

    # Each simple cycle has at most V edges. If its summed step distance is
    # positive, this stride makes its total strict-rank constraint negative.
    # A nonpositive-distance cycle has no increasing rank and is rejected.
    count, visits, rounds = len(points), 0, 0
    stride, offsets = count + 1, [0] * count
    predecessor = [None] * count
    for round_index in range(count):
        changed = None
        rounds = round_index + 1
        for arc in arcs:
            visits += 1
            required = offsets[arc.source] + 1 - stride * arc.steps
            if required > offsets[arc.destination]:
                offsets[arc.destination] = required
                predecessor[arc.destination] = arc
                changed = arc.destination
        if changed is None:
            return ProgressCertificate(
                tuple(points), tuple(arcs), stride, tuple(offsets), rounds, visits
            )
    current = changed
    for _ in range(count):
        current = predecessor[current].source
    witness, first = [], current
    while not witness or current != first:
        arc = predecessor[current]
        source, destination = points[arc.source], points[arc.destination]
        witness.append(
            f"{source.command.task.operation.uid.data}:{source.command.kind}/{source.point} "
            f"-> {destination.command.task.operation.uid.data}:{destination.command.kind}/"
            f"{destination.point}, step distance {arc.steps}: {arc.reason}"
        )
        current = arc.source
    raise CompileError(
        "selected local protocol has no increasing dispatch/completion rank",
        kind=ErrorKind.UNSUPPORTED,
        phase="local-progress",
        witness=tuple(witness),
    )
