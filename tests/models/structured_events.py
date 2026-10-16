"""Independent finite event/access oracle for the final physical SSA IR.

Only small concrete instances are expanded here. This is neither a proof for
arbitrary N nor a numerical/device model. Admitted native modes, geometry and
implicit-completion primitives are trusted; no selector, resource plan,
production progress graph, Expr decoder or protocol name supplies an edge.

The logical traversal binds source read versions. A separate actor projection
executes actual SCF/stage guards and issue orders. Native ends have no FIFO
edge. Sets capture earlier ends, waits consume real IDs, and only hardware
blocking waits gate a controller. Repeating a body invents no host completion
edge, fresh local allocation identity or reset of an event/counter.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING

from xdsl.dialects import arith, scf
from xdsl.ir import Operation

from xdsl_npu.dialects.physical import (
    BufferOp,
    CoreIdOp,
    EndOp,
    FixConfigOp,
    KernelOp,
    Load3DConfigOp,
    MixedOp,
    RuntimeSizeOp,
    TaskOp,
)
from xdsl_npu.dialects.structured import (
    ActorDomainOp,
    AddressOp,
    NotificationOp,
    PipelineLoopOp,
    PipelineStageOp,
)
from xdsl_npu.target.ascend910b import Actor, Engine, SyncKind

if TYPE_CHECKING:
    from xdsl_npu.analysis.structured_memory import NativeMemoryFacts


class EventModelError(AssertionError):
    """An actual finite instruction, token or byte-access counterexample."""


def _check(condition, message):
    if not condition:
        raise EventModelError(message)


@dataclass(frozen=True)
class Occurrence:
    invocation: int
    group: int
    actor: Actor
    operation: Operation
    coordinates: tuple[tuple[Operation, int], ...]

    @property
    def label(self):
        operation = self.operation
        name = operation.uid.data if isinstance(operation, TaskOp) else operation.kind.data
        return f"{self.invocation}/{self.group}/{self.actor.value}/{name}/{tuple(g for _, g in self.coordinates)}"


@dataclass(frozen=True)
class Span:
    root: BufferOp
    group: int | None
    actor: Actor | None
    start: int
    end: int
    mode: str

    @property
    def storage(self):
        return self.root, self.group, self.actor


@dataclass
class Command:
    occurrence: Occurrence
    kind: str
    source: Engine
    destination: Engine
    event: int | None = None
    spans: tuple[Span, ...] = ()
    implicit: bool = False
    issue: int = -1
    begin: int = -1
    end: int = -1


@dataclass(frozen=True)
class Consumption:
    source: Command
    wait: Command
    counter: bool


@dataclass(frozen=True)
class Hazard:
    kind: str
    source: Occurrence
    consumer: Occurrence
    root: BufferOp
    start: int
    end: int


@dataclass(frozen=True)
class FiniteExecution:
    commands: tuple[Command, ...]
    consumptions: tuple[Consumption, ...]
    hazards: tuple[Hazard, ...]
    reachable: tuple[int, ...]
    bytes_read: int
    bytes_written: int

    @property
    def natives(self):
        return tuple(command for command in self.commands if command.kind == "native")

    def happens_before(self, source: int, destination: int):
        return bool(self.reachable[source] & (1 << destination))


class _Interpret:
    def __init__(self, memory, parameters, *, physical, maximum):
        self.memory, self.parameters = memory, parameters
        self.templates = {template.operation: template for template in memory.templates}
        self.physical, self.maximum = physical, maximum
        self.commands = []

    def native(self, operation, values, invocation, group, actor, coordinates):
        _check(actor is not None, "native has no actual actor domain")
        template = self.templates[operation]
        _check(
            template.contract.native == operation.native.data,
            "actual native changed its admitted contract",
        )
        occurrence = Occurrence(invocation, group, actor, operation, coordinates)
        spans = []
        for access in template.accesses:
            ref = operation.refs[access.operand_index]
            _check(ref in self.memory.addresses, "actual operand has no admitted native geometry")
            root, offset, allocation = values[ref]
            geometry = self.memory.addresses[ref].geometry
            mode = access.mode.value
            if (
                operation.native.data in {"mad_first", "mad_cycle"}
                and access is template.accesses[-1]
            ):
                ordinals = [g for loop, g in coordinates if isinstance(loop, PipelineLoopOp)]
                _check(bool(ordinals), "MAD initialization has no actual pipeline coordinate")
                period = 1 if operation.native.data == "mad_first" else template.contract.params[3]
                if (
                    ordinals[-1] == 0
                    or operation.native.data == "mad_cycle"
                    and ordinals[-1] % period == 0
                ):
                    mode = "W"
            private = root.ref.type.bank.data != "GM"
            for row in range(geometry.rows):
                start = offset + row * geometry.row_stride
                end = start + geometry.nbytes
                _check(
                    0 <= start < end <= allocation,
                    f"actual address out of bounds: {root.sym_name.data}",
                )
                spans.append(
                    Span(
                        root,
                        group if private else None,
                        actor if private else None,
                        start,
                        end,
                        mode,
                    )
                )
        self.commands.append(
            Command(
                occurrence,
                "native",
                template.contract.engine,
                template.contract.engine,
                spans=tuple(spans),
                implicit=template.contract.implicit_completion,
            )
        )
        _check(len(self.commands) <= self.maximum, "finite model command budget exceeded")
        for result in operation.results:
            values[result] = occurrence

    def block(self, block, values, invocation, group, actor, coordinates=()):
        for operation in block.ops:
            if isinstance(operation, (EndOp, scf.YieldOp)):
                return tuple(values[value] for value in operation.operands)
            if isinstance(operation, RuntimeSizeOp):
                value = self.parameters[operation.sym_name.data]
                _check(
                    type(value) is int
                    and value >= operation.minimum.data
                    and value % operation.multiple.data == 0,
                    "runtime parameter violates its actual declaration",
                )
                values[operation.value] = value
            elif isinstance(operation, CoreIdOp):
                values[operation.value] = group
            elif isinstance(operation, BufferOp):
                extent = (
                    operation.nbytes.data
                    if operation.extent_count is None
                    else values[operation.extent_count] * operation.count_scale.data
                )
                values[operation.ref] = operation, extent
            elif isinstance(operation, arith.ConstantOp):
                values[operation.result] = operation.value.value.data
            elif isinstance(operation, AddressOp):
                root, extent = values[operation.source]
                values[operation.ref] = root, values[operation.offset], extent
            elif isinstance(operation, TaskOp):
                self.native(operation, values, invocation, group, actor, coordinates)
            elif isinstance(operation, NotificationOp):
                if self.physical and actor in tuple(
                    Actor(value.data) for value in operation.participants
                ):
                    event = None if operation.event is None else values[operation.event]
                    kind = SyncKind(operation.kind.data)
                    if event is not None:
                        limit = 8 if kind in {SyncKind.LOCAL_SET, SyncKind.LOCAL_WAIT} else 65536
                        _check(
                            type(event) is int and 0 <= event < limit,
                            "actual notification has an invalid event ID",
                        )
                    self.commands.append(
                        Command(
                            Occurrence(invocation, group, actor, operation, coordinates),
                            kind.value,
                            Engine(operation.source.data),
                            Engine(operation.destination.data),
                            event,
                        )
                    )
                    _check(
                        len(self.commands) <= self.maximum, "finite model command budget exceeded"
                    )
            elif isinstance(operation, ActorDomainOp):
                participants = (
                    (Actor.AIC,) if operation.kind.data == "cube" else (Actor.AIV0, Actor.AIV1)
                )
                for participant in participants:
                    if actor is None or actor is participant:
                        owner = 0 if participant in {Actor.AIC, Actor.AIV0} else 1
                        self.block(
                            operation.body.block,
                            values | {operation.body.block.args[0]: owner},
                            invocation,
                            group,
                            participant,
                            coordinates,
                        )
            elif isinstance(operation, scf.IfOp):
                region = operation.true_region if values[operation.cond] else operation.false_region
                result = (
                    self.block(region.block, dict(values), invocation, group, actor, coordinates)
                    if region.blocks
                    else ()
                )
                values.update(zip(operation.results, result, strict=True))
            elif isinstance(operation, scf.ForOp):
                lower, upper, step = (
                    values[value] for value in (operation.lb, operation.ub, operation.step)
                )
                _check(
                    step > 0 and len(range(lower, upper, step)) <= self.maximum,
                    "finite serial iteration budget exceeded",
                )
                state = tuple(values[value] for value in operation.iter_args)
                for ordinal, induction in enumerate(range(lower, upper, step)):
                    local = values | dict(
                        zip(operation.body.block.args, (induction, *state), strict=True)
                    )
                    state = self.block(
                        operation.body.block,
                        local,
                        invocation,
                        group,
                        actor,
                        (*coordinates, (operation, ordinal)),
                    )
                values.update(zip(operation.results, state, strict=True))
            elif isinstance(operation, PipelineLoopOp):
                lower, upper, step = (values[value] for value in operation.operands)
                _check(step > 0, "actual pipeline step is not positive")
                count = len(range(lower, upper, step))
                _check(count <= self.maximum, "finite pipeline iteration budget exceeded")
                stages = [
                    stage
                    for stage in operation.body.block.ops
                    if isinstance(stage, PipelineStageOp)
                ]
                if self.physical and operation.issue_order is not None:
                    family = "cube" if actor is Actor.AIC else "vector"
                    if family in operation.issue_order.data:
                        ranks = {
                            label.data: rank
                            for rank, label in enumerate(operation.issue_order.data[family])
                        }
                        stages.sort(key=lambda stage: ranks.get(stage.label.data, len(ranks)))
                horizon = operation.schedule.horizon.data if self.physical else 0
                for clock in range(count + horizon):
                    for stage in stages:
                        ordinal = clock - stage.offset.data if self.physical else clock
                        if 0 <= ordinal < count:
                            local = values | dict(
                                zip(
                                    stage.body.block.args,
                                    (ordinal, lower + ordinal * step),
                                    strict=True,
                                )
                            )
                            self.block(
                                stage.body.block,
                                local,
                                invocation,
                                group,
                                actor,
                                (*coordinates, (operation, ordinal)),
                            )
            elif isinstance(operation, (FixConfigOp, Load3DConfigOp)):
                pass  # Admitted invariant register packets; no tensor access.
            else:
                self.arithmetic(operation, values)
        return ()

    @staticmethod
    def arithmetic(operation, values):
        _check(len(operation.operands) == 2, f"unsupported actual scalar {operation.name}")
        a, b = (values[value] for value in operation.operands)
        if isinstance(operation, arith.CmpiOp):
            predicate = operation.predicate.value.data
            if predicate >= 6:
                a, b = a % (1 << 64), b % (1 << 64)
            result = (a == b, a != b, a < b, a <= b, a > b, a >= b, a < b, a <= b, a > b, a >= b)[
                predicate
            ]
        else:
            functions = {
                arith.AddiOp: lambda: a + b,
                arith.SubiOp: lambda: a - b,
                arith.MuliOp: lambda: a * b,
                arith.MinSIOp: lambda: min(a, b),
                arith.MaxSIOp: lambda: max(a, b),
                arith.DivUIOp: lambda: a // b,
                arith.RemUIOp: lambda: a % b,
                arith.CeilDivUIOp: lambda: a // b + (a % b != 0),
            }
            _check(type(operation) in functions, f"unsupported actual scalar {operation.name}")
            if isinstance(operation, (arith.DivUIOp, arith.RemUIOp, arith.CeilDivUIOp)):
                _check(a >= 0 and b > 0, "guard did not protect unsigned scalar arithmetic")
            result = functions[type(operation)]()
            _check(-(1 << 63) <= result < (1 << 63), "actual scalar overflows signed index")
        values[operation.results[0]] = result


def _causality(commands):
    edges, labels = [], []

    def node(label):
        labels.append(label)
        edges.append(set())
        return len(edges) - 1

    def edge(source, destination):
        if source is not None and source != destination:
            edges[source].add(destination)

    controllers, gates, queues, ends, implicit = {}, {}, {}, defaultdict(list), {}
    local_sets, local_waits, cross_sets, cross_waits = (defaultdict(list) for _ in range(4))
    previous_counter = {}
    for command in commands:
        point = command.occurrence
        controller = point.group, point.actor
        kind = command.kind
        pipe = (
            command.destination
            if kind in {SyncKind.LOCAL_WAIT.value, SyncKind.CROSS_WAIT.value}
            else command.source
        )
        queue = *controller, pipe
        command.issue = node(point.label + ":issue")
        command.begin = node(point.label + ":begin")
        command.end = (
            node(point.label + ":end/effect")
            if kind in {"native", SyncKind.LOCAL_SET.value, SyncKind.CROSS_SET.value}
            else command.begin
        )
        edge(controllers.get(controller), command.issue)
        edge(gates.get(controller), command.issue)
        edge(command.issue, command.begin)
        edge(queues.get(queue), command.begin)
        edge(command.begin, command.end)
        controllers[controller], queues[queue] = command.issue, command.begin
        if kind == "native":
            edge(implicit.get(queue), command.begin)
            ends[queue].append(command.end)
            if command.implicit:
                completed = node(point.label + ":implicit-prefix")
                for previous in ends[queue]:
                    edge(previous, completed)
                implicit[queue] = completed
            if pipe is Engine.S:
                gates[controller] = command.end
        elif kind in {SyncKind.LOCAL_SET.value, SyncKind.CROSS_SET.value, SyncKind.FENCE.value}:
            for previous in ends[queue]:
                edge(previous, command.end)
        if (
            kind == SyncKind.CROSS_WAIT.value
            or kind == SyncKind.LOCAL_WAIT.value
            and pipe is Engine.S
        ):
            gates[controller] = command.begin
        if kind in {SyncKind.LOCAL_SET.value, SyncKind.LOCAL_WAIT.value}:
            bank = *controller, command.source, command.destination, command.event
            (local_sets if kind == SyncKind.LOCAL_SET.value else local_waits)[bank].append(command)
        elif kind == SyncKind.CROSS_SET.value:
            recipients = (Actor.AIV0, Actor.AIV1) if point.actor is Actor.AIC else (Actor.AIC,)
            for recipient in recipients:
                bank = point.group, point.actor, recipient, command.event
                cross_sets[bank].append(command)
                # FIFO counter contributions on one actual source queue. This
                # orders effects, never future source-native admissions/ends.
                fifo = *bank, command.source
                edge(previous_counter.get(fifo), command.end)
                previous_counter[fifo] = command.end
        elif kind == SyncKind.CROSS_WAIT.value:
            sources = (Actor.AIV0, Actor.AIV1) if point.actor is Actor.AIC else (Actor.AIC,)
            for source in sources:
                cross_waits[point.group, source, point.actor, command.event].append(command)

    consumptions = []
    for sends, waits, name, counter in (
        (local_sets, local_waits, "binary event", False),
        (cross_sets, cross_waits, "mode-2 quorum", True),
    ):
        _check(sends.keys() == waits.keys(), f"unmatched {name} resource")
        for bank, producers in sends.items():
            consumers = waits[bank]
            _check(len(producers) == len(consumers), f"unconsumed or missing {name} contribution")
            for send, wait in zip(producers, consumers, strict=True):
                edge(send.end, wait.begin)
                consumptions.append(Consumption(send, wait, counter))

    degree = [0] * len(edges)
    for destinations in edges:
        for destination in destinations:
            degree[destination] += 1
    pending = deque(index for index, count in enumerate(degree) if count == 0)
    order = []
    while pending:
        source = pending.popleft()
        order.append(source)
        for destination in edges[source]:
            degree[destination] -= 1
            if degree[destination] == 0:
                pending.append(destination)
    _check(len(order) == len(edges), "actual queue/controller command deadlock")
    reach = [0] * len(edges)
    for source in reversed(order):
        for destination in edges[source]:
            reach[source] |= (1 << destination) | reach[destination]
    before = lambda source, destination: bool(reach[source] & (1 << destination))
    for bank, producers in local_sets.items():
        for wait, send in zip(local_waits[bank], producers[1:]):
            _check(
                before(wait.begin, send.end),
                "binary event overwritten before actual Wait consumption",
            )
    for bank, producers in cross_sets.items():
        for old, new in pairwise(producers):
            _check(before(old.end, new.end), "mode-2 source effects have no FIFO order")
        for send, wait in zip(producers[15:], cross_waits[bank]):
            _check(before(wait.begin, send.end), "mode-2 counter backlog can exceed 15")
    return tuple(reach), tuple(consumptions), {node: index for index, node in enumerate(order)}


def _accesses(logical, physical, reach, order):
    actual = {command.occurrence: command for command in physical if command.kind == "native"}
    expected = {command.occurrence: command for command in logical}
    _check(
        actual.keys() == expected.keys(), "physical projection changed native execution instances"
    )
    boundaries = defaultdict(set)
    for key, command in expected.items():
        _check(
            command.spans == actual[key].spans,
            "physical projection changed an actual native address",
        )
        for span in command.spans:
            boundaries[span.storage].update((span.start, span.end))
    atoms = {storage: tuple(pairwise(sorted(points))) for storage, points in boundaries.items()}
    touches = {}
    for command in logical:
        touched = defaultdict(set)
        for span in command.spans:
            for start, end in atoms[span.storage]:
                if span.start <= start and end <= span.end:
                    touched[span.storage, start, end].update(span.mode)
        touches[command.occurrence] = touched
    bindings, writers, writes = [], {}, defaultdict(list)
    bytes_read = bytes_written = 0
    for command in logical:
        key = command.occurrence
        for cell, mode in touches[key].items():
            storage, start, end = cell
            if "R" in mode:
                _check(
                    cell in writers or storage[0].initialized.data,
                    f"undefined source read: {storage[0].sym_name.data}",
                )
                bindings.append((cell, key, writers.get(cell)))
                bytes_read += end - start
            if "W" in mode:
                writers[cell] = key
                writes[cell].append(key)
                bytes_written += end - start
    hazards, following = [], {}

    def require_order(kind, cell, source, consumer):
        if source == consumer:
            return
        storage, start, end = cell
        old, new = actual[source], actual[consumer]
        _check(
            bool(reach[old.end] & (1 << new.begin)),
            f"unordered physical {kind}: {storage[0].sym_name.data}[{start}:{end}] {source.label} -> {consumer.label}",
        )
        hazards.append(Hazard(kind, source, consumer, storage[0], start, end))

    for cell, versions in writes.items():
        # Independent clears may occupy a slot in a different order from
        # source traversal. Actual causal write order must still be total;
        # source RAW bindings below forbid any intervening wrong version.
        versions.sort(key=lambda key: order[actual[key].begin])
        following[cell, None] = versions[0]
        for old, new in pairwise(versions):
            require_order("WAW", cell, old, new)
            following[cell, old] = new
        if cell[0][0].external.data:
            _check(versions[-1] == writers[cell], "final external writer changed")
    for cell, reader, writer in bindings:
        if writer is not None:
            require_order("RAW", cell, writer, reader)
        next_writer = following.get((cell, writer))
        if next_writer is not None:
            require_order("WAR", cell, reader, next_writer)
    return tuple(hazards), bytes_read, bytes_written


def check_structured_events(
    memory: NativeMemoryFacts,
    parameters: dict[str, int] | None = None,
    *,
    invocations: int = 1,
    groups: tuple[int, ...] = (0,),
    max_commands: int = 5000,
) -> FiniteExecution:
    """Check only the explicitly supplied small runtime instances."""
    _check(type(invocations) is int and invocations > 0, "finite invocations must be positive")
    root = memory.module.body.block.first_op
    _check(isinstance(root, (KernelOp, MixedOp)), "model requires one actual physical root")
    count = 1 if root.blocks is None else root.blocks.data
    _check(
        bool(groups)
        and len(set(groups)) == len(groups)
        and all(0 <= group < count for group in groups),
        "invalid finite physical group selection",
    )
    actors = (
        (Actor.AIC, Actor.AIV0, Actor.AIV1)
        if isinstance(root, MixedOp)
        else (Actor.AIC,)
        if root.core.data == "aic"
        else (Actor.AIV0,)
    )
    physical = _Interpret(memory, parameters or {}, physical=True, maximum=max_commands)
    logical = _Interpret(memory, parameters or {}, physical=False, maximum=max_commands)
    for invocation in range(invocations):
        for group in groups:
            logical.block(
                root.body.block,
                {},
                invocation,
                group,
                None if isinstance(root, MixedOp) else actors[0],
            )
            for actor in actors:
                physical.block(root.body.block, {}, invocation, group, actor)
    reach, consumptions, order = _causality(physical.commands)
    hazards, reads, writes = _accesses(logical.commands, physical.commands, reach, order)
    return FiniteExecution(tuple(physical.commands), consumptions, hazards, reach, reads, writes)
