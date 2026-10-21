"""Independent finite event/access oracle for the final physical SSA IR.

Only small concrete instances are expanded here. This is neither a proof for
arbitrary N nor a numerical/device model. Admitted native modes, geometry and
implicit-completion primitives are trusted; no selector, resource plan,
production progress graph, Expr decoder or protocol name supplies an edge.

An optional independent pre-sync native snapshot supplies the logical traversal
and source read versions. Without it, both traversals read the final IR and do
not check preservation of its native program. A separate actor projection
executes actual SCF/stage guards and issue orders. Native ends have no FIFO
edge. Sets capture earlier ends, waits consume real IDs, and only hardware
blocking waits gate a controller. Repeating a body invents no host completion
edge, fresh local allocation identity or reset of an event/counter.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, replace
from itertools import pairwise
from typing import TYPE_CHECKING

from xdsl.dialects import arith, scf
from xdsl.ir import Operation

from ascendflow.dialects.physical import (
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
from ascendflow.dialects.structured import (
    ActorDomainOp,
    AddressOp,
    NotificationOp,
    PipelineLoopOp,
    PipelineStageOp,
)
from ascendflow.target.ascend910b import Actor, Engine, SyncKind

if TYPE_CHECKING:
    from ascendflow.analysis.structured_memory import NativeMemoryFacts


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
            template.spec.native == operation.native.data,
            "actual native changed its admitted contract",
        )
        occurrence = Occurrence(invocation, group, actor, operation, coordinates)
        spans = []
        for access in template.accesses:
            ref = operation.refs[access.operand_index]
            root, offset, allocation = values[ref]
            geometry = access.address.geometry
            address = ref.owner
            _check(
                isinstance(address, AddressOp)
                and (address.nbytes.data, address.rows.data, address.row_stride.data)
                == (geometry.nbytes, geometry.rows, geometry.row_stride),
                "actual operand changed its admitted native geometry",
            )
            mode = access.mode.value
            if (
                operation.native.data in {"mad_first", "mad_cycle"}
                and access is template.accesses[-1]
            ):
                ordinal = values[operation.refs[-1]]
                period = 1 if operation.native.data == "mad_first" else template.spec.params[3]
                if ordinal == 0 or operation.native.data == "mad_cycle" and ordinal % period == 0:
                    mode = "W"
            private = root.ref.type.bank.data != "GM"
            active_rows = geometry.rows
            active_bytes = geometry.nbytes
            if address.valid_rows is not None:
                count = values[address.valid_rows]
                _check(
                    0 <= count <= access.address.row_capacity, "actual runtime row count exceeds capacity"
                )
                if geometry.layout == "NZ":
                    active_bytes = count * 32
                    active_rows = geometry.rows if count else 0
                else:
                    active_rows = count
            for row in range(active_rows):
                start = offset + row * geometry.row_stride
                end = start + active_bytes
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
                template.spec.engine,
                template.spec.engine,
                spans=tuple(spans),
                implicit=template.spec.implicit_completion,
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
                        # dav-2201 GetfftsConfig retains four flag-ID bits.
                        # IDs >=16 alias hardware resources, not extra slots.
                        limit = 8 if kind in {SyncKind.LOCAL_SET, SyncKind.LOCAL_WAIT} else 16
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
                nested_coordinates = coordinates
                if "npu.stage_offset" in operation.attributes:
                    loop, clock = coordinates[-1]
                    nested_coordinates = (
                        *coordinates[:-1],
                        (loop, clock - operation.attributes["npu.stage_offset"].data),
                    )
                result = (
                    self.block(
                        region.block, dict(values), invocation, group, actor, nested_coordinates
                    )
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
                        (
                            *coordinates,
                            (
                                operation,
                                induction - operation.attributes["npu.pipeline_clock"].data
                                if "npu.pipeline_clock" in operation.attributes
                                else ordinal,
                            ),
                        ),
                    )
                values.update(zip(operation.results, state, strict=True))
            elif isinstance(operation, PipelineLoopOp):
                count = max(0, values[operation.count])
                _check(count <= self.maximum, "finite pipeline iteration budget exceeded")
                family = "cube" if actor is Actor.AIC else "vector"
                stages = [
                    stage
                    for stage in operation.body.block.ops
                    if isinstance(stage, PipelineStageOp)
                    and (actor is None or stage.family is None or stage.family.data == family)
                ]
                if (
                    self.physical
                    and operation.issue_order is not None
                    and family in operation.issue_order.data
                ):
                    ranks = {
                        label.data: rank
                        for rank, label in enumerate(operation.issue_order.data[family])
                    }
                    stages.sort(key=lambda stage: ranks.get(stage.label.data, len(ranks)))
                horizon = (
                    max((stage.offset.data for stage in stages), default=0) if self.physical else 0
                )
                for clock in range(count + horizon):
                    for stage in stages:
                        ordinal = clock - stage.offset.data if self.physical else clock
                        if 0 <= ordinal < count:
                            local = values | dict(
                                zip(
                                    stage.body.block.args,
                                    (ordinal,),
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
        if isinstance(operation, arith.SelectOp):
            condition, left, right = operation.operands
            values[operation.results[0]] = values[left] if values[condition] else values[right]
            return
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
                arith.AndIOp: lambda: a & b,
                arith.OrIOp: lambda: a | b,
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
    prefixes = {}

    def capture(queue, label):
        # Factor the union of prior native completions, not prior Set effects.
        # A synthetic prefix has exactly the incoming native-end edges that
        # each subsequent capture used to repeat. Real notifications remain
        # unordered unless their hardware rule supplies an edge separately.
        if ends[queue]:
            completed = node(label)
            edge(prefixes.get(queue), completed)
            for previous in ends[queue]:
                edge(previous, completed)
            ends[queue].clear()
            prefixes[queue] = completed
        return prefixes.get(queue)

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
                implicit[queue] = capture(queue, point.label + ":implicit-prefix")
            if pipe is Engine.S:
                gates[controller] = command.end
        elif kind in {SyncKind.LOCAL_SET.value, SyncKind.CROSS_SET.value, SyncKind.FENCE.value}:
            edge(capture(queue, point.label + ":captured-prefix"), command.end)
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
    points = {storage: tuple(sorted(values)) for storage, values in boundaries.items()}
    atoms = {storage: tuple(pairwise(values)) for storage, values in points.items()}
    positions = {storage: {value: i for i, value in enumerate(values)} for storage, values in points.items()}
    touches = {}
    for command in logical:
        touched = defaultdict(set)
        for span in command.spans:
            # Every endpoint was inserted above. The exact same partition
            # atoms are covered; avoid scanning the entire tensor's atoms for
            # every small native row (quadratic in large sparse/Gram cases).
            first, last = positions[span.storage][span.start], positions[span.storage][span.end]
            for start, end in atoms[span.storage][first:last]:
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
        if cell[0][0].write_disjoint is not None:
            _check(
                len({key.invocation for key in versions}) == len(versions),
                "write_disjoint contract violated by actual overlapping writes",
            )
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


def _reference_correspondence(memory, reference):
    """Map independent snapshots; synchronization may only add scalar/sync IR.

    Native UIDs and allocation symbols identify the retained source objects.
    Loops have no UID, so match their ordered native-bearing nesting paths,
    ignoring notification-only wrappers. This deliberately rejects native or
    loop restructuring; such a transformation needs its own reference mapping.
    """
    _check(memory.module is not reference.module, "logical reference needs an independent snapshot")

    def indexed(module, operation_type, name):
        result = {}
        for operation in module.walk():
            if isinstance(operation, operation_type):
                key = getattr(operation, name).data
                _check(key not in result, f"ambiguous logical reference identity: {key}")
                result[key] = operation
        return result

    actual_tasks = indexed(memory.module, TaskOp, "uid")
    source_tasks = indexed(reference.module, TaskOp, "uid")
    _check(actual_tasks.keys() == source_tasks.keys(), "logical reference native UID set changed")
    tasks = {}
    for uid, source in source_tasks.items():
        actual = actual_tasks[uid]
        _check(
            all(
                source.attributes[name] == actual.attributes[name]
                for name in ("native", "params", "scalar")
            )
            and tuple(value.type for value in source.refs)
            == tuple(value.type for value in actual.refs),
            f"logical reference native contract changed: {uid}",
        )
        tasks[source] = actual

    actual_roots = indexed(memory.module, BufferOp, "sym_name")
    source_roots = indexed(reference.module, BufferOp, "sym_name")
    _check(actual_roots.keys() == source_roots.keys(), "logical reference allocation roots changed")
    roots = {}
    for name, source in source_roots.items():
        actual = actual_roots[name]
        _check(
            source.ref.type == actual.ref.type
            and all(
                source.attributes.get(attribute) == actual.attributes.get(attribute)
                for attribute in ("nbytes", "external", "initialized", "count_scale")
            ),
            f"logical reference allocation contract changed: {name}",
        )
        roots[source] = actual

    def loops(module):
        return [
            op
            for op in module.walk()
            if isinstance(op, (PipelineLoopOp, scf.ForOp))
            and any(isinstance(child, TaskOp) for child in op.walk())
        ]

    actual_loops, source_loops = loops(memory.module), loops(reference.module)
    loop_map = {}
    for source in source_loops:
        source_id = source.attributes.get("npu.loop_id")
        source_uids = frozenset(op.uid.data for op in source.walk() if isinstance(op, TaskOp))
        candidates = [
            actual
            for actual in actual_loops
            if (source_id is not None and actual.attributes.get("npu.loop_id") == source_id)
            or (
                source_id is None
                and frozenset(op.uid.data for op in actual.walk() if isinstance(op, TaskOp))
                == source_uids
            )
        ]
        _check(bool(candidates), "logical reference native loop paths changed")
        for task in (op for op in source.walk() if isinstance(op, TaskOp)):
            containing = [actual for actual in candidates if tasks[task] in set(actual.walk())]
            _check(len(containing) == 1, "logical reference native loop paths changed")
            loop_map[source, task] = containing[0]

    def command(value):
        point = value.occurrence
        return replace(
            value,
            occurrence=replace(
                point,
                operation=tasks[point.operation],
                coordinates=tuple(
                    (loop_map[loop, point.operation], ordinal)
                    for loop, ordinal in point.coordinates
                ),
            ),
            spans=tuple(replace(span, root=roots[span.root]) for span in value.spans),
        )

    return command


def check_compiled_events(compilation, parameters=None, **options):
    """Use the independent pre-sync native snapshot for a terminal SCF plan."""
    from ascendflow.analysis.structured_memory import collect_native_memory
    from ascendflow.dialects.io import parse

    original = collect_native_memory(parse(compilation.before_ir))
    return check_structured_events(
        collect_native_memory(compilation.module), parameters, logical_reference=original, **options
    )


def check_structured_events(
    memory: NativeMemoryFacts,
    parameters: dict[str, int] | None = None,
    *,
    logical_reference: NativeMemoryFacts | None = None,
    invocations: int = 1,
    groups: tuple[int, ...] = (0,),
    max_commands: int = 5000,
) -> FiniteExecution:
    """Check explicitly supplied finite instances, optionally against pre-sync IR.

    ``logical_reference`` must own an independent native snapshot made before
    synchronization, with stable UIDs, allocation symbols and retained loops.
    It supplies expected native instances, byte addresses and read versions;
    the actual final IR alone supplies queue/event causality. Both snapshots
    still trust admitted native geometry and hardware completion contracts.
    """
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
    reference = memory if logical_reference is None else logical_reference
    source_root = reference.module.body.block.first_op
    _check(
        type(source_root) is type(root)
        and all(
            source_root.attributes.get(name) == root.attributes.get(name)
            for name in ("sym_name", "target", "core", "blocks", "disjoint_external")
        ),
        "logical reference physical invocation changed",
    )
    correspondence = (
        None if logical_reference is None else _reference_correspondence(memory, reference)
    )
    physical = _Interpret(memory, parameters or {}, physical=True, maximum=max_commands)
    logical = _Interpret(reference, parameters or {}, physical=False, maximum=max_commands)
    for invocation in range(invocations):
        for group in groups:
            logical.block(
                source_root.body.block,
                {},
                invocation,
                group,
                None if isinstance(root, MixedOp) else actors[0],
            )
            for actor in actors:
                physical.block(root.body.block, {}, invocation, group, actor)
    expected = (
        logical.commands
        if correspondence is None
        else [correspondence(command) for command in logical.commands]
    )
    reach, consumptions, order = _causality(physical.commands)
    hazards, reads, writes = _accesses(expected, physical.commands, reach, order)
    return FiniteExecution(tuple(physical.commands), consumptions, hazards, reach, reads, writes)
