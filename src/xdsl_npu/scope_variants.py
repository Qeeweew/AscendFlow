"""Finite parent cases on one retained child native template.

Filtering changes analysis clones only. Native instructions in the actual
stream stay in their original blocks; planning installs only case-labelled
channel declarations and synchronization operations beside those natives.
"""

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

from xdsl.dialects.builtin import ArrayAttr, IndexType, IntAttr, StringAttr

from .budget import bounded, poll
from .ir import (
    BoundaryOp,
    CallOp,
    EndOp,
    LoopChannelOp,
    LoopSyncOp,
    OnceOp,
    ScopeGuardOp,
    ScopeSyncOp,
    StreamOp,
    TaskOp,
    ViewOp,
)
from .model import require
from .scope_predicates import PARENT_CASES, guard_active


def operation_cases(op):
    """Read actual labels; absence means all cases, never an empty case set."""
    attr = op.attributes.get("scope_cases")
    if attr is None:
        return PARENT_CASES
    require(
        isinstance(attr, ArrayAttr) and all(isinstance(value, StringAttr) for value in attr),
        "scope cases must be an array of parent case names",
    )
    cases = tuple(value.data for value in attr)
    require(
        cases and cases == tuple(case for case in PARENT_CASES if case in cases),
        "scope cases must be nonempty, known, distinct and in parent case order",
    )
    return cases


def scope_operations(stream):
    """Visit actual operations and their enclosing parent guard kinds.

    Wrapper terminators are structural and do not escape their body. Inner
    last-only boundaries remain unsupported in retained parent-case children;
    they must not be confused with a guard on the parent clock.
    """

    def visit(block, guards):
        for op in block.ops:
            require(
                not isinstance(op, BoundaryOp),
                "parent-case children do not support child-local boundary regions",
            )
            if isinstance(op, ScopeGuardOp):
                require(
                    len(op.body.blocks) == 1
                    and not op.body.block.args
                    and isinstance(op.body.block.last_op, EndOp),
                    "scope guard needs a terminated body without block arguments",
                )
                yield from visit(op.body.block, (*guards, op.kind.data))
            elif not (guards and isinstance(op, EndOp)):
                yield op, guards

    yield from visit(stream.body.block, ())


def _case_operations(block, case):
    for op in block.ops:
        if isinstance(op, ScopeGuardOp):
            if guard_active(op.kind.data, case):
                yield from _case_operations(op.body.block, case)
        elif isinstance(op, BoundaryOp):
            yield from _case_operations(op.body.block, case)
        elif isinstance(op, (LoopChannelOp, LoopSyncOp)):
            if case in operation_cases(op):
                yield op
        else:
            yield op


def _pairing(stream, case):
    declarations, syncs, placements, tasks = {}, [], {}, {}
    pending, previous = [], None
    for op in _case_operations(stream.body.block, case):
        if isinstance(op, LoopChannelOp):
            name = op.sym_name.data
            require(name not in declarations, "duplicate scoped child channel")
            declarations[name] = op
        elif isinstance(op, TaskOp):
            uid = op.uid.data
            require(uid not in tasks, "duplicate scoped child native uid")
            tasks[uid] = op
            for sync in pending:
                placements[sync] = uid
            pending, previous = [], uid
        elif isinstance(op, LoopSyncOp):
            action = op.action.data
            require(action in {"set", "wait", "barrier"}, "unknown scoped child sync action")
            syncs.append(op)
            if action == "set":
                require(previous is not None and not pending, "scoped Set has no native producer")
                placements[op] = previous
            else:
                pending.append(op)
    require(not pending, "scoped acquisition has no active native consumer")
    counts = defaultdict(lambda: defaultdict(int))
    for op in syncs:
        name, action = op.channel.data, op.action.data
        if action == "barrier":
            require(not name, "scoped barrier cannot acquire a hidden channel")
            continue
        require(name in declarations, "scoped sync refers to an inactive or unknown child channel")
        channel = declarations[name]
        require(channel.kind.data == "local", "parent cases require local child event channels")
        endpoints = channel.sources if action == "set" else channel.destinations
        require(
            len(endpoints) == 1
            and endpoints.data[0].data.count("/") == 1
            and endpoints.data[0].data.split("/", 1)[1] == placements[op],
            "scoped child channel endpoint disagrees with its actual native placement",
        )
        counts[name][action] += 1
    for name in declarations:
        require(
            counts[name] == {"set": 1, "wait": 1},
            "scoped child channel needs one active Set and Wait in every declared case",
        )


@bounded
def validate_child_scopes(stream, call):
    """Validate actual clocks and pairing before any case can discard an op."""
    require(
        isinstance(stream, (StreamOp, OnceOp))
        and isinstance(call, CallOp)
        and call.parent is not None
        and len(call.parent.args) == 1
        and call.generation is call.parent.args[0]
        and stream.parent is call.body.block
        and len(stream.body.blocks) == 1
        and len(stream.body.block.args) == 1
        and isinstance(stream.body.block.args[0].type, IndexType)
        and stream.limit is None
        and stream.count is not None
        and stream.count.data >= 0
        and isinstance(stream.body.block.last_op, EndOp),
        "scope cases need the actual fixed child in its owning CallOp",
    )
    coordinate = stream.body.block.args[0]
    guards, names = [], set()

    def visit(block, enclosing=None):
        for index, op in enumerate(block.ops):
            if index % 128 == 0:
                poll("child parent-case validation", operations=index)
            require(
                isinstance(
                    op,
                    (
                        ScopeGuardOp,
                        BoundaryOp,
                        ViewOp,
                        TaskOp,
                        LoopChannelOp,
                        LoopSyncOp,
                        ScopeSyncOp,
                        EndOp,
                    ),
                ),
                f"unsupported scoped child effect {op.name}",
            )
            if isinstance(op, ScopeGuardOp):
                require(
                    enclosing is None
                    and op.parent_generation is call.generation
                    and op.child_generation is coordinate
                    and op.kind.data in {"first", "last"}
                    and len(op.body.blocks) == 1
                    and not op.body.block.args
                    and isinstance(op.body.block.last_op, EndOp),
                    "scope guard needs first/last on its actual parent and child clocks",
                )
                nested = set(op.walk())
                require(
                    all(
                        use.operation in nested
                        for child in nested
                        for result in child.results
                        for use in result.uses
                    ),
                    "a value defined inside a scope guard escapes its conditional body",
                )
                guards.append(op)
                visit(op.body.block, op)
            elif isinstance(op, BoundaryOp):
                require(False, "parent-case children do not support child-local boundary regions")
            elif isinstance(op, LoopChannelOp):
                cases = operation_cases(op)
                require(op.sym_name.data not in names, "duplicate scoped child channel")
                names.add(op.sym_name.data)
                if enclosing is not None and isinstance(enclosing, ScopeGuardOp):
                    require(
                        any(guard_active(enclosing.kind.data, case) for case in cases),
                        "child channel cases contradict its enclosing scope guard",
                    )
            elif isinstance(op, LoopSyncOp):
                cases = operation_cases(op)
                parent = op.parent_generation
                require(
                    op.generation is coordinate
                    and (
                        parent is call.generation
                        if "scope_cases" in op.attributes
                        else parent is None
                    ),
                    "child sync parent cases need its actual parent and child clocks",
                )
                if enclosing is not None and isinstance(enclosing, ScopeGuardOp):
                    require(
                        any(guard_active(enclosing.kind.data, case) for case in cases),
                        "child sync cases contradict its enclosing scope guard",
                    )
            elif isinstance(op, ScopeSyncOp):
                require(
                    op.parent_generation is call.generation and op.child_generation is coordinate,
                    "scope synchronization uses a different parent or child clock",
                )
            elif isinstance(op, ViewOp):
                require(op.generation is coordinate, "child view uses a different inner clock")

    visit(stream.body.block)
    for case in PARENT_CASES:
        _pairing(stream, case)
    return tuple(guards)


def child_is_empty(stream):
    """An explicitly empty analysis case has no surviving native template."""
    return not any(isinstance(op, TaskOp) for op in stream.walk())


def describe_child(stream, call, *, value_mapper=None):
    """Describe every native and actual sync for emission, never for proof.

    Conditional verification exclusively uses filter_child. This union view
    supplies ordinary native operand/command rendering after those proofs.
    """
    validate_child_scopes(stream, call)
    clone = stream.clone(value_mapper=value_mapper)

    def flatten(block):
        for op in tuple(block.ops):
            if isinstance(op, ScopeGuardOp):
                flatten(op.body.block)
                children = tuple(
                    child for child in op.body.block.ops if not isinstance(child, EndOp)
                )
                for child in children:
                    op.body.block.detach_op(child)
                block.insert_ops_before(children, op)
                block.erase_op(op)
            elif isinstance(op, (LoopChannelOp, LoopSyncOp)):
                op.attributes.pop("scope_cases", None)
                if isinstance(op, LoopSyncOp):
                    op.operands = (op.generation,)

    flatten(clone.body.block)
    return clone


@bounded
def filter_child(stream, call, case, *, value_mapper=None):
    """Clone one case, unwrap active guards, and remove inactive side effects.

    Empty cases return a genuinely empty stream; callers must use
    ``child_is_empty`` instead of inventing a native command for analysis.
    Parent ScopeSyncOps remain beside their selected native instructions for
    the scope model to check and erase separately.
    """
    require(case in PARENT_CASES, "unknown parent child case")
    validate_child_scopes(stream, call)
    clone = stream.clone(value_mapper=value_mapper)

    def filter_block(block):
        for op in tuple(block.ops):
            if isinstance(op, ScopeGuardOp):
                if guard_active(op.kind.data, case):
                    filter_block(op.body.block)
                    children = tuple(
                        child for child in op.body.block.ops if not isinstance(child, EndOp)
                    )
                    for child in children:
                        op.body.block.detach_op(child)
                    block.insert_ops_before(children, op)
                block.erase_op(op)
            elif isinstance(op, BoundaryOp):
                filter_block(op.body.block)
            elif isinstance(op, (LoopChannelOp, LoopSyncOp)):
                if case not in operation_cases(op):
                    block.erase_op(op)
                    continue
                op.attributes.pop("scope_cases", None)
                if isinstance(op, LoopSyncOp):
                    op.operands = (op.generation,)

    filter_block(clone.body.block)
    return clone


@dataclass(frozen=True)
class VariantPlan:
    cases: tuple[str, ...]
    empty_cases: tuple[str, ...]
    channel_count: int
    sync_count: int
    banks: tuple[tuple[tuple[str, str, str], int], ...]


def _attributes(op, excluded=()):
    return tuple(
        sorted((name, value) for name, value in op.attributes.items() if name not in excluded)
    )


def _placements(stream):
    """Read selected sync order at actual native before/after positions."""
    before, after, pending, previous = defaultdict(list), defaultdict(list), [], None
    for op, _ in scope_operations(stream):
        if isinstance(op, TaskOp):
            before[op.uid.data].extend(pending)
            pending, previous = [], op.uid.data
        elif isinstance(op, LoopSyncOp):
            if op.action.data == "set":
                require(previous is not None and not pending, "selected Set has no native producer")
                after[previous].append(op)
            else:
                pending.append(op)
    require(not pending, "selected acquisition has no native consumer")
    return tuple(
        (uid, side, index, sync)
        for side, positions in (("before", before), ("after", after))
        for uid, operations in positions.items()
        for index, sync in enumerate(operations)
    )


@bounded
def plan_child_variants(stream, call, loops, *, live_in=(), banks=None):
    """Plan normalized nonempty cases and install only their selected syncs.

    ``loops`` maps effective parent labels to normalized one-actor Loops;
    ``None`` explicitly identifies an empty case. Each leaf planner receives
    independent temporary banks. Equal channel configurations and exact sync
    placements share one declaration; other configurations reserve separate
    IDs, including when their parent cases are mutually exclusive.

    ``live_in`` is either one iterable shared by cases or a per-case mapping.
    ``banks`` is the caller's actor-specific next-ID map, as for _plan_leaf.
    Source natives are neither replaced nor cloned into the actual stream.
    """
    from .loop_sync import _plan_leaf

    validate_child_scopes(stream, call)
    require(
        isinstance(loops, Mapping) and all(case in PARENT_CASES for case in loops),
        "child variant plans need known parent cases",
    )
    require(
        not any(isinstance(op, (LoopChannelOp, LoopSyncOp)) for op in stream.walk()),
        "parent-case AutoSync expects an unsynchronized actual child",
    )
    baseline = defaultdict(int, {} if banks is None else banks)
    require(
        all(
            isinstance(key, tuple)
            and len(key) == 3
            and all(isinstance(part, str) for part in key)
            and type(value) is int
            and 0 <= value <= 8
            for key, value in baseline.items()
        ),
        "invalid parent-case event bank reservation",
    )
    if isinstance(live_in, Mapping):
        require(
            all(case in live_in for case, loop in loops.items() if loop is not None),
            "child variant is missing its explicit live-in assumptions",
        )
    else:
        live_in = tuple(live_in)
    actual = {op.uid.data: op for op, _ in scope_operations(stream) if isinstance(op, TaskOp)}
    groups, selected, empty = {}, [], []
    for case in PARENT_CASES:
        if case not in loops:
            continue
        loop = loops[case]
        active = {
            op.uid.data: op
            for op in _case_operations(stream.body.block, case)
            if isinstance(op, TaskOp)
        }
        if loop is None:
            require(not active, "a nonempty actual child case cannot be omitted from planning")
            empty.append(case)
            continue
        require(
            len(loop.actors) == 1 and loop.constant_count == stream.count.data,
            "parent-case planning needs one actor and the actual fixed child count",
        )
        actor, data = next(iter(loop.actors.items()))
        natives = {op.uid.data: op for op in data.stream.walk() if isinstance(op, TaskOp)}
        require(
            natives.keys() == active.keys()
            and all(natives[uid].attributes == op.attributes for uid, op in active.items()),
            "normalized child case changed its actual native templates",
        )
        if data.program.config:
            baseline[actor, "S", "FIX"] = max(baseline[actor, "S", "FIX"], 1)
        if data.program.load3d is not None:
            baseline[actor, "S", "MTE1"] = max(baseline[actor, "S", "MTE1"], 1)
        temporary = defaultdict(int, baseline)
        poll("native parent-case planning", case=case)
        _plan_leaf(
            loop,
            live_in=live_in[case] if isinstance(live_in, Mapping) else live_in,
            banks=temporary,
        )
        selected.append(case)
        placements = _placements(data.stream)
        channels = tuple(op for op in loop.protocol_scope.walk() if isinstance(op, LoopChannelOp))
        for declaration in channels:
            require(
                declaration.kind.data == "local" and declaration.carry is None,
                "parent cases require local, non-carried child events",
            )
            name = declaration.sym_name.data
            positioned = tuple(item for item in placements if item[-1].channel.data == name)
            width = len(declaration.event_ids)
            require(width > 0, "selected parent-case channel has no locally proved ID width")
            descriptor = (
                _attributes(declaration, ("sym_name", "event_ids", "scope_cases")),
                width,
                tuple(
                    (uid, side, index, _attributes(op, ("channel", "scope_cases")))
                    for uid, side, index, op in positioned
                ),
            )
            if descriptor not in groups:
                (source,), (destination,) = (
                    declaration.sources.data,
                    declaration.destinations.data,
                )
                source_uid, destination_uid = (
                    source.data.split("/", 1)[1],
                    destination.data.split("/", 1)[1],
                )
                tasks = {task.uid: task for task in data.program.tasks}
                bank = actor, tasks[source_uid].pipe, tasks[destination_uid].pipe
                groups[descriptor] = declaration, positioned, bank, width, []
            groups[descriptor][-1].append(case)
        for uid, side, index, op in placements:
            if op.action.data != "barrier":
                continue
            descriptor = "barrier", uid, side, index, _attributes(op, ("scope_cases",))
            if descriptor not in groups:
                groups[descriptor] = None, ((uid, side, index, op),), None, 0, []
            groups[descriptor][-1].append(case)

    assignments, channel_count = [], 0
    for declaration, positioned, bank, width, cases in groups.values():
        name, ids = "", ()
        if declaration is not None:
            name = f"case_c{channel_count}"
            channel_count += 1
            base = baseline[bank]
            require(
                base + width <= 8,
                f"{bank}: parent-case event bank exhausted; no serialization fallback",
            )
            baseline[bank] = base + width
            ids = tuple(range(base, base + width))
        assignments.append((declaration, positioned, cases, name, ids))

    declarations, locations = [], defaultdict(list)
    for declaration, positioned, cases, name, ids in assignments:
        case_attr = ArrayAttr(StringAttr(case) for case in PARENT_CASES if case in cases)
        if declaration is not None:
            cloned = declaration.clone()
            cloned.attributes.update(
                sym_name=StringAttr(name),
                event_ids=ArrayAttr(IntAttr(value) for value in ids),
                scope_cases=case_attr,
            )
            declarations.append(cloned)
        for uid, side, index, operation in positioned:
            op = operation.clone()
            op.attributes.update(channel=StringAttr(name), scope_cases=case_attr)
            op.operands = (stream.body.block.args[0], call.generation)
            locations[uid, side].append((index, len(locations[uid, side]), op))

    # All bank and template checks finish before the actual stream changes.
    for (uid, side), operations in locations.items():
        target = actual[uid]
        ordered = [op for _, _, op in sorted(operations, key=lambda item: item[:2])]
        if side == "before":
            target.parent.insert_ops_before(ordered, target)
        else:
            anchor = target
            for op in ordered:
                target.parent.insert_op_after(op, anchor)
                anchor = op
    stream.body.block.insert_ops_before(declarations, stream.body.block.first_op)
    if banks is not None:
        banks.update(baseline)
    return VariantPlan(
        tuple(selected),
        tuple(empty),
        len(declarations),
        sum(len(operations) for operations in locations.values()),
        tuple(sorted(baseline.items())),
    )
