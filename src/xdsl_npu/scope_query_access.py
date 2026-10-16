"""Address equivalences for the memory analysis of one actual serial query.

These descriptors never replace physical buffers, captured pointers, or the
GM ABI. A ring normalization is one common permutation of its slots for a
fixed query. Reused storage remains reused; none of these contracts proves
completion, retirement, or safe reuse between queries.
"""

from dataclasses import dataclass, replace

from .budget import bounded, poll
from .diagnostics import require
from .dialects.physical import BufferOp, CallOp, CaptureOp, CoreIdOp, RuntimeSizeOp, TaskOp, ViewOp
from .indexing import arguments, constant_ratio, equivalent, polynomial
from .loop_model import _storage
from .model import Storage, optional_int
from .periodic import bounded_period
from .scalar import Expr, expression, facts_from_sizes, from_ssa, literal
from .scope_invocation import extract_scope_invocation, verify_query_domains
from .scope_model import Captured, ScopedProgram, _uniforms
from .target.native import native_contract

CORE = Expr("argument", ("$core",))
KV = Expr("argument", ("$parent0",))
QUERY = Expr("argument", ("$parent1",))


@dataclass(frozen=True)
class QueryRoot:
    """A proved coordinate contract tied to its original allocation identity.

    ``frame_bytes`` is absent for dynamically sized stationary storage.
    ``phase`` is the actual query's ring rotation, retained rather than erased.
    ``biases`` describes static offsets within normalized frames when there is
    such a frame. Readonly frames are analysis aliases and have zero biases.
    """

    kind: str
    actual_root: BufferOp
    frame_bytes: int | None
    slots: int
    phase: Expr | None
    groups: int
    total: Expr | None
    actual_extent: Expr
    biases: dict[str, int]


@dataclass(frozen=True)
class QueryAccessMap:
    # Only changed GM analysis descriptors, keyed by actual actor/root name.
    storages: dict[tuple[str, str], Storage]
    # Every capture, with GM names unchanged and local names actor-qualified.
    offsets: dict[tuple[str, str], Expr]
    contracts: dict[str, QueryRoot]


@dataclass(frozen=True)
class _Alias:
    uid: str
    actor: str
    capture: Captured
    storage: Storage


def _snapshot(op, cloned, values):
    require(
        type(op) is type(cloned)
        and op.attributes == cloned.attributes
        and tuple(values.get(value, value) for value in op.operands) == tuple(cloned.operands),
        "query access has a stale native/view descriptor; extract actual scopes again",
    )


def _native_modes(leaf):
    """Read every actual native template, including first/last guarded work."""
    require(len(leaf.loop.actors) == 1, "query access leaf needs one actual actor")
    _, data = next(iter(leaf.loop.actors.items()))
    actual = [op for op in leaf.stream.walk() if isinstance(op, TaskOp)]
    require(
        len(actual) == len(data.program.tasks)
        and {op.uid.data for op in actual} == data.operations.keys(),
        "query access has stale actual native templates",
    )
    captures = {captured.op.ref: captured for captured in leaf.captures}
    for op in leaf.stream.walk():
        if isinstance(op, ViewOp):
            mapped = leaf.value_map.get(op.ref)
            require(mapped is not None, "query access has a stale actual view")
            _snapshot(op, mapped.owner, leaf.value_map)
            require(
                op.generation is leaf.stream.body.block.args[0],
                "query access view uses a different actual child clock",
            )
    tasks = {task.uid: task for task in data.program.tasks}
    modes = {}
    for op in actual:
        task = tasks[op.uid.data]
        _snapshot(op, data.operations[op.uid.data], leaf.value_map)
        pipe, effects = native_contract(op.native.data, task.refs, tuple(p.data for p in op.params))
        require(
            (pipe, effects) == (task.pipe, task.modes),
            "query access has stale native memory modes",
        )
        operands = tuple(op.refs)
        if op.native.data in {"scale_value", "sub_value"}:
            operands = operands[:-1]
        require(len(operands) == len(task.refs), "query access loses a native memory operand")
        for operand, ref, mode in zip(operands, task.refs, effects):
            seen = set()
            while isinstance(operand.owner, ViewOp):
                require(operand not in seen, "query access has a cyclic view alias")
                seen.add(operand)
                operand = operand.owner.source
            require(operand in captures, "query access native loses its actual capture root")
            captured = captures[operand]
            require(
                ref.storage.name == captured.op.sym_name.data,
                "query access native descriptor changed its actual capture alias",
            )
            modes.setdefault(captured.root, set()).update(mode)
    return modes


def _actual_domains(program, facts):
    """Recheck actual clocks without cloning or re-extracting native regions."""
    require(
        program.module.body.block.first_op is program.root
        and optional_int(program.root.blocks, 1) == program.groups
        and tuple(op for op in program.root.body.block.ops if isinstance(op, RuntimeSizeOp))
        == program.parameters,
        "query access has a stale actual launch or scalar domain",
    )
    uniforms = _uniforms(tuple(program.root.body.block.ops), facts, allow_local=True)
    require(
        program.parents.keys() == program.kernels.keys() == program.invocations.keys(),
        "query access requires actual serial query invocations for every actor",
    )
    invocations, coordinates = {}, {}
    for actor, kernel in program.kernels.items():
        invocation, parent = extract_scope_invocation(kernel, uniforms, facts, program.groups)
        cached = program.invocations[actor]
        require(
            invocation is not None
            and invocation.op is cached.op
            and invocation.bound == cached.bound
            and invocation.cyclic == cached.cyclic
            and parent is program.parents[actor]
            and optional_int(kernel.blocks, 1) == program.groups,
            "query access has a stale actual query/parent domain",
        )
        require(
            invocation.cyclic is not None,
            "query access currently requires actual cyclic query ownership",
        )
        if parent.limit is not None:
            require(
                parent.limit in uniforms
                and program.bound is not None
                and equivalent(uniforms[parent.limit], program.bound, facts)
                and program.count is None,
                "query access has a stale actual KV bound",
            )
        else:
            require(
                program.bound is None and parent.count.data == program.count,
                "query access has a stale actual fixed KV count",
            )
        cores = [op for op in kernel.body.block.ops if isinstance(op, CoreIdOp)]
        require(len(cores) == 1, "query access needs its actual core coordinate")
        coordinates[actor] = {
            cores[0].value: "$core",
            parent.body.block.args[0]: "$parent0",
            invocation.op.body.block.args[0]: "$parent1",
        }
        require(
            {op for op in parent.body.block.ops if isinstance(op, CallOp)}
            == {leaf.call for leaf in program.leaves.values() if leaf.actor == actor},
            "query access has stale actual parent calls",
        )
        invocations[actor] = invocation
    require(invocations, "query access requires an actual serial query")
    verify_query_domains(invocations, program.kernels)
    return uniforms, coordinates, next(iter(invocations.values())).cyclic.total


def _static_bias(offset, base, facts):
    difference = polynomial(expression("sub", offset, base), facts)
    if not difference:
        return 0
    if difference.keys() == {()} and difference[()].denominator == 1:
        return int(difference[()])
    return None


def _frame_biases(aliases, base, tile, facts):
    biases = {}
    for alias in aliases:
        bias = _static_bias(alias.capture.offset, base, facts)
        if bias is None or bias < 0 or bias + alias.capture.op.nbytes.data > tile:
            return None
        biases[alias.uid] = bias
    return biases


def _remainders(expr):
    # Inspect the original form: polynomial normalization expands remainders.
    if expr.kind == "rem" and "$parent1" in arguments(expr):
        yield expr
    for arg in expr.args:
        if isinstance(arg, Expr):
            yield from _remainders(arg)


def _epoch_ring(name, root, aliases, epoch, groups, facts):
    remainders = [tuple(_remainders(alias.capture.offset)) for alias in aliases]
    require(
        root.extent_count is None and all(remainders),
        f"{name}: unsupported query-dependent root; no common actual epoch ring or query frame",
    )
    periods = set()
    for expressions in remainders:
        for remainder in expressions:
            numerator, divisor = remainder.args
            require(
                divisor.constant is not None
                and divisor.constant > 0
                and equivalent(numerator, epoch, facts),
                f"{name}: actual ring alias has a different query/KV epoch phase",
            )
            periods.add(divisor.constant)
    require(len(periods) == 1, f"{name}: actual ring aliases have inconsistent slot periods")
    slots = bounded_period(next(iter(periods)), f"query ring {name}")
    owners = groups if root.external.data else 1
    capacity = root.nbytes.data
    require(
        capacity > 0 and capacity % (owners * slots) == 0,
        f"{name}: actual ring capacity does not partition into owner slots",
    )
    tile = capacity // (owners * slots)
    owner_base = (
        expression("mul", CORE, literal(slots * tile)) if root.external.data else literal(0)
    )
    actual_base = expression(
        "add",
        owner_base,
        expression("mul", literal(tile), expression("rem", epoch, literal(slots))),
    )
    biases = _frame_biases(aliases, actual_base, tile, facts)
    require(
        biases is not None,
        f"{name}: actual ring aliases change their slot coefficient, owner partition, or frame bounds",
    )
    # Keep the periodic atom intact for the existing parent-address analysis.
    normalized_base = expression(
        "add", owner_base, expression("mul", literal(tile), expression("rem", KV, literal(slots)))
    )
    return tile, slots, biases, normalized_base


@bounded
def normalize_query_access(program: ScopedProgram) -> QueryAccessMap:
    """Prove a bounded memory-coordinate map for one fixed actual query.

    Runtime query counts and KV bounds remain symbolic. Only initialized GM
    whose actual native modes are all R may collapse to a read-only frame.
    All written query-dependent aliases must prove a common cyclic query frame
    or the exact global epoch ring. Unknown relations fail explicitly.
    """
    require(isinstance(program, ScopedProgram), "query accesses need an extracted ScopedProgram")
    facts = facts_from_sizes(program.parameters)
    uniforms, coordinates, total = _actual_domains(program, facts)
    aliases, roots, modes = {}, {}, {}
    for uid, leaf in program.leaves.items():
        parent = program.parents[leaf.actor]
        require(
            uid == leaf.call.uid.data
            and leaf.call.parent_op() is parent
            and leaf.call.generation is parent.body.block.args[0]
            and leaf.stream.parent_op() is leaf.call
            and leaf.loop.constant_count == leaf.stream.count.data,
            "query access uses a different actual call, KV clock, or child domain",
        )
        require(
            tuple(op for op in leaf.call.body.block.ops if isinstance(op, CaptureOp))
            == tuple(capture.op for capture in leaf.captures)
            and len({capture.root for capture in leaf.captures}) == len(leaf.captures),
            "query access has stale or duplicate actual capture roots",
        )
        for root, effects in _native_modes(leaf).items():
            modes.setdefault(root, set()).update(effects)
        for captured in leaf.captures:
            cap, root = captured.op, captured.root
            actual = from_ssa(cap.offset, coordinates=coordinates[leaf.actor], facts=facts)
            alias_root = leaf.value_map.get(cap.ref)
            span = cap.nbytes.data if root.external.data else (cap.nbytes.data + 31) // 32 * 32
            require(
                cap.source is root.ref
                and root.parent_op()
                is (program.root if root.external.data else program.kernels[leaf.actor])
                and alias_root is not None
                and isinstance(alias_root.owner, BufferOp)
                and alias_root.owner.sym_name == cap.sym_name
                and alias_root.owner.nbytes.data == span
                and cap.nbytes.data > 0
                and equivalent(actual, captured.offset, facts),
                "query access has a stale actual capture root, span, or offset",
            )
            storage = _storage(root, parent.limit, uniforms)
            name = storage.name if storage.external else leaf.actor + "/" + storage.name
            require(
                name not in roots or roots[name] is root,
                "query access root name loses actual allocation identity",
            )
            roots[name] = root
            aliases.setdefault(name, []).append(_Alias(uid, leaf.actor, captured, storage))
        poll("query capture contracts", call=uid)
    kv_bound = program.bound if program.bound is not None else literal(program.count)
    query_epoch = expression("mul", QUERY, kv_bound)
    epoch = expression("add", query_epoch, KV)
    flat = expression("add", CORE, expression("mul", literal(program.groups), QUERY))
    storages, offsets, contracts = {}, {}, {}
    for name, members in aliases.items():
        root = roots[name]
        extent = (
            literal(root.nbytes.data)
            if root.extent_count is None
            else expression(
                "mul", from_ssa(root.extent_count, facts=facts), literal(root.count_scale.data)
            )
        )
        replacements = None
        if root.external.data and modes.get(root) == {"R"}:
            require(
                root.initialized.data == 1, f"{name}: readonly GM needs actual initialized storage"
            )
            kind, tile, slots, phase = (
                "readonly",
                max(a.capture.op.nbytes.data for a in members),
                1,
                None,
            )
            biases, base = {alias.uid: 0 for alias in members}, literal(0)
            replacements = tile
        elif all("$parent1" not in arguments(alias.capture.offset) for alias in members):
            kind, tile, slots, phase = "reused", extent.constant, 1, None
            biases, base = {}, None
        else:
            tile = constant_ratio(extent, total, facts) if root.external.data else None
            biases = (
                _frame_biases(members, expression("mul", flat, literal(tile)), tile, facts)
                if tile is not None and tile > 0
                else None
            )
            if biases is not None:
                kind, slots, phase = "query_frame", 1, None
                base = expression("mul", CORE, literal(tile))
                replacements = program.groups * tile
            else:
                tile, slots, biases, base = _epoch_ring(
                    name, root, members, epoch, program.groups, facts
                )
                kind = "epoch_ring"
                phase = expression("rem", query_epoch, literal(slots))
        for alias in members:
            offsets[alias.uid, name] = (
                alias.capture.offset
                if base is None
                else expression("add", base, literal(biases[alias.uid]))
            )
            if replacements is not None:
                storage = replace(alias.storage, nbytes=replacements, count_scale=0, extent=None)
                if storage != alias.storage:
                    storages[alias.actor, name] = storage
        contracts[name] = QueryRoot(
            kind, root, tile, slots, phase, program.groups, total, extent, biases
        )
        poll("query root contracts", root=name, kind=kind)
    return QueryAccessMap(storages, offsets, contracts)
