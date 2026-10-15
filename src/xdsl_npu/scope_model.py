"""Extract actual nested calls and address aliases before synchronization.

A leaf analysis rebinds its captured aliases to bounded descriptors once. It
does not instantiate iterations, allocate device storage, or certify parent
ownership/completion. Every final check rebuilds this view from actual IR.
"""

import re
from dataclasses import dataclass

from xdsl.dialects import arith
from xdsl.dialects.builtin import IndexType, IntAttr, ModuleOp, StringAttr
from xdsl.ir import Block, Region

from .indexing import check_axis, interval, nonnegative
from .ir import (
    BufferOp,
    CallOp,
    CaptureOp,
    CoreIdOp,
    EndOp,
    FixConfigOp,
    KernelOp,
    Load3DConfigOp,
    LoopChannelOp,
    LoopSyncOp,
    MixedOp,
    OnceOp,
    RuntimeSizeOp,
    ScopeChannelOp,
    ScopeSyncOp,
    StreamOp,
    TaskOp,
    ViewOp,
)
from .loop_boundary import stream_operations
from .loop_model import _storage, extract_loop
from .model import CAPACITY, require, storage_layout
from .scalar import (
    ARITHMETIC,
    INT64_MAX,
    Expr,
    Fact,
    expression,
    facts_from_sizes,
    from_ssa,
    infer,
    literal,
    uniform,
)


@dataclass(frozen=True)
class Captured:
    op: CaptureOp
    root: BufferOp
    offset: Expr


@dataclass(frozen=True)
class ScopeLeaf:
    actor: str
    call: CallOp
    stream: StreamOp | OnceOp
    captures: tuple[Captured, ...]
    module: ModuleOp
    loop: object
    # Mapping only for moving selected leaf protocols back to their actual IR.
    value_map: dict


@dataclass(frozen=True)
class ScopedProgram:
    module: ModuleOp
    root: KernelOp | MixedOp
    kernels: dict[str, KernelOp]
    parents: dict[str, StreamOp]
    leaves: dict[str, ScopeLeaf]
    parameters: tuple[RuntimeSizeOp, ...]
    bound: Expr | None
    count: int | None
    groups: int


def is_scoped(module):
    return any(isinstance(op, CallOp) for op in module.walk())


def _identifier(name, description):
    require(re.fullmatch(r"[A-Za-z_]\w*", name), f"invalid nested {description} identifier")


def _terminated(block, description):
    require(isinstance(block.last_op, EndOp), f"nested {description} requires an actual EndOp")


def _clock(stream, description):
    _terminated(stream.body.block, description)
    require(
        len(stream.body.block.args) == 1 and isinstance(stream.body.block.args[0].type, IndexType),
        f"{description} needs one Index logical clock",
    )
    if isinstance(stream, StreamOp):
        require(
            stream.symbolic is None or stream.symbolic.data == 1,
            f"{description} has an invalid symbolic marker",
        )
        require(
            (stream.count is None) != (stream.limit is None),
            f"{description} requires exactly one fixed count or uniform limit",
        )
        require(
            stream.count is None or 0 <= stream.count.data <= INT64_MAX,
            f"{description} fixed count must fit nonnegative int64",
        )
    if stream.uid is not None:
        _identifier(stream.uid.data, "stream")


def _index_operation(op, coordinates, domains, facts):
    # Inspect each actual division, even if a later add/subtract cancels it out
    # of the captured address. Algebra is not a proof of its unsigned dividend.
    if isinstance(op, (arith.DivUIOp, arith.RemUIOp, arith.CeilDivUIOp)):
        dividend = from_ssa(op.operands[0], coordinates=coordinates, facts=facts)
        lower, _ = interval(dividend, domains, facts)
        require(
            nonnegative(lower, facts),
            "scope unsigned division has an unproven nonnegative dividend",
        )
    return from_ssa(op.results[0], coordinates=coordinates, facts=facts)


def _uniforms(operations, facts):
    values = {}
    for op in operations:
        if isinstance(op, RuntimeSizeOp):
            values[op.value] = from_ssa(op.value, facts=facts)
        elif isinstance(op, arith.ConstantOp) or type(op) in ARITHMETIC:
            require(
                all(value in values for value in op.operands), "scope uniform has nonuniform inputs"
            )
            expr = from_ssa(op.results[0], facts=facts)
            require(uniform(expr, facts), "scope bound must be a nonnegative uniform expression")
            values[op.results[0]] = expr
        elif isinstance(op, BufferOp):
            require(
                op.extent_count is None or op.extent_count in values,
                "scope allocation extent must be an earlier declared uniform SSA value",
            )
    return values


def _child_clock(call, stream, channels):
    _clock(stream, "nested child")
    require(
        stream.limit is None and stream.count is not None,
        "nested child needs its actual fixed iteration domain",
    )
    coordinate = stream.body.block.args[0]
    maximum = 0
    for op, _ in stream_operations(stream):
        require(
            isinstance(op, (ViewOp, TaskOp, LoopChannelOp, LoopSyncOp, ScopeSyncOp, EndOp)),
            f"unsupported nested child effect {op.name}",
        )
        if isinstance(op, (ViewOp, LoopSyncOp)):
            require(op.generation is coordinate, "nested child uses a different child clock")
        elif isinstance(op, TaskOp):
            require(0 <= op.stage.data <= 32, "invalid nested child native stage")
            maximum = max(maximum, op.stage.data)
        elif isinstance(op, ScopeSyncOp):
            require(
                op.parent_generation is call.generation and op.child_generation is coordinate,
                "scope synchronization uses a different parent or child clock",
            )
            require(
                op.channel.data in channels and op.action.data in {"set", "wait", "barrier"},
                "scope synchronization has an unknown channel or action",
            )
            require(
                0 <= op.position.data < stream.count.data,
                "scope synchronization has no active inner position",
            )
    require(
        stream.count.data <= INT64_MAX - maximum,
        "fixed child count plus native stage overflows int64",
    )


def _leaf(kernel, call, stream, captures):
    """Normalize one actual leaf with one alias per underlying allocation."""
    block, values = Block(), {}
    for captured in captures:
        cap, root = captured.op, captured.root
        external = bool(root.external.data)
        span = cap.nbytes.data if external else (cap.nbytes.data + 31) // 32 * 32
        alias = BufferOp.create(
            result_types=[cap.ref.type],
            attributes={
                "sym_name": cap.sym_name,
                "nbytes": IntAttr(span),
                "external": root.external,
                "initialized": root.initialized,
                "count_scale": IntAttr(0),
            },
        )
        block.add_op(alias)
        values[cap.ref] = alias.ref
    for op in kernel.body.block.ops:
        if isinstance(op, (FixConfigOp, Load3DConfigOp)):
            block.add_op(op.clone())
    child = stream.clone(value_mapper=values)
    for op in list(child.walk()):
        if isinstance(op, ScopeSyncOp):
            op.parent.erase_op(op)  # Parent graph checks these actual insertions separately.
    block.add_ops([child, EndOp()])
    attrs = dict(kernel.attributes)
    attrs.update(sym_name=call.uid, blocks=IntAttr(1), disjoint_external=IntAttr(1))
    attrs.pop("source_schedule", None)
    normalized = ModuleOp([KernelOp.create(attributes=attrs, regions=[Region(block)])])
    loop = extract_loop(normalized, region=child)
    # Padding in the temporary local descriptor does not authorize an actual
    # native access beyond the captured span.
    spans = {captured.op.sym_name.data: captured.op.nbytes.data for captured in captures}
    for data in loop.actors.values():
        for task in data.program.tasks:
            for ref in task.refs:
                require(
                    not ref.core_terms and not ref.count_offset and not ref.count_core_stride,
                    "native child retains ancestor core/count translation; use an actual capture",
                )
                period = loop.storage_periods[loop.root(next(iter(loop.actors)), ref)]
                count = loop.constant_count
                for residue in range(min(period, count)):
                    last = residue + (count - 1 - residue) // period * period
                    require(
                        all(hi <= spans[ref.storage.name] for _, hi in ref.segments(last)),
                        "native child footprint exceeds captured span",
                    )
    return normalized, loop, values


def extract_scopes(module):
    module.verify()
    require(len(list(module.body.block.ops)) == 1, "expected one nested physical kernel")
    root = module.body.block.first_op
    require(
        isinstance(root, (KernelOp, MixedOp)) and root.target.data == "dav-2201",
        "nested physical kernel must target dav-2201",
    )
    _identifier(root.sym_name.data, "kernel")
    _terminated(root.body.block, "kernel")
    require(not root.body.block.args, "nested kernel cannot have region arguments")
    groups = 1 if root.blocks is None else root.blocks.data
    require(
        1 <= groups <= 1024 and root.disjoint_external.data in (0, 1),
        "invalid nested launch contract",
    )
    mixed = isinstance(root, MixedOp)
    declarations = tuple(root.body.block.ops)
    if mixed:
        members = [op for op in declarations if isinstance(op, KernelOp)]
        require(
            [op.sym_name.data for op in members] == ["aic", "aiv0", "aiv1"],
            "nested CV needs actual unique AIC/two-AIV participants",
        )
        kernels = {op.sym_name.data: op for op in members}
        entered = False
        for op in declarations:
            if isinstance(op, KernelOp):
                entered = True
            elif not isinstance(op, EndOp):
                require(not entered, "mixed declarations must precede every actual actor")
                require(
                    isinstance(op, (RuntimeSizeOp, BufferOp, ScopeChannelOp, arith.ConstantOp))
                    or type(op) in ARITHMETIC,
                    f"unsupported nested mixed root effect {op.name}",
                )
                if isinstance(op, BufferOp):
                    require(
                        op.external.data == 1 and op.ref.type.bank.data == "GM",
                        "mixed root buffers must be actual shared GM allocations",
                    )
    else:
        kernels = {"aic" if root.core.data == "aic" else "aiv0": root}
    parameters = tuple(op for op in declarations if isinstance(op, RuntimeSizeOp))
    require(
        len({op.sym_name.data for op in parameters}) == len(parameters),
        "duplicate nested runtime scalar name",
    )
    facts = facts_from_sizes(parameters)
    uniforms = _uniforms(declarations, facts)
    shared = {op.ref: op for op in declarations if isinstance(op, BufferOp) and op.external.data}
    require(
        len({op.sym_name.data for op in shared.values()}) == len(shared),
        "duplicate nested shared GM root name",
    )
    require(
        len(shared) <= 1 or root.disjoint_external.data == 1,
        "nested GM roots need a disjoint alias contract",
    )
    channels = {op.sym_name.data for op in declarations if isinstance(op, ScopeChannelOp)}
    parents, leaves, bounds = {}, {}, set()
    for actor, kernel in kernels.items():
        require(
            kernel.core.data == ("aic" if actor == "aic" else "aiv"),
            "nested actor owns wrong engine family",
        )
        require(
            kernel.target.data == "dav-2201"
            and kernel.disjoint_external.data in (0, 1)
            and kernel.epoch_divisor is None,
            "nested actor needs a valid target/alias contract without invocation epochs",
        )
        _terminated(kernel.body.block, "actor")
        require(not kernel.body.block.args, "nested actor cannot have region arguments")
        require(
            (1 if kernel.blocks is None else kernel.blocks.data) == groups,
            "nested actors disagree on groups",
        )
        parent_loops = [op for op in kernel.body.block.ops if isinstance(op, StreamOp)]
        require(len(parent_loops) == 1, "nested actor requires one actual parent stream")
        parent = parents[actor] = parent_loops[0]
        _clock(parent, "parent pipeline")
        if parent.limit is not None:
            require(
                parent.limit in uniforms and parent.count is None,
                "parent count must be an actual uniform SSA value",
            )
            bound, count = uniforms[parent.limit], None
        else:
            bound, count = None, parent.count.data
        bounds.add((bound, count))
        roots = dict(shared)
        local_roots = [op for op in kernel.body.block.ops if isinstance(op, BufferOp)]
        require(
            not mixed
            or all(op.external.data == 0 and op.ref.type.bank.data != "GM" for op in local_roots),
            "mixed actors cannot add private GM declarations; use actual shared GM roots",
        )
        roots.update({op.ref: op for op in local_roots})
        storages = [_storage(op, parent.limit, uniforms) for op in roots.values()]
        require(
            len({storage.name for storage in storages}) == len(storages),
            "duplicate nested physical allocation name",
        )
        _, peaks = storage_layout(storages)
        require(
            all(
                size <= CAPACITY[bank] and ((actor != "aic") == (bank == "UB"))
                for bank, size in peaks.items()
            ),
            "nested allocations exceed actor capacity/ownership",
        )
        cores = [op for op in kernel.body.block.ops if isinstance(op, CoreIdOp)]
        require(len(cores) == 1, "nested actor needs its actual group coordinate")
        coordinates = {cores[0].value: "$core", parent.body.block.args[0]: "$parent0"}
        domains = [
            (Expr("argument", ("$core",)), literal(0), literal(groups - 1)),
            (
                Expr("argument", ("$parent0",)),
                literal(0),
                expression("sub", bound if bound is not None else literal(count), literal(1)),
            ),
        ]
        entered = False
        for op in kernel.body.block.ops:
            if op is parent:
                entered = True
                continue
            require(
                not entered or isinstance(op, EndOp),
                "nested declarations/configuration must precede the parent stream",
            )
            require(
                isinstance(
                    op,
                    (
                        BufferOp,
                        CoreIdOp,
                        RuntimeSizeOp,
                        StreamOp,
                        FixConfigOp,
                        Load3DConfigOp,
                        ScopeChannelOp,
                        EndOp,
                        arith.ConstantOp,
                    ),
                )
                or type(op) in ARITHMETIC,
                f"unsupported nested kernel effect {op.name}",
            )
            if mixed:
                require(
                    not isinstance(op, (RuntimeSizeOp, ScopeChannelOp)),
                    "nested actor parameters/channels must be actual shared root declarations",
                )
                if isinstance(op, arith.ConstantOp) or type(op) in ARITHMETIC:
                    require(
                        all(value in uniforms for value in op.results),
                        "nested actor arithmetic must use actual shared uniform declarations",
                    )
        calls = [op for op in parent.body.block.ops if isinstance(op, CallOp)]
        require(calls, "nested actor needs an actual child call")
        require(
            count is None or count <= INT64_MAX - max(call.stage.data for call in calls),
            "fixed parent count plus call stage overflows int64",
        )
        for call in parent.body.block.ops:
            if isinstance(call, EndOp):
                continue
            require(isinstance(call, CallOp), "parent stream must retain real child calls")
            require(
                call.generation is parent.body.block.args[0], "call uses a different parent clock"
            )
            require(
                re.fullmatch(r"[A-Za-z_]\w*", call.uid.data) and call.uid.data not in leaves,
                "invalid/duplicate nested call identity",
            )
            require(
                0 <= call.stage.data <= 32
                and 0 <= call.logical_phase.data <= 255
                and 0 <= call.issue_order.data < (1 << 31),
                "invalid nested call schedule",
            )
            _terminated(call.body.block, "call")
            require(not call.body.block.args, "nested call cannot have region arguments")
            children = [op for op in call.body.block.ops if isinstance(op, (StreamOp, OnceOp))]
            require(len(children) == 1, "call needs exactly one retained child region")
            stream = children[0]
            _child_clock(call, stream, channels)
            captures, seen, available = [], set(), set(uniforms) | set(coordinates)
            entered = False
            for op in call.body.block.ops:
                if op is stream:
                    entered = True
                elif isinstance(op, CaptureOp):
                    require(
                        not entered and op.source in roots and op.source not in seen,
                        "child capture must retain one distinct actual allocation root",
                    )
                    require(
                        op.offset in available
                        and op.ref.type == op.source.type
                        and 0 < op.nbytes.data <= INT64_MAX,
                        "invalid captured address/type/span",
                    )
                    offset = from_ssa(op.offset, coordinates=coordinates, facts=facts)
                    source = roots[op.source]
                    require(
                        source.extent_count is not None or op.nbytes.data <= source.nbytes.data,
                        "captured span exceeds its actual allocation root",
                    )
                    extent = (
                        literal(source.nbytes.data)
                        if source.extent_count is None
                        else expression(
                            "mul", uniforms[source.extent_count], literal(source.count_scale.data)
                        )
                    )
                    if count != 0:
                        check_axis(offset, op.nbytes.data, extent, domains, facts)
                    captures.append(Captured(op, source, offset))
                    seen.add(op.source)
                elif isinstance(op, arith.ConstantOp) or type(op) in ARITHMETIC:
                    require(
                        not entered and all(value in available for value in op.operands),
                        "capture arithmetic has a non-dominating/child input",
                    )
                    _index_operation(op, coordinates, domains, facts)
                    available.update(op.results)
                else:
                    require(isinstance(op, EndOp), f"unknown callsite effect {op.name}")
            child_module, loop, values = _leaf(kernel, call, stream, captures)
            # Normalizing a captured base to zero does not prove the actual
            # native origin alignment. Check the frozen ancestor offset too;
            # boundary-cell cuts deliberately do not carry this native rule.
            alignments = {}
            for data in loop.actors.values():
                for task in data.program.tasks:
                    for ref in task.refs:
                        alignment = (
                            512
                            if task.pipe in {"M", "MTE1", "FIX"} and not ref.storage.external
                            else 4
                            if task.pipe == "S"
                            else 32
                        )
                        alignments[ref.storage.name] = max(
                            alignments.get(ref.storage.name, 1), alignment
                        )
            address_facts = facts | {"$parent0": Fact(), "$core": Fact(0, 1, groups - 1)}
            for captured in captures:
                alignment = alignments.get(captured.op.sym_name.data, 1)
                offset_fact = infer(captured.offset, address_facts)
                require(
                    captured.offset.constant == 0
                    or offset_fact is not None
                    and offset_fact.multiple % alignment == 0,
                    f"{captured.op.sym_name.data}: captured base violates native {alignment}-byte alignment",
                )
            leaves[call.uid.data] = ScopeLeaf(
                actor, call, stream, tuple(captures), child_module, loop, values
            )
    require(
        len(bounds) == 1 and leaves, "nested actors need one shared parent domain and real work"
    )
    bound, count = next(iter(bounds))
    return ScopedProgram(module, root, kernels, parents, leaves, parameters, bound, count, groups)


def compose_scope_actors(program, modules):
    """Share actual GM/uniform SSA while retaining each actor's nested regions."""
    block, sizes, expressions, roots, kernels = Block(), {}, {}, {}, []
    for actor, module in zip(("aic", "aiv0", "aiv1"), modules):
        kernel = module.body.block.first_op
        for op in list(kernel.body.block.ops):
            if isinstance(op, RuntimeSizeOp):
                key = op.sym_name.data
                if key not in sizes:
                    shared = op.clone()
                    sizes[key] = shared
                    block.add_op(shared)
                shared = sizes[key]
                require(
                    (op.minimum, op.multiple) == (shared.minimum, shared.multiple),
                    "actor scalar contracts disagree",
                )
                op.value.replace_all_uses_with(shared.value)
                kernel.body.block.erase_op(op)
            elif isinstance(op, arith.ConstantOp) or type(op) in ARITHMETIC:
                expr = from_ssa(op.results[0], facts=facts_from_sizes(sizes.values()))
                if expr not in expressions:
                    shared = op.clone()
                    expressions[expr] = shared.results[0]
                    block.add_op(shared)
                op.results[0].replace_all_uses_with(expressions[expr])
                kernel.body.block.erase_op(op)
            elif isinstance(op, BufferOp) and op.external.data:
                key = op.sym_name.data
                if key not in roots:
                    shared = op.clone()
                    roots[key] = shared
                    block.add_op(shared)
                shared = roots[key]
                require(
                    (op.ref.type, op.nbytes, op.count_scale, op.initialized, op.extent_count)
                    == (
                        shared.ref.type,
                        shared.nbytes,
                        shared.count_scale,
                        shared.initialized,
                        shared.extent_count,
                    ),
                    "nested actors disagree on shared allocation",
                )
                op.ref.replace_all_uses_with(shared.ref)
                kernel.body.block.erase_op(op)
        kernel.attributes["sym_name"] = StringAttr(actor)
        module.body.block.detach_op(kernel)
        kernels.append(kernel)
    block.add_ops([*kernels, EndOp()])
    return ModuleOp(
        [
            MixedOp.create(
                attributes={
                    "sym_name": program.sym_name,
                    "target": StringAttr("dav-2201"),
                    "blocks": program.groups,
                    "disjoint_external": program.disjoint_external,
                },
                regions=[Region(block)],
            )
        ]
    )
