"""Retain parent pipeline callsites and child clocks in physical IR.

One child template is lowered once. Captures are address aliases of original
allocations, never private copies or completion tokens. Synchronization must
subsequently compose the actual child effects at the declared parent schedule.
"""

from itertools import groupby

from xdsl.dialects import arith, scf
from xdsl.dialects.builtin import ArrayAttr, IndexType, IntAttr, StringAttr
from xdsl.ir import Block, Region

from .indexing import equivalent, validate_source_arithmetic
from .ir import CallOp, CaptureOp, CoreIdOp, EndOp, StreamOp, ViewOp
from .lowering_context import ActorBindings
from .model import require
from .scalar import ARITHMETIC, from_ssa
from .scope_predicates import classify_guard
from .source_ir import AllocationOp, MatmulOp, PipelineOp
from .source_scopes import ClockFrame, Scope, Work, collect_source_scopes


def has_nested_pipelines(program):
    return any(
        isinstance(op, PipelineOp)
        and any(isinstance(parent, PipelineOp) for parent in _parents(op))
        for op in program.walk()
    )


def _parents(op):
    while (op := op.parent_op()) is not None:
        yield op


def _expr_value(expr, block, names, cache):
    if expr in cache:
        return cache[expr]
    if expr.kind == "argument":
        require(expr.args[0] in names, "capture references a clock outside its callsite")
        return names[expr.args[0]]
    if expr.constant is not None:
        op = arith.ConstantOp.from_int_and_width(expr.constant, IndexType())
    else:
        classes = {name: cls for cls, name in ARITHMETIC.items()}
        require(expr.kind in classes, "capture has unsupported index arithmetic")
        op = classes[expr.kind](*(_expr_value(arg, block, names, cache) for arg in expr.args))
    block.add_op(op)
    cache[expr] = op.results[0]
    return op.results[0]


def _segments(scope, family):
    """Ordinary runs split at real children, ancestor Stage and guard changes."""
    for is_child, items in groupby(
        scope.items_for(family), key=lambda item: isinstance(item, Scope)
    ):
        if is_child:
            for item in items:
                yield item, tuple(item.items_for(family)), False
            continue
        for _, run in groupby(
            items, key=lambda item: (item.context.clocks, item.context.guards, item.context.domain)
        ):
            yield scope, tuple(run), True


def _capture_extent(capture, stream):
    # Physical view coefficients are nonnegative. Its maximum affine/modulo
    # displacement is a safe span; leaf extraction independently checks exact
    # per-row footprints and native recipe extents in that span.
    count = 1 if stream.count is None else stream.count.data
    extent = 0
    for op in stream.walk():
        if not isinstance(op, ViewOp) or op.source is not capture.ref:
            continue
        require(op.extent_count is None, "captured child cannot retain an outer count translation")
        terms = [(op.stride.data, 1 if op.divisor is None else op.divisor.data, op.modulus.data)]
        terms.extend(tuple(v.data for v in term) for term in op.index_terms or ())
        require(
            op.offset.data >= 0 and all(s >= 0 and d > 0 and m >= 0 for s, d, m in terms),
            "captured child address must have nonnegative static coefficients",
        )
        maximum = op.offset.data + sum(
            stride
            * (
                min(max(0, count - 1) // divisor, modulus - 1)
                if modulus
                else max(0, count - 1) // divisor
            )
            for stride, divisor, modulus in terms
        )
        rows = 1 if op.rows is None else op.rows.data
        pitch = 0 if op.row_stride is None else op.row_stride.data
        extent = max(extent, maximum + (rows - 1) * pitch + op.nbytes.data)
    require(extent > 0, "capture has no physical child view")
    return extent


def lower_scopes(program):
    from .lowering import _Actor, _check_loop_values, _work
    from .scope_model import compose_scope_actors, extract_scopes

    validate_source_arithmetic(program)
    tree = collect_source_scopes(program)
    require(
        len(tree.items) == 1
        and isinstance(tree.items[0], Scope)
        and isinstance(tree.items[0].op, PipelineOp),
        "nested AST pipelines require hierarchical synchronization: one parent Pipelined is required",
    )
    parent = tree.items[0]
    parent_guards = {
        op: kind
        for op in program.walk()
        if isinstance(op, scf.IfOp)
        if (kind := classify_guard(op, parent.op)) is not None
    }
    allocations = tuple(op for op in program.walk() if isinstance(op, AllocationOp))
    for op in program.walk():
        if isinstance(op, MatmulOp):
            require(op.a.type.memory.data == "L0A", "nested AST matmul needs explicit L0 operands")
        if isinstance(op, PipelineOp) and op is not parent.op:
            require(
                op.limit is None
                and not any(
                    isinstance(child, PipelineOp) for child in op.walk() if child is not op
                ),
                "nested AST pipelines currently require direct fixed children",
            )
    domains = (
        (("aic", "cube", 0), ("aiv0", "vector", 0), ("aiv1", "vector", 1))
        if program.kind.data == "mixed"
        else (("aic" if program.kind.data == "cube" else "aiv0", program.kind.data, 0),)
    )
    modules, lowered = [], {}
    for actor, family, lane in domains:
        bindings = ActorBindings(program, family, allocations)
        kernel = bindings.kernel
        core = CoreIdOp.create(result_types=[IndexType()])
        kernel.block.add_op(core)
        limit = bindings.runtime_bound(parent.op) if parent.op.limit is not None else None
        outer = StreamOp.create(
            operands=[] if limit is None else [limit],
            attributes={"uid": StringAttr("parent"), "symbolic": IntAttr(1)}
            | ({"count": IntAttr(parent.op.count.data)} if limit is None else {}),
            regions=[Region(Block(arg_types=[IndexType()]))],
        )
        names = {"$parent0": outer.body.block.args[0], "$core": core.value}
        for source, target in bindings.uniform.items():
            expr = from_ssa(source, facts=bindings.facts)
            if expr.kind == "argument":
                names[expr.args[0]] = target
        policies = {} if parent.op.issue_order is None else parent.op.issue_order.data
        labels = [label.data for label in policies.get(family, ())]
        ranks = {label: rank for rank, label in enumerate(labels)}
        calls, leaves, used_labels = [], [], set()
        for index, (scope, items, once) in enumerate(_segments(parent, family)):
            require(
                items and all(isinstance(item, Work) for item in items),
                "child must contain native work",
            )
            context = items[0].context if once else scope.context
            frame = context.clocks[-1]
            require(frame.op is parent.op, "child callsite changed its parent clock")
            stage = frame.stage
            label = "pipeline" if stage is None else stage.label.data
            used_labels.add(label)
            uid = f"{actor}_call{index}"
            call = CallOp.create(
                operands=[outer.body.block.args[0]],
                attributes={
                    "uid": StringAttr(uid),
                    "stage": IntAttr(frame.offset),
                    "logical_phase": IntAttr(parent.items.index(items[0] if once else scope)),
                    "issue_order": IntAttr(ranks.get(label, 0) * 65536 + index),
                },
                regions=[Region(Block())],
            )
            captures, cache = {}, {}

            def capture(
                allocation,
                offset,
                *,
                call=call,
                captures=captures,
                cache=cache,
                bindings=bindings,
                names=names,
            ):
                if allocation in captures:
                    previous, op = captures[allocation]
                    require(
                        equivalent(previous, offset, bindings.facts),
                        "one child uses incompatible ancestor bases of the same allocation",
                    )
                    return op.ref
                source = bindings.roots[allocation]
                value = _expr_value(offset, call.body.block, names, cache)
                name = f"cap_{call.uid.data}_{allocation.sym_name.data}"
                op = CaptureOp.create(
                    operands=[source, value],
                    result_types=[source.type],
                    attributes={"sym_name": StringAttr(name), "nbytes": IntAttr(0)},
                )
                call.body.block.add_op(op)
                captures[allocation] = offset, op
                return op.ref

            pipeline = parent.op if once else scope.op
            records = _work(
                program,
                pipeline,
                family,
                lane,
                operations=tuple(item.op for item in items) if once else None,
                scoped=items,
                scope=scope,
                nested=True,
                parent_guards=parent_guards,
            )
            leaf = _Actor(
                program,
                pipeline,
                actor,
                family,
                lane,
                allocations,
                symbolic=True,
                records=records,
                once=once,
                region_name=uid,
                bindings=bindings,
                ambient_frames=(ClockFrame(parent.op),),
                capture=capture,
                parent_generation=call.generation,
            )
            leaf.emit()
            lowered[uid] = family, leaf
            for _, op in captures.values():
                op.attributes["nbytes"] = IntAttr(_capture_extent(op, leaf.region))
            calls.append(call)
            leaves.append(leaf)
        require(leaves, f"{actor}: nested schedule has no participating work")
        require(
            not ranks or set(ranks) == used_labels,
            f"{family}: issue_order must name all active parent stages",
        )
        kernel.finish_region()
        for call, leaf in zip(calls, leaves):
            kernel.block.detach_op(leaf.region)
            call.body.block.add_ops([leaf.region, EndOp()])
            outer.body.block.add_op(call)
        outer.body.block.add_op(EndOp())
        kernel.block.add_ops([outer, EndOp()])
        kernel.op.attributes["source_schedule"] = ArrayAttr(
            item for leaf in leaves for item in leaf.schedule
        )
        modules.append(kernel.module)
    physical = modules[0] if len(modules) == 1 else compose_scope_actors(program, modules)
    extracted = extract_scopes(physical)
    for uid, physical_leaf in extracted.leaves.items():
        family, source_leaf = lowered[uid]
        roots = {
            physical_leaf.loop.root(actor, ref)
            for actor, data in physical_leaf.loop.actors.items()
            for task in data.program.tasks
            for ref in task.refs
        }
        _check_loop_values(
            physical_leaf.module,
            {family: source_leaf},
            source_leaf.pipeline,
            extracted=physical_leaf.loop,
            live_in=roots,
        )
    return physical
