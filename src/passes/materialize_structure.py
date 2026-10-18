"""Insert selected notifications into their owned physical SSA control tree.

The returned module is the input module. Native packets, addresses and loop
controllers keep their identities; only scalar arithmetic, short-circuit SCF
guards and observable notifications are inserted. No optimizer sees async work.
"""

from collections.abc import Iterable

from xdsl.dialects import arith, scf
from xdsl.dialects.builtin import ArrayAttr, IndexType, ModuleOp, StringAttr
from xdsl.ir import Block, Region

from ..analysis.structured_memory import NativeMemoryFacts
from ..analysis.structured_schedule import Placement, SyncAction
from ..diagnostics import require
from ..dialects.physical import BufferOp, CoreIdOp, EndOp, KernelOp, MixedOp, RuntimeSizeOp, TaskOp
from ..dialects.structured import ActorDomainOp, NotificationOp, PipelineLoopOp, PipelineStageOp
from ..scalar import literal
from ..target.ascend910b import Actor

_BINARY = {
    "add": arith.AddiOp,
    "mul": arith.MuliOp,
    "min": arith.MinSIOp,
    "max": arith.MaxSIOp,
    "floordiv": arith.DivUIOp,
    "ceildiv": arith.CeilDivUIOp,
    "rem": arith.RemUIOp,
}
_COMPARE = {"==": "eq", "!=": "ne", "<": "slt", "<=": "sle", ">": "sgt", ">=": "sge"}


def _entry_target(root):
    """Invocation entry follows its existing uniform scalar declarations.

    RuntimeSize/CoreId values are real SSA definitions, not implicit function
    parameters in this dialect. An entry notification may use them only after
    their definitions. Existing pure prefix work and allocation declarations
    stay in place, before the first executing controller or native operation.
    """
    for operation in root.body.block.ops:
        if not isinstance(
            operation,
            (BufferOp, CoreIdOp, RuntimeSizeOp, *tuple(arith.Arith.operations)),
        ):
            return operation
    raise ValueError("physical invocation has no terminator")


def _bindings(memory, block, before, container):
    """Bind only original SSA values which dominate this insertion point."""
    control = memory.control
    memo = {}

    def bind(value):
        if isinstance(value.type, IndexType):
            memo[control.environment.expression(value)] = value

    def visit_scope(current, stop):
        owner = current.parent_op()
        if owner is not None and owner.parent is not None:
            visit_scope(owner.parent, owner)
        for argument in current.args:
            bind(argument)
        if isinstance(owner, scf.ForOp):
            logical = control.loops[owner]
            induction = current.args[0]
            if logical.lower.constant == 0 and logical.step == 1:
                memo[logical.ordinal] = induction
            else:
                delta = arith.SubiOp(induction, owner.lb)
                division = arith.DivUIOp(delta.result, owner.step)
                container.add_ops((delta, division))
                memo[logical.ordinal] = division.result
        elif isinstance(owner, PipelineStageOp):
            logical = control.loops[owner.parent_op()]
            memo[logical.ordinal] = current.args[0]
        for operation in current.ops:
            if operation is stop:
                break
            # Newly inserted operations are deliberately absent from the old
            # environment. They belong to another action's evaluation scope.
            if operation in control.frames:
                for value in operation.results:
                    bind(value)

    visit_scope(block, before)
    return memo


def materialize_notifications(memory: NativeMemoryFacts, actions: Iterable[SyncAction]) -> ModuleOp:
    """Materialize once, in place, without cloning or changing owned facts."""
    require(
        isinstance(memory, NativeMemoryFacts), "notification materialization needs owned memory"
    )
    require(memory.control.module is memory.module, "control belongs to another owned module")
    actions = tuple(actions)
    root = memory.module.body.block.first_op
    require(isinstance(root, (KernelOp, MixedOp)), "unsupported physical invocation")
    for action in actions:
        require(
            action.anchor in memory.control.frames, "sync action anchor belongs to another module"
        )
        participants = memory.control.frames[action.anchor].participants
        require(
            not participants or set(action.participants) <= set(participants),
            "sync action participants belong to another actual actor scope",
        )
        require(
            action.participants
            and (
                action.participants == (Actor.AIC,)
                or set(action.participants) <= {Actor.AIV0, Actor.AIV1}
            ),
            "sync action must belong to one actual actor family",
        )
        require(
            isinstance(action.anchor, TaskOp)
            and action.placement in {Placement.BEFORE, Placement.AFTER}
            or isinstance(action.anchor, (KernelOp, MixedOp, PipelineLoopOp, scf.ForOp, scf.IfOp))
            and action.placement in {Placement.ENTRY, Placement.EXIT},
            "sync action needs a native packet or invocation boundary placement",
        )

    # All actions at one real insertion point consume the same dominating
    # original SSA environment. Build it once, then append actions in order.
    # Contiguous common guard prefixes share their original short-circuit SCF.
    groups = {}
    for action in actions:
        family = "cube" if action.participants == (Actor.AIC,) else "vector"
        groups.setdefault((action.anchor, action.placement, family), []).append(action)
    after = {}
    entry_target = _entry_target(root)
    for (anchor, placement, family), group in groups.items():
        if anchor is root:
            block = root.body.block
            before = entry_target if placement is Placement.ENTRY else block.last_op
        elif placement in {Placement.BEFORE, Placement.ENTRY}:
            block, before = anchor.parent, anchor
        else:
            previous = after.get(anchor, anchor)
            block, before = previous.parent, previous.next_op
        assert block is not None
        container, domain = Block(), block.parent_op()
        while domain is not None and not isinstance(domain, ActorDomainOp):
            domain = domain.parent_op()
        current, owner, created_domain = container, None, False
        if isinstance(root, MixedOp) and domain is None:
            body = Block(arg_types=[IndexType()])
            domain = ActorDomainOp.build(
                attributes={"kind": StringAttr(family)}, regions=[Region(body)]
            )
            container.add_op(domain)
            current, created_domain = body, True
            if family == "vector":
                owner = body.args[0]
        memo = _bindings(memory, block, before, current)
        if owner is not None:
            memo[memory.control.sid] = owner

        def scalar(expr, current, memo):
            if expr in memo:
                return memo[expr]
            if expr.constant is not None:
                operation = arith.ConstantOp.from_int_and_width(expr.constant, IndexType())
            else:
                require(
                    expr.kind != "argument",
                    f"sync materialization has unbound symbol {expr.text()}",
                )
                operation = _BINARY[expr.kind](
                    *(scalar(child, current, memo) for child in expr.args)
                )
            current.add_op(operation)
            memo[expr] = operation.results[0]
            return operation.results[0]

        prefixes, scopes = [], [(current, memo)]
        for action in group:
            tests = action.guard.comparisons
            if (
                isinstance(root, MixedOp)
                and len(action.participants) == 1
                and action.participants[0] is not Actor.AIC
            ):
                from ..analysis.structured_schedule import Comparison

                tests = (
                    Comparison(
                        memory.control.sid, "==", literal(int(action.participants[0] is Actor.AIV1))
                    ),
                ) + tests
            common = 0
            while common < min(len(prefixes), len(tests)) and prefixes[common] == tests[common]:
                common += 1
            while len(prefixes) > common:
                scopes.pop()[0].add_op(scf.YieldOp())
                prefixes.pop()
            for test in tests[common:]:
                current, memo = scopes[-1]
                comparison = arith.CmpiOp(
                    scalar(test.lhs, current, memo),
                    scalar(test.rhs, current, memo),
                    _COMPARE[test.predicate],
                )
                nested = Block()
                current.add_ops(
                    (comparison, scf.IfOp(comparison.result, [], Region(nested), Region()))
                )
                prefixes.append(test)
                scopes.append((nested, dict(memo)))
            current, memo = scopes[-1]
            event = None if action.event is None else scalar(action.event, current, memo)
            current.add_op(
                NotificationOp.build(
                    operands=[event],
                    attributes={
                        "kind": StringAttr(action.kind.value),
                        # Participant restrictions have already become the
                        # surrounding SID control. The instruction executes in
                        # that actual family region, with no second backend
                        # participant filter. Shared actions need neither.
                        "participants": ArrayAttr(
                            StringAttr(actor.value)
                            for actor in (
                                (Actor.AIV0, Actor.AIV1)
                                if isinstance(root, MixedOp) and family == "vector"
                                else action.participants
                            )
                        ),
                        "source": StringAttr(action.source.value),
                        "destination": StringAttr(action.destination.value),
                        "protocol": StringAttr(", ".join(action.protocols)),
                    },
                )
            )
        while len(scopes) > 1:
            scopes.pop()[0].add_op(scf.YieldOp())
        if created_domain:
            scopes[0][0].add_op(EndOp())
        emitted = list(container.ops)
        for operation in emitted:
            container.detach_op(operation)
        if before is None:
            block.add_ops(emitted)
        else:
            block.insert_ops_before(emitted, before)
        if anchor is not root and placement in {Placement.AFTER, Placement.EXIT}:
            after[anchor] = emitted[-1]
    return memory.module
