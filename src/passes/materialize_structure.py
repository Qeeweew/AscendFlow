"""Insert selected notifications into their owned physical SSA control tree.

The returned module is the input module. Native packets, addresses and loop
controllers keep their identities; only scalar arithmetic, short-circuit SCF
guards and observable notifications are inserted. No optimizer sees async work.
"""

from collections.abc import Iterable

from xdsl.dialects import arith, scf
from xdsl.dialects.builtin import ArrayAttr, IndexType, ModuleOp, StringAttr, i1
from xdsl.ir import Block, Region

from ..analysis.integer_facts import IntegerFacts
from ..analysis.structured_memory import NativeMemoryFacts
from ..analysis.structured_schedule import Comparison, Placement, SyncAction
from ..diagnostics import require
from ..dialects.physical import BufferOp, CoreIdOp, EndOp, KernelOp, MixedOp, RuntimeSizeOp, TaskOp
from ..dialects.structured import ActorDomainOp, NotificationOp, PipelineLoopOp
from ..scalar import DIVISION, INT64_MAX, expression, literal
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
    memo, ordinals = {}, {}

    def bind(value):
        if isinstance(value.type, IndexType) or value.type == i1:
            expression = control.environment.expression(value)
            # A known mathematical value is not necessarily literal SSA.
            # Keep planned literals constant so unrelated source CSE cannot
            # change immediate event lowering into dynamic-ID selection.
            if expression.constant is None or isinstance(value.owner, arith.ConstantOp):
                memo[expression] = value

    for value in control.dominating_scalars(block, before):
        bind(value)
        owner = value.owner
        if isinstance(owner, Block) and isinstance(owner.parent_op(), scf.ForOp):
            loop = owner.parent_op()
            if value is owner.args[0]:
                logical = control.loops[loop]
                if logical.lower.constant != 0 or logical.step != 1:
                    ordinals[logical.ordinal] = expression(
                        "floordiv",
                        expression("sub", logical.induction, logical.lower),
                        literal(logical.step),
                    )
    return memo, ordinals


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
        memo, ordinals = _bindings(memory, block, before, current)
        if owner is not None:
            memo[memory.control.sid] = owner
        frame = memory.control.frames[anchor]
        if frame.operation is anchor and (
            frame.loop is not None
            or isinstance(anchor, TaskOp)
            and anchor.native.data == "index_load"
            and placement is Placement.BEFORE
        ):
            frame = frame.parent
        premises = (
            tuple(
                Comparison(value, predicate, bound)
                for value, lower, upper in frame.domains()
                for predicate, bound in ((">=", lower), ("<=", upper))
            )
            + frame.predicates()
        )
        arithmetic = IntegerFacts(memory.control.environment.facts)

        def residue(value, modulus, memo, assumptions, arithmetic=arithmetic):
            """Bound a newly generated nonnegative ring calculation before %.

            For small constant moduli, reducing add/mul operands keeps every
            intermediate representable even when an unbounded rank would not.
            Existing SSA evaluations are always reused as written.
            """
            if value.constant is not None:
                return literal(value.constant % modulus) if value.constant >= 0 else None
            divisor = literal(modulus)
            if value.kind == "rem" and value.args[1] == divisor:
                return value
            if value not in memo and value.kind in {"add", "mul"}:
                if not all(
                    arithmetic.prove(Comparison(v, ">=", literal(0)), assumptions)
                    for v in value.args
                ):
                    return None
                children = tuple(residue(v, modulus, memo, assumptions) for v in value.args)
                if all(v is not None for v in children):
                    return expression("rem", expression(value.kind, *children), divisor)
            return expression("rem", value, divisor)

        def scalar(
            expr,
            current,
            memo,
            prefix,
            *,
            arithmetic=arithmetic,
            ordinals=ordinals,
            premises=premises,
        ):
            if expr in memo:
                return memo[expr]
            if expr in ordinals:
                memo[expr] = scalar(ordinals[expr], current, memo, prefix)
                return memo[expr]
            if expr.constant is not None:
                require(
                    -INT64_MAX - 1 <= expr.constant <= INT64_MAX, "generated index exceeds int64"
                )
                operation = arith.ConstantOp.from_int_and_width(expr.constant, IndexType())
            else:
                require(
                    expr.kind != "argument",
                    f"sync materialization has unbound symbol {expr.text()}",
                )
                assumptions = (
                    *premises,
                    *prefix,
                    # Only bindings in this actual scope supply machine
                    # ranges. Sibling guarded definitions cannot escape.
                    *(
                        Comparison(value, predicate, literal(bound))
                        for value, ssa in memo.items()
                        for predicate, bound in (
                            (">=", 0 if ssa.type == i1 else -INT64_MAX - 1),
                            ("<=", 1 if ssa.type == i1 else INT64_MAX),
                        )
                    ),
                )
                if expr.kind == "rem" and (modulus := expr.args[1].constant) is not None:
                    numerator = expr.args[0]
                    if (
                        0 < modulus <= 1 << 31
                        and numerator not in memo
                        and not arithmetic.prove(
                            Comparison(numerator, "<=", literal(INT64_MAX)), assumptions
                        )
                    ):
                        reduced = residue(numerator, modulus, memo, assumptions)
                        if reduced is not None and reduced != expr:
                            memo[expr] = scalar(reduced, current, memo, prefix)
                            return memo[expr]
                operands = tuple(scalar(child, current, memo, prefix) for child in expr.args)
                assumptions += tuple(
                    Comparison(child, predicate, literal(bound))
                    for child in expr.args
                    for predicate, bound in ((">=", -INT64_MAX - 1), ("<=", INT64_MAX))
                )
                if expr.kind in DIVISION:
                    numerator, divisor = expr.args
                    require(
                        arithmetic.prove(Comparison(numerator, ">=", literal(0)), assumptions)
                        and arithmetic.prove(Comparison(divisor, ">", literal(0)), assumptions),
                        "new notification arithmetic is not defined in its evaluation scope",
                    )
                if expr.kind in {"add", "mul", "ceildiv"}:
                    require(
                        arithmetic.prove(
                            Comparison(expr, ">=", literal(-INT64_MAX - 1)), assumptions
                        )
                        and arithmetic.prove(
                            Comparison(expr, "<=", literal(INT64_MAX)), assumptions
                        ),
                        "new notification arithmetic may overflow int64",
                    )
                operation = _BINARY[expr.kind](*operands)
            current.add_op(operation)
            memo[expr] = operation.results[0]
            return operation.results[0]

        prefixes, scopes = [], [(current, memo)]
        for action in group:
            active_prefix = []
            tests = action.guard.comparisons
            if (
                isinstance(root, MixedOp)
                and len(action.participants) == 1
                and action.participants[0] is not Actor.AIC
            ):
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
                active_prefix = prefixes[:]
                left, right = (
                    scalar(test.lhs, current, memo, active_prefix),
                    scalar(test.rhs, current, memo, active_prefix),
                )
                if left.type != right.type and left.type == i1 and test.rhs.constant in {0, 1}:
                    boolean = arith.ConstantOp.from_int_and_width(test.rhs.constant, i1)
                    current.add_op(boolean)
                    right = boolean.result
                comparison = arith.CmpiOp(
                    left,
                    right,
                    _COMPARE[test.predicate],
                )
                nested = Block()
                current.add_ops(
                    (comparison, scf.IfOp(comparison.result, [], Region(nested), Region()))
                )
                prefixes.append(test)
                scopes.append((nested, dict(memo)))
            current, memo = scopes[-1]
            active_prefix = prefixes[:]
            event = (
                None if action.event is None else scalar(action.event, current, memo, active_prefix)
            )
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
