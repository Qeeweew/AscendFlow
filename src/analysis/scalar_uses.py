"""Actual per-family SSA liveness and use counts for terminal lowering/output."""

from collections import defaultdict

from xdsl.dialects import arith, scf

from ..dialects.physical import FixConfigOp, Load3DConfigOp, MixedOp, TaskOp
from ..dialects.structured import ActorDomainOp, AddressOp, NotificationOp, PipelineLoopOp
from ..target.ascend910b import Actor


class ScalarUses:
    def __init__(self, module, natives, *, metadata=False):
        self.operations = defaultdict(set)
        self.values = set()
        self.reads = defaultdict(int)
        root = module.body.block.first_op

        def mark(op, family):
            while op is not None:
                self.operations[op].add(family)
                op = op.parent_op()

        for template in natives:
            for actor in template.participants:
                mark(template.operation, "cube" if actor is Actor.AIC else "vector")
        for op in module.walk():
            if isinstance(op, (FixConfigOp, Load3DConfigOp)):
                mark(op, "cube")
            elif isinstance(op, NotificationOp):
                for actor in op.participants:
                    mark(op, "cube" if Actor(actor.data) is Actor.AIC else "vector")
            elif isinstance(op, (scf.ForOp, scf.IfOp)) and op.results:
                parent = op.parent_op()
                while parent is not None and not isinstance(parent, ActorDomainOp):
                    parent = parent.parent_op()
                families = (
                    (parent.kind.data,)
                    if parent is not None
                    else (
                        ("cube", "vector")
                        if isinstance(root, MixedOp)
                        else ("cube" if root.core.data == "aic" else "vector",)
                    )
                )
                for family in families:
                    mark(op, family)

        def scalar(value, family):
            self.reads[value, family] += 1
            if (value, family) in self.values:
                return
            self.values.add((value, family))
            owner = value.owner
            if isinstance(owner, arith.ConstantOp):
                return
            if isinstance(owner, AddressOp):
                operands = (
                    (
                        owner.offset,
                        *owner.origins,
                        *owner.extents,
                        *((owner.valid_rows,) if owner.valid_rows is not None else ()),
                    )
                    if metadata
                    else (owner.offset,)
                )
            elif isinstance(owner, (scf.IfOp, scf.ForOp)):
                mark(owner, family)
                operands = tuple(owner.operands)
                for region in owner.regions:
                    for block in region.blocks:
                        operands += tuple(block.last_op.arguments)
            elif hasattr(owner, "name") and owner.name.startswith("arith."):
                operands = owner.operands
            else:
                return
            self.operations[owner].add(family)
            for operand in operands:
                scalar(operand, family)

        for op in module.walk():
            for family in tuple(self.operations[op]):
                if isinstance(op, TaskOp):
                    operands = op.refs
                elif isinstance(op, NotificationOp):
                    operands = () if op.event is None else (op.event,)
                elif isinstance(op, (scf.ForOp, PipelineLoopOp, scf.IfOp)):
                    operands = tuple(op.operands)
                    for region in op.regions:
                        for block in region.blocks:
                            if isinstance(block.last_op, scf.YieldOp):
                                operands += tuple(block.last_op.arguments)
                else:
                    continue
                for operand in operands:
                    scalar(operand, family)
