"""Recursive physical pipeline templates and SSA byte addresses.

These operations preserve control and aliases, without completing any native
work. Address geometry verification does not establish dynamic bounds, native
alignment, initialized data, buffer ownership, or asynchronous retirement.
"""

from collections import Counter

from xdsl.dialects import arith
from xdsl.dialects.builtin import (
    ArrayAttr,
    DictionaryAttr,
    IndexType,
    IntAttr,
    IntegerAttr,
    StringAttr,
)
from xdsl.irdl import (
    AttrSizedOperandSegments,
    IRDLOperation,
    attr_def,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
    opt_operand_def,
    region_def,
    result_def,
    traits_def,
    var_operand_def,
)
from xdsl.traits import RecursiveMemoryEffect
from xdsl.utils.exceptions import VerifyException

from .loops import LoopScheduleAttr
from .types import AsyncEffects, RefType


@irdl_op_definition
class ActorDomainOp(IRDLOperation):
    """One actor family's code inside a mixed control tree.

    The owner argument is zero for Cube and the runtime sub-block ID in
    {0, 1} for Vector on dav-2201's 1:2 launch. The Vector region is one body
    executed by both AIV participants, each with private local-bank instances.
    GM roots retain their shared identity. Region exit certifies no completion.
    """

    name = "npu.actor_domain"
    kind = attr_def(StringAttr)
    body = region_def("single_block")
    traits = traits_def(RecursiveMemoryEffect())

    def verify_(self):
        from .physical import EndOp, MixedOp

        if self.kind.data not in {"cube", "vector"}:
            raise VerifyException("actor domain kind must be cube or vector")
        parent, mixed = self.parent_op(), False
        while parent is not None:
            if isinstance(parent, ActorDomainOp):
                raise VerifyException("actor domains cannot nest")
            mixed |= isinstance(parent, MixedOp)
            parent = parent.parent_op()
        if not mixed:
            raise VerifyException("actor domain requires a MixedOp ancestor")
        block = self.body.block
        if len(block.args) != 1 or not isinstance(block.args[0].type, IndexType):
            raise VerifyException("actor domain needs one index owner block argument")
        if not isinstance(block.last_op, EndOp):
            raise VerifyException("actor domain body must end with npu.end")


@irdl_op_definition
class PipelineLoopOp(IRDLOperation):
    """One logical pipeline template, with explicit physical-step semantics.

    Let count = ceildiv(max(upper-lower, 0), step). At physical step j, a
    stage at offset s executes only if g = j-s satisfies 0 <= g < count.
    Its two local coordinates are (g, lower + g*step). The canonical schedule
    supplies the physical-step phases; exiting a phase or loop is no drain.

    In a mixed control tree, each actor controller runs its own physical step
    and projects its family's issue_order from this shared template. Equal j
    does not imply global lockstep, ordering, or cross-actor completion.
    The optional order lists existing, unambiguous stage labels per family;
    ownership-aware completeness is checked by the lowerer.
    """

    name = "npu.pipeline_loop"
    lower = operand_def(IndexType)
    upper = operand_def(IndexType)
    step = operand_def(IndexType)
    schedule = attr_def(LoopScheduleAttr)
    issue_order = opt_attr_def(DictionaryAttr)
    body = region_def("single_block")
    traits = traits_def(RecursiveMemoryEffect())

    def verify_(self):
        from .physical import EndOp

        self.schedule.verify()
        constant = self.step.owner
        if (
            not isinstance(constant, arith.ConstantOp)
            or not isinstance(constant.value, IntegerAttr)
            or constant.value.value.data <= 0
        ):
            raise VerifyException("pipeline step must be a positive integer constant")
        block = self.body.block
        if block.args:
            raise VerifyException("pipeline loop body has no block arguments")
        if not isinstance(block.last_op, EndOp):
            raise VerifyException("pipeline loop body must end with npu.end")
        if any(not isinstance(operation, PipelineStageOp) for operation in tuple(block.ops)[:-1]):
            raise VerifyException("pipeline loop body contains only pipeline stages and npu.end")
        if self.issue_order is not None:
            labels = Counter(operation.label.data for operation in tuple(block.ops)[:-1])
            for family, order in self.issue_order.data.items():
                if (
                    family not in {"cube", "vector"}
                    or not isinstance(order, ArrayAttr)
                    or any(not isinstance(label, StringAttr) for label in order)
                ):
                    raise VerifyException("pipeline issue_order needs family stage-label arrays")
                names = tuple(label.data for label in order)
                if len(set(names)) != len(names):
                    raise VerifyException("pipeline issue_order stage labels must not repeat")
                if any(labels[name] != 1 for name in names):
                    raise VerifyException(
                        "pipeline issue_order must name existing unambiguous stages"
                    )


@irdl_op_definition
class PipelineStageOp(IRDLOperation):
    """A stage's logical ordinal and source induction value, with no results.

    The block arguments are (ordinal, iv), bound by the enclosing pipeline's
    guarded physical-step execution. They identify neither completion facts
    nor values of another stage's current physical dispatch.
    """

    name = "npu.pipeline_stage"
    offset = attr_def(IntAttr)
    label = attr_def(StringAttr)
    body = region_def("single_block")
    traits = traits_def(RecursiveMemoryEffect())

    def verify_(self):
        from .physical import EndOp

        parent = self.parent_op()
        if not isinstance(parent, PipelineLoopOp):
            raise VerifyException("pipeline stage must belong directly to a pipeline loop")
        if not 0 <= self.offset.data <= parent.schedule.horizon.data:
            raise VerifyException("pipeline stage offset lies outside its loop schedule")
        block = self.body.block
        if len(block.args) != 2 or any(not isinstance(arg.type, IndexType) for arg in block.args):
            raise VerifyException("pipeline stage needs index ordinal and iv block arguments")
        if not isinstance(block.last_op, EndOp):
            raise VerifyException("pipeline stage body must end with npu.end")


@irdl_op_definition
class AddressOp(IRDLOperation):
    """An alias of an original allocation at a real SSA byte offset.

    nbytes is the static width of each row. Offsets, total allocation bounds,
    and native-specific restrictions are admitted by subsequent analysis.
    An address freezes no data and creates no storage or completion token.

    Optional axis metadata retains origins/extents before linearization and
    widths/axes of the original source view. Displacement records a subview's
    byte displacement relative to that view. Metadata is complete or absent;
    neither form certifies safety, and absence supplies no axis bounds proof.
    """

    name = "npu.address"
    source = operand_def(RefType)
    offset = operand_def(IndexType)
    origins = var_operand_def(IndexType)
    extents = var_operand_def(IndexType)
    nbytes = attr_def(IntAttr)
    rows = attr_def(IntAttr)
    row_stride = attr_def(IntAttr)
    widths = opt_attr_def(ArrayAttr[IntAttr])
    axes = opt_attr_def(ArrayAttr[IntAttr])
    displacement = opt_attr_def(IntAttr)
    ref = result_def(RefType)
    irdl_options = (AttrSizedOperandSegments(as_property=False),)

    def verify_(self):
        from .physical import BufferOp

        owner = self.source.owner
        if not isinstance(owner, BufferOp) or self.source is not owner.ref:
            raise VerifyException("address source must be its original BufferOp allocation root")
        if self.ref.type != self.source.type:
            raise VerifyException("address must preserve its allocation reference type")
        width, rows, stride = self.nbytes.data, self.rows.data, self.row_stride.data
        if width <= 0 or rows <= 0 or stride < 0 or (rows > 1 and stride < width):
            raise VerifyException("address needs positive row width/count and non-overlapping rows")
        metadata = (self.widths, self.axes, self.displacement)
        if not self.origins and not self.extents and all(value is None for value in metadata):
            return  # Structurally valid external input, with no bounds admission.
        if (
            not self.origins
            or len(self.origins) != len(self.extents)
            or any(value is None for value in metadata)
        ):
            raise VerifyException("address axis metadata must be complete or entirely absent")
        assert self.widths is not None and self.axes is not None and self.displacement is not None
        rank = len(self.origins)
        widths = tuple(width.data for width in self.widths)
        axes = tuple(axis.data for axis in self.axes)
        if len(widths) != rank or any(width <= 0 for width in widths):
            raise VerifyException(
                "address axis widths must be positive and match its original rank"
            )
        if axes != tuple(sorted(set(axes))) or any(not 0 <= axis < rank for axis in axes):
            raise VerifyException(
                "address axes must be ordered unique indices of its original rank"
            )
        if self.displacement.data < 0:
            raise VerifyException("address displacement must be nonnegative")


@irdl_op_definition
class NotificationOp(IRDLOperation):
    """One selected instruction at an actual control insertion point.

    Guards are ordinary surrounding SCF control and IDs are scalar SSA values.
    This op is observable even without tensor results. Structural verification
    establishes neither matching partners nor progress or an event lease.
    """

    name = "npu.notification"
    event = opt_operand_def(IndexType)
    kind = attr_def(StringAttr)
    participants = attr_def(ArrayAttr[StringAttr])
    source = attr_def(StringAttr)
    destination = attr_def(StringAttr)
    protocol = attr_def(StringAttr)
    traits = traits_def(AsyncEffects())

    def verify_(self):
        from ..target.ascend910b import Actor, Engine, SyncKind, supports_engine

        try:
            kind = SyncKind(self.kind.data)
            participants = tuple(Actor(value.data) for value in self.participants)
            source, destination = Engine(self.source.data), Engine(self.destination.data)
        except ValueError as error:
            raise VerifyException(
                "notification has an invalid participant, kind, or engine"
            ) from error
        if not participants or len(set(participants)) != len(participants):
            raise VerifyException("notification needs distinct participants")
        if Actor.AIC in participants and len(participants) != 1:
            raise VerifyException("notification cannot span both Cube and Vector code")
        if kind is SyncKind.FENCE:
            if self.event is not None or source is not destination:
                raise VerifyException("pipe fence needs one engine and no event ID")
        elif self.event is None:
            raise VerifyException("notification needs an actual event ID SSA value")
        if kind in {SyncKind.LOCAL_SET, SyncKind.LOCAL_WAIT} and source is destination:
            raise VerifyException("local notification needs two different engines")
        engines = (
            (source,)
            if kind is SyncKind.CROSS_SET
            else (destination,)
            if kind is SyncKind.CROSS_WAIT
            else (source, destination)
        )
        if any(not supports_engine(actor, engine) for actor in participants for engine in engines):
            raise VerifyException("notification engine belongs to another participant")
