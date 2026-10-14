"""Small physical dialect. SSA task identity is deliberately not a completion token.

All executing operations have observable effects. Generic CSE/DCE must not treat
an unused task result as dead asynchronous work. Precise effects live in model.py;
the ordinary xDSL interface conservatively exposes an unqualified write too.
"""

from io import StringIO

from xdsl.context import Context
from xdsl.dialects.builtin import (
    ArrayAttr,
    Builtin,
    FloatAttr,
    IndexType,
    IntAttr,
    ModuleOp,
    StringAttr,
    f32,
)
from xdsl.ir import Dialect, Operation, ParametrizedAttribute, TypeAttribute
from xdsl.irdl import (
    IRDLOperation,
    attr_def,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
    opt_operand_def,
    opt_result_def,
    region_def,
    result_def,
    traits_def,
    var_operand_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.traits import (
    EffectInstance,
    IsTerminator,
    MemoryEffect,
    MemoryEffectKind,
    RecursiveMemoryEffect,
)


@irdl_attr_definition
class RefType(ParametrizedAttribute, TypeAttribute):
    name = "npu.ref"
    bank: StringAttr
    dtype: StringAttr
    layout: StringAttr


@irdl_attr_definition
class TaskType(ParametrizedAttribute, TypeAttribute):
    name = "npu.task_id"


class AsyncEffects(MemoryEffect):
    @classmethod
    def get_effects(cls, op: Operation):
        return frozenset(
            {EffectInstance(MemoryEffectKind.READ), EffectInstance(MemoryEffectKind.WRITE)}
        )


@irdl_op_definition
class KernelOp(IRDLOperation):
    name = "npu.kernel"
    sym_name = attr_def(StringAttr)
    target = attr_def(StringAttr)
    core = attr_def(StringAttr)
    disjoint_external = attr_def(IntAttr)
    blocks = opt_attr_def(IntAttr)
    epoch_divisor = opt_attr_def(IntAttr)
    body = region_def("single_block")
    traits = traits_def(RecursiveMemoryEffect())


@irdl_op_definition
class MixedOp(IRDLOperation):
    """One dav-2201 group: one AIC and two participating, private-UB AIVs."""

    name = "npu.mixed"
    sym_name = attr_def(StringAttr)
    target = attr_def(StringAttr)
    blocks = attr_def(IntAttr)
    disjoint_external = attr_def(IntAttr)
    body = region_def("single_block")
    traits = traits_def(RecursiveMemoryEffect())


@irdl_op_definition
class StaticDispatchOp(IRDLOperation):
    """One launch with an exact static partition of mixed group IDs.

    All actors select the same case with their common logical block index.
    Cases bind their GM roots to the same named launch arguments; the final
    proof checks accesses across cases as well as within each mixed group.
    """

    name = "npu.static_dispatch"
    sym_name = attr_def(StringAttr)
    target = attr_def(StringAttr)
    blocks = attr_def(IntAttr)
    body = region_def("single_block")
    traits = traits_def(RecursiveMemoryEffect())


@irdl_op_definition
class StaticCaseOp(IRDLOperation):
    name = "npu.static_case"
    lower = attr_def(IntAttr)
    upper = attr_def(IntAttr)
    body = region_def("single_block")
    traits = traits_def(RecursiveMemoryEffect())


@irdl_op_definition
class CrossChannelOp(IRDLOperation):
    """Mode-2 counting notification, with explicit collective participants."""

    name = "npu.cross_channel"
    sym_name = attr_def(StringAttr)
    sources = attr_def(ArrayAttr[StringAttr])
    destinations = attr_def(ArrayAttr[StringAttr])
    flag_id = attr_def(IntAttr)
    count = attr_def(IntAttr)
    traits = traits_def(AsyncEffects())


@irdl_op_definition
class CrossSyncOp(IRDLOperation):
    name = "npu.cross_sync"
    generation = operand_def(IndexType)
    action = attr_def(StringAttr)
    channel = attr_def(StringAttr)
    lower = attr_def(IntAttr)
    upper = attr_def(IntAttr)
    stride = attr_def(IntAttr)
    epoch_divisor = attr_def(IntAttr)
    epoch_offset = attr_def(IntAttr)
    traits = traits_def(AsyncEffects())


@irdl_op_definition
class CrossDrainOp(IRDLOperation):
    """Consume the remaining returns after dispatching this actor's stream."""

    name = "npu.cross_drain"
    channel = attr_def(StringAttr)
    lower = attr_def(IntAttr)
    upper = attr_def(IntAttr)
    traits = traits_def(AsyncEffects())


@irdl_op_definition
class BufferOp(IRDLOperation):
    name = "npu.buffer"
    sym_name = attr_def(StringAttr)
    nbytes = attr_def(IntAttr)
    external = attr_def(IntAttr)
    initialized = attr_def(IntAttr)
    extent_count = opt_operand_def(IndexType)
    count_scale = opt_attr_def(IntAttr)
    ref = result_def(RefType)
    traits = traits_def(AsyncEffects())


@irdl_op_definition
class FrameOp(IRDLOperation):
    """One owner/invocation's contiguous GM frame; this is an alias, no DMA."""

    name = "npu.frame"
    source = operand_def(RefType)
    invocation = operand_def(IndexType)
    extent = operand_def(IndexType)
    scale = attr_def(IntAttr)
    ref = result_def(RefType)


@irdl_op_definition
class ViewOp(IRDLOperation):
    name = "npu.view"
    source = operand_def(RefType)
    generation = operand_def(IndexType)
    extent_count = opt_operand_def(IndexType)
    offset = attr_def(IntAttr)
    stride = attr_def(IntAttr)
    modulus = attr_def(IntAttr)
    divisor = opt_attr_def(IntAttr)
    nbytes = attr_def(IntAttr)
    tail_nbytes = attr_def(IntAttr)
    rows = opt_attr_def(IntAttr)
    row_stride = opt_attr_def(IntAttr)
    core_stride = opt_attr_def(IntAttr)
    core_divisor = opt_attr_def(IntAttr)
    core_modulus = opt_attr_def(IntAttr)
    index_terms = opt_attr_def(ArrayAttr[ArrayAttr[IntAttr]])
    core_terms = opt_attr_def(ArrayAttr[ArrayAttr[IntAttr]])
    count_offset = opt_attr_def(IntAttr)
    count_core_stride = opt_attr_def(IntAttr)
    prefix_elements = opt_attr_def(IntAttr)
    ref = result_def(RefType)


@irdl_op_definition
class StreamOp(IRDLOperation):
    name = "npu.stream"
    count = opt_attr_def(IntAttr)
    limit = opt_operand_def(IndexType)
    symbolic = opt_attr_def(IntAttr)
    body = region_def("single_block")
    traits = traits_def(RecursiveMemoryEffect())


@irdl_op_definition
class BoundaryOp(IRDLOperation):
    """Conditional native region in its enclosing stream's logical clock.

    kind=last executes only at generation=N-1 with N>0. It is not a drain
    and cannot certify the completion of any enclosed asynchronous effect.
    """

    name = "npu.boundary"
    generation = operand_def(IndexType)
    kind = attr_def(StringAttr)
    body = region_def("single_block")
    traits = traits_def(RecursiveMemoryEffect())


@irdl_op_definition
class RuntimeSizeOp(IRDLOperation):
    """A uniform by-value kernel argument; not a mutable metadata load."""

    name = "npu.runtime_size"
    sym_name = attr_def(StringAttr)
    multiple = attr_def(IntAttr)
    value = result_def(IndexType)


@irdl_op_definition
class LoopChannelOp(IRDLOperation):
    name = "npu.loop_channel"
    sym_name = attr_def(StringAttr)
    kind = attr_def(StringAttr)
    sources = attr_def(ArrayAttr[StringAttr])
    destinations = attr_def(ArrayAttr[StringAttr])
    distance = attr_def(IntAttr)
    period = attr_def(IntAttr)
    residues = attr_def(ArrayAttr[IntAttr])
    event_ids = attr_def(ArrayAttr[IntAttr])
    last = opt_attr_def(IntAttr)
    carry = opt_attr_def(IntAttr)
    traits = traits_def(AsyncEffects())


@irdl_op_definition
class InvocationChannelOp(IRDLOperation):
    """Local event from a child tail to the next invocation's head.

    A Set captures the producer pipe prefix after source at N-1-source_gap;
    its Wait precedes destination at destination_index in the next call.
    Neither endpoint executes outside its real count/parent activation.
    """

    name = "npu.invocation_channel"
    sym_name = attr_def(StringAttr)
    source = attr_def(StringAttr)
    destination = attr_def(StringAttr)
    source_gap = attr_def(IntAttr)
    destination_index = attr_def(IntAttr)
    event_id = attr_def(IntAttr)
    traits = traits_def(AsyncEffects())


@irdl_op_definition
class InvocationSyncOp(IRDLOperation):
    name = "npu.invocation_sync"
    generation = operand_def(IndexType)
    action = attr_def(StringAttr)
    channel = attr_def(StringAttr)
    traits = traits_def(AsyncEffects())


@irdl_op_definition
class LoopSyncOp(IRDLOperation):
    name = "npu.loop_sync"
    last_gap = opt_attr_def(IntAttr)
    generation = operand_def(IndexType)
    action = attr_def(StringAttr)
    channel = attr_def(StringAttr)
    lower = attr_def(IntAttr)
    upper_gap = attr_def(IntAttr)
    period = attr_def(IntAttr)
    residues = attr_def(ArrayAttr[IntAttr])
    traits = traits_def(AsyncEffects())


@irdl_op_definition
class TaskOp(IRDLOperation):
    name = "npu.task"
    refs = var_operand_def()
    uid = attr_def(StringAttr)
    native = attr_def(StringAttr)
    stage = attr_def(IntAttr)
    lower = opt_attr_def(IntAttr)
    upper = opt_attr_def(IntAttr)
    issue_order = opt_attr_def(IntAttr)
    period = opt_attr_def(IntAttr)
    phase = opt_attr_def(IntAttr)
    phase_width = opt_attr_def(IntAttr)
    logical_phase = opt_attr_def(IntAttr)
    params = attr_def(ArrayAttr[IntAttr])
    scalar = attr_def(FloatAttr)
    task = result_def(TaskType)
    value = opt_result_def(f32)
    traits = traits_def(AsyncEffects())


@irdl_op_definition
class FixConfigOp(IRDLOperation):
    """One real invariant S initialization + one S->FIX ready handshake.

    It has its own bank/ID0, executes even on an empty stream, and cannot be
    moved past a Fixpipe reader. No external/native state clobbers are accepted.
    """

    name = "npu.init_fixpipe"
    traits = traits_def(AsyncEffects())


@irdl_op_definition
class ChannelOp(IRDLOperation):
    name = "npu.channel"
    sym_name = attr_def(StringAttr)
    source = attr_def(StringAttr)
    destination = attr_def(StringAttr)
    distance = attr_def(IntAttr)
    lower = attr_def(IntAttr)
    upper = attr_def(IntAttr)
    stride = opt_attr_def(IntAttr)
    width = attr_def(IntAttr)
    base = attr_def(IntAttr)
    event_ids = opt_attr_def(ArrayAttr[IntAttr])
    reason = attr_def(StringAttr)
    traits = traits_def(AsyncEffects())


@irdl_op_definition
class SyncOp(IRDLOperation):
    """Task-local set/wait or a conflict-only same-pipe barrier.

    [lower,upper) is in the attached task's logical generation, not its stage.
    A wait uses the producing generation g-distance when selecting an ID.
    """

    name = "npu.sync"
    generation = operand_def(IndexType)
    action = attr_def(StringAttr)
    channel = attr_def(StringAttr)
    pipe = attr_def(StringAttr)
    lower = attr_def(IntAttr)
    upper = attr_def(IntAttr)
    distance = attr_def(IntAttr)
    stride = opt_attr_def(IntAttr)
    traits = traits_def(AsyncEffects())


@irdl_op_definition
class EndOp(IRDLOperation):
    name = "npu.end"
    traits = traits_def(IsTerminator())


NPU = Dialect(
    "npu",
    [
        KernelOp,
        MixedOp,
        StaticDispatchOp,
        StaticCaseOp,
        BufferOp,
        FrameOp,
        ViewOp,
        StreamOp,
        BoundaryOp,
        RuntimeSizeOp,
        LoopChannelOp,
        InvocationChannelOp,
        InvocationSyncOp,
        LoopSyncOp,
        TaskOp,
        FixConfigOp,
        ChannelOp,
        SyncOp,
        CrossChannelOp,
        CrossSyncOp,
        CrossDrainOp,
        EndOp,
    ],
    [RefType, TaskType],
)


def dump(module: ModuleOp) -> str:
    output = StringIO()
    Printer(stream=output).print_op(module)
    return output.getvalue() + "\n"


def parse(text: str) -> ModuleOp:
    from xdsl.dialects.arith import Arith
    from xdsl.dialects.scf import Scf

    from .source_ir import SOURCE

    ctx = Context()
    ctx.load_dialect(Builtin)
    ctx.load_dialect(NPU)
    ctx.load_dialect(SOURCE)
    ctx.load_dialect(Arith)
    ctx.load_dialect(Scf)
    module = Parser(ctx, text).parse_module()
    module.verify()
    return module


def ref_type(bank: str, dtype: str, layout: str = "linear") -> RefType:
    return RefType(StringAttr(bank), StringAttr(dtype), StringAttr(layout))


def task_attrs(uid: str, native: str, stage: int, params=(), scalar=0.0):
    return {
        "uid": StringAttr(uid),
        "native": StringAttr(native),
        "stage": IntAttr(stage),
        "params": ArrayAttr(IntAttr(p) for p in params),
        "scalar": FloatAttr(scalar, f32),
    }
