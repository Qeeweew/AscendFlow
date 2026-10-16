"""Physical asynchronous operations and explicit synchronization structure.

SSA task identity is deliberately not a completion token. Executing operations
have observable effects, so generic CSE/DCE cannot erase asynchronous work.
"""

from xdsl.dialects.builtin import (
    ArrayAttr,
    DictionaryAttr,
    FloatAttr,
    IndexType,
    IntAttr,
    StringAttr,
    f32,
)
from xdsl.ir import Dialect
from xdsl.irdl import (
    IRDLOperation,
    attr_def,
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
from xdsl.traits import IsTerminator, RecursiveMemoryEffect
from xdsl.utils.exceptions import VerifyException

from .loops import LoopScheduleAttr
from .structured import AddressOp, PipelineLoopOp, PipelineStageOp
from .types import AsyncEffects, RefType, TaskType, verify_runtime_contract


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
class CoreIdOp(IRDLOperation):
    """Logical launch group coordinate, shared by its AIC and participating AIVs."""

    name = "npu.core_id"
    value = result_def(IndexType)


@irdl_op_definition
class CaptureOp(IRDLOperation):
    """An address frozen at a parent callsite; alias only, with no completion.

    The byte offset is pure index SSA in the actual parent clock. The child
    accesses a bounded static span relative to it. Parent extraction proves
    bounds and ownership against source; captures never create fresh storage.
    """

    name = "npu.capture"
    source = operand_def(RefType)
    offset = operand_def(IndexType)
    nbytes = attr_def(IntAttr)
    sym_name = attr_def(StringAttr)
    ref = result_def(RefType)


@irdl_op_definition
class CallOp(IRDLOperation):
    """One retained child invocation at an actual parent software-pipeline slot.

    Its body contains address captures and one fixed Stream/Once region. Stage
    shifts the parent generation only; exiting this call retires no child work.
    """

    name = "npu.call"
    generation = operand_def(IndexType)
    uid = attr_def(StringAttr)
    stage = attr_def(IntAttr)
    logical_phase = attr_def(IntAttr)
    issue_order = attr_def(IntAttr)
    body = region_def("single_block")
    traits = traits_def(RecursiveMemoryEffect())


@irdl_op_definition
class ScopeChannelOp(IRDLOperation):
    """Parent-generation protocol between actual fixed child native endpoints."""

    name = "npu.scope_channel"
    sym_name = attr_def(StringAttr)
    kind = attr_def(StringAttr)
    sources = attr_def(ArrayAttr[StringAttr])
    destinations = attr_def(ArrayAttr[StringAttr])
    distance = attr_def(IntAttr)
    period = attr_def(IntAttr)
    residues = attr_def(ArrayAttr[IntAttr])
    event_ids = attr_def(ArrayAttr[IntAttr])
    activation = opt_attr_def(DictionaryAttr)
    traits = traits_def(AsyncEffects())


@irdl_op_definition
class ScopeSyncOp(IRDLOperation):
    """A real native insertion at inner position, activated on its parent clock."""

    name = "npu.scope_sync"
    parent_generation = operand_def(IndexType)
    child_generation = operand_def(IndexType)
    action = attr_def(StringAttr)
    channel = attr_def(StringAttr)
    position = attr_def(IntAttr)
    traits = traits_def(AsyncEffects())


@irdl_op_definition
class QueryChannelOp(IRDLOperation):
    """Local event between actual native endpoints in distinct serial queries.

    ``distance`` advances the query clock. Both parent positions and ``count``
    refer independently to the KV clock; the encoded native positions refer
    to the fixed child clock. No flattened generation replaces these axes.
    """

    name = "npu.query_channel"
    sym_name = attr_def(StringAttr)
    source = attr_def(StringAttr)
    target = attr_def(StringAttr)
    distance = attr_def(IntAttr)
    source_parent = attr_def(DictionaryAttr)
    target_parent = attr_def(DictionaryAttr)
    count = attr_def(DictionaryAttr)
    source_point = attr_def(StringAttr)
    event_ids = attr_def(ArrayAttr[IntAttr])
    traits = traits_def(AsyncEffects())


@irdl_op_definition
class QuerySyncOp(IRDLOperation):
    """Actual local Set/Wait carrying the real query, KV and child clocks."""

    name = "npu.query_sync"
    query_generation = operand_def(IndexType)
    parent_generation = operand_def(IndexType)
    child_generation = operand_def(IndexType)
    action = attr_def(StringAttr)
    channel = attr_def(StringAttr)
    position = attr_def(IntAttr)
    traits = traits_def(AsyncEffects())


@irdl_op_definition
class ScopeGuardOp(IRDLOperation):
    """Native work selected by the retained parent's first or last generation.

    The child coordinate remains independent: a last-parent condition never
    means the last row/chunk of this fixed native software pipeline.
    """

    name = "npu.scope_guard"
    parent_generation = operand_def(IndexType)
    child_generation = operand_def(IndexType)
    kind = attr_def(StringAttr)
    body = region_def("single_block")
    traits = traits_def(RecursiveMemoryEffect())

    def verify_(self):
        if self.kind.data not in {"first", "last"}:
            raise VerifyException("scope guard requires parent first or last")
        if self.body.block.args or not isinstance(self.body.block.last_op, EndOp):
            raise VerifyException("scope guard needs no block arguments and an EndOp terminator")


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
    uid = opt_attr_def(StringAttr)
    count = opt_attr_def(IntAttr)
    limit = opt_operand_def(IndexType)
    symbolic = opt_attr_def(IntAttr)
    body = region_def("single_block")
    traits = traits_def(RecursiveMemoryEffect())


@irdl_op_definition
class OnceOp(IRDLOperation):
    """One ordinary block per enclosing invocation, including when a child is empty.

    The single index argument is exactly zero, not a runtime loop or completion
    token. Exiting the block does not retire its asynchronous operations.
    """

    name = "npu.once"
    uid = attr_def(StringAttr)
    body = region_def("single_block")
    traits = traits_def(RecursiveMemoryEffect())

    @property
    def count(self):
        return IntAttr(1)

    @property
    def limit(self):
        return None


@irdl_op_definition
class RegionChannelOp(IRDLOperation):
    """A guarded single notification (or same-pipe fence) across local regions."""

    name = "npu.region_channel"
    sym_name = attr_def(StringAttr)
    source_region = attr_def(StringAttr)
    source_task = attr_def(StringAttr)
    source_end = attr_def(IntAttr)
    source_offset = attr_def(IntAttr)
    destination_region = attr_def(StringAttr)
    destination_task = attr_def(StringAttr)
    destination_end = attr_def(IntAttr)
    destination_offset = attr_def(IntAttr)
    # (period, residue, exact or -1, minimum); guards refer to the shared N.
    domains = attr_def(ArrayAttr[ArrayAttr[IntAttr]])
    event_id = attr_def(IntAttr)  # -1 denotes a same-pipe fence.
    carried = opt_attr_def(IntAttr)  # Set in invocation p, Wait in p+1; no seed.
    traits = traits_def(AsyncEffects())


@irdl_op_definition
class RegionSyncOp(IRDLOperation):
    name = "npu.region_sync"
    generation = operand_def(IndexType)
    action = attr_def(StringAttr)
    channel = attr_def(StringAttr)
    traits = traits_def(AsyncEffects())


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
    minimum = attr_def(IntAttr)
    multiple = attr_def(IntAttr)
    value = result_def(IndexType)

    def verify_(self):
        verify_runtime_contract(self.sym_name.data, self.minimum.data, self.multiple.data)


@irdl_op_definition
class CyclicCountOp(IRDLOperation):
    """Owner-local count derived from a uniform total and actual Kernel.blocks.

    For owner c in [0, groups), count = ceildiv(max(0, total-c), groups).
    The containing physical loop validates the scope and ownership relation.
    """

    name = "npu.cyclic_count"
    total = operand_def(IndexType)
    count = result_def(IndexType)


def _verify_conditional_loop_context(op):
    ancestor = op.parent_op()
    while ancestor is not None:
        if isinstance(ancestor, CallOp):
            return
        ancestor = ancestor.parent_op()
    raise VerifyException("conditional loop protocol requires an actual enclosing CallOp")


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
    scope_cases = opt_attr_def(ArrayAttr[StringAttr])
    traits = traits_def(AsyncEffects())

    def verify_(self):
        if self.scope_cases is not None:
            _verify_conditional_loop_context(self)


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
    parent_generation = opt_operand_def(IndexType)
    action = attr_def(StringAttr)
    channel = attr_def(StringAttr)
    lower = attr_def(IntAttr)
    upper_gap = attr_def(IntAttr)
    period = attr_def(IntAttr)
    residues = attr_def(ArrayAttr[IntAttr])
    scope_cases = opt_attr_def(ArrayAttr[StringAttr])
    traits = traits_def(AsyncEffects())

    def verify_(self):
        if self.parent_generation is not None or self.scope_cases is not None:
            _verify_conditional_loop_context(self)


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
class Load3DConfigOp(IRDLOperation):
    """Invariant FMATRIX_A/B and Load3D state plus an entry S->MTE1 handshake."""

    name = "npu.init_load3d"
    a_width = attr_def(IntAttr)
    b_width = attr_def(IntAttr)
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
        AddressOp,
        PipelineLoopOp,
        PipelineStageOp,
        FrameOp,
        CoreIdOp,
        CaptureOp,
        CallOp,
        ScopeChannelOp,
        ScopeSyncOp,
        QueryChannelOp,
        QuerySyncOp,
        ScopeGuardOp,
        ViewOp,
        StreamOp,
        OnceOp,
        RegionChannelOp,
        RegionSyncOp,
        BoundaryOp,
        RuntimeSizeOp,
        CyclicCountOp,
        LoopChannelOp,
        InvocationChannelOp,
        InvocationSyncOp,
        LoopSyncOp,
        TaskOp,
        FixConfigOp,
        Load3DConfigOp,
        ChannelOp,
        SyncOp,
        CrossChannelOp,
        CrossSyncOp,
        CrossDrainOp,
        EndOp,
    ],
    [RefType, TaskType, LoopScheduleAttr],
)


def task_attrs(uid: str, native: str, stage: int, params=(), scalar=0.0):
    return {
        "uid": StringAttr(uid),
        "native": StringAttr(native),
        "stage": IntAttr(stage),
        "params": ArrayAttr(IntAttr(p) for p in params),
        "scalar": FloatAttr(scalar, f32),
    }
