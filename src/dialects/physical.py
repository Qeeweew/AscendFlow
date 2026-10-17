"""Physical native work and the single structured execution dialect.

SSA task identity is deliberately not a completion token. Executing operations
have observable effects, so generic CSE/DCE cannot erase asynchronous work.
"""

from xdsl.dialects.builtin import ArrayAttr, FloatAttr, IndexType, IntAttr, StringAttr, f32
from xdsl.ir import Dialect
from xdsl.irdl import (
    IRDLOperation,
    attr_def,
    irdl_op_definition,
    opt_attr_def,
    opt_operand_def,
    opt_result_def,
    region_def,
    result_def,
    traits_def,
    var_operand_def,
)
from xdsl.traits import IsTerminator, RecursiveMemoryEffect

from .structured import ActorDomainOp, AddressOp, NotificationOp, PipelineLoopOp, PipelineStageOp
from .types import AsyncEffects, RefType, TaskType, verify_runtime_contract


@irdl_op_definition
class KernelOp(IRDLOperation):
    name = "npu.kernel"
    sym_name = attr_def(StringAttr)
    target = attr_def(StringAttr)
    core = attr_def(StringAttr)
    disjoint_external = attr_def(IntAttr)
    blocks = opt_attr_def(IntAttr)
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
class CoreIdOp(IRDLOperation):
    """Logical launch group coordinate, shared by its AIC and participating AIVs."""

    name = "npu.core_id"
    value = result_def(IndexType)


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
class TaskOp(IRDLOperation):
    name = "npu.task"
    refs = var_operand_def()
    uid = attr_def(StringAttr)
    native = attr_def(StringAttr)
    stage = attr_def(IntAttr)
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
class EndOp(IRDLOperation):
    name = "npu.end"
    traits = traits_def(IsTerminator())


NPU = Dialect(
    "npu",
    [
        KernelOp,
        MixedOp,
        BufferOp,
        CoreIdOp,
        RuntimeSizeOp,
        TaskOp,
        FixConfigOp,
        Load3DConfigOp,
        EndOp,
        ActorDomainOp,
        AddressOp,
        PipelineLoopOp,
        PipelineStageOp,
        NotificationOp,
    ],
    [RefType, TaskType],
)


def task_attrs(uid: str, native: str, stage: int, params=(), scalar=0.0):
    return {
        "uid": StringAttr(uid),
        "native": StringAttr(native),
        "stage": IntAttr(stage),
        "params": ArrayAttr(IntAttr(p) for p in params),
        "scalar": FloatAttr(scalar, f32),
    }
