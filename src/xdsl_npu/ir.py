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
    body = region_def("single_block")
    traits = traits_def(RecursiveMemoryEffect())


@irdl_op_definition
class BufferOp(IRDLOperation):
    name = "npu.buffer"
    sym_name = attr_def(StringAttr)
    nbytes = attr_def(IntAttr)
    external = attr_def(IntAttr)
    initialized = attr_def(IntAttr)
    ref = result_def(RefType)
    traits = traits_def(AsyncEffects())


@irdl_op_definition
class ViewOp(IRDLOperation):
    name = "npu.view"
    source = operand_def(RefType)
    generation = operand_def(IndexType)
    offset = attr_def(IntAttr)
    stride = attr_def(IntAttr)
    modulus = attr_def(IntAttr)
    nbytes = attr_def(IntAttr)
    tail_nbytes = attr_def(IntAttr)
    ref = result_def(RefType)


@irdl_op_definition
class StreamOp(IRDLOperation):
    name = "npu.stream"
    count = attr_def(IntAttr)
    body = region_def("single_block")
    traits = traits_def(RecursiveMemoryEffect())


@irdl_op_definition
class TaskOp(IRDLOperation):
    name = "npu.task"
    refs = var_operand_def(RefType)
    uid = attr_def(StringAttr)
    native = attr_def(StringAttr)
    stage = attr_def(IntAttr)
    params = attr_def(ArrayAttr[IntAttr])
    scalar = attr_def(FloatAttr)
    task = result_def(TaskType)
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
    width = attr_def(IntAttr)
    base = attr_def(IntAttr)
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
    traits = traits_def(AsyncEffects())


@irdl_op_definition
class EndOp(IRDLOperation):
    name = "npu.end"
    traits = traits_def(IsTerminator())


NPU = Dialect(
    "npu",
    [KernelOp, BufferOp, ViewOp, StreamOp, TaskOp, FixConfigOp, ChannelOp, SyncOp, EndOp],
    [RefType, TaskType],
)


def dump(module: ModuleOp) -> str:
    output = StringIO()
    Printer(stream=output).print_op(module)
    return output.getvalue() + "\n"


def parse(text: str) -> ModuleOp:
    ctx = Context()
    ctx.load_dialect(Builtin)
    ctx.load_dialect(NPU)
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
