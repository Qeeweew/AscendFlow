"""Shared NPU types, conservative effects, and runtime input contracts."""

from xdsl.dialects.builtin import ArrayAttr, IntAttr, StringAttr
from xdsl.ir import Operation, ParametrizedAttribute, TypeAttribute
from xdsl.irdl import irdl_attr_definition
from xdsl.traits import EffectInstance, MemoryEffect, MemoryEffectKind
from xdsl.utils.exceptions import VerifyException


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


def verify_runtime_contract(name, minimum, multiple):
    if not name.isidentifier():
        raise VerifyException("runtime size requires a parameter name")
    if type(minimum) is not int or not 0 <= minimum <= (1 << 63) - 1:
        raise VerifyException("runtime minimum must be a nonnegative int64 integer")
    if type(multiple) is not int or not 0 < multiple <= (1 << 63) - 1:
        raise VerifyException("runtime multiple must be a positive int64 integer")
    if ((minimum + multiple - 1) // multiple) * multiple > (1 << 63) - 1:
        raise VerifyException("runtime input contract has no representable int64 value")


def ref_type(bank: str, dtype: str, layout: str = "linear") -> RefType:
    return RefType(StringAttr(bank), StringAttr(dtype), StringAttr(layout))


@irdl_attr_definition
class TensorType(ParametrizedAttribute, TypeAttribute):
    name = "npu_source.tensor"
    shape: ArrayAttr[IntAttr]
    dtype: StringAttr
    memory: StringAttr
    layout: StringAttr


@irdl_attr_definition
class TileType(ParametrizedAttribute, TypeAttribute):
    name = "npu_source.tile"
    shape: ArrayAttr[IntAttr]
    dtype: StringAttr
