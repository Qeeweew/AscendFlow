"""IEEE literal representations for source and native scalar operands."""

import struct
from math import isfinite, isinf

from ..diagnostics import CompileError, require


def round_f32(value, *, allow_infinity=False):
    """The IR literal and emitted C++ float must have the same value."""
    if allow_infinity and isinf(value):
        return float(value)
    require(isfinite(value), "Vector scalar must be finite")
    try:
        result = struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError as error:
        raise CompileError("Vector scalar is outside finite f32 range") from error
    require(isfinite(result), "Vector scalar is outside finite f32 range")
    return result


def b16_fill_bits(value, dtype):
    """Round the admitted f32 literal to IEEE b16, RN ties to even.

    Vector Dup writes these bits without another device-side conversion.
    Preserve signed zero, subnormals and infinity; finite overflow rounds to
    infinity. NaN is rejected by the common source scalar contract.
    """
    require(dtype in {"f16", "bf16"}, "b16 fill needs f16 or bf16")
    value = round_f32(value, allow_infinity=True)
    if dtype == "bf16":
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        return (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    try:
        return struct.unpack("<H", struct.pack("<e", value))[0]
    except OverflowError:
        return 0xFC00 if value < 0 else 0x7C00
