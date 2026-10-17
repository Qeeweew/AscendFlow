"""IEEE f32 literals shared by source capture and physical native checks."""

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
