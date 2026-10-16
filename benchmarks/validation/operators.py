"""Shared explicit FP32 rounding for independent numerical references."""

import struct


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]
