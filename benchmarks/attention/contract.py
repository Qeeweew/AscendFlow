"""Declared runner policies for the single public attention example."""

import math
import struct
from dataclasses import asdict, dataclass

from xdsl_npu.model import require


@dataclass(frozen=True)
class TensorContract:
    name: str
    key: str
    shape: tuple[int, ...]
    dtype: str
    extent: str | None = None

    @property
    def element_bytes(self):
        return 4 if self.dtype == "f32" else 2

    @property
    def fixed_bytes(self):
        return 0 if self.extent else math.prod(self.shape) * self.element_bytes

    @property
    def bytes_per_extent(self):
        return (
            math.prod(n for n in self.shape if n != -1) * self.element_bytes if self.extent else 0
        )

    def runtime_shape(self, sq, sk):
        count = sq if self.extent == "Sq" else sk
        return tuple(count if n == -1 else n for n in self.shape)

    def nbytes(self, sq, sk):
        count = sq if self.extent == "Sq" else sk
        return self.fixed_bytes + self.bytes_per_extent * count

    def metadata(self):
        return {
            "shape": self.shape,
            "dtype": self.dtype,
            "extent": self.extent,
            "fixed_bytes": self.fixed_bytes,
            "bytes_per_extent": self.bytes_per_extent,
        }


@dataclass(frozen=True)
class AttentionPolicy:
    batch: int = 1
    heads: int = 8
    kv_heads: int = 8
    dim: int = 128
    groups: int = 20
    dtype: str = "f16"

    def __post_init__(self):
        require(
            all(
                type(v) is int and v > 0
                for v in (self.batch, self.heads, self.kv_heads, self.dim, self.groups)
            ),
            "attention policy requires positive integer dimensions, not bool",
        )
        declared = "runner requires the declared fixed attention policy B1/HQ8/D128/groups20, HKV8/2/1 and f16/bf16"
        require(
            (self.batch, self.heads, self.dim, self.groups) == (1, 8, 128, 20)
            and type(self.dtype) is str
            and self.dtype in {"f16", "bf16"},
            declared,
        )
        require(self.heads % self.kv_heads == 0, "attention policy requires HQ divisible by HKV")
        require(self.kv_heads in {8, 2, 1}, declared)

    def as_dict(self):
        return asdict(self)

    def parameters(self):
        return {
            "B": self.batch,
            "HQ": self.heads,
            "HKV": self.kv_heads,
            "D": self.dim,
            "BQ": 128,
            "BK": 1024,
            "CK": 128,
            "VR": 8,
            "GROUPS": self.groups,
            "DTYPE": self.dtype,
            "SCALE": 0.08838834764831845,
        }

    @property
    def scale(self):
        return struct.unpack("<f", struct.pack("<f", self.parameters()["SCALE"]))[0]

    @property
    def tolerances(self):
        return (0.002, 0.0002) if self.dtype == "f16" else (0.01, 0.001)

    @property
    def tensors(self):
        return (
            TensorContract("Q", "q", (self.batch, self.heads, -1, self.dim), self.dtype, "Sq"),
            TensorContract("K", "k", (self.batch, self.kv_heads, -1, self.dim), self.dtype, "Sk"),
            TensorContract("V", "v", (self.batch, self.kv_heads, -1, self.dim), self.dtype, "Sk"),
            TensorContract("O", "o", (self.batch, self.heads, -1, self.dim), self.dtype, "Sq"),
            TensorContract("Score", "score", (self.groups, 2, 128, 1024), "f32"),
            TensorContract("P", "p", (self.groups, 2, 128, 1024), self.dtype),
            TensorContract("PV", "pv", (self.groups, 2, 128, self.dim), "f32"),
        )

    def validate_case(self, sq, sk):
        require(
            type(sq) is int
            and type(sk) is int
            and sq > 0
            and sk > 0
            and sq % 128 == sk % 1024 == 0,
            "attention cases require positive complete BQ128/BK1024 tiles",
        )
        require(
            all(tensor.nbytes(sq, sk) <= 2**63 - 1 for tensor in self.tensors),
            "attention allocation bytes exceed int64 extents",
        )

    def allocation_bytes(self, sq, sk):
        self.validate_case(sq, sk)
        return {tensor.key: tensor.nbytes(sq, sk) for tensor in self.tensors}

    def shape(self, sq, sk):
        self.validate_case(sq, sk)
        return self.batch, self.heads, self.kv_heads, sq, sk, self.dim

    def reference_parameters(self):
        return {key: value for key, value in self.as_dict().items() if key != "dim"}


DEFAULT_POLICY = AttentionPolicy()
