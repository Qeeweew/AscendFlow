"""Declared runner policies for the single public attention example."""

import math
import struct
from dataclasses import asdict, dataclass

from ascendflow.diagnostics import require


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
    query_tile: int = 256
    kv_tile: int = 1024
    vector_rows: int = 8
    causal: bool = False

    def __post_init__(self):
        require(
            all(
                type(v) is int and v > 0
                for v in (
                    self.batch,
                    self.heads,
                    self.kv_heads,
                    self.dim,
                    self.groups,
                    self.query_tile,
                    self.kv_tile,
                    self.vector_rows,
                )
            ),
            "attention policy requires positive integer dimensions, not bool",
        )
        declared = "runner requires the declared fixed attention policy B1/HQ8/D128/groups20, HKV8/2/1, BQ128/256 and f16/bf16"
        require(
            (self.batch, self.heads, self.dim, self.groups) == (1, 8, 128, 20)
            and self.query_tile in {128, 256}
            and type(self.dtype) is str
            and self.dtype in {"f16", "bf16"},
            declared,
        )
        require(self.heads % self.kv_heads == 0, "attention policy requires HQ divisible by HKV")
        require(self.kv_heads in {8, 2, 1}, declared)
        require(
            type(self.causal) is bool and self.kv_tile in {128, 256, 512, 1024},
            "attention requires a boolean causal policy and BK128/256/512/1024",
        )
        require(self.vector_rows in {8, 16}, "attention policy requires VR8/16")

    def as_dict(self):
        return asdict(self)

    def parameters(self):
        return {
            "B": self.batch,
            "HQ": self.heads,
            "HKV": self.kv_heads,
            "D": self.dim,
            "BQ": self.query_tile,
            "BM": 128,
            "BK": self.kv_tile,
            "CK": 128,
            "VR": self.vector_rows,
            "GROUPS": self.groups,
            "DTYPE": self.dtype,
            "SCALE": 0.08838834764831845,
            "CAUSAL": self.causal,
        }

    @property
    def minimum_sk(self):
        return 2 * self.kv_tile if self.causal else 1

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
            TensorContract(
                "Score", "score", (self.groups, 2, self.query_tile, self.kv_tile), "f32"
            ),
            TensorContract("P", "p", (self.groups, 2, self.query_tile, self.kv_tile), self.dtype),
            TensorContract("PV", "pv", (self.groups, 2, self.query_tile, self.dim), "f32"),
        )

    @property
    def local_footprints(self):
        """Declared source allocations; UB storage is private to each AIV."""
        bq, bm, bk, vr, dtype = self.query_tile, 128, self.kv_tile, self.vector_rows, self.dtype
        return (
            ((bm, 128) if bq == bm else (bq // bm, bm, 128), dtype, "L1", "NZ"),
            ((2, 128, 128), dtype, "L1", "NZ"),
            ((2, bq, 128) if bq == bm else (2, bq // bm, bm, 128), dtype, "L1", "NZ"),
            ((2, 128, 128), dtype, "L1", "NZ"),
            *(
                (((bm, 128), dtype, "L0A", "ZZ"),) * 2
                if bq == bm
                else (((2, bm, 128), dtype, "L0A", "ZZ"),)
            ),
            ((2, 128, 128), dtype, "L0B", "ZN"),
            ((2, bm, 128), "f32", "L0C", "NZ"),
            ((2, vr, bk), "f32", "UB", "linear"),
            *((((vr, bk), "f32", "UB", "linear"),) if bq == bm else ()),
            ((2, vr, bk), dtype, "UB", "linear"),
            ((2, vr, 128), "f32", "UB", "linear"),
            ((1 if self.causal else 2, vr, 128), dtype, "UB", "linear"),
            ((bq // 2,), "f32", "UB", "linear"),
            ((bq // 2,), "f32", "UB", "linear"),
            ((bq // 2, 128), "f32", "UB", "linear"),
            ((2, bq // 2), "f32", "UB", "linear"),
            ((vr * (bk // 8 + (bk // 64 if bk > 512 else 0)),), "f32", "UB", "linear"),
            *(((vr,), "f32", "UB", "linear"),) * 4,
            ((vr, 8), "f32", "UB", "linear"),
            ((bq // 2, 8), "f32", "UB", "linear"),
        )

    def validate_case(self, sq, sk):
        require(
            type(sq) is int
            and type(sk) is int
            and sq > 0
            and sk >= self.minimum_sk
            and sq % self.query_tile == sk % 1024 == 0,
            f"attention cases require positive complete BQ{self.query_tile}/Sk1024 tiles",
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
        return {
            key: value
            for key, value in self.as_dict().items()
            if key not in {"dim", "query_tile", "kv_tile", "vector_rows", "causal"}
        }


DEFAULT_POLICY = AttentionPolicy()
