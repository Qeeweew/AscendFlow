"""xDSL NPU: explicit physical programs, compiler-owned synchronization."""

from .builder import Kernel
from .codegen import emit_asc
from .kernels import batched_matmul_nt, rms_norm, rope, softmax, swiglu
from .sync import autosync, verify_plan

__all__ = [
    "Kernel",
    "autosync",
    "batched_matmul_nt",
    "emit_asc",
    "rms_norm",
    "rope",
    "softmax",
    "swiglu",
    "verify_plan",
]
