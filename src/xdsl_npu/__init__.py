"""xDSL NPU: explicit physical programs, compiler-owned synchronization."""

from .builder import Kernel
from .codegen import emit_asc
from .kernels import batched_matmul_nt, gemm_bias_silu, gemm_nt, rms_norm, rope, softmax, swiglu
from .recurrent import recurrent_gdn
from .sync import autosync, verify_plan

__all__ = [
    "Kernel",
    "autosync",
    "batched_matmul_nt",
    "emit_asc",
    "gemm_bias_silu",
    "gemm_nt",
    "recurrent_gdn",
    "rms_norm",
    "rope",
    "softmax",
    "swiglu",
    "verify_plan",
]
