"""xDSL NPU: explicit physical programs, compiler-owned synchronization."""

from .builder import Kernel
from .codegen import emit_asc
from .compilation import Compilation, compile
from .sync import autosync, verify_plan

__all__ = [
    "Compilation",
    "Kernel",
    "autosync",
    "compile",
    "emit_asc",
    "verify_plan",
]
