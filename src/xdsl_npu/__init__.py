"""xDSL NPU: explicit physical programs, compiler-owned synchronization."""

from .codegen import emit_asc
from .compilation import Compilation, compile
from .sync import autosync, verify_plan

__all__ = [
    "Compilation",
    "autosync",
    "compile",
    "emit_asc",
    "verify_plan",
]
