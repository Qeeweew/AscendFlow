"""xDSL NPU: explicit physical programs, compiler-owned synchronization."""

from .builder import Kernel
from .codegen import emit_asc
from .sync import autosync, verify_plan

__all__ = ["Kernel", "autosync", "emit_asc", "verify_plan"]
