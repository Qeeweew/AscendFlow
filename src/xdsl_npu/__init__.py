"""xDSL NPU: source programs with compiler-owned physical synchronization."""

from .compilation import Compilation, compile

__all__ = ["Compilation", "compile"]
