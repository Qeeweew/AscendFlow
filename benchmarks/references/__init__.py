"""Handwritten baselines; not part of the compiler package."""

import ctypes as ct
from pathlib import Path

from ascendflow.diagnostics import require


def ascendc_source(name: str) -> str:
    return (Path(__file__).parent / "ascendc" / name).read_text()


def reference_kernel(runtime, library, name, tensors, *, parameters=()):
    """Bind the explicit pointer ABI of handwritten AscendC references."""
    module = ct.CDLL(str(Path(library).resolve()))
    function = getattr(module, "launch_" + name)
    function.argtypes = [ct.c_void_p] * (1 + tensors) + [ct.c_int64] * len(parameters)
    function.restype = ct.c_int

    def launch(*pointers):
        require(len(pointers) == tensors, "reference tensor count differs from its ABI")
        require(function(runtime.stream, *pointers, *parameters) == 0, "reference launch failed")

    return launch
