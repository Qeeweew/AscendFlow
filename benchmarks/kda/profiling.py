"""Mark one complete KDA invocation, excluding setup and oracle work."""

import ctypes as ct
from contextlib import contextmanager

from ascendflow.backend.cann import CANN


@contextmanager
def invocation(enabled):
    if not enabled:
        yield
        return
    import torch

    library = ct.CDLL(str(CANN / "tools/mstx/lib64/libms_tools_ext.so"))
    library.mstxRangeStartA.argtypes = [ct.c_char_p, ct.c_void_p]
    library.mstxRangeStartA.restype = ct.c_uint64
    library.mstxRangeEnd.argtypes = [ct.c_uint64]
    library.mstxRangeEnd.restype = None
    marker = library.mstxRangeStartA(b"kda", torch.npu.current_stream().npu_stream)
    try:
        yield
    finally:
        library.mstxRangeEnd(marker)
