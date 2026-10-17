"""Independent finite b16 storage conversion, with RN ties-to-even."""

from ascendflow.diagnostics import require


def encode(values, dtype):
    import numpy as np

    values = np.asarray(values, dtype=np.float32)
    require(dtype in {"f16", "bf16"}, "oracle requires an explicit b16 dtype")
    if dtype == "f16":
        return values.astype("<f2").tobytes()
    bits = values.view(np.uint32)
    upper = ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) >> 16).astype("<u2")
    return upper.tobytes()


def decode(data, dtype, shape=None):
    import numpy as np

    if dtype == "f16":
        result = np.frombuffer(data, dtype="<f2").astype(np.float32)
    else:
        require(dtype == "bf16", "oracle requires an explicit b16 dtype")
        result = (np.frombuffer(data, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
    return result.reshape(shape) if shape is not None else result
