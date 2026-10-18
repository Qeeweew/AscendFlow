"""Import the unmodified reference files without importing vLLM's engine.

The namespace adapters expose only Triton and its two integer math helpers.
All operator wrappers, JIT bodies, tiling and numerical work come from the
selected vllm-ascend checkout. This module is a benchmark dependency only.
"""

import hashlib
import importlib
import importlib.metadata
import subprocess
import sys
import types
from pathlib import Path


def load_reference(checkout):
    import triton
    import triton.language as tl

    checkout = Path(checkout).resolve()
    for name, directory in (
        ("vllm", None),
        ("vllm.utils", None),
        ("vllm_ascend", checkout / "vllm_ascend"),
        ("vllm_ascend.ops", checkout / "vllm_ascend/ops"),
        ("vllm_ascend.ops.triton", checkout / "vllm_ascend/ops/triton"),
        ("vllm_ascend.ops.triton.kda", checkout / "vllm_ascend/ops/triton/kda"),
    ):
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [] if directory is None else [str(directory)]
            sys.modules[name] = module
    if "vllm.triton_utils" not in sys.modules:
        module = types.ModuleType("vllm.triton_utils")
        module.triton, module.tl, module.HAS_TRITON = triton, tl, True
        sys.modules[module.__name__] = module
    if "vllm.utils.math_utils" not in sys.modules:
        module = types.ModuleType("vllm.utils.math_utils")
        module.cdiv, module.next_power_of_2 = triton.cdiv, triton.next_power_of_2
        sys.modules[module.__name__] = module
    operators = importlib.import_module("vllm_ascend.ops.triton.kda.kda")
    files = sorted((checkout / "vllm_ascend/ops/triton/kda").glob("*.py"))
    files.append(checkout / "vllm_ascend/ops/triton/triton_utils.py")
    provenance = {
        "checkout": str(checkout),
        "commit": subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
        ).strip(),
        "status": subprocess.check_output(
            ["git", "-C", str(checkout), "status", "--short"], text=True
        ),
        "files": {
            str(p.relative_to(checkout)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
        },
        "import_adapters": [
            "vllm.triton_utils",
            "vllm.utils.math_utils",
            "namespace packages without engine initialization",
        ],
        "triton": triton.__version__,
        "distribution": importlib.metadata.version("triton-ascend"),
        "installation": json_installation(),
        "runtime_files": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for name in ("driver.py", "utils.py", "compiler.py", "npu_utils.cpp")
            if (path := Path(triton.__file__).parent / "backends/ascend" / name).exists()
        },
    }
    return operators, provenance


def json_installation():
    import json

    value = importlib.metadata.distribution("triton-ascend").read_text("direct_url.json")
    return None if value is None else json.loads(value)


def recurrent_oracle(q, k, v, g, beta, initial, lengths, indices, *, normalize=True, scale=None):
    """Independent CPU FP32 recurrence, including each snapshot and null slot."""
    import torch

    output_dtype = v.dtype
    q, k, v, g, beta = [value.float().cpu() for value in (q, k, v, g, beta)]
    state = initial.cpu().clone()
    snapshots = torch.zeros((q.shape[0], *state.shape[1:]), dtype=initial.dtype)
    output = torch.full_like(v, float("nan"))
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    bos = 0
    for seq, length in enumerate(lengths):
        if not length:
            continue
        slot = int(indices[seq, 0])
        if slot <= 0:
            bos += length
            continue
        carry = state[slot].float().clone()
        for token in range(bos, bos + length):
            qt, kt = q[token], k[token]
            if normalize:
                qt = qt * torch.rsqrt(qt.square().sum(-1, keepdim=True) + 1e-6)
                kt = kt * torch.rsqrt(kt.square().sum(-1, keepdim=True) + 1e-6)
            carry = carry * g[token].exp()[:, None, :]
            residual = (v[token] - (carry * kt[:, None, :]).sum(-1)) * beta[token, :, None]
            carry = carry + residual[:, :, None] * kt[:, None, :]
            output[token] = (carry * (qt * scale)[:, None, :]).sum(-1)
            destination = int(indices[seq, token - bos])
            if destination > 0:
                state[destination] = carry.to(state.dtype)
            snapshots[token] = carry.to(state.dtype)
        bos += length
    return output.to(output_dtype), state, snapshots
