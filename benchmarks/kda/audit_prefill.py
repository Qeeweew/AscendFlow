"""Offline precision audit of the unmodified Triton FP32 carry projection."""

import argparse
import importlib
import json
import os
from dataclasses import asdict
from pathlib import Path

from benchmarks.kda.reference import load_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str((args.output / "triton-cache").resolve())
    _, provenance = load_reference(Path(__file__).resolve().parents[3] / "reference/vllm-ascend")
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    module = importlib.import_module("vllm_ascend.ops.triton.kda.chunk_delta_h")
    function = module.chunk_gated_delta_rule_fwd_kernel_h_blockdim64_kda
    while not isinstance(function, triton.runtime.JITFunction):
        function = function.fn
    constants = {
        "H": 32,
        "Hg": 32,
        "K": 128,
        "V": 128,
        "BT": 64,
        "BV": 64,
        "USE_G": False,
        "USE_GK": True,
        "USE_INITIAL_STATE": True,
        "STORE_FINAL_STATE": True,
        "SAVE_NEW_VALUE": True,
        "IS_VARLEN": True,
    }
    signature = {
        name: "constexpr"
        if name in constants
        else "i32"
        if name == "T"
        else "*i64"
        if name in {"cu_seqlens", "chunk_offsets"}
        else "*fp32"
        if name in {"g", "gk", "h0", "ht"}
        else "*fp16"
        for name in function.arg_names
    }
    kernel = triton.compile(
        ASTSource(function, signature=signature, constexprs=constants),
        target=GPUTarget("npu", "Ascend910B3", 0),
        options={"num_warps": 4, "num_stages": 3},
    )
    (args.output / "audit.json").write_text(
        json.dumps(
            {
                "reference": provenance,
                "signature": signature,
                "constexprs": constants,
                "metadata": kernel.metadata._asdict(),
                "stages": list(kernel.asm),
            },
            indent=2,
            default=asdict,
        )
        + "\n"
    )
    print(json.dumps({"status": "compiled", "stages": list(kernel.asm)}))


if __name__ == "__main__":
    main()
