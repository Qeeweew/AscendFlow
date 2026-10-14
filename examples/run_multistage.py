"""Independent numerical checks for the typed stage examples, without tuning."""

import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path

import numpy as np

from xdsl_npu import emit_asc
from xdsl_npu.device import ACL, health_snapshot
from xdsl_npu.ir import parse
from xdsl_npu.source_ir import AllocationOp


def run(directory, *, physical, repeats=10):
    plan = parse((directory / "after.xdsl").read_text())
    source = parse((directory / "source.xdsl").read_text())
    name = plan.body.block.first_op.sym_name.data
    manifest = json.loads((directory / f"{name}.compile.json").read_text())
    library = directory / f"{name}.so"
    if hashlib.sha256(emit_asc(plan).encode()).hexdigest() != manifest["source_sha256"]:
        raise ValueError("ASC does not match the verified plan; compile again")
    if hashlib.sha256(library.read_bytes()).hexdigest() != manifest["shared_sha256"]:
        raise ValueError("shared library differs from the recorded compilation")
    allocations = [op for op in source.walk() if isinstance(op, AllocationOp)]
    external = [op for op in allocations if op.ref.type.memory.data == "GM"]
    contents = {}
    arrays = {}
    for number, op in enumerate(external):
        shape = tuple(n.data for n in op.ref.type.shape)
        if op.access.data == "read":
            data = ((np.arange(np.prod(shape)) * (number + 3) % 17 - 8) / 8).reshape(shape)
            arrays[op.sym_name.data] = data.astype(
                np.float16 if op.ref.type.dtype.data == "f16" else np.float32
            )
            contents[op.sym_name.data] = arrays[op.sym_name.data].tobytes()
        else:
            size = np.prod(shape) * (2 if op.ref.type.dtype.data == "f16" else 4)
            if op.ring.data:
                size *= op.versions.data * source.body.block.first_op.groups.data
            contents[op.sym_name.data] = b"\xcd" * int(size)
    if name == "scale":
        output_name = "Y"
        expected = arrays["X"] * np.float32(2)
        contract = "FP32 elementwise multiply by 2"
    elif name in {"cv", "segmented_cv"}:
        output_name = "O" if name == "cv" else "Output"
        q = arrays["Q"].astype(np.float32)
        k, v = (arrays[arg].astype(np.float32) for arg in ("K", "V"))
        if name == "cv":
            k, v = k[:, :, None, :, :], v[:, :, None, :, :]
        expected = np.zeros_like(q)
        for part in range(k.shape[2]):
            score = q @ k[:, :, part, :, :].swapaxes(-1, -2)
            p = (score * np.float32(2)).astype(np.float16).astype(np.float32)
            partial = p @ v[:, :, part, :, :]
            expected = partial if part == 0 else (expected + partial).astype(np.float32)
        contract = "chunk-ordered FP32 sum of RN_f16(2 * QK^T) @ V; not attention"
    else:
        raise ValueError(f"unknown multistage example {name}")
    count = next(op for op in source.walk() if op.name == "npu_source.pipeline").count.data
    if count == 0:
        raise ValueError("zero-trip programs are checked in IR; numerical check requires real work")
    health = health_snapshot(physical, directory)
    runtime = ACL()
    try:
        pointers = {name: runtime.allocate(data) for name, data in contents.items()}
        module = ct.CDLL(str(library.resolve()))
        launch = getattr(module, f"launch_{name}")
        launch.argtypes, launch.restype = [ct.c_void_p] * (len(pointers) + 1), None
        for _ in range(repeats):
            launch(runtime.stream, *pointers.values())
        runtime.synchronize()
        raw = runtime.read(pointers[output_name], expected.nbytes)
        actual = np.frombuffer(raw, dtype=np.float32).reshape(expected.shape)
        (directory / "device-output.bin").write_bytes(raw)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
        record = {
            "health": health,
            "kernel": name,
            "repeats": repeats,
            "shape": list(expected.shape),
            "dtype": "f32 output",
            "contract": contract,
            "max_abs_error": float(np.max(np.abs(actual - expected))),
            "source_sha256": manifest["source_sha256"],
            "shared_sha256": manifest["shared_sha256"],
            "argument_sha256": {
                name: hashlib.sha256(data).hexdigest() for name, data in contents.items()
            },
            "rtol": 1e-5,
            "atol": 1e-6,
            "performance_measured": False,
        }
        (directory / "device.json").write_text(json.dumps(record, indent=2) + "\n")
        return record
    finally:
        runtime.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--physical", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    print(json.dumps(run(args.directory, physical=args.physical, repeats=args.repeats), indent=2))


if __name__ == "__main__":
    main()
