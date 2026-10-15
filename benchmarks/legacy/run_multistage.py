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
from xdsl_npu.loop_model import is_symbolic
from xdsl_npu.source_ir import AllocationOp


def _case(source, external, count):
    contents, arrays = {}, {}
    guard = b"\x5a" * 128
    groups = source.body.block.first_op.groups.data
    for number, op in enumerate(external):
        shape = tuple(count if n.data == -1 else n.data for n in op.ref.type.shape)
        if op.access.data == "read":
            data = ((np.arange(np.prod(shape)) * (number + 3) % 17 - 8) / 8).reshape(shape)
            arrays[op.sym_name.data] = data.astype(
                np.float16 if op.ref.type.dtype.data == "f16" else np.float32
            )
            contents[op.sym_name.data] = arrays[op.sym_name.data].tobytes() + guard
        else:
            size = np.prod(shape) * (2 if op.ref.type.dtype.data == "f16" else 4)
            if op.ring.data:
                size *= op.versions.data * groups
            contents[op.sym_name.data] = b"\xcd" * int(size) + guard
    name = source.body.block.first_op.sym_name.data
    if name == "scale":
        output_name, expected = "Y", arrays["X"] * np.float32(2)
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
        contract = "chunk-ordered FP32 sum of RN_f16(2 * (Q @ K.T)) @ V; not attention"
    else:
        raise ValueError(f"unknown multistage example {name}")
    return contents, output_name, expected, contract, guard


def run(directory, *, physical, repeats=10, counts=None):
    from xdsl_npu.source_ir import PipelineOp, SizeOp

    plan = parse((directory / "after.xdsl").read_text())
    source = parse((directory / "source.xdsl").read_text())
    name = plan.body.block.first_op.sym_name.data
    manifest = json.loads((directory / f"{name}.compile.json").read_text())
    library = directory / f"{name}.so"
    if hashlib.sha256(emit_asc(plan).encode()).hexdigest() != manifest["source_sha256"]:
        raise ValueError("ASC does not match the verified plan; compile again")
    if hashlib.sha256(library.read_bytes()).hexdigest() != manifest["shared_sha256"]:
        raise ValueError("shared library differs from the recorded compilation")
    pipeline = next(op for op in source.walk() if isinstance(op, PipelineOp))
    dynamic = pipeline.limit is not None
    if dynamic:
        if counts is None or not counts or any(count < 0 for count in counts):
            raise ValueError("runtime programs require explicit nonnegative --counts")
        size = pipeline.limit.owner
        if not isinstance(size, SizeOp) or any(count % size.multiple.data for count in counts):
            raise ValueError("lengths violate the uniform runtime size contract")
    else:
        if counts is not None and counts != [pipeline.count.data]:
            raise ValueError("static binary can only run its compiled count")
        counts = [pipeline.count.data]
        if counts == [0]:
            raise ValueError("use a runtime binary to check empty arrays and zero-trip guards")
    if repeats <= 0:
        raise ValueError("positive repeats required")
    external = [
        op
        for op in source.walk()
        if isinstance(op, AllocationOp) and op.ref.type.memory.data == "GM"
    ]
    health = health_snapshot(physical, directory)
    runtime = ACL()
    cases = []
    try:
        module = ct.CDLL(str(library.resolve()))
        launch = getattr(module, f"launch_{name}")
        launch.argtypes = [ct.c_void_p] * (len(external) + 1) + ([ct.c_int64] if dynamic else [])
        # Parametric launchers return a contract status; old finite launchers
        # remain void. Both go through the public independent ASC verifier.
        launch.restype = ct.c_int if is_symbolic(plan) else None
        for count in counts:
            contents, output_name, expected, contract, guard = _case(source, external, count)
            pointers = {arg: runtime.allocate(data) for arg, data in contents.items()}
            arguments = [*pointers.values()] + ([count] if dynamic else [])
            for _ in range(repeats):
                status = launch(runtime.stream, *arguments)
                if launch.restype is ct.c_int and status != 0:
                    raise ValueError(f"launcher rejected runtime length {count}")
            runtime.synchronize()
            raw = runtime.read(pointers[output_name], expected.nbytes)
            actual = np.frombuffer(raw, dtype=np.float32).reshape(expected.shape)
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
            for arg, data in contents.items():
                if (
                    runtime.read(
                        ct.c_void_p(pointers[arg].value + len(data) - len(guard)), len(guard)
                    )
                    != guard
                ):
                    raise ValueError(f"{arg}: allocation bounds guard changed")
                if count == 0 and runtime.read(pointers[arg], len(data)) != data:
                    raise ValueError(f"{arg}: zero-trip stream changed memory")
            (directory / f"device-output-{count}.bin").write_bytes(raw)
            case = {
                "count": count,
                "repeats": repeats,
                "shape": list(expected.shape),
                "dtype": "f32 output",
                "contract": contract,
                "max_abs_error": float(np.max(np.abs(actual - expected))) if actual.size else 0.0,
                "bounds_guards": "all arguments checked",
                "rtol": 1e-5,
                "atol": 1e-6,
                "argument_sha256": {
                    arg: hashlib.sha256(data).hexdigest() for arg, data in contents.items()
                },
            }
            cases.append(case)
            print(json.dumps(case), flush=True)
        record = {
            "health": health,
            "kernel": name,
            "source_sha256": manifest["source_sha256"],
            "shared_sha256": manifest["shared_sha256"],
            "same_binary_for_all_lengths": True,
            "cases": cases,
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
    parser.add_argument("--counts", type=int, nargs="+", help="runtime tile counts for one binary")
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.directory, physical=args.physical, repeats=args.repeats, counts=args.counts),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
