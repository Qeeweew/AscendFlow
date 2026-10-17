"""Contiguous NN b16 GEMM with exact edge guards and installed ACLNN."""

import argparse
import ctypes as ct
import hashlib
import json
import subprocess
from pathlib import Path

from ascendflow.backend.cann import CANN
from ascendflow.build import build_file
from ascendflow.device import ACL, health_snapshot
from ascendflow.diagnostics import require
from ascendflow.dialects.io import parse
from ascendflow.dialects.physical import BufferOp
from ascendflow.dialects.source import AllocationOp
from ascendflow.performance import hardware_snapshot
from benchmarks.artifacts import read_artifact, verified_source
from benchmarks.references.aclnn import GemmReference
from benchmarks.validation.dtypes import decode, encode
from benchmarks.work import structure_inventory, work_record

EXAMPLE = Path(__file__).resolve().parents[2] / "examples/gemm/matmul.py"
GUARD = b"\x5a" * 512


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def load_contract(directory):
    metadata = json.loads((directory / "benchmark.json").read_text())
    for name, digest in metadata["files"].items():
        require(sha((directory / name).read_bytes()) == digest, f"GEMM artifact changed: {name}")
    for name, digest in metadata["compiler_files"].items():
        require(
            sha((directory / "compiler-source" / name).read_bytes()) == digest,
            f"compiler snapshot changed: {name}",
        )
    for name, digest in metadata.get("algorithm_files", {}).items():
        require(
            sha((directory / "algorithm-source" / name).read_bytes()) == digest,
            f"algorithm snapshot changed: {name}",
        )
    artifact = read_artifact(directory)
    manifest = json.loads((directory / "gemm.compile.json").read_text())
    require(
        verified_source(manifest) == directory / "gemm.asc"
        and manifest["source_sha256"] == metadata["files"]["gemm.asc"]
        and manifest["target"] == "dav-2201",
        "GEMM compiled identity differs",
    )
    policy = metadata["policy"]
    m, n, k = (policy[key] for key in ("M", "N", "K"))
    dtype = policy["DTYPE"]
    require(dtype in {"f16", "bf16"}, "GEMM needs b16 inputs and output")
    expected = {
        "A": ((m, k), m * k * 2, dtype),
        "B": ((k, n), k * n * 2, dtype),
        "C": ((m, n), m * n * 2, dtype),
    }
    tensors = {
        op.sym_name.data: (tuple(size.data for size in op.ref.type.shape), op.ref.type.dtype.data)
        for op in parse((directory / "source.xdsl").read_text()).walk()
        if isinstance(op, AllocationOp) and op.ref.type.memory.data == "GM"
    }
    require(
        tensors == {name: (shape, typ) for name, (shape, _, typ) in expected.items()},
        "GEMM source layout/dtype differs",
    )
    buffers = {
        op.sym_name.data: (op.nbytes.data, op.ref.type.dtype.data)
        for op in artifact.plan.walk()
        if isinstance(op, BufferOp) and op.external.data
    }
    require(
        buffers == {name: (size, typ) for name, (_, size, typ) in expected.items()},
        "GEMM physical output/input dtype differs",
    )
    inventory = structure_inventory(artifact.plan)
    require(inventory["groups"] == policy["GROUPS"], "GEMM grid differs")
    return metadata, artifact, manifest, inventory


def inputs_and_oracle(m, n, k, dtype, *, seed, data, scope):
    import numpy as np

    rng = np.random.default_rng(seed)
    if data == "rounding":
        values = [np.zeros(shape, dtype=np.float32) for shape in ((m, k), (k, n))]
        values[0][:, 0] = np.where(np.arange(m) % 2, -1.5, 1.5)
        eps = 2**-10 if dtype == "f16" else 2**-7
        values[1][0, :] = 1 + eps * (1 + 2 * (np.arange(n) % 2))
    elif data == "exact":
        require(k <= 65536, "exact FP32 accumulation bound exceeded")
        values = [
            rng.integers(-2, 3, size=shape).astype(np.float32) / 8 for shape in ((m, k), (k, n))
        ]
    else:
        values = [rng.normal(0, 0.5, shape) for shape in ((m, k), (k, n))]
    a, b = (encode(value, dtype) for value in values)
    rows = (
        np.arange(m)
        if scope == "full"
        else np.unique(
            [row for start in range(0, m, 128) for row in range(start, min(start + 6, m))] + [m - 1]
        )
    )
    expected = decode(a, dtype, (m, k))[rows].astype(np.float64) @ decode(b, dtype, (k, n)).astype(
        np.float64
    )
    rounded = decode(encode(expected, dtype), dtype, expected.shape)
    return a, b, rows, expected, rounded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build", action="store_true")
    for key, default in {
        "m": 4096,
        "n": 4096,
        "k": 4096,
        "bm": 128,
        "bn": 128,
        "bk": 256,
        "ck": 128,
        "col_parts": 2,
        "l2_tile": 0,
        "groups": 20,
    }.items():
        parser.add_argument("--" + key.replace("_", "-"), type=int, default=default)
    parser.add_argument("--dtype", choices=("f16", "bf16"), default="f16")
    parser.add_argument("--output-cache", choices=("auto", "normal", "bypass"), default="auto")
    parser.add_argument("--backend", choices=("generated", "aclnn"), default="generated")
    parser.add_argument("--physical", type=int)
    parser.add_argument("--record", type=Path)
    parser.add_argument("--scope", choices=("full", "sampled"), default="full")
    parser.add_argument("--data", choices=("normal", "exact", "rounding"), default="normal")
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--offset-elements", type=int, default=0)
    args = parser.parse_args()
    directory = args.output.resolve()
    if args.build:
        policy = {
            key.upper(): getattr(args, key)
            for key in (
                "m",
                "n",
                "k",
                "bm",
                "bn",
                "bk",
                "ck",
                "col_parts",
                "l2_tile",
                "groups",
                "dtype",
                "output_cache",
            )
        }
        policy["L2_BYTES"] = 201326592
        require(1 <= args.groups <= 20, "invalid Cube count")
        root = EXAMPLE.parents[2]
        definition = EXAMPLE.read_bytes()
        algorithm = {
            str(path.relative_to(root)): path.read_bytes()
            for path in (EXAMPLE, EXAMPLE.with_name("schedule.py"))
        }
        compiler = {
            str(p.relative_to(root)): p.read_bytes() for p in sorted((root / "src").rglob("*.py"))
        }
        build_file(EXAMPLE, directory, entry="gemm", parameters=policy, shared=True, timeout_s=300)
        require(
            EXAMPLE.read_bytes() == definition
            and all((root / name).read_bytes() == contents for name, contents in algorithm.items())
            and all((root / name).read_bytes() == contents for name, contents in compiler.items()),
            "GEMM source changed during build",
        )
        (directory / "definition.py").write_bytes(definition)
        for name, contents in algorithm.items():
            target = directory / "algorithm-source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(contents)
        for name, contents in compiler.items():
            target = directory / "compiler-source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(contents)
        write_json(
            directory / "benchmark.json",
            {
                "policy": policy,
                "git_head": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=root, text=True
                ).strip(),
                "compiler_files": {name: sha(contents) for name, contents in compiler.items()},
                "algorithm_files": {name: sha(contents) for name, contents in algorithm.items()},
                "files": {
                    name: sha((directory / name).read_bytes())
                    for name in (
                        "definition.py",
                        "source.xdsl",
                        "before.xdsl",
                        "after.xdsl",
                        "gemm.asc",
                    )
                },
            },
        )
    metadata, artifact, manifest, inventory = load_contract(directory)
    if args.physical is None:
        print(json.dumps({"policy": metadata["policy"], "compilation": artifact.timings}))
        return
    require(
        args.warmup >= 0 and args.repeats > 0 and 0 <= args.offset_elements <= 15,
        "invalid launch counts or pointer offset",
    )
    record = (args.record or directory / f"device-{args.backend}.json").resolve()
    health = health_snapshot(args.physical, record.parent)
    hardware = hardware_snapshot(args.physical)
    policy = metadata["policy"]
    if policy["OUTPUT_CACHE"] == "auto":
        require(policy["L2_BYTES"] == hardware["l2_bytes"], "L2 cache capacity differs")
    m, n, k = (policy[key] for key in ("M", "N", "K"))
    dtype = policy["DTYPE"]
    a, b, rows, expected, rounded = inputs_and_oracle(
        m, n, k, dtype, seed=args.seed, data=args.data, scope=args.scope
    )
    import numpy as np

    runtime, operator = ACL(), None
    prefix = GUARD + b"\x5a" * (args.offset_elements * 2)
    try:
        caches = {
            op.sym_name.data: op.cache.data if op.cache is not None else "normal"
            for op in artifact.plan.walk()
            if isinstance(op, BufferOp) and op.external.data
        }
        alias = None
        if args.backend == "generated" and "bypass" in caches.values():
            library_path = (CANN / "lib64/libruntime.so").resolve()
            rts = ct.CDLL(str(library_path))
            query = rts.rtGetL2CacheOffset
            query.argtypes, query.restype = [ct.c_uint32, ct.POINTER(ct.c_uint64)], ct.c_int32
            offset = ct.c_uint64()
            require(
                query(0, ct.byref(offset)) == 0 and offset.value > 0,
                "cache bypass alias unavailable",
            )
            alias = {
                "offset": offset.value,
                "runtime_library": str(library_path),
                "runtime_sha256": sha(library_path.read_bytes()),
            }
        contents = (a, b, b"\xcd" * (m * n * 2))
        handles = [runtime.allocate(prefix + value + GUARD) for value in contents]
        pointers = [ct.c_void_p(handle.value + len(prefix)) for handle in handles]
        if args.backend == "generated":
            library = directory / "gemm.so"
            module = ct.CDLL(str(library))
            launcher = module.launch_gemm
            launcher.argtypes, launcher.restype = [ct.c_void_p] * 4, ct.c_int

            def launch():
                require(launcher(runtime.stream, *pointers) == 0, "GEMM launch rejected")

            identity = {
                "library": str(library),
                "shared_sha256": sha(library.read_bytes()),
                "manifest": manifest,
            }
        else:
            operator = GemmReference(
                runtime, *pointers, m, n, k, dtype=dtype, output_dtype=dtype, transpose_b=False
            )
            launch = operator
            identity = {
                "interface": operator.interface,
                "tensor_dtypes": operator.tensor_dtypes,
                "adapter_sha256": sha(
                    (Path(__file__).parents[1] / "references/aclnn.py").read_bytes()
                ),
                "workspace_bytes": operator.workspace_size.value,
                "libraries": {
                    name: {
                        "path": str((CANN / "lib64" / name).resolve()),
                        "sha256": sha((CANN / "lib64" / name).read_bytes()),
                    }
                    for name in ("libopapi.so", "libnnopbase.so")
                },
            }
        for _ in range(args.warmup):
            launch()
        runtime.synchronize()
        start, stop = runtime.event(), runtime.event()
        runtime.record(start)
        for _ in range(args.repeats):
            launch()
        runtime.record(stop)
        runtime.synchronize()
        raw = runtime.read(handles[2], len(prefix) + m * n * 2 + len(GUARD))
        output = raw[len(prefix) : -len(GUARD)]
        actual = decode(output, dtype, (m, n))
        selected = actual[rows].astype(np.float64)
        rtol = (2**-10 if dtype == "f16" else 2**-7) / 2 + 1e-5
        require(np.isfinite(actual).all(), "nonfinite GEMM output")
        errors = (
            selected != rounded
            if args.data != "normal"
            else np.abs(selected - expected) > (0.001 + rtol * np.abs(expected))
        )
        if errors.any():
            positions = np.argwhere(errors)
            write_json(
                record.with_suffix(".failure.json"),
                {
                    "status": "failed",
                    "shape_mnk": [m, n, k],
                    "dtype": dtype,
                    "offset_elements": args.offset_elements,
                    "backend": args.backend,
                    "bad_elements": int(errors.sum()),
                    "max_abs_error": float(np.abs(selected - expected).max()),
                    "output_guards_unchanged": raw[: len(prefix)] == prefix
                    and raw[-len(GUARD) :] == GUARD,
                    "samples": [
                        {
                            "row": int(rows[i]),
                            "column": int(j),
                            "actual": float(selected[i, j]),
                            "fp64": float(expected[i, j]),
                            "rounded": float(rounded[i, j]),
                        }
                        for i, j in positions[:32]
                    ],
                },
            )
            record.with_suffix(".output.bin").write_bytes(output)
        require(not errors.any(), "GEMM numerical contract failed")
        require(
            raw[: len(prefix)] == prefix and raw[-len(GUARD) :] == GUARD,
            "GEMM output guard changed",
        )
        require(
            all(
                runtime.read(handle, len(prefix) + len(value) + len(GUARD))
                == prefix + value + GUARD
                for handle, value in zip(handles[:2], (a, b), strict=True)
            ),
            "GEMM input or guard changed",
        )
        elapsed = ct.c_float()
        runtime.call(
            "aclrtEventElapsedTime",
            [ct.POINTER(ct.c_float), ct.c_void_p, ct.c_void_p],
            ct.byref(elapsed),
            start,
            stop,
        )
        case = {
            "backend": args.backend,
            "shape_mnk": [m, n, k],
            "layout": "NN contiguous",
            "input_dtype": dtype,
            "output_dtype": dtype,
            "data": args.data,
            "scope": args.scope,
            "seed": args.seed,
            "oracle_rows": rows.tolist(),
            "checked_elements": int(expected.size),
            "max_abs_error": float(np.abs(selected - expected).max()),
            "rounded_mismatches": int(np.count_nonzero(selected != rounded)),
            "rtol": rtol,
            "atol": 0.001,
            "comparison": "exact rounded FP64"
            if args.data != "normal"
            else "FP64 with output rounding tolerance",
            "finite_output": True,
            "inputs_unchanged": True,
            "guards_unchanged": True,
            "pointer_alignment_mod32": [p.value % 32 for p in pointers],
            "offset_elements": args.offset_elements,
            "input_sha256": [sha(a), sha(b)],
            "output_sha256": sha(output),
            "warmup": args.warmup,
            "repeats": args.repeats,
            "stream_interval_us": elapsed.value * 1000 / args.repeats,
            "timing_scope": "ACL stream interval including host submission gaps; not kernel-only",
            "cache_policy": "same allocations reused, no flush",
            "generated_buffer_cache_policies": caches,
            "runtime_cache_alias": alias,
            "work": work_record(
                inventory,
                algorithm_counts={
                    "fma_flops": 2 * m * n * k,
                    "logical_input_bytes": len(a) + len(b),
                    "logical_output_bytes": m * n * 2,
                },
                algorithm_scope="dense NN b16 GEMM, two FLOPs per scalar FMA",
            ),
        }
        write_json(
            record,
            {
                "status": "passed",
                "runner_sha256": sha(Path(__file__).read_bytes()),
                "storage_oracle_sha256": sha(
                    (Path(__file__).parents[1] / "validation/dtypes.py").read_bytes()
                ),
                "numpy_version": np.__version__,
                "policy": policy,
                "artifact": metadata,
                "health": health,
                "hardware": hardware,
                "backend": identity,
                "cases": [case],
            },
        )
        print(
            json.dumps(
                {key: value for key, value in case.items() if key not in {"oracle_rows", "work"}}
            ),
            flush=True,
        )
    finally:
        if operator is not None:
            operator.close()
        runtime.close()


if __name__ == "__main__":
    main()
