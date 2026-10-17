"""Contiguous f16 NT GEMM versus installed ACLNN, with frozen build identity.

One process measures one backend and one shape. Device-task profiling is
external; ACL stream intervals include host submission gaps.
"""

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
from benchmarks.work import structure_inventory, work_record

EXAMPLE = Path(__file__).resolve().parents[2] / "examples/gemm/dense.py"
GUARD = b"\x5a" * 512


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def load_contract(directory):
    metadata = json.loads((directory / "benchmark.json").read_text())
    for name, digest in metadata["files"].items():
        require(sha((directory / name).read_bytes()) == digest, f"GEMM artifact changed: {name}")
    for name, digest in metadata.get("compiler_files", {}).items():
        require(
            sha((directory / "compiler-source" / name).read_bytes()) == digest,
            f"compiler snapshot changed: {name}",
        )
    artifact = read_artifact(directory)
    manifest = json.loads((directory / f"{artifact.name}.compile.json").read_text())
    source = verified_source(manifest)
    require(
        source == directory / f"{artifact.name}.asc"
        and manifest["source_sha256"] == metadata["files"][f"{artifact.name}.asc"],
        "GEMM library does not belong to these IR/ASC snapshots",
    )
    require(manifest["target"] == "dav-2201", "GEMM target differs")
    policy = metadata["policy"]
    m, n, k = (policy[key] for key in ("M", "N", "K"))
    tensors = {
        op.sym_name.data: tuple(size.data for size in op.ref.type.shape)
        for op in parse((directory / "source.xdsl").read_text()).walk()
        if isinstance(op, AllocationOp) and op.ref.type.memory.data == "GM"
    }
    require(tensors == {"A": (m, k), "Bt": (n, k), "C": (m, n)}, "GEMM source shapes differ")
    externals = {
        op.sym_name.data: (op.nbytes.data, op.ref.type.dtype.data)
        for op in artifact.plan.walk()
        if isinstance(op, BufferOp) and op.external.data
    }
    require(
        externals == {"A": (m * k * 2, "f16"), "Bt": (n * k * 2, "f16"), "C": (m * n * 4, "f32")},
        "GEMM shape/dtype differs from actual buffers",
    )
    inventory = structure_inventory(artifact.plan)
    require(inventory["groups"] == policy["GROUPS"], "GEMM grid differs")
    signature = (
        'extern "C" int launch_dense_gemm(void* stream, void* arg_A, void* arg_Bt, void* arg_C)'
    )
    require(signature in artifact.asc, "GEMM launcher ABI changed")
    return metadata, artifact, manifest, inventory


def inputs_and_oracle(m, n, k, *, seed, data, scope):
    import numpy as np

    rng = np.random.default_rng(seed)
    if data == "exact":
        require(k <= 65536, "exact GEMM oracle bound exceeded")
        a = (rng.integers(-2, 3, size=(m, k)).astype(np.float32) / 8).astype(np.float16)
        b = (rng.integers(-2, 3, size=(n, k)).astype(np.float32) / 8).astype(np.float16)
    else:
        a = rng.normal(0, 0.5, (m, k)).astype(np.float16)
        b = rng.normal(0, 0.5, (n, k)).astype(np.float16)
    rows = (
        np.arange(m)
        if scope == "full"
        else np.unique(
            [row for start in range(0, m, 128) for row in range(start, min(start + 6, m))] + [m - 1]
        )
    )
    # Independent FP64 oracle; no production packing or Cube recurrence used.
    expected = a[rows].astype(np.float64) @ b.astype(np.float64).T
    return a.tobytes(), b.tobytes(), rows, expected


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
        "groups": 20,
    }.items():
        parser.add_argument("--" + key, type=int, default=default)
    parser.add_argument("--a-slots", type=int, default=2)
    parser.add_argument("--b-slots", type=int, default=2)
    parser.add_argument("--c-slots", type=int, default=1)
    parser.add_argument("--prefetch", type=int, default=1)
    parser.add_argument("--col-parts", type=int, default=2)
    parser.add_argument("--output-cache", choices=["auto", "normal", "bypass"], default="auto")
    parser.add_argument("--backend", choices=["generated", "aclnn"], default="generated")
    parser.add_argument("--physical", type=int)
    parser.add_argument("--record", type=Path)
    parser.add_argument("--scope", choices=["full", "sampled"], default="sampled")
    parser.add_argument("--data", choices=["normal", "exact"], default="normal")
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    directory = args.output.resolve()
    if args.build:
        policy = {
            key: getattr(args, key.lower())
            for key in (
                "M",
                "N",
                "K",
                "BM",
                "BN",
                "BK",
                "CK",
                "A_SLOTS",
                "B_SLOTS",
                "C_SLOTS",
                "PREFETCH",
                "COL_PARTS",
                "GROUPS",
            )
        }
        policy.update(OUTPUT_CACHE=args.output_cache, L2_BYTES=201326592)
        require(1 <= policy["GROUPS"] <= 20, "invalid physical group count")
        root = EXAMPLE.parents[2]
        definition = EXAMPLE.read_bytes()
        compiler = {
            str(path.relative_to(root)): path.read_bytes()
            for path in sorted((root / "src").rglob("*.py"))
        }
        build_file(
            EXAMPLE, directory, entry="dense_gemm", parameters=policy, shared=True, timeout_s=300
        )
        require(
            EXAMPLE.read_bytes() == definition
            and all((root / name).read_bytes() == data for name, data in compiler.items()),
            "GEMM source changed during the build",
        )
        (directory / "definition.py").write_bytes(definition)
        for name, data in compiler.items():
            target = directory / "compiler-source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        (directory / "source.diff").write_bytes(
            subprocess.check_output(["git", "diff", "HEAD"], cwd=root)
        )
        metadata = {
            "policy": policy,
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "compiler_files": {name: sha(data) for name, data in compiler.items()},
            "files": {
                name: sha((directory / name).read_bytes())
                for name in (
                    "definition.py",
                    "source.xdsl",
                    "before.xdsl",
                    "after.xdsl",
                    "dense_gemm.asc",
                )
            },
        }
        write_json(directory / "benchmark.json", metadata)
    metadata, artifact, manifest, inventory = load_contract(directory)
    if args.physical is None:
        print(json.dumps({"policy": metadata["policy"], "compilation": artifact.timings}))
        return
    require(args.warmup >= 0 and args.repeats > 0, "invalid launch counts")
    record = (args.record or directory / f"device-{args.backend}.json").resolve()
    health = health_snapshot(args.physical, record.parent)
    hardware = hardware_snapshot(args.physical)
    if metadata["policy"].get("OUTPUT_CACHE") == "auto":
        require(
            metadata["policy"]["L2_BYTES"] == hardware["l2_bytes"],
            "GEMM cache policy uses a different L2 capacity",
        )
    m, n, k = (metadata["policy"][key] for key in ("M", "N", "K"))
    a, b, rows, expected = inputs_and_oracle(
        m, n, k, seed=args.seed, data=args.data, scope=args.scope
    )
    import numpy as np

    runtime = ACL()
    operator = None
    try:
        caches = {
            op.sym_name.data: op.cache.data if op.cache is not None else "normal"
            for op in artifact.plan.walk()
            if isinstance(op, BufferOp) and op.external.data
        }
        alias = None
        if args.backend == "generated" and "bypass" in caches.values():
            rts = ct.CDLL(str(CANN / "lib64/libruntime.so"))
            get_offset = rts.rtGetL2CacheOffset
            get_offset.argtypes, get_offset.restype = (
                [ct.c_uint32, ct.POINTER(ct.c_uint64)],
                ct.c_int32,
            )
            value = ct.c_uint64()
            require(
                get_offset(0, ct.byref(value)) == 0 and value.value > 0,
                "cache bypass alias is unavailable",
            )
            alias = {
                "offset": value.value,
                "runtime_library": str((CANN / "lib64/libruntime.so").resolve()),
                "runtime_sha256": sha((CANN / "lib64/libruntime.so").read_bytes()),
            }
        ap, bp, cp = (
            runtime.allocate(value)
            for value in (a + GUARD, b + GUARD, b"\xcd" * (m * n * 4) + GUARD)
        )
        if args.backend == "generated":
            library = directory / "dense_gemm.so"
            module = ct.CDLL(str(library))
            launcher = module.launch_dense_gemm
            launcher.argtypes, launcher.restype = [ct.c_void_p] * 4, ct.c_int

            def launch():
                status = launcher(runtime.stream, ap, bp, cp)
                require(status == 0, f"GEMM launch rejected: {status}")

            identity = {
                "manifest": manifest,
                "library": str(library),
                "shared_sha256": sha(library.read_bytes()),
            }
        else:
            operator = GemmReference(runtime, ap, bp, cp, m, n, k)
            launch = operator
            identity = {
                "interface": operator.interface,
                "adapter_sha256": sha(
                    (Path(__file__).parents[1] / "references/aclnn.py").read_bytes()
                ),
                "libraries": {
                    name: {
                        "path": str((CANN / "lib64" / name).resolve()),
                        "sha256": sha((CANN / "lib64" / name).read_bytes()),
                    }
                    for name in ("libopapi.so", "libnnopbase.so")
                },
                "workspace_bytes": operator.workspace_size.value,
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
        raw = runtime.read(cp, m * n * 4 + len(GUARD))
        actual = np.frombuffer(raw[: m * n * 4], dtype=np.float32).reshape(m, n)
        require(np.isfinite(actual).all(), "nonfinite GEMM output")
        selected = actual[rows].astype(np.float64)
        require(
            np.array_equal(selected, expected)
            if args.data == "exact"
            else np.allclose(selected, expected, rtol=0.001, atol=0.001),
            "GEMM numerical contract failed",
        )
        require(
            raw[m * n * 4 :] == GUARD
            and runtime.read(ap, len(a) + len(GUARD)) == a + GUARD
            and runtime.read(bp, len(b) + len(GUARD)) == b + GUARD,
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
            "data": args.data,
            "comparison": "exact" if args.data == "exact" else "rtol_atol",
            "seed": args.seed,
            "scope": args.scope,
            "oracle_rows": rows.tolist(),
            "checked_elements": int(expected.size),
            "max_abs_error": float(np.abs(selected - expected).max()),
            "rtol": 0.001,
            "atol": 0.001,
            "finite_output": True,
            "guards_unchanged": True,
            "inputs_unchanged": True,
            "input_sha256": [sha(a), sha(b)],
            "output_sha256": sha(raw[: m * n * 4]),
            "warmup": args.warmup,
            "repeats": args.repeats,
            "stream_interval_us": elapsed.value * 1000 / args.repeats,
            "timing_scope": "ACL stream event interval including host submission gaps; not kernel-only",
            "cache_policy": "same contiguous allocations reused; no flush",
            "generated_buffer_cache_policies": caches,
            "runtime_cache_alias": alias,
            "work": work_record(
                inventory,
                algorithm_counts={
                    "fma_flops": 2 * m * n * k,
                    "logical_input_bytes": len(a) + len(b),
                    "logical_output_bytes": m * n * 4,
                },
                algorithm_scope="dense NT GEMM; two FLOPs per scalar FMA",
            ),
        }
        write_json(
            record,
            {
                "status": "passed",
                "runner_sha256": sha(Path(__file__).read_bytes()),
                "policy": metadata["policy"],
                "artifact": metadata,
                "health": health,
                "hardware": hardware,
                "backend": identity,
                "cases": [case],
            },
        )
        print(
            json.dumps(
                {key: value for key, value in case.items() if key not in {"work", "oracle_rows"}}
            ),
            flush=True,
        )
    finally:
        if operator is not None:
            operator.close()
        runtime.close()


if __name__ == "__main__":
    main()
