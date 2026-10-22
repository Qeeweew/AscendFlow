"""Check the aligned teaching attention against independent FP32 NumPy math."""

import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path

import numpy as np

from ascendflow.device import ACL, health_snapshot
from ascendflow.diagnostics import require
from ascendflow.dialects.physical import BufferOp
from benchmarks.artifacts import read_artifact
from benchmarks.validation.dtypes import decode, encode

GUARD = b"\x5a" * 512


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--physical", type=int, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frozen = read_artifact(args.artifact)
    manifest = json.loads((args.artifact / "simple_attention.compile.json").read_text())
    library = args.artifact / "simple_attention.so"
    require(frozen.name == "simple_attention", "expected the teaching attention")
    require(
        hashlib.sha256(frozen.asc.encode()).hexdigest() == manifest["source_sha256"]
        and hashlib.sha256(library.read_bytes()).hexdigest() == manifest["shared_sha256"],
        "source/library differs from its published artifact",
    )
    # This runner qualifies the default H4/S256/D128 specialization.
    h, s, d, keys = 4, 256, 128, 128
    rng = np.random.default_rng(9102201)
    data = {
        name: (rng.standard_normal(shape) * 0.5).astype(np.float16).astype(np.float32)
        for name, shape in (("Q", (h, s, d)), ("K", (h, keys, d)), ("V", (h, keys, d)))
    }
    raw = {name: encode(value, "f16") for name, value in data.items()}
    shapes = {"O": (h, s, d), "Score": (h, 2, 64, keys), "P": (h, 2, 64, keys), "PV": (h, 2, 64, d)}
    sizes = {
        name: int(np.prod(shape)) * (4 if name in {"Score", "PV"} else 2)
        for name, shape in shapes.items()
    }
    external = {
        op.sym_name.data: (op.nbytes.data, op.ref.type.dtype.data)
        for op in frozen.plan.walk()
        if isinstance(op, BufferOp) and op.external.data
    }
    expected_storage = {
        **{name: (len(blob), "f16") for name, blob in raw.items()},
        **{
            name: (size, "f32" if name in {"Score", "PV"} else "f16")
            for name, size in sizes.items()
        },
    }
    require(external == expected_storage, "runner requires the default H4/S256/D128 artifact")
    signature = "void* stream, " + ", ".join(
        "void* arg_" + name for name in ("Q", "K", "V", "O", "Score", "P", "PV")
    )
    require(
        f'extern "C" int launch_simple_attention({signature})' in frozen.asc,
        "unexpected teaching attention launcher ABI",
    )
    record = {
        "status": "running",
        "shape": {"H": h, "Sq": s, "Sk": keys, "D": d},
        "rtol": 0.002,
        "atol": 0.0005,
        "reference": "independent dense FP32 NumPy QK, stable softmax, PV; no causal mask",
        "manifest": manifest,
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_ir_sha256": hashlib.sha256(
            (args.artifact / "source.xdsl").read_bytes()
        ).hexdigest(),
        "physical_ir_sha256": hashlib.sha256(frozen.plan_ir.encode()).hexdigest(),
        "cases": [],
    }
    runtime = None
    try:
        record["health"] = health_snapshot(args.physical, args.output)
        module = ct.CDLL(str(library.resolve()))
        launch = module.launch_simple_attention
        launch.argtypes, launch.restype = [ct.c_void_p] * 8, ct.c_int
        runtime = ACL()
        pointers = {name: runtime.allocate(blob + GUARD) for name, blob in raw.items()}
        pointers.update(
            {name: runtime.allocate(b"\xcd" * size + GUARD) for name, size in sizes.items()}
        )
        original_k = data["K"].copy()
        for case in ("random", "uniform"):
            if case == "uniform":
                data["K"] = np.zeros_like(original_k)
                raw["K"] = encode(data["K"], "f16")
                pointers["K"] = runtime.allocate(raw["K"] + GUARD)
            scores = (data["Q"] @ data["K"].transpose(0, 2, 1)) * np.float32(d**-0.5)
            probability = np.exp(scores - scores.max(axis=-1, keepdims=True))
            probability /= probability.sum(axis=-1, keepdims=True)
            expected = probability @ data["V"]
            for _ in range(3):
                require(
                    launch(
                        runtime.stream,
                        *(pointers[name] for name in ("Q", "K", "V", "O", "Score", "P", "PV")),
                    )
                    == 0,
                    "attention launch failed",
                )
                runtime.synchronize()
                for name, size in sizes.items():
                    blob = runtime.read(pointers[name], size + len(GUARD))
                    require(blob[size:] == GUARD, f"{name} bounds guard changed")
                    if name == "O":
                        actual = decode(blob[:size], "f16", shapes["O"])
                require(
                    np.allclose(actual, expected, rtol=record["rtol"], atol=record["atol"]),
                    f"{case}: output differs",
                )
            for name, blob in raw.items():
                require(
                    runtime.read(pointers[name], len(blob) + len(GUARD)) == blob + GUARD,
                    f"{name} input changed",
                )
            np.savez(args.output / f"{case}.npz", **data, actual=actual, expected=expected)
            record["cases"].append(
                {
                    "case": case,
                    "launches": 3,
                    "max_abs_error": float(np.max(np.abs(actual - expected))),
                    "input_sha256": {
                        name: hashlib.sha256(blob).hexdigest() for name, blob in raw.items()
                    },
                    "output_sha256": hashlib.sha256(encode(actual, "f16")).hexdigest(),
                    "input_and_guards": "unchanged",
                }
            )
        record["status"] = "passed"
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (args.output / "device.json").write_text(json.dumps(record, indent=2) + "\n")
        if runtime is not None:
            runtime.close()
    print(json.dumps({"status": record["status"], "cases": record["cases"]}), flush=True)


if __name__ == "__main__":
    main()
