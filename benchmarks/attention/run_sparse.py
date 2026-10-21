"""Build and validate the paged V4.1 sparse MLA decode example on 910B."""

import argparse
import ctypes as ct
import hashlib
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ascendflow.build import build_file
from ascendflow.device import ACL, health_snapshot
from ascendflow.diagnostics import require
from ascendflow.performance import hardware_snapshot
from benchmarks.artifacts import read_artifact
from benchmarks.attention.sparse_reference import CASES, gather_oracle, inputs, oracle
from benchmarks.validation.dtypes import decode, encode

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {"sparse_mla_decode": "decode"}
GUARD = b"\x5a" * 512
RTOL, ATOL = 0.02, 0.005


def digest(data):
    return hashlib.sha256(data).hexdigest()


def source_identity():
    files = [
        *sorted((ROOT / "src").rglob("*.py")),
        *sorted((ROOT / "examples/sparse_attention").glob("*.py")),
    ]
    return {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "python_sha256": {str(path.relative_to(ROOT)): digest(path.read_bytes()) for path in files},
    }


def policy_for(args):
    page, bk = 128, 512
    require(
        args.batch > 0 and args.heads > 0 and args.heads % 32 == 0, "complete head tiles required"
    )
    require(args.max_seq > 0 and args.max_seq % page == 0, "complete logical pages required")
    require(args.topk > 0 and args.window > 0, "positive topk and window capacities required")
    require(args.features > 0 and args.features % 128 == 0, "complete feature tiles required")
    require(args.repeats > 0 and args.warmup >= 0, "invalid repeat count")
    bm = getattr(args, "head_tile", None) or min(64, args.heads)
    require(args.heads % bm == 0, "head tile must divide heads")
    require(args.window == page, "CSA currently requires a 128-token window capacity")
    parts = getattr(args, "kv_parts", None) or 1
    require(parts == 1, "CSA baseline has one request owner")
    selected = args.window + ((args.topk + bk - 1) // bk) * bk
    require(selected <= 4096, "resident sparse softmax requires at most 4096 padded columns")
    return {
        "B": args.batch,
        "H": args.heads,
        "D": args.features,
        "PAGE": page,
        "RATIO": args.ratio,
        "MAX_SEQ": args.max_seq,
        "ORI_BLOCKS": args.batch * (args.max_seq // page),
        "CMP_BLOCKS": args.batch * (args.max_seq // page),
        "WINDOW": args.window,
        "TOPK": args.topk,
        "S": selected,
        "BM": bm,
        "VR": 16 if selected <= 1024 else 8,
        "PARTS": parts,
        "BK": bk,
        "C": 128,
        "DTYPE": args.dtype,
        "SCALE": args.features**-0.5,
    }


def sources_for(policy):
    return SOURCES


def workspace_shapes(policy):
    groups = policy["B"] * (policy["H"] // policy["BM"])
    return {
        "KV": (groups, 3, policy["S"] - policy["WINDOW"], policy["D"]),
        "Bias": (groups, 3, policy["S"] - policy["WINDOW"]),
        "O": (policy["B"], policy["H"], policy["D"]),
        "Score": (groups, 2, policy["BM"], policy["S"]),
        "P": (groups, 2, policy["BM"], policy["S"]),
        "PV": (groups, 2, policy["BM"], policy["D"]),
    }


def check_gathered(gathered, bias, expected_kv, expected_bias, policy, workers):
    owners = policy["B"] * (policy["H"] // policy["BM"])
    require(0 < workers <= owners, "invalid artifact worker count")
    for group in range(workers):
        retained = {}
        for request, owner in enumerate(range(group, owners, workers)):
            retained[request % 3] = owner // (policy["H"] // policy["BM"])
        for slot, batch in retained.items():
            selection = slice(policy["WINDOW"], policy["S"])
            require(
                np.array_equal(gathered[group, slot], expected_kv[batch, selection]),
                "gathered KV differs",
            )
            require(
                np.array_equal(bias[group, slot], expected_bias[batch, selection]),
                "gathered mask differs",
            )


def build(output, policy, shared):
    identity = source_identity()
    for entry, filename in sources_for(policy).items():
        keys = tuple(policy) if filename == "decode" else ("B", "H", "D", "BM", "PARTS", "DTYPE")
        build_file(
            ROOT / f"examples/sparse_attention/{filename}.py",
            output / filename,
            entry=entry,
            parameters={key: policy[key] for key in keys},
            shared=shared,
            timeout_s=300,
        )
    require(source_identity() == identity, "compiler/DSL source changed during this build")
    identity["built_utc"] = datetime.now(UTC).isoformat()
    (output / "source-identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    (output / "policy.json").write_text(json.dumps(policy, indent=2) + "\n")


def artifact(output, entry, filename):
    directory = output / filename
    frozen = read_artifact(directory)
    manifest = json.loads((directory / f"{entry}.compile.json").read_text())
    library = directory / f"{entry}.so"
    require(
        frozen.name == entry and digest(frozen.asc.encode()) == manifest["source_sha256"],
        "ASC identity changed",
    )
    require(digest(library.read_bytes()) == manifest["shared_sha256"], "library identity changed")
    signatures = re.findall(r'extern "C" int launch_\w+\(([^()]*)\)', frozen.asc)
    require(len(signatures) == 1, "expected one launcher")
    names = [
        parameter.strip().split()[-1].removeprefix("arg_") for parameter in signatures[0].split(",")
    ]
    require(names[0] == "stream", "unexpected launcher stream position")
    module = ct.CDLL(str(library.resolve()))
    launch = getattr(module, f"launch_{entry}")
    launch.argtypes, launch.restype = [ct.c_void_p] * len(names), ct.c_int
    identity = {
        "blocks": frozen.plan.body.block.first_op.blocks.data,
        "manifest": manifest,
        "source_ir_sha256": digest((directory / "source.xdsl").read_bytes()),
        "physical_ir_sha256": digest((directory / "after.xdsl").read_bytes()),
    }
    return module, launch, names[1:], identity


def run_device(args, policy):
    output = args.output.resolve()
    record = {"status": "running", "policy": policy, "rtol": RTOL, "atol": ATOL, "cases": []}
    record_path = output / "device.json"
    runtime = None
    try:
        record["health"] = health_snapshot(args.physical, output)
        record["hardware"] = hardware_snapshot(args.physical)
        record["source_identity"] = json.loads((output / "source-identity.json").read_text())
        kernels = {
            entry: artifact(output, entry, filename)
            for entry, filename in sources_for(policy).items()
        }
        record["artifacts"] = {entry: value[3] for entry, value in kernels.items()}
        runtime = ACL()
        dtype = policy["DTYPE"]
        shapes = workspace_shapes(policy)
        capacities = {
            name: int(np.prod(shape)) * (2 if name in {"KV", "O", "P"} else 4)
            for name, shape in shapes.items()
        }
        workspace = {
            name: runtime.allocate(b"\xcd" * size + GUARD) for name, size in capacities.items()
        }
        for case in args.cases:
            data = inputs(policy, case)
            directory = output / "validation" / case
            directory.mkdir(parents=True, exist_ok=True)
            if case == args.cases[0]:
                np.savez(output / "cache-inputs.npz", Ori=data["Ori"], Cmp=data["Cmp"])
            np.savez(
                directory / "inputs.npz",
                **{key: value for key, value in data.items() if key not in {"Ori", "Cmp"}},
            )
            raw = {
                key: encode(value, dtype) if key in {"Q", "Ori", "Cmp"} else value.tobytes()
                for key, value in data.items()
            }
            pointers = {
                **workspace,
                **{key: runtime.allocate(value + GUARD) for key, value in raw.items()},
            }

            def launch_pair(pointers=pointers):
                for entry in sources_for(policy):
                    _module, launch, names, _identity = kernels[entry]
                    require(
                        launch(runtime.stream, *(pointers[key] for key in names)) == 0,
                        f"{entry} launch failed",
                    )

            for _ in range(args.warmup):
                launch_pair()
            runtime.synchronize()
            start, end = runtime.event(), runtime.event()
            runtime.record(start)
            for _ in range(args.repeats):
                launch_pair()
            runtime.record(end)
            runtime.synchronize()
            elapsed = ct.c_float()
            runtime.call(
                "aclrtEventElapsedTime",
                [ct.POINTER(ct.c_float), ct.c_void_p, ct.c_void_p],
                ct.byref(elapsed),
                start,
                end,
            )
            results = {}
            for name, size in capacities.items():
                blob = runtime.read(pointers[name], size + len(GUARD))
                require(blob[size:] == GUARD, f"{case}: {name} bounds guard changed")
                results[name] = blob[:size]
            for name, blob in raw.items():
                require(
                    runtime.read(pointers[name], len(blob) + len(GUARD)) == blob + GUARD,
                    f"{case}: {name} input changed",
                )
            expected_kv, expected_bias = gather_oracle(data, policy)
            gathered = decode(results["KV"], dtype, shapes["KV"])
            gathered_bias = np.frombuffer(results["Bias"], dtype=np.float32).reshape(shapes["Bias"])
            check_gathered(
                gathered,
                gathered_bias,
                expected_kv,
                expected_bias,
                policy,
                kernels["sparse_mla_decode"][3]["blocks"],
            )
            actual = decode(results["O"], dtype, shapes["O"])
            expected = oracle(data, policy)
            np.savez(directory / "outputs.npz", actual=actual, expected=expected)
            require(
                np.allclose(actual, expected, rtol=RTOL, atol=ATOL),
                f"{case}: sparse attention differs; max_abs={np.max(np.abs(actual - expected))}",
            )
            row = {
                "case": case,
                "max_abs_error": float(np.max(np.abs(actual - expected))),
                "input_sha256": {key: digest(value) for key, value in raw.items()},
                "output_sha256": digest(results["O"]),
                "stream_interval_us": elapsed.value * 1000 / args.repeats,
                "timing_scope": "fused kernel plus host submission gaps; single probe, no performance qualification",
                "input_and_guards": "unchanged",
                "gather": "exact",
                "launch_pairs": args.repeats + args.warmup,
            }
            record["cases"].append(row)
            record_path.write_text(json.dumps(record, indent=2) + "\n")
            print(
                json.dumps(
                    {key: row[key] for key in ("case", "max_abs_error", "stream_interval_us")}
                ),
                flush=True,
            )
            # Reuse the same output/workspace and library across all lengths;
            # release only the current case's input allocations.
            for handle in runtime.allocations[len(workspace) :]:
                runtime.call("aclrtFree", [ct.c_void_p], handle)
            del runtime.allocations[len(workspace) :]
        record["status"] = "passed"
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        if runtime is not None:
            runtime.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--ir-only", action="store_true")
    parser.add_argument("--physical", type=int)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--head-tile", type=int, choices=(32, 64))
    parser.add_argument("--kv-parts", type=int)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--features", type=int, default=512)
    parser.add_argument("--topk", type=int, default=512)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--max-seq", type=int, default=4096)
    parser.add_argument("--ratio", type=int, choices=(1, 2), default=2)
    parser.add_argument("--dtype", choices=("bf16", "f16"), default="bf16")
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args(argv)
    policy = policy_for(args)
    require(not args.ir_only or args.physical is None, "IR-only does not produce a device library")
    if args.build:
        build(args.output.resolve(), policy, not args.ir_only)
    if args.physical is not None:
        require(
            json.loads((args.output / "policy.json").read_text()) == policy,
            "artifact specialization differs",
        )
        run_device(args, policy)
    elif not args.build:
        parser.error("select --build or --physical")


if __name__ == "__main__":
    main()
