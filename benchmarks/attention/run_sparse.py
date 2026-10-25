"""Build and validate paged V4.1 sparse MLA attention on 910B."""

import argparse
import ctypes as ct
import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ascendflow.build import build_file
from ascendflow.device import ACL, health_snapshot
from ascendflow.diagnostics import require
from ascendflow.performance import hardware_snapshot
from benchmarks.artifacts import read_artifact
from benchmarks.attention.sparse_reference import (
    CASES,
    gather_oracle,
    inputs,
    npu_reference_batches,
    selected_rows,
)
from benchmarks.references.sparse_mla_aclnn import SparseMlaReference
from benchmarks.validation.dtypes import decode, encode

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {"sparse_mla": "attention"}
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
    query_lens = getattr(args, "query_lens", None)
    if query_lens is None:
        query_lens = [getattr(args, "query_tokens", 4)] * args.batch
    require(
        len(query_lens) == args.batch and all(0 <= n <= args.max_seq for n in query_lens),
        "query lengths must have B entries, each within [0,MAX_SEQ]",
    )
    require(getattr(args, "groups", 20) > 0, "worker count must be positive")
    require(sum(query_lens) > 0, "TND must contain at least one query token")
    return {
        "B": args.batch,
        "T": sum(query_lens),
        "GROUPS": getattr(args, "groups", 20),
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


def workspace_shapes(policy):
    groups = min(policy["GROUPS"], policy["T"] * (policy["H"] // policy["BM"]))
    return {
        "KV": (groups, 3, policy["S"] - policy["WINDOW"], policy["D"]),
        "Bias": (groups, 3, policy["S"] - policy["WINDOW"]),
        "attention_out": (policy["T"], policy["H"], policy["D"]),
        "Score": (groups, 2, policy["BM"], policy["S"]),
        "P": (groups, 2, policy["BM"], policy["S"]),
        "PV": (groups, 2, policy["BM"], policy["D"]),
    }


def check_gathered(gathered, bias, expected, policy, workers):
    owners = policy["T"] * (policy["H"] // policy["BM"])
    require(0 < workers <= owners, "invalid artifact worker count")
    for group in range(workers):
        retained = {}
        for request, owner in enumerate(range(group, owners, workers)):
            retained[request % 3] = owner // (policy["H"] // policy["BM"])
        for slot, batch in retained.items():
            expected_kv, expected_bias = expected(batch)
            selection = slice(policy["WINDOW"], policy["S"])
            active = np.isfinite(expected_bias[selection])
            require(
                np.array_equal(gathered[group, slot, active], expected_kv[selection][active]),
                "gathered KV differs",
            )
            require(np.isfinite(gathered[group, slot]).all(), "gathered KV must remain finite")
            require(
                np.array_equal(bias[group, slot], expected_bias[selection]),
                "gathered mask differs",
            )


def build(output, policy, shared):
    identity = source_identity()
    for entry, filename in SOURCES.items():
        build_file(
            ROOT / f"examples/sparse_attention/{filename}.py",
            output / filename,
            entry=entry,
            parameters=policy,
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
    names = [tensor["name"] for tensor in frozen.report["abi"]["tensors"]]
    identity = {
        "blocks": frozen.plan.body.block.first_op.blocks.data,
        "manifest": manifest,
        "source_ir_sha256": digest((directory / "source.xdsl").read_bytes()),
        "physical_ir_sha256": digest((directory / "after.xdsl").read_bytes()),
        "workspace": frozen.report["workspace"],
    }
    return library, names, identity


def run_device(args, policy):
    output = args.output.resolve()
    record = {"status": "running", "policy": policy, "rtol": RTOL, "atol": ATOL, "cases": []}
    record_path = output / "device.json"
    runtime = None
    reference = None
    try:
        record["health"] = health_snapshot(args.physical, output)
        record["hardware"] = hardware_snapshot(args.physical)
        record["source_identity"] = json.loads((output / "source-identity.json").read_text())
        kernels = {entry: artifact(output, entry, filename) for entry, filename in SOURCES.items()}
        record["artifacts"] = {entry: value[2] for entry, value in kernels.items()}
        if args.reference_artifacts is not None:
            os.environ["ASCEND_CUSTOM_OPP_PATH"] = str(args.reference_artifacts.resolve())
        runtime = ACL()
        dtype = policy["DTYPE"]
        shapes = workspace_shapes(policy)
        capacities = {
            name: int(np.prod(shape)) * (2 if name in {"KV", "attention_out", "P"} else 4)
            for name, shape in shapes.items()
        }
        bindings = {
            entry: runtime.kernel(library, entry)
            for entry, (library, _names, _identity) in kernels.items()
        }
        workspace = {
            "attention_out": runtime.allocate(b"\xcd" * capacities["attention_out"] + GUARD)
        }
        arena = bindings["sparse_mla"]
        offset = 0
        for allocation in kernels["sparse_mla"][2]["workspace"]["allocations"]:
            name, nbytes = allocation["name"], allocation["byte_coefficient"]
            require(
                not allocation["dimensions"] and nbytes == capacities[name],
                "workspace layout differs",
            )
            workspace[name] = ct.c_void_p(arena.workspace.value + offset)
            offset += (nbytes + 511) // 512 * 512
        require(offset == arena.workspace_nbytes, "workspace total differs")
        record["workspace_bytes"] = arena.workspace_nbytes
        retained_handles = len(runtime.allocations)
        for case in args.cases:
            data = inputs(policy, case, args.query_lens)
            directory = output / "validation" / case
            directory.mkdir(parents=True, exist_ok=True)
            raw = {
                key: encode(value, dtype) if key in {"q", "ori_kv", "cmp_kv"} else value.tobytes()
                for key, value in data.items()
            }
            pointers = {
                **workspace,
                **{key: runtime.allocate(value + GUARD) for key, value in raw.items()},
            }

            def launch_attention(pointers=pointers):
                for entry in SOURCES:
                    _library, names, _identity = kernels[entry]
                    bindings[entry](*(pointers[key] for key in names))

            for _ in range(args.warmup):
                launch_attention()
            runtime.synchronize()
            start, end = runtime.event(), runtime.event()
            runtime.record(start)
            for _ in range(args.repeats):
                launch_attention()
            runtime.record(end)
            runtime.synchronize()
            elapsed_us = runtime.elapsed_us(start, end)
            results = {}
            for name, size in capacities.items():
                blob = runtime.read(
                    pointers[name], size + (len(GUARD) if name == "attention_out" else 0)
                )
                if name == "attention_out":
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
                lambda token, kv=expected_kv, mask=expected_bias: (kv[token], mask[token]),
                policy,
                kernels["sparse_mla"][2]["blocks"],
            )
            actual = decode(results["attention_out"], dtype, shapes["attention_out"])
            expected = np.empty_like(actual)
            for first, chunk in npu_reference_batches(
                data["q"],
                data["sinks"],
                lambda token, data=data: selected_rows(data, policy, token)[0],
                policy,
            ):
                expected[first : first + len(chunk)] = chunk
            require(
                np.allclose(actual, expected, rtol=RTOL, atol=ATOL),
                f"{case}: sparse attention differs; max_abs={np.max(np.abs(actual - expected))}",
            )
            official = None
            if args.reference_artifacts is not None:
                size = capacities["attention_out"]
                pointer = runtime.allocate(b"\xcd" * size + GUARD)
                reference = SparseMlaReference(
                    runtime,
                    pointers | {"attention_out": pointer},
                    policy,
                    max(1, int(data["seqused_ori_kv"].max())),
                    args.reference_artifacts,
                    query_lens=np.diff(data["cu_seqlens_q"]),
                )
                reference()
                runtime.synchronize()
                blob = runtime.read(pointer, size + len(GUARD))
                require(blob[size:] == GUARD, "official output guard changed")
                official = decode(blob[:size], dtype, shapes["attention_out"])
                require(
                    np.allclose(official, expected, rtol=RTOL, atol=ATOL),
                    f"{case}: official differs from independent oracle",
                )
                reference_record = reference.identity()
                reference.close()
                reference = None
            row = {
                "case": case,
                "max_abs_error": float(np.max(np.abs(actual - expected))),
                "input_sha256": {key: digest(value) for key, value in raw.items()},
                "output_sha256": digest(results["attention_out"]),
                "stream_interval_us": elapsed_us / args.repeats,
                "timing_scope": "fused kernel plus host submission gaps; single probe, no performance qualification",
                "input_and_guards": "unchanged",
                "gather": "exact",
                "launches": args.repeats + args.warmup,
                "numerical_reference": "NPU FP32 QK/softmax/PV; HF32 disabled",
            }
            if official is not None:
                row["official"] = reference_record
                row["official_max_abs_error"] = float(np.max(np.abs(official - expected)))
                row["max_abs_difference_from_official"] = float(np.max(np.abs(actual - official)))
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
            for handle in runtime.allocations[retained_handles:]:
                runtime.call("aclrtFree", [ct.c_void_p], handle)
            del runtime.allocations[retained_handles:]
        record["status"] = "passed"
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        if runtime is not None:
            if reference is not None:
                reference.close()
            runtime.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--ir-only", action="store_true")
    parser.add_argument("--physical", type=int)
    parser.add_argument(
        "--reference-artifacts",
        type=Path,
        help="also check the official operator on identical inputs",
    )
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument(
        "--query-tokens",
        type=int,
        default=4,
        help="query tokens per request; B remains the request count",
    )
    parser.add_argument(
        "--query-lens",
        nargs="+",
        type=int,
        help="runtime query lengths, B entries; overrides --query-tokens",
    )
    parser.add_argument("--groups", type=int, default=20)
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
