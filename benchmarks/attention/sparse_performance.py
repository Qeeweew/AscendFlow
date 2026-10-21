"""One fully populated long-context case for msprof, without host timing claims."""

import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from ascendflow.device import ACL, health_snapshot
from ascendflow.diagnostics import require
from ascendflow.performance import hardware_snapshot
from benchmarks.attention.run_sparse import (
    ATOL,
    GUARD,
    ROOT,
    RTOL,
    artifact,
    check_gathered,
    policy_for,
    sources_for,
    workspace_shapes,
)
from benchmarks.references.sparse_mla_aclnn import SparseMlaReference
from benchmarks.validation.dtypes import decode, encode


def make_inputs(policy, context, seed):
    b, h, d, topk, page, ratio, maximum, window = (
        policy[key] for key in ("B", "H", "D", "TOPK", "PAGE", "RATIO", "MAX_SEQ", "WINDOW")
    )
    require(
        window <= context <= maximum and topk <= context // ratio,
        "case must populate every selected row",
    )
    rng = np.random.default_rng(seed)
    # Physical pages form disjoint request pools, with a real full-capacity
    # paged allocation. Only values that attention reads need random patches;
    # every remaining cache byte is explicitly initialized to zero.
    tables = np.stack(
        [rng.permutation(maximum // page) + batch * (maximum // page) for batch in range(b)]
    ).astype(np.int32)
    indices = np.stack(
        [np.sort(rng.choice(context // ratio, topk, replace=False)) for _ in range(b)]
    ).astype(np.int32)
    selected = rng.normal(0, 0.5, (b, window + topk, d)).astype(np.float32)
    selected = decode(encode(selected, "bf16"), "bf16", selected.shape)
    q = rng.normal(0, 0.5, (b, h, d)).astype(np.float32)
    q = decode(encode(q, "bf16"), "bf16", q.shape)
    arrays = {
        "Q": q,
        "Sinks": np.linspace(-2, 2, h, dtype=np.float32),
        "OriTable": tables,
        "CmpTable": tables.copy(),
        "Indices": indices,
        "Lengths": np.full(b, context, dtype=np.int32),
        "WindowLengths": np.full(b, window, dtype=np.int32),
        "Counts": np.full(b, topk, dtype=np.int32),
        "Cu": np.arange(b + 1, dtype=np.int32),
        "CmpLengths": np.full(b, context // ratio, dtype=np.int32),
        "Residual": np.full(b, context % ratio, dtype=np.int32),
    }
    patches = {"Ori": [], "Cmp": []}
    for batch in range(b):
        # Consecutive window rows are uploaded as one or two page fragments.
        window_positions = np.arange(context - window, context)
        for key, positions, values, block in (
            ("Ori", window_positions, selected[batch, :window], page),
            ("Cmp", indices[batch], selected[batch, window:], page // ratio),
        ):
            for position, value in zip(positions, values, strict=True):
                physical = int(tables[batch, position // block]) * block + int(position % block)
                patches[key].append((physical * d * 2, encode(value, "bf16")))
    return arrays, patches, selected


def zero_cache(runtime, nbytes, patches):
    pointer = runtime.allocate_uninitialized(nbytes + len(GUARD))
    runtime.call(
        "aclrtMemset",
        [ct.c_void_p, ct.c_size_t, ct.c_int32, ct.c_size_t],
        pointer,
        nbytes + len(GUARD),
        0,
        nbytes,
    )
    # This canonical sparse-patch identity describes every initialized byte,
    # including the zero background, without saving multi-GB cache replicas.
    identity = hashlib.sha256(str(nbytes).encode() + b"/zero-background/bf16/patches-v1")
    for offset, blob in sorted(patches):
        source = ct.create_string_buffer(blob)
        runtime.call(
            "aclrtMemcpy",
            [ct.c_void_p, ct.c_size_t, ct.c_void_p, ct.c_size_t, ct.c_int],
            ct.c_void_p(pointer.value + offset),
            len(blob),
            source,
            len(blob),
            1,
        )
        identity.update(offset.to_bytes(8, "little"))
        identity.update(blob)
    guard = ct.create_string_buffer(GUARD)
    runtime.call(
        "aclrtMemcpy",
        [ct.c_void_p, ct.c_size_t, ct.c_void_p, ct.c_size_t, ct.c_int],
        ct.c_void_p(pointer.value + nbytes),
        len(GUARD),
        guard,
        len(GUARD),
        1,
    )
    return pointer, identity.hexdigest()


def expected_output(arrays, selected, policy):
    result = np.empty_like(arrays["Q"])
    for batch, (q, kv) in enumerate(zip(arrays["Q"], selected, strict=True)):
        scores = (q.astype(np.float64) @ kv.astype(np.float64).T) * policy["SCALE"]
        maximum = np.maximum(scores.max(axis=1), arrays["Sinks"])
        weight = np.exp(scores - maximum[:, None])
        denominator = weight.sum(axis=1) + np.exp(arrays["Sinks"] - maximum)
        result[batch] = (weight @ kv.astype(np.float64)) / denominator[:, None]
    return decode(encode(result, "bf16"), "bf16", result.shape)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--reference-artifacts", type=Path)
    parser.add_argument("--backend", choices=("ascendflow", "vllm-ascend"), required=True)
    parser.add_argument("--physical", type=int, required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--head-tile", type=int, choices=(32, 64))
    parser.add_argument("--kv-parts", type=int)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--features", type=int, default=512)
    parser.add_argument("--topk", type=int, default=512)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--max-seq", type=int, default=131072)
    parser.add_argument("--context", type=int, default=65536)
    parser.add_argument("--ratio", type=int, choices=(1, 2), default=2)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--rotations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=4102026)
    parser.add_argument("--dtype", choices=("bf16",), default="bf16")
    args = parser.parse_args(argv)
    require(0 < args.rotations <= args.warmup + args.repeats, "every input rotation must run")
    args.output.mkdir(parents=True, exist_ok=True)
    policy = policy_for(args)
    record = {
        "status": "running",
        "backend": args.backend,
        "policy": policy,
        "context": args.context,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "seed": args.seed,
        "rotations": args.rotations,
        "cache_scope": "cyclic independent cache allocations; no cache flush or cold-HBM claim",
        "benchmark_sources_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__).resolve(),
                ROOT / "benchmarks/attention/run_sparse.py",
                ROOT / "benchmarks/references/sparse_mla_aclnn.py",
            )
        },
    }
    runtime, references, cases = None, [], []
    try:
        record["health"] = health_snapshot(args.physical, args.output)
        record["hardware"] = hardware_snapshot(args.physical)
        if args.backend == "ascendflow":
            require(args.artifacts is not None, "generated backend needs --artifacts")
            require(
                json.loads((args.artifacts / "policy.json").read_text()) == policy,
                "artifact policy differs",
            )
            kernels = {
                entry: artifact(args.artifacts, entry, name)
                for entry, name in sources_for(policy).items()
            }
            record["artifacts"] = {entry: kernel[3] for entry, kernel in kernels.items()}
            record["source_identity"] = json.loads(
                (args.artifacts / "source-identity.json").read_text()
            )
        else:
            require(args.reference_artifacts is not None, "official custom OPP package is required")
            os.environ["ASCEND_CUSTOM_OPP_PATH"] = str(args.reference_artifacts.resolve())
        runtime = ACL()
        shapes = workspace_shapes(policy)
        capacities = (
            {
                name: int(np.prod(shape)) * (2 if name in {"KV", "P"} else 4)
                for name, shape in shapes.items()
                if name != "O"
            }
            if args.backend == "ascendflow"
            else {}
        )
        scratch = {
            name: runtime.allocate(b"\xcd" * size + GUARD) for name, size in capacities.items()
        }
        sizes = {
            "Ori": policy["ORI_BLOCKS"] * policy["PAGE"] * policy["D"] * 2,
            "Cmp": policy["CMP_BLOCKS"] * (policy["PAGE"] // policy["RATIO"]) * policy["D"] * 2,
        }
        output_bytes = policy["B"] * policy["H"] * policy["D"] * 2
        for rotation in range(args.rotations):
            arrays, patches, selected = make_inputs(policy, args.context, args.seed + rotation)
            np.savez(args.output / f"inputs-{rotation}.npz", **arrays, selected=selected)
            raw = {
                key: encode(value, "bf16") if key == "Q" else value.tobytes()
                for key, value in arrays.items()
            }
            pointers = scratch | {key: runtime.allocate(blob + GUARD) for key, blob in raw.items()}
            cache_identity = {}
            for name, size in sizes.items():
                pointers[name], cache_identity[name] = zero_cache(runtime, size, patches[name])
            pointers["O"] = runtime.allocate(b"\xcd" * output_bytes + GUARD)
            identity = {
                "arguments": {key: hashlib.sha256(blob).hexdigest() for key, blob in raw.items()},
                "cache_zero_patch_sha256": cache_identity,
                "cache_bytes": sizes,
            }
            if args.backend == "ascendflow":

                def launch(pointers=pointers):
                    for _module, fn, names, _identity in kernels.values():
                        require(
                            fn(runtime.stream, *(pointers[key] for key in names)) == 0,
                            "generated launch failed",
                        )
            else:
                reference = SparseMlaReference(
                    runtime, pointers, policy, args.context, args.reference_artifacts
                )
                references.append(reference)
                launch = reference
            cases.append(
                {
                    "pointers": pointers,
                    "arrays": arrays,
                    "raw": raw,
                    "selected": selected,
                    "identity": identity,
                    "launch": launch,
                }
            )
        record["input_identity"] = [case["identity"] for case in cases]
        record["selected_kv_working_bytes"] = (
            args.rotations * policy["B"] * (policy["WINDOW"] + policy["TOPK"]) * policy["D"] * 2
        )
        record["allocated_cache_bytes"] = args.rotations * sum(sizes.values())
        record["launch_rotation"] = "invocation % rotations; includes warmup"
        if references:
            record["reference"] = [reference.identity() for reference in references]
        for invocation in range(args.warmup + args.repeats):
            cases[invocation % args.rotations]["launch"]()
        runtime.synchronize()
        errors, outputs = [], []
        for rotation, case in enumerate(cases):
            pointers, arrays, raw, selected = (
                case[key] for key in ("pointers", "arrays", "raw", "selected")
            )
            blob = runtime.read(pointers["O"], output_bytes + len(GUARD))
            require(blob[output_bytes:] == GUARD, "output guard changed")
            actual = decode(blob[:output_bytes], "bf16", (policy["B"], policy["H"], policy["D"]))
            expected = expected_output(arrays, selected, policy)
            np.savez(args.output / f"outputs-{rotation}.npz", actual=actual, expected=expected)
            require(
                np.allclose(actual, expected, rtol=RTOL, atol=ATOL),
                f"numerical mismatch: {np.max(np.abs(actual - expected))}",
            )
            errors.append(float(np.max(np.abs(actual - expected))))
            outputs.append(hashlib.sha256(blob[:output_bytes]).hexdigest())
            for name, data in raw.items():
                require(
                    runtime.read(pointers[name], len(data) + len(GUARD)) == data + GUARD,
                    f"{name} input changed",
                )
            for name, size in sizes.items():
                require(
                    runtime.read(ct.c_void_p(pointers[name].value + size), len(GUARD)) == GUARD,
                    f"{name} guard changed",
                )
        for name, size in capacities.items():
            require(
                runtime.read(ct.c_void_p(scratch[name].value + size), len(GUARD)) == GUARD,
                f"{name} guard changed",
            )
        if args.backend == "ascendflow":
            # Scratch is shared sequentially, so its retained tiles belong to
            # the final rotation. Every rotation's output is checked above.
            last = cases[(args.warmup + args.repeats - 1) % args.rotations]
            packed = decode(runtime.read(scratch["KV"], capacities["KV"]), "bf16", shapes["KV"])
            bias = np.frombuffer(
                runtime.read(scratch["Bias"], capacities["Bias"]), np.float32
            ).reshape(shapes["Bias"])
            expected_rows = np.zeros((policy["B"], policy["S"], policy["D"]), np.float32)
            expected_rows[:, : last["selected"].shape[1]] = last["selected"]
            expected_bias = np.full((policy["B"], policy["S"]), -np.inf, np.float32)
            expected_bias[:, : last["selected"].shape[1]] = 0
            check_gathered(
                packed,
                bias,
                expected_rows,
                expected_bias,
                policy,
                kernels["sparse_mla_decode"][3]["blocks"],
            )
        record.update(
            status="passed",
            max_abs_error=max(errors),
            per_rotation_max_abs_error=errors,
            rtol=RTOL,
            atol=ATOL,
            checked_metadata_and_guards="unchanged",
            output_sha256=outputs,
        )
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (args.output / "device.json").write_text(json.dumps(record, indent=2) + "\n")
        for reference in references:
            reference.close()
        if runtime is not None:
            runtime.close()
    print(
        json.dumps({key: record[key] for key in ("status", "backend", "context", "max_abs_error")}),
        flush=True,
    )


if __name__ == "__main__":
    main()
