"""One fully populated long-context case for msprof, without host timing claims."""

import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
from time import monotonic

import numpy as np

from ascendflow.device import ACL, health_snapshot
from ascendflow.diagnostics import require
from ascendflow.performance import hardware_snapshot
from benchmarks.attention.run_sparse import (
    ATOL,
    GUARD,
    ROOT,
    RTOL,
    SOURCES,
    artifact,
    check_gathered,
    policy_for,
    workspace_shapes,
)
from benchmarks.attention.sparse_reference import npu_reference_batches
from benchmarks.references.sparse_mla_aclnn import SparseMlaReference
from benchmarks.validation.dtypes import decode, encode


class ReferenceRows:
    """Retain shared cache rows once; select a single query on demand."""

    def __init__(self, arrays, planes, policy):
        self.arrays, self.planes, self.policy = arrays, planes, policy

    def split(self, token):
        cu = self.arrays["cu_seqlens_q"]
        batch = int(np.searchsorted(cu, token, side="right")) - 1
        right = int(self.arrays["seqused_ori_kv"][batch]) - int(cu[batch + 1]) + token + 1
        positions = np.arange(max(0, right - self.policy["WINDOW"]), right)
        indices = self.arrays["cmp_sparse_indices"][token, 0]
        indices = indices[indices >= 0]
        w_positions, w_values = self.planes[batch]["ori_kv"]
        c_positions, c_values = self.planes[batch]["cmp_kv"]
        return (
            w_values[np.searchsorted(w_positions, positions)],
            c_values[np.searchsorted(c_positions, indices)],
        )

    def rows(self, token):
        return np.concatenate(self.split(token))

    def gathered(self, token):
        window, compressed = self.split(token)
        rows = np.zeros((self.policy["S"], self.policy["D"]), np.float32)
        bias = np.full(self.policy["S"], -np.inf, np.float32)
        first = self.policy["WINDOW"]
        rows[: len(window)] = window
        rows[first : first + len(compressed)] = compressed
        bias[: len(window)] = 0
        bias[first : first + len(compressed)] = 0
        return rows, bias


def make_inputs(policy, context, seed, query_lens=None):
    from benchmarks.attention.sparse_reference import query_counts

    b, t, h, d, topk, page, ratio, maximum, window = (
        policy[key] for key in ("B", "T", "H", "D", "TOPK", "PAGE", "RATIO", "MAX_SEQ", "WINDOW")
    )
    counts = query_counts(policy) if query_lens is None else np.array(query_lens, np.int32)
    require(len(counts) == b and counts.sum() == t, "query lengths differ")
    require(int(counts.max()) <= context <= maximum, "context must contain the query tokens")
    rng = np.random.default_rng(seed)
    tables = np.stack(
        [rng.permutation(maximum // page) + batch * (maximum // page) for batch in range(b)]
    ).astype(np.int32)
    cu = np.r_[0, counts.cumsum()].astype(np.int32)
    indices = np.full((t, 1, topk), -1, dtype=np.int32)
    valid = np.zeros((t, window + topk), dtype=bool)
    for batch in range(b):
        for offset, token in enumerate(range(cu[batch], cu[batch + 1])):
            right = context - int(counts[batch]) + offset + 1
            count = min(topk, right // ratio)
            indices[token, 0, :count] = np.sort(rng.choice(right // ratio, count, replace=False))
            valid[token, : min(window, right)] = True
            valid[token, window : window + count] = True
    q = rng.normal(0, 0.5, (t, h, d)).astype(np.float32)
    q = decode(encode(q, "bf16"), "bf16", q.shape)
    arrays = {
        "q": q,
        "sinks": np.linspace(-2, 2, h, dtype=np.float32),
        "ori_block_table": tables,
        "cmp_block_table": tables.copy(),
        "cmp_sparse_indices": indices,
        "seqused_ori_kv": np.full(b, context, dtype=np.int32),
        "ori_topk_length": np.zeros((t, 1), dtype=np.int32),
        "cu_seqlens_q": cu,
        "seqused_cmp_kv": np.full(b, context // ratio, dtype=np.int32),
        "cmp_residual_kv": np.full(b, context % ratio, dtype=np.int32),
    }
    patches = {"ori_kv": [], "cmp_kv": []}
    planes = [{} for _ in range(b)]
    for batch in range(b):
        if not counts[batch]:
            continue
        window_union = np.arange(max(0, context - int(counts[batch]) + 1 - window), context)
        compressed_union = np.unique(indices[cu[batch] : cu[batch + 1]])
        compressed_union = compressed_union[compressed_union >= 0]
        for key, positions, block in (
            ("ori_kv", window_union, page),
            ("cmp_kv", compressed_union, page // ratio),
        ):
            values = rng.normal(0, 0.5, (len(positions), d)).astype(np.float32)
            values = decode(encode(values, "bf16"), "bf16", values.shape)
            planes[batch][key] = positions, values
            # Overlapping windows/indices within a request refer to one cache
            # value. No per-query patch may overwrite a peer query's KV.
            for position, value in zip(positions, values, strict=True):
                physical = int(tables[batch, position // block]) * block + int(position % block)
                patches[key].append((physical * d * 2, encode(value, "bf16")))
    return arrays, patches, ReferenceRows(arrays, planes, policy), valid


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
    # Prefill can touch millions of rows. Stage nearby patches together;
    # holes keep the same zero background, outside the measured kernel.
    transfer_start, transfer = 0, bytearray()

    def publish():
        source = ct.create_string_buffer(bytes(transfer))
        runtime.call(
            "aclrtMemcpy",
            [ct.c_void_p, ct.c_size_t, ct.c_void_p, ct.c_size_t, ct.c_int],
            ct.c_void_p(pointer.value + transfer_start),
            len(transfer),
            source,
            len(transfer),
            1,
        )

    for offset, blob in sorted(patches):
        if not transfer or offset + len(blob) > transfer_start + len(transfer):
            if transfer:
                publish()
            transfer_start = offset
            transfer = bytearray(min(max(1024 * 1024, len(blob)), nbytes - offset))
        transfer[offset - transfer_start : offset - transfer_start + len(blob)] = blob
        identity.update(offset.to_bytes(8, "little"))
        identity.update(blob)
    if transfer:
        publish()
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--reference-artifacts", type=Path)
    parser.add_argument("--backend", choices=("ascendflow", "vllm-ascend"), required=True)
    parser.add_argument("--physical", type=int, required=True)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--query-tokens", type=int, default=4)
    parser.add_argument("--query-lens", type=int, nargs="+")
    parser.add_argument("--groups", type=int, default=20)
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
    parser.add_argument("--cache-evict-artifacts", type=Path)
    parser.add_argument("--seed", type=int, default=4102026)
    parser.add_argument("--dtype", choices=("bf16",), default="bf16")
    args = parser.parse_args(argv)
    started = monotonic()
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
        "input_encoding": "rounded query/metadata plus BF16 byte-offset patches over zero caches",
        "benchmark_sources_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__).resolve(),
                ROOT / "benchmarks/attention/run_sparse.py",
                ROOT / "benchmarks/attention/sparse_reference.py",
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
                entry: artifact(args.artifacts, entry, name) for entry, name in SOURCES.items()
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
                if name != "attention_out"
            }
            if args.backend == "ascendflow"
            else {}
        )
        scratch = {
            name: runtime.allocate(b"\xcd" * size + GUARD) for name, size in capacities.items()
        }
        sizes = {
            "ori_kv": policy["ORI_BLOCKS"] * policy["PAGE"] * policy["D"] * 2,
            "cmp_kv": policy["CMP_BLOCKS"] * (policy["PAGE"] // policy["RATIO"]) * policy["D"] * 2,
        }
        output_bytes = policy["T"] * policy["H"] * policy["D"] * 2
        for rotation in range(args.rotations):
            arrays, patches, selected, valid = make_inputs(
                policy, args.context, args.seed + rotation, args.query_lens
            )
            # Capture real cache bytes once, instead of duplicating the same
            # KV row in every prefill query's selected matrix on disk.
            snapshot = dict(arrays)
            snapshot["selected_valid"] = valid
            for name, values in patches.items():
                snapshot[name + "_patch_offsets"] = np.array([p for p, _ in values], np.uint64)
                snapshot[name + "_patch_bf16"] = np.frombuffer(
                    b"".join(blob for _, blob in values), dtype="<u2"
                ).reshape(-1, policy["D"])
            np.savez(args.output / f"inputs-{rotation}.npz", **snapshot)
            raw = {
                key: encode(value, "bf16") if key == "q" else value.tobytes()
                for key, value in arrays.items()
            }
            pointers = scratch | {key: runtime.allocate(blob + GUARD) for key, blob in raw.items()}
            cache_identity = {}
            for name, size in sizes.items():
                pointers[name], cache_identity[name] = zero_cache(runtime, size, patches[name])
            pointers["attention_out"] = runtime.allocate(b"\xcd" * output_bytes + GUARD)
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
                    runtime,
                    pointers,
                    policy,
                    args.context,
                    args.reference_artifacts,
                    query_lens=np.diff(arrays["cu_seqlens_q"]),
                )
                references.append(reference)
                launch = reference
            cases.append(
                {
                    "pointers": pointers,
                    "arrays": arrays,
                    "raw": raw,
                    "selected": selected,
                    "valid": valid,
                    "identity": identity,
                    "selected_bytes": sum(
                        sum(len(blob) for _offset, blob in values) for values in patches.values()
                    ),
                    "launch": launch,
                }
            )
        record["input_identity"] = [case["identity"] for case in cases]
        record["selected_kv_logical_bytes"] = (
            sum(int(case["valid"].sum()) for case in cases) * policy["D"] * 2
        )
        record["selected_kv_working_bytes"] = sum(case["selected_bytes"] for case in cases)
        record["allocated_cache_bytes"] = args.rotations * sum(sizes.values())
        record["launch_rotation"] = "invocation % rotations; includes warmup"
        record["preparation_seconds"] = monotonic() - started
        print(
            json.dumps({"phase": "prepared", "seconds": record["preparation_seconds"]}), flush=True
        )
        if references:
            record["reference"] = [reference.identity() for reference in references]
        eviction = None
        if args.cache_evict_artifacts is not None:
            from benchmarks.cache import CacheEviction

            eviction = CacheEviction(
                runtime, args.cache_evict_artifacts, record["hardware"]["l2_bytes"]
            )
            record["cache_scope"] = "disjoint full-buffer L2 eviction before every attention launch"
            record["cache_eviction"] = eviction.identity
        for invocation in range(args.warmup + args.repeats):
            if eviction is not None:
                eviction()
            cases[invocation % args.rotations]["launch"]()
        runtime.synchronize()
        if eviction is not None:
            record["cache_eviction"]["launches"] = eviction.calls
        errors, outputs = [], []
        for rotation, case in enumerate(cases):
            pointers, arrays, raw, selected = (
                case[key] for key in ("pointers", "arrays", "raw", "selected")
            )
            blob = runtime.read(pointers["attention_out"], output_bytes + len(GUARD))
            require(blob[output_bytes:] == GUARD, "output guard changed")
            actual_bf16 = np.frombuffer(blob, "<u2", count=output_bytes // 2).reshape(
                policy["T"], policy["H"], policy["D"]
            )
            reference_started = monotonic()
            maximum_error = 0.0
            expected_sha = hashlib.sha256()
            for first, expected in npu_reference_batches(
                arrays["q"], arrays["sinks"], selected.rows, policy
            ):
                actual_chunk = actual_bf16[first : first + len(expected)]
                actual = decode(actual_chunk.tobytes(), "bf16", actual_chunk.shape)
                error = float(np.max(np.abs(actual - expected)))
                require(
                    np.allclose(actual, expected, rtol=RTOL, atol=ATOL),
                    f"numerical mismatch at query {first}: {error}",
                )
                maximum_error = max(maximum_error, error)
                expected_sha.update(expected.tobytes())
            print(
                json.dumps(
                    {
                        "phase": "reference",
                        "rotation": rotation,
                        "seconds": monotonic() - reference_started,
                    }
                ),
                flush=True,
            )
            np.savez(args.output / f"outputs-{rotation}.npz", actual_bf16=actual_bf16)
            case["reference_fp32_sha256"] = expected_sha.hexdigest()
            errors.append(maximum_error)
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
            check_gathered(
                packed,
                bias,
                last["selected"].gathered,
                policy,
                kernels["sparse_mla"][3]["blocks"],
            )
        record.update(
            status="passed",
            max_abs_error=max(errors),
            per_rotation_max_abs_error=errors,
            rtol=RTOL,
            atol=ATOL,
            checked_metadata_and_guards="unchanged",
            output_sha256=outputs,
            numerical_reference={
                "device": "npu:0",
                "dtype": "fp32",
                "allow_hf32": False,
                "chunk_queries": 128,
                "formula": "logical KV selection; FP32 QK, softmax with one sink, FP32 PV",
                "fp32_sha256": [case["reference_fp32_sha256"] for case in cases],
            },
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
