"""Independent AscendC gather2 experiment; checking and profiling are separate."""

import argparse
import csv
import ctypes as ct
import hashlib
import json
import os
import shlex
import statistics
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np

from ascendflow.backend.cann import offline_compile
from ascendflow.device import ACL, health_snapshot
from ascendflow.diagnostics import require
from ascendflow.performance import hardware_snapshot
from ascendflow.runtime.process import kill_process_group
from benchmarks.attention.run_sparse import GUARD
from benchmarks.cache import CacheEviction

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "benchmarks/references/ascendc/ascendc_gather2.asc.in"
NAMES = tuple(
    f"gather_{mode}_{address}"
    for address in ("lookup", "physical")
    for mode in ("single", "pair_single", "pair_dma")
)
GROUPS, QT, TOPK, D, TABLE, PAGE_ROWS = 20, 4, 512, 512, 1024, 64
SOURCE_ROWS = 128 * TABLE * PAGE_ROWS
SOURCE_BYTES = SOURCE_ROWS * D * 2
OUTPUT_SHAPE = (GROUPS, 3, TOPK, D // 2)
OUTPUT_BYTES = int(np.prod(OUTPUT_SHAPE)) * 4


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def inputs():
    rng = np.random.default_rng(4102026)
    pages = np.stack([rng.permutation(TABLE) + b * TABLE for b in range(128)]).astype(np.int32)
    indices = np.stack(
        [
            np.sort(rng.choice((131072 - QT + token % QT + 1) // 2, TOPK, replace=False))
            for token in range(128 * QT)
        ]
    ).astype(np.int32)
    batch = np.arange(128 * QT) // QT
    physical = pages[batch[:, None], indices // PAGE_ROWS] * PAGE_ROWS + indices % PAGE_ROWS
    return pages, indices, physical.astype(np.int32)


def memset(runtime, pointer, size, value):
    runtime.call(
        "aclrtMemset",
        [ct.c_void_p, ct.c_size_t, ct.c_int32, ct.c_size_t],
        pointer,
        size,
        value,
        size,
    )


def worker(args):
    directory = args.output / args.mode
    directory.mkdir(parents=True, exist_ok=False)
    record = {
        "status": "running",
        "mode": args.mode,
        "batches": args.batches,
        "query_tokens": QT,
        "topk": TOPK,
        "features": D,
        "logical_context": 131072,
        "compressed_context": TABLE * PAGE_ROWS,
        "groups": GROUPS,
        "aiv_cores": 40,
        "ub_rows": 32,
        "ub_slots": 2,
        "workspace_slots": 3,
        "row_bytes": D * 2,
        "source_bytes": SOURCE_BYTES,
        "source_pattern": "uint32 word at byte offset 4*i contains i; bit-copy only, no attention math",
        "warmup": args.warmup,
        "repeats": args.repeats,
        "benchmark_sha256": sha(Path(__file__).read_bytes()),
        "numpy_version": np.__version__,
        "health": health_snapshot(args.physical, directory),
        "hardware": hardware_snapshot(args.physical),
    }
    runtime = None
    try:
        artifact = args.output / "kernel"
        manifest = json.loads((artifact / "gather_dma.compile.json").read_text())
        library = artifact / "gather_dma.so"
        require(sha(SOURCE.read_bytes()) == manifest["source_sha256"], "AscendC source changed")
        require(sha(library.read_bytes()) == manifest["shared_sha256"], "library changed")
        record["artifact"] = manifest
        module = ct.CDLL(str(library.resolve()))
        launch = {}
        for name in NAMES:
            fn = getattr(module, "launch_" + name)
            fn.argtypes, fn.restype = [ct.c_void_p] * 5 + [ct.c_int64], ct.c_int
            launch[name] = fn
        init = module.launch_gather_init
        init.argtypes, init.restype = [ct.c_void_p, ct.c_void_p, ct.c_int64], ct.c_int
        runtime = ACL()
        cache = runtime.allocate_uninitialized(SOURCE_BYTES + len(GUARD))
        require(init(runtime.stream, cache, SOURCE_BYTES // 4) == 0, "pattern init failed")
        runtime.synchronize()
        guard = ct.create_string_buffer(GUARD)
        runtime.call(
            "aclrtMemcpy",
            [ct.c_void_p, ct.c_size_t, ct.c_void_p, ct.c_size_t, ct.c_int],
            ct.c_void_p(cache.value + SOURCE_BYTES),
            len(GUARD),
            guard,
            len(GUARD),
            1,
        )
        pages, indices, physical = inputs()
        # Edge correctness runs are separate from performance inputs and timing.
        if args.mode == "check":
            indices[0, :10] = [0, 0, -1, 17, 23, -1, -1, -1, 65535, 1]
            physical = (
                pages[(np.arange(128 * QT) // QT)[:, None], np.maximum(indices, 0) // PAGE_ROWS]
                * PAGE_ROWS
                + np.maximum(indices, 0) % PAGE_ROWS
            )
            physical[indices < 0] = -1
            physical[1, :2] = [0, SOURCE_ROWS - 1]  # >INT32_MAX byte gap: fall back.
        arrays = {"pages": pages, "lookup": indices, "physical": physical.astype(np.int32)}
        blobs = {name: value.tobytes() for name, value in arrays.items()}
        pointers = {name: runtime.allocate(blob + GUARD) for name, blob in blobs.items()}
        out = runtime.allocate_uninitialized(OUTPUT_BYTES + len(GUARD))
        memset(runtime, out, OUTPUT_BYTES + len(GUARD), 0x5A)
        record["input_sha256"] = {name: sha(blob) for name, blob in blobs.items()}
        record["pair_statistics"] = {}
        for b in args.batches:
            pair = arrays["physical"][: b * QT].reshape(-1, 2).astype(np.int64)
            gap = (np.abs(pair[:, 1] - pair[:, 0]) - 1) * D * 2
            record["pair_statistics"][str(b)] = {
                "pairs": len(pair),
                "eligible_two_row": int(
                    np.count_nonzero((pair >= 0).all(axis=1) & (gap >= 0) & (gap < 2**31 - 1))
                ),
                "reverse_address_order": int(np.count_nonzero(pair[:, 0] > pair[:, 1])),
            }
        if args.mode == "check":
            checks = []
            for batch in args.batches:
                for name in NAMES:
                    memset(runtime, out, OUTPUT_BYTES + len(GUARD), 0x5A)
                    address = "physical" if name.endswith("physical") else "lookup"
                    require(
                        launch[name](
                            runtime.stream, cache, pointers[address], pointers["pages"], out, batch
                        )
                        == 0,
                        name,
                    )
                    runtime.synchronize()
                    raw = runtime.read(out, OUTPUT_BYTES + len(GUARD))
                    require(raw[OUTPUT_BYTES:] == GUARD, "output guard changed")
                    actual = np.frombuffer(raw[:OUTPUT_BYTES], np.uint32).reshape(OUTPUT_SHAPE)
                    for group in range(min(GROUPS, batch * QT)):
                        retained = {
                            j % 3: q for j, q in enumerate(range(group, batch * QT, GROUPS))
                        }
                        for slot, token in retained.items():
                            ids = arrays["physical"][token].astype(np.int64).copy()
                            if address == "lookup":
                                logical = indices[token]
                                ids = (
                                    pages[token // QT, np.maximum(logical, 0) // PAGE_ROWS]
                                    * PAGE_ROWS
                                    + np.maximum(logical, 0) % PAGE_ROWS
                                )
                                ids = ids.astype(np.int64)
                                ids[logical < 0] = -1
                            if "pair_" in name:
                                pair = ids.reshape(-1, 2)
                                valid_pair = (pair >= 0).all(axis=1)
                                pair[valid_pair] = np.sort(pair[valid_pair], axis=1)
                            active = ids >= 0
                            expected = (ids[active, None] * (D // 2) + np.arange(D // 2)).astype(
                                np.uint32
                            )
                            require(
                                np.array_equal(actual[group, slot, active], expected),
                                f"{name}: wrong row/stride/order",
                            )
                    checks.append({"batch": batch, "kernel": name, "bit_exact_valid_rows": True})
            for name, blob in blobs.items():
                require(
                    runtime.read(pointers[name], len(blob) + len(GUARD)) == blob + GUARD,
                    "metadata changed",
                )
            require(
                runtime.read(ct.c_void_p(cache.value + SOURCE_BYTES), len(GUARD)) == GUARD,
                "cache guard changed",
            )
            record.update(status="passed", checks=checks)
        else:
            eviction = CacheEviction(
                runtime, args.cache_evict_artifacts, record["hardware"]["l2_bytes"]
            )
            sequence = []
            for batch in args.batches:
                for iteration in range(args.warmup + args.repeats):
                    order = NAMES if iteration % 2 == 0 else tuple(reversed(NAMES))
                    for name in order:
                        eviction()
                        address = "physical" if name.endswith("physical") else "lookup"
                        require(
                            launch[name](
                                runtime.stream,
                                cache,
                                pointers[address],
                                pointers["pages"],
                                out,
                                batch,
                            )
                            == 0,
                            name,
                        )
                        sequence.append({"batch": batch, "iteration": iteration, "kernel": name})
            runtime.synchronize()
            record.update(
                status="measured",
                launch_sequence=sequence,
                numerical_validation="not_run",
                cache_eviction=eviction.identity,
            )
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (directory / "device.json").write_text(json.dumps(record, indent=2) + "\n")
        if runtime is not None:
            runtime.close()
    print(json.dumps({"status": record["status"], "mode": args.mode}), flush=True)


def profile(args):
    worker = [
        str(ROOT / "benchmarks/profile.sh"),
        "benchmarks.attention.gather_dma",
        "--mode",
        "measure",
        "--output",
        str(args.output),
        "--physical",
        str(args.physical),
        "--batches",
        *map(str, args.batches),
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
        "--cache-evict-artifacts",
        str(args.cache_evict_artifacts),
    ]
    command = [
        "msprof",
        "--output=" + str(args.output / "profile"),
        "--aic-metrics=" + args.metrics,
        "--application=" + shlex.join(worker),
    ]
    env = os.environ | {
        "ASCENDFLOW_PYTHON": sys.executable,
        "ASCEND_RT_VISIBLE_DEVICES": str(args.physical),
        "ASCEND_DEVICE_ID": "0",
        "OPENBLAS_NUM_THREADS": "1",
    }
    (args.output / "profile-command.json").write_text(json.dumps(command, indent=2) + "\n")
    with (args.output / "msprof.log").open("w") as log:
        proc = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            code = proc.wait(timeout=300)
        finally:
            kill_process_group(proc.pid)
        require(code == 0, "msprof failed; see msprof.log")
    summarize(args.output)


def summarize(output):
    measurement = output / "measure/device.json"
    d = json.loads(measurement.read_text())
    require(d["status"] == "measured", "measurement did not finish")
    paths = list((output / "profile").rglob("op_summary*.csv"))
    require(len(paths) == 1, "expected one device summary")
    with paths[0].open() as stream:
        rows = list(csv.DictReader(stream))
    # Epoch timestamps exceed float's sub-microsecond precision. Keep exact decimal
    # arithmetic until after subtracting them, including for the eviction boundary.
    rows.sort(key=lambda row: Decimal(row["Task Start Time(us)"]))
    sequence = [row for row in rows if row["OP Type"] in NAMES or row["OP Type"] == "state_sum"]
    calls = [row for row in sequence if row["OP Type"] in NAMES]
    require(
        [row["OP Type"] for row in calls] == [r["kernel"] for r in d["launch_sequence"]],
        "recorded launch order differs from device execution",
    )
    require(
        sum(row["OP Type"] == "state_sum" for row in sequence) == len(calls) + 1,
        "eviction launch count differs",
    )
    for i, row in enumerate(sequence):
        if row["OP Type"] in NAMES:
            require(
                i > 0 and sequence[i - 1]["OP Type"] == "state_sum",
                "gather was not immediately preceded by eviction",
            )
            previous = sequence[i - 1]
            require(
                Decimal(previous["Task Start Time(us)"]) + Decimal(previous["Task Duration(us)"])
                <= Decimal(row["Task Start Time(us)"]),
                "eviction overlapped gather",
            )
            require(row["Task Type"] == "AI_VECTOR_CORE", "unexpected device task type")
    first = Decimal(calls[0]["Task Start Time(us)"])
    require(
        all(
            row["OP Type"] in NAMES
            or row["OP Type"] == "state_sum"
            or Decimal(row["Task Start Time(us)"]) < first
            for row in rows
        ),
        "unexpected device compute after timing started",
    )
    groups = {name: [row for row in calls if row["OP Type"] == name] for name in NAMES}
    results = []
    count = d["warmup"] + d["repeats"]
    for batch_index, batch in enumerate(d["batches"]):
        for name, calls in groups.items():
            require(len(calls) == len(d["batches"]) * count, "launch count differs")
            measured = calls[batch_index * count + d["warmup"] : (batch_index + 1) * count]
            samples = [float(row["Task Duration(us)"]) for row in measured]
            median = statistics.median(samples)
            results.append(
                {
                    "batch": batch,
                    "kernel": name,
                    "median_us": median,
                    "min_us": min(samples),
                    "max_us": max(samples),
                    "samples_us": samples,
                    "logical_kv_read_GB_s": batch
                    * d["query_tokens"]
                    * d["topk"]
                    * d["row_bytes"]
                    / (median * 1000),
                    "metrics_median": {
                        key: statistics.median(float(row[key]) for row in measured)
                        for key in measured[0]
                        if key.startswith(("aiv_", "hbm_", "ddr_", "l2_"))
                        and all(row[key] not in {"", "N/A"} for row in measured)
                    },
                }
            )
    result = {
        "scope": "gather plus GM ring publication; device task duration, excludes eviction/setup/host",
        "summary_csv": str(paths[0]),
        "summary_sha256": sha(paths[0].read_bytes()),
        "measurement_sha256": sha(measurement.read_bytes()),
        "analyzer_sha256": sha(Path(__file__).read_bytes()),
        "results": results,
    }
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(results), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("build", "check", "measure", "profile", "summarize"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--physical", type=int, default=3)
    parser.add_argument("--batches", nargs="+", type=int, default=[64, 128])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--metrics", default="PipeUtilization")
    parser.add_argument("--cache-evict-artifacts", type=Path)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    require(all(1 <= b <= 128 for b in args.batches), "batch outside experiment range")
    if args.mode == "build":
        print(
            offline_compile(
                SOURCE.read_text(), args.output / "kernel", "gather_dma", shared=True, timeout_s=300
            )
        )
    elif args.mode == "profile":
        require(args.cache_evict_artifacts is not None, "explicit eviction artifact required")
        profile(args)
    elif args.mode == "summarize":
        summarize(args.output)
    else:
        if args.mode == "measure":
            require(args.cache_evict_artifacts is not None, "explicit eviction artifact required")
        worker(args)


if __name__ == "__main__":
    main()
