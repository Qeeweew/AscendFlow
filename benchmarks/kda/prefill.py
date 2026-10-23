"""Build, validate and time the chunk KDA prefill pipeline."""

import argparse
import ctypes as ct
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

from ascendflow.build import build_file
from ascendflow.device import health_snapshot
from ascendflow.diagnostics import require
from ascendflow.dialects.io import parse
from ascendflow.dialects.source import AllocationOp
from ascendflow.performance import hardware_snapshot
from ascendflow.runtime.process import bounded_process
from benchmarks.kda.reference import load_reference
from benchmarks.kda.run import digest, error_metrics

PHASES = ("prepare", "scan", "output")
KERNELS = {p: "prefill_" + p for p in PHASES}
SOURCES = {p: "examples/kda/prefill_" + p + ".py" for p in PHASES}
RUNTIME_PHASES = {"prepare", "output"}
ORDERS = {
    "prepare": (
        "Q",
        "Key",
        "Value",
        "Gate",
        "Beta",
        "Qp",
        "Kp",
        "Vp",
        "Gp",
        "LocalGp",
        "Bp",
        "Aqk",
        "Diagonal",
        "Ai32",
        "Negative",
        "Inverse",
        "Right",
        "KG",
        "QG",
        "A16",
        "U",
        "W32",
        "GramLeft",
        "GramRight",
        "GramProducts",
        "Merge16",
        "Merge32",
        "Off32",
        "W16Ring",
        "Starts",
        "Lengths",
        "Lower",
    ),
    "scan": (
        "U",
        "KG",
        "Gp",
        "Initial",
        "Final",
        "History",
        "Snapshot",
        "W32",
        "Projection",
        "Residual",
        "Delta",
        "Lengths",
    ),
    "output": (
        "QG",
        "A16",
        "Snapshot",
        "Residual",
        "Output",
        "OutputProducts",
        "Starts",
        "Lengths",
    ),
}


def workspace_layout(chunks, heads, sequences, dtype):
    layouts = {}

    def add(names, shape, storage):
        for name in names.split():
            layouts[name] = (shape, storage)

    add("KG QG Residual U", (chunks, heads, 64, 128), dtype)
    add("Qp Kp Vp", (20, 1, 64, 128), dtype)
    add("LocalGp", (20, 1, 64, 128), "f32")
    add("Gp W32 Projection", (chunks, heads, 64, 128), "f32")
    add("Aqk Ai32 Negative Diagonal", (20, 1, 64, 64), "f32")
    add("Bp", (20, 1, 64), "f32")
    add("Merge16", (20, 2, 16, 16), "f32")
    add("Merge32 Off32", (20, 2, 32, 32), "f32")
    add("Inverse", (20, 1, 64, 64), dtype)
    add("A16", (chunks, heads, 64, 64), dtype)
    add("OutputProducts", (20, 2, 64, 256), "f32")
    add("Right", (20, 1, 2, 64, 128), dtype)
    add("W16Ring", (20, 2, 64, 128), dtype)
    add("Snapshot", (chunks, heads, 128, 128), dtype)
    add("Delta", (chunks, heads, 128, 128), "f32")
    add("GramLeft", (20, 2, 1, 128, 128), "f32")
    add("GramRight", (20, 2, 1, 64, 128), "f32")
    add("GramProducts", (20, 2, 32, 96), "f32")
    add("Initial History Final", (sequences, heads, 128, 128), "f32")
    add("Lower", (2, 64, 64), "f32")
    return layouts


def oracle(host, lengths):
    import torch

    q, k, v, g, beta = [host[n].float() for n in ("Q", "Key", "Value", "Gate", "Beta")]
    q = q * (q.square().sum(-1, keepdim=True) + 1e-6).rsqrt()
    k = k * (k.square().sum(-1, keepdim=True) + 1e-6).rsqrt()
    output = torch.empty_like(v)
    final = host["Initial"].clone()
    start = 0
    for seq, length in enumerate(lengths):
        state = final[seq]
        for t in range(start, start + length):
            state = state * g[t].exp().unsqueeze(-2)
            projected = (state * k[t].unsqueeze(-2)).sum(-1)
            residual = (v[t] - projected) * beta[t].unsqueeze(-1)
            state = state + residual.unsqueeze(-1) * k[t].unsqueeze(-2)
            output[t] = (state * (q[t] * (q.shape[-1] ** -0.5)).unsqueeze(-2)).sum(-1)
        final[seq] = state
        start += length
    return output.to(host["Value"].dtype), final


def build(args):
    (args.artifacts / "pipeline.json").unlink(missing_ok=True)
    nc = (args.lengths[0] + 63) // 64
    require(
        all((length + 63) // 64 == nc and length > 0 for length in args.lengths),
        "scan currently requires the same nonzero chunk count per sequence",
    )
    parameters = {"CHUNKS": nc * len(args.lengths), "H": args.heads, "DTYPE": args.dtype}
    for phase in args.phases:
        print(f"build prefill_{phase}", flush=True)
        options = parameters | (
            {"SEQUENCES": len(args.lengths), "NC": nc}
            if phase in {"prepare", "scan", "output"}
            else {}
        )
        build_file(
            Path(__file__).resolve().parents[2] / SOURCES[phase],
            args.artifacts / phase,
            entry=KERNELS[phase],
            parameters=options,
            shared=True,
            timeout_s=300,
        )
    identity = validate_artifacts(
        args.artifacts, args.heads, args.dtype, nc * len(args.lengths), len(args.lengths)
    )
    (args.artifacts / "pipeline.json").write_text(
        json.dumps(
            {
                "lengths": args.lengths,
                "heads": args.heads,
                "dimension": 128,
                "dtype": args.dtype,
                "chunk_size": 64,
                "chunks_per_sequence": nc,
                "parameters": parameters,
                "phases": PHASES,
                "front_end_task_policy": "chunk_head_local",
                "artifacts": identity,
            },
            indent=2,
        )
        + "\n"
    )


def validate_artifacts(directory, heads, dtype, chunks, sequences):
    identity = {}
    layouts = workspace_layout(chunks, heads, sequences, dtype)
    for phase in PHASES:
        path = directory / phase
        source = parse((path / "source.xdsl").read_text())
        buffers = {
            op.sym_name.data: op.ref.type
            for op in source.walk()
            if isinstance(op, AllocationOp) and op.ref.type.memory.data == "GM"
        }
        require(tuple(buffers) == ORDERS[phase], f"{phase} library ABI differs")
        for name, tensor in buffers.items():
            if name in layouts:
                shape, storage = layouts[name]
                require(
                    tuple(d.data for d in tensor.shape.data) == shape
                    and tensor.dtype.data == storage,
                    f"{phase} {name} specialization differs",
                )
        primary = buffers["KG" if phase == "prepare" else "QG" if phase == "output" else "U"]
        require(
            primary.shape.data[0].data == chunks
            and primary.shape.data[1].data == heads
            and primary.dtype.data == dtype,
            f"{phase} specialization differs",
        )
        kernel = KERNELS[phase]
        manifest = json.loads((path / f"{kernel}.compile.json").read_text())
        require(
            digest(path / f"{kernel}.asc") == manifest["source_sha256"]
            and digest(path / f"{kernel}.so") == manifest["shared_sha256"],
            "artifact changed",
        )
        identity[phase] = {
            "source_xdsl_sha256": digest(path / "source.xdsl"),
            "asc_sha256": manifest["source_sha256"],
            "shared_sha256": manifest["shared_sha256"],
        }
    return identity


def worker(args):
    health = health_snapshot(args.physical, args.output)
    hardware = hardware_snapshot(args.physical)
    if args.backend in {"ascendc", "both"}:
        from benchmarks.references.kda_aclnn import configure_reference

        reference = configure_reference(args.reference_artifacts)
    else:
        reference = None
    import torch
    import torch_npu  # noqa: F401

    torch.set_num_threads(1)
    config = json.loads((args.artifacts / "pipeline.json").read_text())
    if args.run_lengths is not None:
        require(
            len(args.run_lengths) == len(config["lengths"])
            and all(
                n > 0 and (n + 63) // 64 == config["chunks_per_sequence"] for n in args.run_lengths
            ),
            "runtime lengths exceed the compiled chunk domain",
        )
        config["build_lengths"] = config["lengths"]
        config["lengths"] = args.run_lengths
    identity = validate_artifacts(
        args.artifacts,
        config["heads"],
        config["dtype"],
        len(config["lengths"]) * config["chunks_per_sequence"],
        len(config["lengths"]),
    )
    require(config.get("artifacts", identity) == identity, "pipeline artifacts changed")
    lengths = config["lengths"]
    tokens, batch, heads, dimension = sum(lengths), len(lengths), config["heads"], 128
    chunks, bt = batch * config["chunks_per_sequence"], 64
    require(tuple(config["phases"]) == PHASES, "expected the three-stage pipeline")
    dtype = {"f16": torch.float16, "bf16": torch.bfloat16}[config["dtype"]]
    generator = torch.Generator().manual_seed(42)
    host = {
        name: torch.randn(tokens, heads, dimension, generator=generator).to(dtype)
        for name in ("Q", "Key", "Value")
    }
    host["Gate"] = torch.nn.functional.logsigmoid(
        torch.randn(tokens, heads, dimension, generator=generator)
    )
    host["Beta"] = torch.rand(tokens, heads, generator=generator)
    host["Initial"] = torch.randn(batch, heads, dimension, dimension, generator=generator)
    starts, valid, chunk_indices, cu = [], [], [], [0]
    for seq, length in enumerate(lengths):
        for local, row in enumerate(range(0, length, bt)):
            starts.append(cu[-1] + row)
            valid.append(min(bt, length - row))
            chunk_indices.append((seq, local))
        cu.append(cu[-1] + length)
    host["Starts"] = torch.tensor(starts, dtype=torch.int64)
    host["Lengths"] = torch.tensor(valid, dtype=torch.int64)
    input_sha = {
        n: hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        for n, t in host.items()
    }
    expected, final_expected = oracle(host, lengths)
    torch.npu.set_device(0)
    device = {n: t.npu() for n, t in host.items()}
    constants = {
        "Lower": torch.stack((torch.tril(torch.ones(bt, bt, dtype=torch.float32)), torch.eye(bt))),
    }
    constant_sha = {
        name: hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()
        for name, t in constants.items()
    }
    device.update({name: t.npu() for name, t in constants.items()})

    def workspace(name, shape, storage_dtype):
        device[name] = torch.empty(shape, dtype=storage_dtype, device="npu")

    storage_types = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}
    for name, (shape, storage) in workspace_layout(chunks, heads, batch, config["dtype"]).items():
        if name not in device:
            workspace(name, shape, storage_types[storage])
    workspace("Output", (tokens, heads, dimension), dtype)
    stream = torch.npu.current_stream()
    libraries, launchers, artifacts = [], {}, {}
    for phase in PHASES:
        directory = (args.artifacts / phase).resolve()
        kernel = KERNELS[phase]
        manifest = json.loads((directory / f"{kernel}.compile.json").read_text())
        require(
            digest(directory / f"{kernel}.so") == manifest["shared_sha256"]
            and digest(directory / f"{kernel}.asc") == manifest["source_sha256"],
            "prefill artifact changed",
        )
        library = ct.CDLL(str(directory / f"{kernel}.so"))
        libraries.append(library)
        launch = getattr(library, "launch_" + kernel)
        launch.argtypes = [ct.c_void_p] * (len(ORDERS[phase]) + 1) + (
            [ct.c_int64] if phase in RUNTIME_PHASES else []
        )
        launch.restype = ct.c_int
        launchers[phase] = launch
        artifacts[phase] = {
            "directory": str(directory),
            "asc_sha256": manifest["source_sha256"],
            "shared_sha256": manifest["shared_sha256"],
            "commands": manifest["commands"],
        }

    def launch_phase(phase):
        runtime = [tokens] if phase in RUNTIME_PHASES else []
        require(
            launchers[phase](
                stream.npu_stream, *(device[n].data_ptr() for n in ORDERS[phase]), *runtime
            )
            == 0,
            f"prefill {phase} launch failed",
        )
        if args.phase_sync:
            torch.npu.synchronize()
            print("completed " + phase, flush=True)

    def generated():
        for phase in PHASES:
            launch_phase(phase)
        return device["Output"], device["Final"]

    functions = {"ascendflow": generated}
    reference_call = None
    if args.backend in {"ascendc", "both"}:
        from benchmarks.references.kda_aclnn import KdaReference

        reference_call = KdaReference.prefill(args.reference_artifacts, device, cu, chunk_indices)
        reference["interface"] = reference_call.operator
        functions["ascendc"] = reference_call
    value_initial = device["Value"].clone()
    if args.backend == "triton":
        original, reference = load_reference(args.reference)
        cu_device = torch.tensor(cu, dtype=torch.int64, device="npu")
        meta = SimpleNamespace(
            chunk_indices_chunk64=torch.tensor(chunk_indices, dtype=torch.int64, device="npu"),
            chunk_offsets_chunk64=torch.tensor(
                [seq * config["chunks_per_sequence"] for seq in range(batch + 1)],
                dtype=torch.int64,
                device="npu",
            ),
        )

        def triton_call():
            out, state = original.chunk_kda(
                *(device[n].unsqueeze(0) for n in ("Q", "Key", "Value", "Gate", "Beta")),
                initial_state=device["Initial"],
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_device,
                prebuilt_meta=meta,
            )
            return out.squeeze(0), state

        functions["triton"] = triton_call
    results = {}
    for backend in functions if args.backend == "both" else (args.backend,):
        device["Value"].copy_(value_initial)
        torch.npu.synchronize()
        from benchmarks.kda.profiling import invocation

        with invocation(args.profile_invocation):
            output, state = functions[backend]()
        torch.npu.synchronize()
        correctness = {
            "output": error_metrics(output, expected),
            "state": error_metrics(state, final_expected),
        }
        if backend == "ascendflow" and args.save_intermediates:
            torch.save({n: t.cpu() for n, t in device.items()}, args.output / "intermediates.pt")
        require(
            all(m["rmse_ratio"] < 0.005 for m in correctness.values()),
            f"prefill {backend} mismatch: {correctness}",
        )
        for _ in range(args.warmup):
            device["Value"].copy_(value_initial)
            functions[backend]()
        times = []
        for _ in range(args.repeats):
            device["Value"].copy_(value_initial)
            begin, end = [torch.npu.Event(enable_timing=True) for _ in range(2)]
            begin.record(stream)
            functions[backend]()
            end.record(stream)
            end.synchronize()
            times.append(begin.elapsed_time(end) * 1000)
        results[backend] = {"correctness": correctness, "stream_interval_us": times}
    if reference_call is not None:
        torch.npu.synchronize()
        reference_call.close()
    (args.output / "results.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "configuration": config,
                "health": health,
                "hardware": hardware,
                "input_sha256": input_sha,
                "constant_sha256": constant_sha,
                "generated": artifacts,
                "reference": reference,
                "results": results,
                "tolerance_rmse_ratio": 0.005,
                "timing_scope": "stream intervals include host submission; kernel comparisons use profiling",
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(results, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--physical", type=int)
    parser.add_argument("--lengths", nargs="+", type=int, default=[128])
    parser.add_argument(
        "--run-lengths",
        nargs="+",
        type=int,
        help="reuse libraries with different lengths in the same compiled chunk domain",
    )
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--dtype", choices=("f16", "bf16"), default="bf16")
    parser.add_argument(
        "--backend", choices=("ascendflow", "ascendc", "triton", "both"), default="ascendflow"
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "reference/vllm-ascend",
    )
    parser.add_argument("--reference-artifacts", type=Path)
    parser.add_argument("--profile-invocation", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--build-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--phases", nargs="+", choices=PHASES, default=PHASES)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--save-intermediates", action="store_true")
    parser.add_argument("--phase-sync", action="store_true")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    require(args.build_only or args.physical is not None, "--physical is required for device runs")
    if args.build or args.build_only:
        if args.build_worker:
            build(args)
        else:
            bounded_process(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "benchmarks.kda.prefill",
                    "--build-only",
                    "--build-worker",
                    "--artifacts",
                    str(args.artifacts),
                    "--output",
                    str(args.output),
                    "--heads",
                    str(args.heads),
                    "--dtype",
                    args.dtype,
                    "--lengths",
                    *(str(n) for n in args.lengths),
                    "--phases",
                    *args.phases,
                ],
                args.artifacts / "build-watchdog",
                timeout_s=300,
            )
    if args.build_only:
        return
    if args.worker:
        worker(args)
    else:
        environment = {"ASCEND_RT_VISIBLE_DEVICES": str(args.physical), "ASCEND_DEVICE_ID": "0"}
        os.environ.update(environment)
        os.environ["TRITON_CACHE_DIR"] = str((args.output / "triton-cache").resolve())
        command = [
            sys.executable,
            "-m",
            "benchmarks.kda.prefill",
            "--artifacts",
            str(args.artifacts),
            "--output",
            str(args.output),
            "--physical",
            str(args.physical),
            "--backend",
            args.backend,
            "--reference",
            str(args.reference),
            "--warmup",
            str(args.warmup),
            "--repeats",
            str(args.repeats),
            "--worker",
        ]
        if args.reference_artifacts is not None:
            command.extend(["--reference-artifacts", str(args.reference_artifacts.resolve())])
        if args.save_intermediates:
            command.append("--save-intermediates")
        if args.phase_sync:
            command.append("--phase-sync")
        if args.run_lengths is not None:
            command.extend(["--run-lengths", *(str(n) for n in args.run_lengths)])
        bounded_process(command, args.output / "watchdog", timeout_s=300)


if __name__ == "__main__":
    main()
