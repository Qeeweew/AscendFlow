"""Validate a published KDA library and compare upstream AscendC KDA."""

import argparse
import ctypes as ct
import hashlib
import json
import os
import sys
from pathlib import Path

from ascendflow.device import health_snapshot
from ascendflow.diagnostics import require
from ascendflow.dialects.io import parse
from ascendflow.dialects.source import AllocationOp, SizeOp
from ascendflow.performance import hardware_snapshot
from ascendflow.runtime.process import bounded_process
from benchmarks.kda.reference import load_reference, recurrent_oracle


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def error_metrics(actual, expected):
    import torch

    actual, expected = actual.detach().float().cpu(), expected.float().cpu()
    require(torch.isfinite(actual).all().item(), "KDA produced a nonfinite value")
    if actual.numel() == 0:
        return {"max_abs": 0.0, "rmse_ratio": 0.0}
    diff = actual - expected
    return {
        "max_abs": diff.abs().max().item(),
        "rmse_ratio": (
            diff.square().mean().sqrt() / (expected.square().mean().sqrt() + 1e-8)
        ).item(),
    }


def worker(args):
    # Check all physical cards before anything initializes the NPU runtime.
    health = health_snapshot(args.physical, args.output)
    hardware = hardware_snapshot(args.physical)
    if args.backend in {"ascendc", "both"}:
        from benchmarks.references.kda_aclnn import configure_reference

        reference = configure_reference(args.reference_artifacts)
    else:
        reference = None
    import torch
    import torch_npu

    torch.set_num_threads(1)
    directory = args.artifact.resolve()
    kernel = json.loads((directory / "verification.json").read_text())["kernel"]
    manifest = json.loads((directory / f"{kernel}.compile.json").read_text())
    require(
        digest(directory / f"{kernel}.asc") == manifest["source_sha256"]
        and digest(directory / f"{kernel}.so") == manifest["shared_sha256"],
        "KDA source/library changed after compilation",
    )
    source = parse((directory / "source.xdsl").read_text())
    allocations = {
        op.sym_name.data: op
        for op in source.walk()
        if isinstance(op, AllocationOp) and op.ref.type.memory.data == "GM"
    }
    tensors = {name: op.ref.type for name, op in allocations.items()}
    sizes = [op for op in source.walk() if isinstance(op, SizeOp)]
    lengths = args.lengths or [1] * (tensors["Cu"].shape.data[0].data - 1)
    runtime_values = {"TOKENS": sum(lengths)}
    require(
        all(op.sym_name.data in runtime_values for op in sizes),
        "KDA runner does not know this runtime parameter",
    )
    shapes = {}
    for name, op in allocations.items():
        shape = [n.data for n in op.ref.type.shape]
        for axis, dimension in zip(op.dynamic_axes or (), op.dimensions, strict=True):
            require(isinstance(dimension.owner, SizeOp), "expected a direct runtime extent")
            shape[axis.data] = runtime_values[dimension.owner.sym_name.data]
        shapes[name] = tuple(shape)
    tokens, heads, dimension = shapes["Q"]
    require(
        heads == shapes["Value"][1] and dimension == shapes["Value"][2],
        "the KDA benchmark currently requires H=HV and K=V",
    )
    require(
        sum(lengths) == tokens
        and len(lengths) == shapes["Cu"][0] - 1
        and min(lengths) >= 0
        and max(lengths) <= shapes["Indices"][1],
        "lengths do not match the compiled KDA contract",
    )
    dtypes = {
        "f16": torch.float16,
        "bf16": torch.bfloat16,
        "f32": torch.float32,
        "i64": torch.int64,
    }
    generator = torch.Generator().manual_seed(42)
    host = {}
    for name in ("Q", "Key", "Value", "State"):
        host[name] = torch.randn(shapes[name], generator=generator, dtype=torch.float32).to(
            dtypes[tensors[name].dtype.data]
        )
    host["Gate"] = torch.nn.functional.logsigmoid(
        torch.randn(shapes["Gate"], generator=generator)
    ).to(dtypes[tensors["Gate"].dtype.data])
    host["Beta"] = torch.rand(shapes["Beta"], generator=generator).to(
        dtypes[tensors["Beta"].dtype.data]
    )
    host["Cu"] = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int64)
    host["Indices"] = torch.zeros(shapes["Indices"], dtype=torch.int64)
    for seq, length in enumerate(lengths):
        host["Indices"][seq, :length] = seq + 1
    host["Accepted"] = torch.ones(shapes["Accepted"], dtype=torch.int64)
    host["State"][0].zero_()
    for name in ("Snapshots", "Output"):
        host[name] = torch.zeros(shapes[name], dtype=dtypes[tensors[name].dtype.data])
    inputs_sha = {
        name: hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        for name, t in host.items()
    }
    oracle, expected_state, _ = recurrent_oracle(
        *(host[n] for n in ("Q", "Key", "Value", "Gate", "Beta", "State")), lengths, host["Indices"]
    )
    torch.npu.set_device(0)
    device = {name: tensor.npu() for name, tensor in host.items()}
    stream = torch.npu.current_stream()
    library = ct.CDLL(str(directory / f"{kernel}.so"))
    launch = getattr(library, f"launch_{kernel}")
    order = (
        "Q",
        "Key",
        "Value",
        "Gate",
        "Beta",
        "State",
        "Snapshots",
        "Output",
        "Cu",
        "Indices",
        "Accepted",
    )
    launch.argtypes = [ct.c_void_p] * (len(order) + 1) + [ct.c_int64] * len(sizes)
    launch.restype = ct.c_int

    def generated():
        result = launch(
            ct.c_void_p(stream.npu_stream),
            *(ct.c_void_p(device[name].data_ptr()) for name in order),
            *(runtime_values[op.sym_name.data] for op in sizes),
        )
        require(result == 0, f"KDA launch returned {result}")
        return device["Output"], device["State"]

    functions = {"ascendflow": generated}
    reference_call = None
    if args.backend in {"both", "ascendc"}:
        from benchmarks.references.kda_aclnn import KdaReference

        reference_call = KdaReference.recurrent(args.reference_artifacts, device)
        reference["interface"] = reference_call.operator
        functions["ascendc"] = reference_call
    if args.backend == "triton":
        original, reference = load_reference(args.reference)

        def triton_call():
            output, _ = original.fused_recurrent_kda(
                *(device[n].unsqueeze(0) for n in ("Q", "Key", "Value", "Gate", "Beta")),
                initial_state=device["State"],
                inplace_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=device["Cu"],
                ssm_state_indices=device["Indices"],
            )
            return output.squeeze(0), device["State"]

        functions["triton"] = triton_call
    selected = functions.keys() if args.backend == "both" else [args.backend]
    initial = device["State"].clone()
    results = {}
    for backend in selected:
        device["State"].copy_(initial)
        from benchmarks.kda.profiling import invocation

        with invocation(args.profile_invocation):
            output, state = functions[backend]()
        torch.npu.synchronize()
        metrics = {
            "output": error_metrics(output, oracle),
            "state": error_metrics(state, expected_state),
        }
        require(
            all(m["rmse_ratio"] < 0.005 for m in metrics.values()),
            f"{backend} KDA correctness failed: {metrics}",
        )
        require(
            torch.equal(device["State"][0].cpu(), host["State"][0]), "KDA overwrote NULL state slot"
        )
        for _ in range(args.warmup):
            device["State"].copy_(initial)
            functions[backend]()
        torch.npu.synchronize()
        times = []
        for _ in range(args.repeats):
            device["State"].copy_(initial)
            start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
            start.record(stream)
            functions[backend]()
            end.record(stream)
            end.synchronize()
            times.append(start.elapsed_time(end) * 1000)
        results[backend] = {"correctness": metrics, "stream_interval_us": times}
    if reference_call is not None:
        torch.npu.synchronize()
        reference_call.close()
    record = {
        "status": "passed",
        "operator": kernel,
        "shapes": shapes,
        "lengths": lengths,
        "runtime_parameters": {op.sym_name.data: runtime_values[op.sym_name.data] for op in sizes},
        "dtypes": {n: t.dtype.data for n, t in tensors.items()},
        "tolerance_rmse_ratio": 0.005,
        "input_sha256": inputs_sha,
        "health": health,
        "hardware": hardware,
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "generated": {
            "directory": str(directory),
            "asc_sha256": manifest["source_sha256"],
            "shared_sha256": manifest["shared_sha256"],
            "commands": manifest["commands"],
        },
        "reference": reference,
        "results": results,
        "timing_scope": "NPU stream interval, includes host submission gaps; kernel performance requires device profiling",
        "warmup": args.warmup,
        "repeats": args.repeats,
    }
    (args.output / "results.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(results, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--reference", type=Path, default=Path("/workspace/xdsl-910b/reference/vllm-ascend")
    )
    parser.add_argument("--reference-artifacts", type=Path)
    parser.add_argument("--profile-invocation", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--physical", type=int, required=True)
    parser.add_argument(
        "--backend", choices=("both", "ascendflow", "ascendc", "triton"), default="ascendflow"
    )
    parser.add_argument("--lengths", nargs="+", type=int)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    require(args.warmup >= 0 and args.repeats >= 0, "invalid KDA timing counts")
    os.environ.update(ASCEND_RT_VISIBLE_DEVICES=str(args.physical), ASCEND_DEVICE_ID="0")
    os.environ.setdefault("TRITON_CACHE_DIR", str(args.output.resolve() / "triton-cache"))
    if args.worker:
        worker(args)
    else:
        require(not args.output.exists(), "KDA results require a new output directory")
        record = bounded_process(
            [sys.executable, "-m", "benchmarks.kda.run", *sys.argv[1:], "--worker"],
            args.output,
            timeout_s=300,
        )
        print(json.dumps(record))


if __name__ == "__main__":
    main()
