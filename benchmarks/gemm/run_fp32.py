"""Check the IEEE FP32 native path with inputs not representable in b16."""

import argparse
import json
from pathlib import Path

from ascendflow import TorchRuntime
from ascendflow.device import health_snapshot
from ascendflow.diagnostics import require
from benchmarks.kda.run import digest, error_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--physical", type=int, required=True)
    args = parser.parse_args()
    health = health_snapshot(args.physical, args.output)
    import torch
    import torch_npu  # noqa: F401

    torch.set_num_threads(1)
    torch.npu.set_device(0)
    generator = torch.Generator().manual_seed(187)
    # KDA projection geometry; perturbations distinguish b16/HF32 from FP32.
    a = torch.randn(1, 3, 64, 64, generator=generator)
    b = torch.randn(1, 3, 128, 64, generator=generator)
    expected = a @ b.transpose(-1, -2)
    aa, bb = a.npu(), b.npu()
    cc = torch.empty((1, 3, 64, 128), dtype=torch.float32, device="npu")
    directory = args.artifact.resolve()
    manifest = json.loads((directory / "batched_matmul.compile.json").read_text())
    require(digest(directory / "batched_matmul.so") == manifest["shared_sha256"], "library changed")
    with TorchRuntime(torch) as runtime:
        launch = runtime.kernel(directory / "batched_matmul.so", "batched_matmul")
        launch(aa, bb, cc)
        torch.npu.synchronize()
    metrics = error_metrics(cc, expected)
    (args.output / "result.json").write_text(
        json.dumps(
            {
                "status": "passed" if metrics["rmse_ratio"] < 2e-6 else "failed",
                "correctness": metrics,
                "health": health,
                "library_sha256": manifest["shared_sha256"],
                "input_dtypes": ["f32", "f32"],
                "oracle": "CPU FP32 matmul",
                "tolerance_rmse_ratio": 2e-6,
            },
            indent=2,
        )
        + "\n"
    )
    require(metrics["rmse_ratio"] < 2e-6, f"IEEE FP32 matrix precision failed: {metrics}")
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
