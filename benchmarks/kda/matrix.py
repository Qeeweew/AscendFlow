"""Build and compare the fixed decode policy across a batch-size matrix."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from ascendflow.build import build_file
from ascendflow.diagnostics import require
from benchmarks.kda.compare import compare
from benchmarks.kda.run import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--physical", required=True, type=int)
    parser.add_argument("--reference-artifacts", required=True, type=Path)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    args = parser.parse_args()
    require(not args.output.exists(), "matrix requires a new output directory")
    require(all(b > 0 for b in args.batches), "batch sizes must be positive")
    root = Path(__file__).resolve().parents[2]
    directory = args.output.resolve()
    directory.mkdir(parents=True)
    rows = []
    source = root / "examples/kda/decode.py"
    for batch in args.batches:
        point = directory / f"b{batch}"
        artifact = point / "artifact"
        print(f"batch={batch}: build", flush=True)
        build_file(
            source,
            artifact,
            entry="decode_kda",
            parameters={"SEQUENCES": batch, "SLOTS": batch + 1},
            shared=True,
            timeout_s=300,
        )
        for backend in ("ascendflow", "ascendc"):
            print(f"batch={batch}: {backend} BasicInfo", flush=True)
            with (point / f"{backend}.log").open("w") as log:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "benchmarks.kda.profile",
                        "--artifact",
                        str(artifact),
                        "--output",
                        str(point / backend),
                        "--physical",
                        str(args.physical),
                        "--backend",
                        backend,
                        "--reference-artifacts",
                        str(args.reference_artifacts.resolve()),
                        "--metrics",
                        "BasicInfo",
                    ],
                    check=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
        report = compare(point / "ascendflow", point / "ascendc", None)
        report_path = point / "comparison.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        row = {
            "batch": batch,
            "ascendflow_us": report["ascendflow"]["kernel_us"],
            "ascendc_us": report["ascendc"]["kernel_us"],
            "speedup": report["speedup"],
            "comparison": str(report_path),
            "comparison_sha256": digest(report_path),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        matrix = {
            "status": "passed" if len(rows) == len(args.batches) else "in_progress",
            "aic_metrics": "BasicInfo",
            "collections_per_point_backend": 1,
            "physical_device": args.physical,
            "source": str(source),
            "source_sha256": digest(source),
            "policy": {"HEAD_GROUPS": 8, "BV": 64, "PIPE_SLOTS": 2, "stages": 3},
            "rows": rows,
        }
        (directory / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
        lines = [
            "| Batch | AscendFlow (μs) | AscendC (μs) | AscendC / AscendFlow |",
            "| ---: | ---: | ---: | ---: |",
        ]
        lines += [
            f"| {r['batch']} | {r['ascendflow_us']:.3f} | {r['ascendc_us']:.3f} | {r['speedup']:.2f}× |"
            for r in rows
        ]
        (directory / "matrix.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
