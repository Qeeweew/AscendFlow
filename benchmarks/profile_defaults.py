"""One default msprof collection each for 4K Flash Attention and normal GEMM."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from ascendflow.device import health_snapshot
from ascendflow.runtime.process import bounded_process
from benchmarks.check_examples import ROOT, run_device

# Algorithm policies and launcher counts use the existing runner defaults.
CASES = (
    ("flash-attention", "benchmarks.attention.run", ("--cases", "4096:4096"), 1, 3),
    ("gemm-normal", "benchmarks.gemm.run_matmul", (), 5, 30),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--physical", type=int, required=True)
    args = parser.parse_args()
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    os.environ["ASCENDFLOW_PYTHON"] = sys.executable
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    report = {
        "status": "running",
        "cases": [],
        "method": "one msprof invocation per case, default collection options; generated only",
    }
    report["head"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    report["sources"] = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for folder in ("src", "examples", "benchmarks")
        for path in sorted((ROOT / folder).rglob("*.py"))
    }

    def save():
        (directory / "results.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        report["health_before"] = health_snapshot(args.physical, directory)
        for name, module, options, warmup, repeats in CASES:
            output = directory / name
            artifact = output / "artifact"
            row = {"name": name, "status": "building", "artifact": str(artifact)}
            report["cases"].append(row)
            command = [sys.executable, "-m", module, "--output", str(artifact), "--build"]
            row["build"] = bounded_process(command, output / "build-step", timeout_s=300)
            profile = output / "profile"
            application = [
                str(ROOT / "benchmarks/profile.sh"),
                module,
                "--output",
                str(artifact),
                "--physical",
                str(args.physical),
                "--scope",
                "sampled",
                "--record",
                str(profile / "device.json"),
                *options,
            ]
            command = ["msprof", f"--output={profile}", *application]
            row["profile_command"] = command
            save()
            run_device(command, output / "profile-step")
            record = json.loads((profile / "device.json").read_text())
            if record.get("status") != "passed" or len(record["cases"]) != 1:
                raise ValueError("profile needs one successful numerical device check")
            case = record["cases"][0]
            if (case["warmup"], case["repeats"]) != (warmup, repeats):
                raise ValueError("runner launch defaults changed; update the collection contract")
            command = [
                sys.executable,
                "-m",
                "benchmarks.profile_summary",
                str(profile),
                "--warmup",
                str(warmup),
                "--repeats",
                str(repeats),
            ]
            subprocess.run(command, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
            summary = json.loads((profile / "summary.json").read_text())
            row.update(
                status="passed",
                kernel_time_us=summary["median_us"],
                warmup=warmup,
                repeats=repeats,
                summary=str(profile / "summary.json"),
                csv=summary["csv"],
                csv_sha256=summary["csv_sha256"],
                device_record=str(profile / "device.json"),
                device_record_sha256=hashlib.sha256(
                    (profile / "device.json").read_bytes()
                ).hexdigest(),
            )
            print(f"{name}: kernel time {row['kernel_time_us']:.3f} us", flush=True)
            save()
        report["health_after"] = health_snapshot(args.physical, directory / "after")
        report["status"] = "passed"
    except Exception as error:  # noqa: BLE001 - publish incomplete collection as failed
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        if report["cases"]:
            report["cases"][-1]["status"] = "failed"
        print(report["error"], file=sys.stderr)
    finally:
        save()
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
