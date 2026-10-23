"""Compare numerically validated msprof op records with identical decode inputs."""

import argparse
import json
from pathlib import Path

from ascendflow.diagnostics import require
from benchmarks.kda.reference import verify_provenance
from benchmarks.kda.run import digest
from benchmarks.references.kda_aclnn import verify_reference


def read_profile(directory, backend):
    metrics = json.loads((directory / "metrics.json").read_text())
    device_path = directory / "device/results.json"
    device = json.loads(device_path.read_text())
    require(
        device["status"] == "passed" and set(device["results"]) == {backend},
        "invalid backend record",
    )
    require(
        metrics["provenance"]["device_record_sha256"] == digest(device_path),
        "numerical record changed after profiling",
    )
    for table in metrics.values():
        if isinstance(table, dict) and "csv" in table:
            require(digest(table["csv"]) == table["csv_sha256"], "profiling CSV changed")
    for relative, sha256 in metrics["provenance"]["triton_cache"].items():
        require(digest(directory / relative) == sha256, "Triton compilation artifact changed")
    numerical = device["results"][backend]["correctness"]
    require(
        all(v["rmse_ratio"] < device["tolerance_rmse_ratio"] for v in numerical.values()),
        "numerical acceptance failed",
    )
    duration = metrics["OpBasicInfo"]["Task Duration(us)"]["mean"]
    require(duration > 0, "invalid kernel duration")
    return device, {
        "kernel_us": duration,
        "correctness": numerical,
        "metrics": str((directory / "metrics.json").resolve()),
        "metrics_sha256": digest(directory / "metrics.json"),
        "device_record": str(device_path.resolve()),
    }


def compare(ascendflow, reference_directory, official_release=None, backend="ascendc"):
    generated, ours = read_profile(ascendflow, "ascendflow")
    original, theirs = read_profile(reference_directory, backend)
    modes = [
        json.loads((directory / "metrics.json").read_text())["provenance"].get(
            "aic_metrics", "Default"
        )
        for directory in (ascendflow, reference_directory)
    ]
    require(modes[0] == modes[1], "profiling configurations differ")
    clocks = [
        json.loads((directory / "metrics.json").read_text())["OpBasicInfo"]["Current Freq"]["mean"]
        for directory in (ascendflow, reference_directory)
    ]
    require(clocks[0] == clocks[1] and clocks[0] > 0, "measured kernel frequencies differ")
    keys = (
        "shapes",
        "lengths",
        "dtypes",
        "input_sha256",
        "runtime_parameters",
        "tolerance_rmse_ratio",
        "generated",
    )
    require(
        all(generated[key] == original[key] for key in keys),
        "decode inputs, contracts or artifact differ",
    )
    require(
        generated["health"]["physical_device"] == original["health"]["physical_device"]
        and generated["hardware"]["platform_sha256"] == original["hardware"]["platform_sha256"],
        "device or platform differs",
    )
    require(
        len(generated["lengths"]) > 0 and max(generated["lengths"]) > 0,
        "comparison requires a nonempty batch",
    )
    artifact = generated["generated"]
    root = Path(artifact["directory"])
    kernel = generated["operator"]
    require(
        digest(root / f"{kernel}.asc") == artifact["asc_sha256"]
        and digest(root / f"{kernel}.so") == artifact["shared_sha256"],
        "compiled artifact changed",
    )
    reference = original["reference"]
    if backend == "ascendc":
        verify_reference(reference)
    else:
        require(official_release is not None, "Triton comparison requires --official-release")
        verify_provenance(reference, official_release)
    report = {
        "status": "passed",
        "scope": f"decode/recurrent; one msprof op {modes[0]} collection per backend",
        "aic_metrics": modes[0],
        "ascendflow": ours,
        backend: theirs,
        "speedup": theirs["kernel_us"] / ours["kernel_us"],
        "configuration": {key: generated[key] for key in keys},
        "physical_device": generated["health"]["physical_device"],
        "reference": reference,
        "official_release": str(official_release.resolve()) if official_release else None,
        "official_release_sha256": digest(official_release) if official_release else None,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ascendflow", required=True, type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ascendc", type=Path)
    group.add_argument("--triton", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--official-release", type=Path)
    args = parser.parse_args()
    backend = "ascendc" if args.ascendc else "triton"
    report = compare(args.ascendflow, args.ascendc or args.triton, args.official_release, backend)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {key: report[key] for key in ("status", "ascendflow", backend, "speedup")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
