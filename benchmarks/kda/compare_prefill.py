"""Compare validated BasicInfo records of identical KDA prefill invocations."""

import argparse
import csv
import json
from pathlib import Path

from ascendflow.diagnostics import require
from benchmarks.kda.reference import verify_provenance
from benchmarks.kda.run import digest
from benchmarks.references.kda_aclnn import verify_reference


def read_profile(directory, backend):
    metrics_path = directory / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    device_path = Path(metrics["device_record"])
    require(digest(device_path) == metrics["device_record_sha256"], "numerical record changed")
    device = json.loads(device_path.read_text())
    require(metrics["status"] == device["status"] == "passed", "profile is not qualified")
    require(set(device["results"]) == {backend}, "profile backend differs")
    require(metrics["aic_metrics"] == "BasicInfo", "profile must use BasicInfo")
    clocks = set()
    for row in metrics["all_collections"]:
        require(digest(row["csv"]) == row["csv_sha256"], "profiling CSV changed")
    for row in metrics["kernels"]:
        with Path(row["csv"]).open() as stream:
            for sample in csv.DictReader(stream):
                clocks.add(int(sample["Current Freq"]))
    for relative, sha in metrics["triton_cache"].items():
        require(digest(directory / relative) == sha, "Triton artifact changed")
    if backend == "ascendflow":
        for artifact in device["generated"].values():
            path = Path(artifact["directory"])
            for suffix, key in ((".asc", "asc_sha256"), (".so", "shared_sha256")):
                files = tuple(path.glob("prefill_*" + suffix))
                require(
                    len(files) == 1 and digest(files[0]) == artifact[key],
                    "AscendFlow artifact changed or disappeared",
                )
    numerical = device["results"][backend]["correctness"]
    require(
        all(m["rmse_ratio"] < device["tolerance_rmse_ratio"] for m in numerical.values()),
        "numerical tolerance failed",
    )
    require(metrics["kernel_us"] > 0, "invalid duration")
    return device, {
        "kernel_us": metrics["kernel_us"],
        "kernels": metrics["kernels"],
        "correctness": numerical,
        "measured_clocks_mhz": sorted(clocks),
        "metrics": str(metrics_path.resolve()),
        "metrics_sha256": digest(metrics_path),
        "replay_mode": metrics.get("replay_mode", "kernel"),
        "phase_host_sync": metrics.get("phase_host_sync", False),
    }


def compare(ascendflow, reference_directory, official_release=None, backend="ascendc"):
    ours_record, ours = read_profile(ascendflow, "ascendflow")
    theirs_record, theirs = read_profile(reference_directory, backend)
    # Each backend may have a different implementation phase graph. These
    # fields and byte digests specify the shared mathematical invocation.
    keys = ("lengths", "heads", "dimension", "dtype", "chunk_size", "chunks_per_sequence")
    configuration = {key: ours_record["configuration"][key] for key in keys}
    require(
        configuration == {key: theirs_record["configuration"][key] for key in keys},
        "prefill configurations differ",
    )
    require(ours_record["input_sha256"] == theirs_record["input_sha256"], "input bytes differ")
    require(
        ours_record["tolerance_rmse_ratio"] == theirs_record["tolerance_rmse_ratio"],
        "numerical tolerances differ",
    )
    require(
        ours_record["health"]["physical_device"] == theirs_record["health"]["physical_device"]
        and ours_record["hardware"]["platform_sha256"]
        == theirs_record["hardware"]["platform_sha256"],
        "device platform differs",
    )
    if backend == "ascendc":
        verify_reference(theirs_record["reference"])
    else:
        require(official_release is not None, "Triton comparison requires --official-release")
        verify_provenance(theirs_record["reference"], official_release)
    require(
        ours["replay_mode"] == theirs["replay_mode"],
        "profiling replay modes differ",
    )
    require(
        ours["measured_clocks_mhz"] == theirs["measured_clocks_mhz"]
        and min(ours["measured_clocks_mhz"]) > 0,
        "measured kernel frequencies differ",
    )
    return {
        "status": "passed",
        "configuration": configuration,
        "physical_device": ours_record["health"]["physical_device"],
        "input_sha256": ours_record["input_sha256"],
        "aic_metrics": "BasicInfo",
        "scope": "sum of device kernel Task Durations; one collection per backend; excludes host gaps",
        "ascendflow": ours,
        backend: theirs,
        "speedup": theirs["kernel_us"] / ours["kernel_us"],
        "reference": theirs_record["reference"],
        "official_release": str(official_release.resolve()) if official_release else None,
        "official_release_sha256": digest(official_release) if official_release else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("ascendflow", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ascendc", type=Path)
    group.add_argument("--triton", type=Path)
    parser.add_argument("--official-release", type=Path)
    args = parser.parse_args()
    backend = "ascendc" if args.ascendc else "triton"
    report = compare(args.ascendflow, args.ascendc or args.triton, args.official_release, backend)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("configuration", "speedup")}, indent=2))


if __name__ == "__main__":
    main()
