"""Qualify three alternating device-task trials against installed ACLNN."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median

from ascendflow.diagnostics import require
from ascendflow.dialects.physical import BufferOp
from ascendflow.performance import profile_evidence
from benchmarks.gemm.run_dense import load_contract
from benchmarks.references.aclnn import GemmReference
from benchmarks.work import read_work


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def qualification(directory):
    directory = Path(directory).resolve()
    results, common = [], None
    for trial in range(3):
        pair = {}
        generated_path = directory / f"trial-{trial}" / "generated"
        generated = read_json(generated_path / "device/device.json")
        artifact_directory = Path(generated["backend"]["library"]).parent
        metadata, artifact, manifest, inventory = load_contract(artifact_directory)
        caches = {
            op.sym_name.data: op.cache.data if op.cache is not None else "normal"
            for op in artifact.plan.walk()
            if isinstance(op, BufferOp) and op.external.data
        }
        for backend in ("generated", "aclnn"):
            profile = directory / f"trial-{trial}" / backend
            record = read_json(profile / "device/device.json")
            summary = read_json(profile / "summary.json")
            require(
                record["status"] == "passed" and len(record["cases"]) == 1, "GEMM run did not pass"
            )
            case = record["cases"][0]
            m, n, k = case["shape_mnk"]
            require(
                case["backend"] == backend
                and case["generated_buffer_cache_policies"] == caches
                and case["data"] == "normal"
                and case["comparison"] == "rtol_atol"
                and case["rtol"] == case["atol"] == 0.001
                and case["finite_output"]
                and case["inputs_unchanged"]
                and case["guards_unchanged"]
                and case["scope"] == "sampled"
                and case["checked_elements"] == len(case["oracle_rows"]) * n
                and case["warmup"] == 5
                and case["repeats"] == 30,
                "GEMM numerical, sampling or launch contract differs",
            )
            require(
                record["artifact"] == metadata
                and record["policy"] == metadata["policy"]
                and [metadata["policy"][key] for key in ("M", "N", "K")] == [m, n, k],
                "GEMM run does not belong to the frozen source/IR/ASC",
            )
            physical = record["health"]["physical_device"]
            require(
                record["hardware"]["physical_device"] == physical
                and record["health"]["logical_device"] == 0,
                "GEMM physical mapping differs",
            )
            work = read_work(case, required_counts=("fma_flops",))
            require(
                work["algorithm_work"]["counts"]["fma_flops"] == 2 * m * n * k,
                "GEMM FLOP contract differs",
            )
            require(len(summary["operators"]) == 1, "require one GEMM task per launch")
            kernel, operator = next(iter(summary["operators"].items()))
            require(operator["blocks"] == [inventory["groups"]], "profile grid differs")
            if backend == "generated":
                require(
                    kernel == artifact.name
                    and record["backend"]["manifest"] == manifest
                    and record["backend"]["shared_sha256"] == manifest["shared_sha256"],
                    "generated GEMM library differs",
                )
                alias = case["runtime_cache_alias"]
                if "bypass" in case["generated_buffer_cache_policies"].values():
                    require(
                        alias is not None
                        and alias["offset"] > 0
                        and digest(alias["runtime_library"]) == alias["runtime_sha256"],
                        "runtime cache alias identity differs",
                    )
                else:
                    require(alias is None, "unexpected cache alias for normal GM")
            else:
                require(
                    kernel.startswith("aclnnMatmul_")
                    and record["backend"]["interface"] == GemmReference.interface,
                    "ACLNN interface differs",
                )
                require(
                    set(record["backend"]["libraries"]) == {"libopapi.so", "libnnopbase.so"}
                    and record["backend"]["adapter_sha256"]
                    == digest(Path(__file__).parents[1] / "references/aclnn.py"),
                    "ACLNN libraries or adapter identity missing",
                )
                for library in record["backend"]["libraries"].values():
                    require(
                        digest(library["path"]) == library["sha256"],
                        "installed ACLNN library changed",
                    )
            identity = {
                "shape_mnk": [m, n, k],
                "input_sha256": case["input_sha256"],
                "seed": case["seed"],
                "oracle_rows": case["oracle_rows"],
                "physical": physical,
                "soc": record["hardware"]["soc"],
                "platform_sha256": record["hardware"]["platform_sha256"],
                "runner_sha256": record["runner_sha256"],
            }
            require(
                common is None or identity == common,
                "GEMM input, numerical scope, runner or device differs across trials",
            )
            common = identity
            evidence = profile_evidence(
                profile,
                summary,
                physical=physical,
                warmup=5,
                repeats=30,
                kernel=kernel,
                task_type="AI_CORE",
            )
            pair[backend] = {
                "profile": str(profile),
                "median_us": summary["median_us"],
                "evidence": evidence,
                "max_abs_error": case["max_abs_error"],
            }
        pair["ratio"] = pair["generated"]["median_us"] / pair["aclnn"]["median_us"]
        results.append(pair)
    generated_us = median(pair["generated"]["median_us"] for pair in results)
    aclnn_us = median(pair["aclnn"]["median_us"] for pair in results)
    worst_separation = max(pair["generated"]["median_us"] for pair in results) / min(
        pair["aclnn"]["median_us"] for pair in results
    )
    return {
        "status": "passed" if worst_separation <= 1.0 else "failed",
        "limit": 1.0,
        "generated_median_us": generated_us,
        "aclnn_median_us": aclnn_us,
        "ratio": generated_us / aclnn_us,
        "worst_generated_over_best_aclnn_trial": worst_separation,
        "timing_scope": "median of three task p50s; warm allocations, no flush; excludes host submission gaps",
        "common": common,
        "trials": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = qualification(args.directory)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key not in {"common", "trials"}},
            indent=2,
        )
    )
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
