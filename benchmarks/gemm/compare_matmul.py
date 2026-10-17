"""Revalidate three NN b16 GEMM trials and enforce generated/ACLNN <= 1."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median

from ascendflow.diagnostics import require
from ascendflow.dialects.physical import BufferOp
from ascendflow.performance import profile_evidence
from benchmarks.gemm.run_matmul import load_contract


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def qualification(directory):
    directory = Path(directory).resolve()
    results, common = [], None
    for trial in range(3):
        pair = {}
        record = read_json(directory / f"trial-{trial}/generated/device/device.json")
        artifact_directory = Path(record["backend"]["library"]).parent
        metadata, artifact, manifest, inventory = load_contract(artifact_directory)
        policy = metadata["policy"]
        dtype = policy["DTYPE"]
        acl_dtype = {"f16": 1, "bf16": 27}[dtype]
        m, n, k = (policy[name] for name in ("M", "N", "K"))
        rows = sorted(
            {row for start in range(0, m, 128) for row in range(start, min(start + 6, m))} | {m - 1}
        )
        caches = {
            op.sym_name.data: op.cache.data if op.cache is not None else "normal"
            for op in artifact.plan.walk()
            if isinstance(op, BufferOp) and op.external.data
        }
        for backend in ("generated", "aclnn"):
            profile = directory / f"trial-{trial}" / backend
            record = read_json(profile / "device/device.json")
            summary = read_json(profile / "summary.json")
            require(record["status"] == "passed" and len(record["cases"]) == 1, "GEMM run failed")
            case = record["cases"][0]
            require(
                case["backend"] == backend
                and case["shape_mnk"] == [m, n, k]
                and case["layout"] == "NN contiguous"
                and case["input_dtype"] == case["output_dtype"] == dtype
                and case["data"] == "normal"
                and case["scope"] == "sampled"
                and case["oracle_rows"] == rows
                and case["checked_elements"] == len(rows) * n
                and case["rtol"] == (2**-10 if dtype == "f16" else 2**-7) / 2 + 1e-5
                and case["atol"] == 0.001
                and case["finite_output"]
                and case["inputs_unchanged"]
                and case["guards_unchanged"]
                and case["warmup"] == 5
                and case["repeats"] == 30
                and case["offset_elements"] == 0
                and case["pointer_alignment_mod32"] == [0, 0, 0]
                and case["generated_buffer_cache_policies"] == caches,
                "GEMM layout/dtype/numerical or launch contract differs",
            )
            require(
                record["artifact"] == metadata and record["policy"] == policy,
                "GEMM frozen artifacts differ",
            )
            require(
                record["runner_sha256"] == digest(Path(__file__).with_name("run_matmul.py"))
                and record["storage_oracle_sha256"]
                == digest(Path(__file__).parents[1] / "validation/dtypes.py"),
                "GEMM runner or storage oracle changed",
            )
            physical = record["health"]["physical_device"]
            require(
                record["health"]["logical_device"] == 0
                and record["hardware"]["physical_device"] == physical,
                "device mapping differs",
            )
            require(len(summary["operators"]) == 1, "GEMM must issue one device task per launch")
            kernel, operator = next(iter(summary["operators"].items()))
            require(operator["blocks"] == [inventory["groups"]], "profile grid differs")
            identity = record["backend"]
            if backend == "generated":
                require(
                    kernel == "gemm"
                    and identity["manifest"] == manifest
                    and digest(identity["library"])
                    == identity["shared_sha256"]
                    == manifest["shared_sha256"],
                    "compiled GEMM identity differs",
                )
                alias = case["runtime_cache_alias"]
                if "bypass" in caches.values():
                    require(
                        alias is not None
                        and alias["offset"] > 0
                        and digest(alias["runtime_library"]) == alias["runtime_sha256"],
                        "runtime cache alias differs",
                    )
                else:
                    require(alias is None, "unexpected cache alias")
            else:
                require(
                    kernel.startswith("aclnnMatmul_")
                    and identity["interface"]
                    == f"aclnnMatmul; {dtype} A and B, {dtype} C; cubeMathType=0"
                    and identity["tensor_dtypes"] == {name: acl_dtype for name in ("A", "B", "C")},
                    "ACLNN actual output/input dtype differs",
                )
                require(
                    identity["adapter_sha256"]
                    == digest(Path(__file__).parents[1] / "references/aclnn.py"),
                    "ACLNN adapter changed",
                )
                require(
                    set(identity["libraries"]) == {"libopapi.so", "libnnopbase.so"},
                    "ACLNN libraries missing",
                )
                for library in identity["libraries"].values():
                    require(digest(library["path"]) == library["sha256"], "ACLNN library changed")
            value = {
                "shape_mnk": [m, n, k],
                "dtype": dtype,
                "seed": case["seed"],
                "input_sha256": case["input_sha256"],
                "physical": physical,
                "platform_sha256": record["hardware"]["platform_sha256"],
                "numpy_version": record["numpy_version"],
            }
            require(common is None or common == value, "GEMM input or device differs across trials")
            common = value
            pair[backend] = {
                "profile": str(profile),
                "median_us": summary["median_us"],
                "evidence": profile_evidence(
                    profile,
                    summary,
                    physical=physical,
                    warmup=5,
                    repeats=30,
                    kernel=kernel,
                    task_type="AI_CORE",
                ),
                "output_sha256": case["output_sha256"],
                "max_abs_error": case["max_abs_error"],
            }
        pair["ratio"] = pair["generated"]["median_us"] / pair["aclnn"]["median_us"]
        results.append(pair)
    generated = median(p["generated"]["median_us"] for p in results)
    aclnn = median(p["aclnn"]["median_us"] for p in results)
    worst = max(p["generated"]["median_us"] for p in results) / min(
        p["aclnn"]["median_us"] for p in results
    )
    return {
        "status": "passed" if worst <= 1.0 else "failed",
        "limit": 1.0,
        "common": common,
        "generated_median_us": generated,
        "aclnn_median_us": aclnn,
        "ratio": generated / aclnn,
        "worst_generated_over_best_aclnn_trial": worst,
        "timing_scope": "median of three device-task p50s; warm allocations, no flush; excludes host gaps",
        "trials": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
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
