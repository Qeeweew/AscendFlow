"""Build upstream KDA's custom OPP with CANN host packaging and selected dynamic AOT signatures."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ascendflow.diagnostics import require
from ascendflow.runtime.process import bounded_process
from benchmarks.references.kda_aclnn import file_digest


def publish(checkout, output, build_record):
    require(build_record["status"] == "passed", "upstream build failed")
    stage = checkout / "build/_CPack_Packages/Linux/External"
    packages = list(stage.glob("*/packages/vendors/fla_npu_transformer"))
    require(len(packages) == 1, "expected one upstream KDA OPP")
    vendor = output / "opp/vendors/fla_npu_transformer"
    require(not vendor.exists(), "reference publication requires a new directory")
    vendor.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(packages[0]), vendor)
    commands, parameters = compile_specializations(checkout, vendor, output)
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=checkout).decode().split("\0")
    record = {
        "status": "passed",
        "soc": "ascend910b",
        "checkout": str(checkout),
        "upstream_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
        ).strip(),
        "source_status": subprocess.check_output(
            ["git", "status", "--short"], cwd=checkout, text=True
        ),
        "source_files_sha256": {
            name: file_digest(checkout / name)
            for name in tracked
            if name and (checkout / name).is_file()
        },
        "runtime_files_sha256": {
            str(p.relative_to(vendor)): file_digest(p)
            for p in sorted(vendor.rglob("*"))
            if p.is_file()
        },
        "build": build_record,
        "specialization_commands": commands,
        "specialization_parameters_sha256": {str(p): file_digest(p) for p in parameters},
        "compiler_options": (checkout / "build/autogen/custom_compile_options.ini").read_text(),
        "cann": os.environ.get("ASCEND_HOME_PATH"),
        "route": "upstream build.sh --jit host package; CANN asc_opc BF16 QKV / FP32 gate, beta and state dynamic signatures",
    }
    require(record["source_status"] == "", "upstream source must be unmodified")
    (vendor / "reference-build.json").write_text(json.dumps(record, indent=2) + "\n")
    return vendor


def compile_specializations(checkout, vendor, output):
    """Compile dtype signatures, with dynamic shapes and upstream options."""
    from ascendflow.backend.cann import CANN

    operations = {
        "RecurrentKda": "recurrent_kda",
        "ChunkKdaFwdPrepare": "chunk_kda_fwd_prepare",
        "ChunkFwdH": "chunk_fwd_h",
        "ChunkKdaFwdFinalize": "chunk_kda_fwd_finalize",
    }
    fp32 = {
        "gate",
        "beta",
        "initial_state",
        "final_state",
        "g",
        "gk",
        "q_rstd",
        "k_rstd",
        "beta_eff",
        "a_log",
        "A_log",
        "dt_bias",
    }
    parameters = output / "parameters"
    parameters.mkdir(parents=True, exist_ok=True)
    impl = vendor / "op_impl/ai_core/tbe/fla_npu_transformer_impl"
    kernel = vendor / "op_impl/ai_core/tbe/kernel"

    def compile_one(item):
        op, name = item
        templates = sorted((checkout / "build/binary/ascend910b/gen").glob(op + "_*_param.json"))
        require(templates, f"upstream did not generate {op} compilation parameters")
        data = json.loads(templates[0].read_text())
        signature = data["op_list"][0]
        for field in signature["inputs"] + signature["outputs"]:
            if field["dtype"] in {"int32", "int64"}:
                field["dtype"] = "int64"
            else:
                field["dtype"] = "float32" if field["name"] in fp32 else "bfloat16"
        signature["bin_filename"] = (
            op
            + "_"
            + hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:32]
        )
        path = parameters / (op + ".json")
        path.write_text(json.dumps(data, indent=2) + "\n")
        target = kernel / "ascend910b" / name
        target.mkdir(parents=True, exist_ok=True)
        command = [
            "asc_opc",
            str(impl / "dynamic" / (name + ".py")),
            "--main_func=" + name,
            "--input_param=" + str(path),
            "--soc_version=Ascend910B1",
            "--output=" + str(target),
            "--impl_mode=high_performance,optional",
            "--simplified_key_mode=0",
            "--op_mode=dynamic",
        ]
        with (parameters / (op + ".log")).open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        return command, path

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(compile_one, operations.items()))
    util = CANN / (
        os.uname().machine
        + "-linux/tikcpp/ascendc_kernel_cmake/fwk_modules/util/ascendc_ops_config.py"
    )
    command = [sys.executable, str(util), "-p", str(kernel / "ascend910b"), "-s", "ascend910b"]
    subprocess.run(command, check=True)
    config = kernel / "config/ascend910b"
    config.mkdir(parents=True, exist_ok=True)
    for path in (kernel / "ascend910b").glob("*.json"):
        shutil.move(str(path), config / path.name)
    return [r[0] for r in results] + [command], [r[1] for r in results]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--dependencies", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    checkout, output = args.checkout.resolve(), args.output.resolve()
    dependencies = args.dependencies.resolve()
    if not args.worker:
        require(not output.exists(), "reference build requires a new output directory")
        root = Path(__file__).resolve().parents[2]
        os.environ["PYTHONPATH"] = str(root) + os.pathsep + os.environ.get("PYTHONPATH", "")
        try:
            bounded_process(
                [
                    sys.executable,
                    str(root / "benchmarks/kda/build_reference.py"),
                    "--checkout",
                    str(checkout),
                    "--dependencies",
                    str(dependencies),
                    "--output",
                    str(output),
                    "--worker",
                ],
                output / "build",
                timeout_s=300,
            )
        except BaseException:
            (output / "opp/vendors/fla_npu_transformer/reference-build.json").unlink(
                missing_ok=True
            )
            raise
        return
    os.environ.setdefault("CMAKE_BUILD_PARALLEL_LEVEL", "32")
    os.chdir(checkout)
    command = [
        "bash",
        "build.sh",
        "--jit",
        "--soc=ascend910b",
        "--vendor_name=fla_npu",
        "--ops=recurrent_kda,chunk_kda_fwd",
        "--cann_3rd_lib_path=" + str(dependencies),
        "-j32",
    ]
    subprocess.run(command, check=True)
    print(
        publish(
            checkout,
            output,
            {
                "status": "passed",
                "command": command,
                "deadline_scope": "outer 300s host + all selected device compilations",
            },
        )
    )


if __name__ == "__main__":
    main()
