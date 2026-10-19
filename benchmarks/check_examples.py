"""Compile every runnable example, optionally checking its device output."""

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from ascendflow.device import health_snapshot
from ascendflow.runtime.process import bounded_process, wait_for_process

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Example:
    name: str
    entries: tuple[tuple[str, str], ...]
    build: tuple[str, ...]
    run: tuple[str, ...]
    device_subdir: str = ""
    result_glob: str = "**/device*.json"


def examples():
    result = []
    for name, filename, entry, module, options in (
        ("scale", "scale", "scale", "run", ("--counts", "0", "1", "7", "--repeats", "1")),
        (
            "grouped-scale",
            "grouped",
            "grouped_scale",
            "run",
            ("--example", "grouped", "--counts", "16", "48", "128", "--repeats", "1"),
        ),
        (
            "rings",
            "independent_rings",
            "independent_rings",
            "run_rings",
            ("--counts", "0", "1", "7", "35", "--repeats", "1"),
        ),
        (
            "masks",
            "masks",
            "masks",
            "run_masks",
            (
                "--cases",
                "0:0:0",
                "7:0:0",
                "0:32:96",
                "7:32:96",
                "0:0:1000000000",
                "9223372036854775807:0:0",
                "12345:64:64",
                "3:0:1000000000",
                "--repeats",
                "1",
            ),
        ),
        (
            "mask-rows",
            "mask_rows",
            "mask_rows",
            "run_mask_rows",
            (
                "--cases",
                "3:100:127",
                "129:65:65",
                "0:0:0",
                "300:1000000:1000000",
                "--repeats",
                "1",
            ),
        ),
    ):
        source = f"examples/vector/{filename}.py"
        result.append(
            Example(
                name,
                ((source, entry),),
                ("ascendflow", "--input", source, "--entry", entry, "--shared"),
                (f"benchmarks.vector.{module}", "--library", f"{{output}}/{entry}.so", *options),
            )
        )

    def add(name, source, entry, module, options=(), checks=()):
        result.append(
            Example(
                name,
                ((f"examples/{source}.py", entry),),
                (f"benchmarks.{module}", "--build", *options),
                (f"benchmarks.{module}", *options, *checks),
            )
        )

    for operator, source, entry, rows in (
        ("rms", "rms_norm", "rms_normalize", ("0", "1", "7")),
        ("softmax", "softmax", "softmax", ("0", "8", "24")),
    ):
        add(
            operator,
            f"normalization/{source}",
            entry,
            "normalization.run",
            ("--variant", "generated", "--operator", operator),
            ("--rows", *rows, "--repeats", "1"),
        )
    add(
        "batched-gemm",
        "gemm/batched",
        "batched_matmul",
        "gemm.run",
        ("--variant", "generated"),
        ("--tiles", "0", "1", "7", "--repeats", "1"),
    )
    add(
        "reduction-gemm",
        "gemm/reduction",
        "reduction_gemm",
        "gemm.run_reduction",
        ("--variant", "generated"),
        ("--panels", "0", "1", "7", "--repeats", "1"),
    )
    add(
        "hierarchical-gemm",
        "gemm/hierarchical",
        "hierarchical_gemm",
        "gemm.run_hierarchical",
        ("--variant", "generated"),
        ("--cases", "0:0", "1:0", "1:1", "3:7", "--repeats", "1"),
    )
    add(
        "dense-gemm",
        "gemm/dense",
        "dense_gemm",
        "gemm.run_dense",
        ("--m", "256", "--n", "512", "--k", "256"),
        ("--scope", "full", "--warmup", "1", "--repeats", "1"),
    )
    add(
        "gemm",
        "gemm/matmul",
        "gemm",
        "gemm.run_matmul",
        ("--m", "129", "--n", "257", "--k", "259"),
        ("--scope", "full", "--warmup", "1", "--repeats", "1"),
    )
    add("nested-cv", "cv/nested", "nested_cv", "cv.run")
    add(
        "state",
        "state/recurrence",
        "state_sum",
        "state.run",
        ("--variant", "generated"),
        ("--counts", "0", "1", "7", "--repeats", "1"),
    )
    for name, filename, entry, mode in (
        ("grouped-state", "grouped", "grouped_state_sum", ()),
        ("shared-state", "shared_inputs", "shared_state_sum", ("--shared-inputs",)),
        ("frame-state", "frame_snapshot", "frame_state_sum", ("--frame-snapshot",)),
    ):
        add(
            name,
            f"state/{filename}",
            entry,
            "state.run_grouped",
            ("--variant", "generated", *mode),
            ("--cases", "0:0", "1:0", "1:1", "3:7", "--repeats", "1"),
        )
    add(
        "flash-attention",
        "flash_attention/attention",
        "attention_target",
        "attention.run",
        (),
        ("--cases", "256:1024", "768:3072", "--repeats", "1"),
    )
    source, entry = "examples/vector/scalar_store.py", "scalar_store"
    result.append(
        Example(
            "scalar-store",
            ((source, entry),),
            ("ascendflow", "--input", source, "--entry", entry, "--shared"),
            ("benchmarks.vector.run_scalar_store", "--artifact", "{output}"),
            "validation",
        )
    )
    for filename, entry, parameters, lengths in (
        (
            "recurrent",
            "recurrent_kda",
            {"TOKENS": 4, "SEQUENCES": 4, "SLOTS": 5, "INDEX_WIDTH": 4, "H": 8, "HV": 8},
            ("0", "1", "3", "0"),
        ),
        (
            "decode",
            "decode_kda",
            {"SEQUENCES": 4, "SLOTS": 5, "H": 8, "HV": 8, "HEAD_GROUPS": 2},
            ("1", "1", "1", "1"),
        ),
    ):
        source = f"examples/kda/{filename}.py"
        result.append(
            Example(
                f"kda-{filename}",
                ((source, entry),),
                (
                    "ascendflow",
                    "--input",
                    source,
                    "--entry",
                    entry,
                    "--parameters",
                    json.dumps(parameters),
                    "--shared",
                ),
                (
                    "benchmarks.kda.run",
                    "--worker",
                    "--artifact",
                    "{output}",
                    "--backend",
                    "ascendflow",
                    "--lengths",
                    *lengths,
                    "--warmup",
                    "0",
                    "--repeats",
                    "1",
                ),
                "validation",
                "validation/results.json",
            )
        )
    from benchmarks.kda.prefill import KERNELS, PHASES, SOURCES

    result.append(
        Example(
            "kda-prefill",
            tuple(dict.fromkeys((SOURCES[phase], KERNELS[phase]) for phase in PHASES)),
            (
                "benchmarks.kda.prefill",
                "--build-only",
                "--build-worker",
                "--artifacts",
                "{output}",
                "--heads",
            "8",
                "--lengths",
                "127",
            ),
            (
                "benchmarks.kda.prefill",
                "--worker",
                "--artifacts",
                "{output}",
                "--backend",
                "ascendflow",
                "--warmup",
                "0",
                "--repeats",
                "1",
            ),
            "validation",
            "validation/results.json",
        )
    )
    return tuple(result)


def run_device(command, directory):
    from time import monotonic

    directory.mkdir()
    with (directory / "run.log").open("w") as log:
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        code = wait_for_process(process, monotonic() + 300)
    if code:
        raise RuntimeError(f"device check failed; see {directory / 'run.log'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--physical", type=int, help="explicitly mapped healthy idle card")
    parser.add_argument("--only", nargs="+", choices=[case.name for case in examples()])
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.list:
        for case in examples():
            print(f"{case.name:20} " + ", ".join(f"{p}::{e}" for p, e in case.entries))
        return 0
    if args.output is None:
        parser.error("--output is required")
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "physical": args.physical, "cases": []}
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
        if args.physical is not None:
            report["health_before"] = health_snapshot(args.physical, directory)
        for case in examples():
            if args.only and case.name not in args.only:
                continue
            output = directory / case.name
            row = {
                "name": case.name,
                "entries": [{"source": source, "entry": entry} for source, entry in case.entries],
                "status": "building",
                "output": str(output),
            }
            report["cases"].append(row)
            command = [
                sys.executable,
                "-m",
                *(part.replace("{output}", str(output)) for part in case.build),
                "--output",
                str(output),
            ]
            row["build_command"] = command
            save()
            # Imports, construction, IR, ASC and CANN share this hard deadline.
            row["build"] = bounded_process(command, output / "build-step", timeout_s=300)
            publications = {
                json.loads(path.read_text())["kernel"]: str(path)
                for path in output.rglob("verification.json")
            }
            missing = {entry for _, entry in case.entries} - publications.keys()
            if missing:
                raise RuntimeError(f"missing compiled example entries: {sorted(missing)}")
            row["publications"] = publications
            row["status"] = "compiled"
            if args.physical is not None:
                command = [
                    sys.executable,
                    "-m",
                    *(part.replace("{output}", str(output)) for part in case.run),
                    "--output",
                    str(output / case.device_subdir),
                    "--physical",
                    str(args.physical),
                ]
                row["run_command"] = command
                run_device(command, output / "run-step")
                records = sorted(output.glob(case.result_glob))
                if not records:
                    raise RuntimeError(f"missing device evidence: {output}")
                row["device_records"] = {
                    str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in records
                }
                row["status"] = "device-passed"
            print(f"{case.name}: {row['status']}", flush=True)
            save()
        if args.physical is not None:
            report["health_after"] = health_snapshot(args.physical, directory / "after")
        report["status"] = "passed"
    except Exception as error:  # noqa: BLE001 - retain failures from any build or runner
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        if report["cases"]:
            report["cases"][-1]["status"] = "failed"
        print(report["error"], file=sys.stderr)
    finally:
        save()
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
