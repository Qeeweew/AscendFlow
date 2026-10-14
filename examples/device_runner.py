"""Shared example data, numerical oracles, and device timing recipes."""

import ctypes as ct
import hashlib
import json
import struct
from pathlib import Path

from xdsl_npu.device import ACL, health_snapshot
from xdsl_npu.model import extract, require


def input_and_oracle(program, *, late_reader=False):
    n = program.count
    if program.name in {"vector_pipeline", "scalar_pipeline"}:
        elements = next(s.nbytes for s in program.storages if s.name == "x") // 4
        values = [(i % 127 - 63) / 16 for i in range(elements)]
        if program.name == "scalar_pipeline":
            expected = [v * values[i // 8 * 8] for i, v in enumerate(values)]
        else:
            expected = [v * (3 if late_reader else 2) + 1 for v in values]
        return [struct.pack(f"<{elements}f", *values), b"\xcd" * (elements * 4)], expected
    require(
        program.name == "cube_pipeline", "device oracle supports only the shipped example recipes"
    )
    # For one 16x16 B16 fractal, NZ is row-major; L0A zN->zZ has no transpose.
    # B is supplied as B^T in NZ (NT contract), loaded into ZN without transpose.
    a = [[((3 * i + k) % 7 - 3) / 4 for k in range(16)] for i in range(n * 16)]
    bt = [[((2 * j + k) % 5 - 2) / 4 for k in range(16)] for j in range(n * 16)]
    expected = [
        sum(a[g * 16 + i][k] * bt[g * 16 + j][k] for k in range(16))
        for g in range(n)
        for i in range(16)
        for j in range(16)
    ]
    return [
        struct.pack(f"<{n * 256}e", *(v for row in a for v in row)),
        struct.pack(f"<{n * 256}e", *(v for row in bt for v in row)),
        b"\xcd" * (n * 1024),
    ], expected


def run_example(
    plan,
    shared_library: Path,
    output: Path,
    *,
    physical=0,
    repeats=30,
    warmup=5,
    late_reader=False,
    reference=False,
    data_case="standard",
) -> dict:
    from xdsl_npu.codegen import emit_asc
    from xdsl_npu.sync import verify_plan

    verify_plan(plan)
    from xdsl_npu.ir import MixedOp

    mixed = isinstance(plan.body.block.first_op, MixedOp)
    if mixed:
        from xdsl_npu.mixed import extract_mixed

        group = extract_mixed(plan)
        from types import SimpleNamespace

        p = SimpleNamespace(name=group.op.sym_name.data, core="mixed", count=group.epochs)
    else:
        p = extract(plan)
    manifest = json.loads(shared_library.with_suffix(".compile.json").read_text())
    require(
        hashlib.sha256(shared_library.read_bytes()).hexdigest() == manifest["shared_sha256"],
        "compiled shared library does not match its recorded artifact",
    )
    if not reference:
        require(
            hashlib.sha256(emit_asc(plan).encode()).hexdigest() == manifest["source_sha256"],
            "compiled source does not match this verified plan/backend; recompile it",
        )
    require(
        repeats > 0 and warmup > 0,
        "correctness requires a launch; timing requires positive repeats",
    )
    health = health_snapshot(physical, output)
    tolerance = (0, 0)
    shape = [p.count, p.tasks[0].refs[0].nbytes // 4] if p.core == "aiv" else [p.count, 16, 16, 16]
    if mixed:
        from oracles import mixed_inputs

        require(
            not reference and data_case == "standard", "mixed oracle requires its generated kernel"
        )
        contents, expected, tolerance, shape = mixed_inputs(plan)
    elif p.name == "recurrent_gdn":
        from recurrent_oracle import recurrent_inputs

        require(not reference and data_case == "standard", "GDN runner requires standard inputs")
        contents, expected, tolerance, shape = recurrent_inputs(p)
    elif p.name in {
        "rms_norm",
        "rope",
        "softmax",
        "swiglu",
        "batched_matmul_nt",
        "gemm_nt",
        "phased_pipeline",
    }:
        from oracles import operator_inputs

        require(
            p.name != "phased_pipeline" or not reference,
            "phased pipeline has its own numerical contract; reference adapter unavailable",
        )

        contents, expected, tolerance, shape = operator_inputs(p, data_case=data_case)
    else:
        require(data_case == "standard", "numerical input cases require an operator recipe")
        contents, expected = input_and_oracle(p, late_reader=late_reader)
    runtime = ACL()
    try:
        pointers = [runtime.allocate(data) for data in contents]
        library = ct.CDLL(str(shared_library.resolve()))
        launch = getattr(library, f"launch_{p.name}")
        launch.argtypes, launch.restype = [ct.c_void_p] * (len(pointers) + 1), None
        for _ in range(warmup):
            launch(runtime.stream, *pointers)
        runtime.synchronize()
        if isinstance(expected, dict):
            parameters = [s for s in p.storages if s.external]
            expected_outputs = {
                name: (
                    pointers[next(i for i, s in enumerate(parameters) if s.name == name)],
                    values,
                )
                for name, values in expected.items()
            }
        else:
            expected_outputs = {"output": (pointers[-1], expected)}
        checked_outputs = {}
        for name, (ptr, values) in expected_outputs.items():
            raw = runtime.read(ptr, len(values) * 4)
            actual = struct.unpack(f"<{len(values)}f", raw)
            filename = "device-output.bin" if name == "output" else f"device-{name}.bin"
            (output / filename).write_bytes(raw)
            error = max((abs(a - b) for a, b in zip(actual, values)), default=0.0)
            require(
                all(
                    abs(a - b) <= tolerance[1] + tolerance[0] * abs(b)
                    for a, b in zip(actual, values)
                ),
                f"device numerical mismatch in {name}; max_abs_error={error}, "
                f"first actual={actual[:8]}, expected={values[:8]}",
            )
            checked_outputs[name] = {"elements": len(values), "max_abs_error": error}
        max_error = max(result["max_abs_error"] for result in checked_outputs.values())
        start, stop = runtime.event(), runtime.event()
        runtime.record(start)
        for _ in range(repeats):
            launch(runtime.stream, *pointers)
        runtime.record(stop)
        runtime.synchronize()
        ms = ct.c_float()
        runtime.call(
            "aclrtEventElapsedTime",
            [ct.POINTER(ct.c_float), ct.c_void_p, ct.c_void_p],
            ct.byref(ms),
            start,
            stop,
        )
        record = {
            "health": health,
            "kernel": p.name,
            "count": p.count,
            "dtype": "f32" if p.core == "aiv" else "f16 operands/f32 accumulation and output",
            "correctness": "compared with the independent FP32 CPU oracle",
            "rtol": tolerance[0],
            "atol": tolerance[1],
            "elements": sum(result["elements"] for result in checked_outputs.values()),
            "max_abs_error": max_error,
            "outputs": checked_outputs,
            "warmup": warmup,
            "repeats": repeats,
            "stream_interval_us_per_launch": ms.value * 1000 / repeats,
            "timing_scope": "ACL event stream interval; includes submission gaps; not kernel-only",
            "library": str(shared_library.resolve()),
            "reference": reference,
            "shape": shape,
            "data_case": data_case,
            "argument_sha256": [hashlib.sha256(data).hexdigest() for data in contents],
            "compiled_source_sha256": manifest["source_sha256"],
            "shared_sha256": manifest["shared_sha256"],
            "tail_elements": p.tasks[0].refs[0].tail_nbytes // 4 if p.core == "aiv" else 0,
        }
        (output / "device.json").write_text(json.dumps(record, indent=2) + "\n")
        return record
    finally:
        runtime.close()
