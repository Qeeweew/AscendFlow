"""Small ACL adapter for the single-core examples, with explicit physical mapping.

No Torch, TileLang or cached runtime dependency. Device errors propagate; this
module never resets devices, changes cards, or retries a failed launch.
"""

import ctypes as ct
import hashlib
import json
import os
import re
import struct
import subprocess
from pathlib import Path

from .model import extract, require
from .toolchain import CANN


def health_snapshot(physical: int, output: Path) -> dict:
    proc = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, check=True)
    output.mkdir(parents=True, exist_ok=True)
    (output / "health-before.txt").write_text(proc.stdout)
    rows = proc.stdout.splitlines()
    devices = {}
    current = None
    for line in rows:
        first = re.search(r"\|\s*(\d+)\s+(910B\w*)\s*\|\s*(\w+)\s*\|", line)
        if first:
            current = int(first.group(1))
            devices[current] = {"name": first.group(2), "health": first.group(3)}
        second = re.search(
            r"\|\s*\d+\s*\|\s*[\w:.]+\s*\|\s*(\d+)\s+"
            r"\d+\s*/\s*\d+\s+(\d+)\s*/\s*(\d+)\s*\|",
            line,
        )
        if second and current is not None:
            devices[current].update(
                utilization=int(second.group(1)),
                used_mb=int(second.group(2)),
                total_mb=int(second.group(3)),
            )
    require(physical in devices, f"physical card {physical} is not visible to npu-smi")
    card = devices[physical]
    require(
        card.get("health") == "OK" and card.get("utilization") == 0,
        f"physical card {physical} is not healthy and idle: {card}",
    )
    require(
        f"No running processes found in NPU {physical} " in proc.stdout,
        f"physical card {physical} has active/unknown processes",
    )
    require(
        card.get("total_mb", 0) - card.get("used_mb", 0) > 64, "insufficient free device memory"
    )
    require(
        os.environ.get("ASCEND_RT_VISIBLE_DEVICES") == str(physical),
        "set ASCEND_RT_VISIBLE_DEVICES to exactly the selected physical card before starting Python",
    )
    require(os.environ.get("ASCEND_DEVICE_ID") == "0", "set ASCEND_DEVICE_ID=0 for this process")
    return {"physical_device": physical, "logical_device": 0, "all_cards": devices}


class ACL:
    def __init__(self):
        self.lib = ct.CDLL(str(CANN / "lib64/libascendcl.so"), mode=ct.RTLD_GLOBAL)
        self.allocations = []
        self.events = []
        self.stream = ct.c_void_p()
        self.call("aclInit", [ct.c_char_p], None)
        self.call("aclrtSetDevice", [ct.c_int], 0)
        count = ct.c_uint()
        self.call("aclrtGetDeviceCount", [ct.POINTER(ct.c_uint)], ct.byref(count))
        require(count.value == 1, f"runtime mapping exposed {count.value} cards; expected one")
        self.call("aclrtCreateStream", [ct.POINTER(ct.c_void_p)], ct.byref(self.stream))

    def call(self, name, argtypes, *args):
        fn = getattr(self.lib, name)
        fn.argtypes, fn.restype = argtypes, ct.c_int
        error = fn(*args)
        if error:
            raise RuntimeError(f"{name} failed with ACL error {error}; device retries are disabled")

    def allocate(self, contents: bytes):
        # Align only the newly allocated argument base, retaining the allocation handle.
        size = max(len(contents), 1)
        handle = ct.c_void_p()
        self.call(
            "aclrtMalloc",
            [ct.POINTER(ct.c_void_p), ct.c_size_t, ct.c_int],
            ct.byref(handle),
            size + 511,
            0,
        )
        self.allocations.append(handle)
        ptr = ct.c_void_p((handle.value + 511) // 512 * 512)
        if contents:
            host = ct.create_string_buffer(contents)
            self.call(
                "aclrtMemcpy",
                [ct.c_void_p, ct.c_size_t, ct.c_void_p, ct.c_size_t, ct.c_int],
                ptr,
                size,
                host,
                len(contents),
                1,
            )  # HOST_TO_DEVICE
        return ptr

    def read(self, ptr, nbytes):
        if not nbytes:
            return b""
        host = ct.create_string_buffer(nbytes)
        self.call(
            "aclrtMemcpy",
            [ct.c_void_p, ct.c_size_t, ct.c_void_p, ct.c_size_t, ct.c_int],
            host,
            nbytes,
            ptr,
            nbytes,
            2,
        )
        return host.raw

    def synchronize(self):
        self.call("aclrtSynchronizeStream", [ct.c_void_p], self.stream)

    def event(self):
        event = ct.c_void_p()
        self.call("aclrtCreateEvent", [ct.POINTER(ct.c_void_p)], ct.byref(event))
        self.events.append(event)
        return event

    def record(self, event):
        self.call("aclrtRecordEvent", [ct.c_void_p, ct.c_void_p], event, self.stream)

    def close(self):
        for event in self.events:
            self.call("aclrtDestroyEvent", [ct.c_void_p], event)
        for handle in self.allocations:
            self.call("aclrtFree", [ct.c_void_p], handle)
        self.call("aclrtDestroyStream", [ct.c_void_p], self.stream)
        self.call("aclFinalize", [])


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
) -> dict:
    from .codegen import emit_asc
    from .sync import verify_plan

    verify_plan(plan)
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
        raw = runtime.read(pointers[-1], len(expected) * 4)
        actual = struct.unpack(f"<{len(expected)}f", raw)
        (output / "device-output.bin").write_bytes(raw)
        max_error = max((abs(a - b) for a, b in zip(actual, expected)), default=0.0)
        require(
            all(a == b for a, b in zip(actual, expected)),
            f"device numerical mismatch; max_abs_error={max_error}, "
            f"first actual={actual[:8]}, expected={expected[:8]}",
        )
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
            "correctness": "exact equality on binary-representable finite test inputs",
            "elements": len(expected),
            "max_abs_error": max_error,
            "warmup": warmup,
            "repeats": repeats,
            "stream_interval_us_per_launch": ms.value * 1000 / repeats,
            "timing_scope": "ACL event stream interval; includes submission gaps; not kernel-only",
            "library": str(shared_library.resolve()),
            "reference": reference,
            "shape": [p.count, p.tasks[0].refs[0].nbytes // 4]
            if p.core == "aiv"
            else [p.count, 16, 16, 16],
            "tail_elements": p.tasks[0].refs[0].tail_nbytes // 4 if p.core == "aiv" else 0,
        }
        (output / "device.json").write_text(json.dumps(record, indent=2) + "\n")
        return record
    finally:
        runtime.close()
