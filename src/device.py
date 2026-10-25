"""Small ACL adapter for the single-core examples, with explicit physical mapping.

No Torch, TileLang or cached runtime dependency. Device errors propagate; this
module never resets devices, changes cards, or retries a failed launch.
"""

import ctypes as ct
import os
import re
import subprocess
from functools import cached_property
from pathlib import Path

from .backend.cann import CANN
from .diagnostics import require
from .target.resources import DeviceResources


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
        self._closed = False
        self._bound_kernels = False
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
        ptr = self.allocate_uninitialized(len(contents))
        size = max(len(contents), 1)
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
            )
        return ptr

    def allocate_uninitialized(self, nbytes):
        size = max(nbytes, 1)
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

    @cached_property
    def resources(self):
        """Query the mapped logical device, never a hard-coded 910B core count."""
        require(not self._closed, "kernel runtime is closed")
        return query_device_resources(self.call)

    def kernel(self, library, name, *, parameters=()):
        """Bind a compiled kernel, allocating its workspace once on this stream.

        Only external tensor pointers are passed to the returned callable.
        Uniform size parameters follow source signature order and remain fixed
        for this binding. Another binding owns an independent arena.
        """
        from .runtime.kernel import KernelLaunch

        require(not self._closed, "kernel runtime is closed")
        kernel = KernelLaunch(self, library, name, parameters=parameters)
        self._bound_kernels = True
        return kernel

    def event(self):
        event = ct.c_void_p()
        self.call("aclrtCreateEvent", [ct.POINTER(ct.c_void_p)], ct.byref(event))
        self.events.append(event)
        return event

    def record(self, event):
        self.call("aclrtRecordEvent", [ct.c_void_p, ct.c_void_p], event, self.stream)

    def elapsed_us(self, start, stop):
        """Elapsed ACL stream interval, including host submission gaps."""
        elapsed = ct.c_float()
        self.call(
            "aclrtEventElapsedTime",
            [ct.POINTER(ct.c_float), ct.c_void_p, ct.c_void_p],
            ct.byref(elapsed),
            start,
            stop,
        )
        return elapsed.value * 1000

    def close(self):
        if self._closed:
            return
        if self._bound_kernels:
            self.synchronize()
        for event in self.events:
            self.call("aclrtDestroyEvent", [ct.c_void_p], event)
        for handle in self.allocations:
            self.call("aclrtFree", [ct.c_void_p], handle)
        self.call("aclrtDestroyStream", [ct.c_void_p], self.stream)
        self.call("aclFinalize", [])
        self._closed = True


def query_device_resources(call):
    """Read core counts in an already initialized, explicitly mapped ACL context."""
    counts = []
    # CANN 9.1 aclrtDevAttr: CUBE_CORE_NUM=102, VECTOR_CORE_NUM=201.
    for attr in (102, 201):
        value = ct.c_int64()
        call(
            "aclrtGetDeviceInfo",
            [ct.c_uint32, ct.c_int, ct.POINTER(ct.c_int64)],
            0,
            attr,
            ct.byref(value),
        )
        counts.append(value.value)
    return DeviceResources(*counts)
