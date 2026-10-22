"""Evict prior workloads from L2 with a disjoint, verified Vector sweep."""

import ctypes as ct
import hashlib
import json
import struct
from pathlib import Path

from ascendflow.diagnostics import require
from benchmarks.artifacts import read_artifact


class CacheEviction:
    """Read a large disjoint input; export only one checksum tile per core."""

    def __init__(self, runtime, artifacts, l2_bytes):
        self.runtime = runtime
        directory = Path(artifacts).resolve()
        self.policy = json.loads((directory / "eviction-policy.json").read_text())
        require(
            self.policy["parameters"] == {"B": 8192, "GROUPS": 40},
            "eviction artifact must specialize the declared cache_sweep sweep",
        )
        self.tiles = self.policy["tiles"]
        self.output_size = 40 * 8192 * 4
        self.size = self.tiles * self.output_size
        require(
            self.size >= 4 * l2_bytes and self.tiles > 0,
            "eviction sweep needs complete tiles and at least four L2 capacities",
        )
        frozen = read_artifact(directory)
        manifest = json.loads((directory / "cache_sweep.compile.json").read_text())
        library = directory / "cache_sweep.so"
        require(frozen.name == "cache_sweep", "unexpected eviction kernel")
        require(
            hashlib.sha256(frozen.asc.encode()).hexdigest() == manifest["source_sha256"],
            "eviction source identity changed",
        )
        require(
            hashlib.sha256(library.read_bytes()).hexdigest() == manifest["shared_sha256"],
            "eviction library identity changed",
        )
        self.module = ct.CDLL(str(library))
        self.launch = self.module.launch_cache_sweep
        self.launch.argtypes = [ct.c_void_p] * 3 + [ct.c_int64]
        self.launch.restype = ct.c_int
        self.x = runtime.allocate_uninitialized(self.size + 512)
        self.y = runtime.allocate_uninitialized(self.output_size + 512)
        for pointer, size, value in (
            (self.x, self.size, 0x3F),
            (self.y, self.output_size, 0xCD),
        ):
            runtime.call(
                "aclrtMemset",
                [ct.c_void_p, ct.c_size_t, ct.c_int32, ct.c_size_t],
                pointer,
                size + 512,
                value,
                size + 512,
            )
        self.calls = 0
        self()
        runtime.synchronize()
        value = struct.unpack("f", b"\x3f" * 4)[0]
        checksum = 0.0
        for _ in range(self.tiles):
            checksum = struct.unpack("f", struct.pack("f", checksum + value))[0]
        expected = struct.pack("f", checksum) * (self.output_size // 4)
        require(
            runtime.read(self.y, self.output_size + 512) == expected + b"\xcd" * 512,
            "eviction sweep checksum/guard differs",
        )
        self.identity = {
            "policy": self.policy,
            "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "artifact": str(directory),
            "manifest": manifest,
            "l2_bytes": l2_bytes,
            "read_capacity_ratio": self.size / l2_bytes,
            "read_bytes": self.size,
            "write_bytes": self.output_size,
            "method": "disjoint cacheable GM->UB read, FP32 checksum retained in UB, one output tile per core; same stream immediately before each attention kernel",
            "validation": "all checksum elements and output tail guard checked",
            "timing": "sweep tasks excluded; report attention device task duration only",
        }

    def __call__(self):
        require(
            self.launch(self.runtime.stream, self.x, self.y, self.tiles) == 0,
            "cache eviction kernel launch failed",
        )
        self.calls += 1
