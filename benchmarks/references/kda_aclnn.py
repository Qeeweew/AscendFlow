"""Unmodified flash-linear-attention-npu ACLNN KDA, on the caller's stream.

Build the upstream OPP for ascend910b and pass its vendor directory. No vLLM,
Triton, Python extension, or handwritten reference kernel is involved.
"""

import ctypes as ct
import hashlib
import json
import os
from pathlib import Path

from ascendflow.backend.cann import CANN
from ascendflow.diagnostics import require
from benchmarks.references.aclnn import OperatorReference


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_reference(record):
    vendor = Path(record["vendor"])
    build_path = vendor / "reference-build.json"
    require(file_digest(build_path) == record["build_sha256"], "reference build record changed")
    build = json.loads(build_path.read_text())
    require(build["status"] == "passed" and build["soc"] == "ascend910b", "invalid KDA OPP")
    require(build["source_status"] == "", "reference source was modified")
    for relative, expected in build["source_files_sha256"].items():
        require(
            file_digest(Path(build["checkout"]) / relative) == expected,
            f"KDA reference source changed: {relative}",
        )
    for absolute, expected in build.get("specialization_parameters_sha256", {}).items():
        require(file_digest(absolute) == expected, "KDA reference compilation parameters changed")
    for relative, expected in build["runtime_files_sha256"].items():
        require(file_digest(vendor / relative) == expected, f"KDA reference changed: {relative}")


def configure_reference(vendor):
    """Configure custom OPP discovery before torch_npu initializes CANN."""
    require(vendor is not None, "AscendC requires --reference-artifacts=<OPP vendor directory>")
    vendor = Path(vendor).resolve()
    record = {"vendor": str(vendor), "build_sha256": file_digest(vendor / "reference-build.json")}
    verify_reference(record)
    paths = [str(vendor), os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")]
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = ":".join(path for path in paths if path)
    return record


class _TorchStream:
    def __init__(self):
        import torch

        self.stream = ct.c_void_p(torch.npu.current_stream().npu_stream)
        self.allocations = []

    def allocate_uninitialized(self, size):
        import torch

        tensor = torch.empty(max(size, 1), dtype=torch.uint8, device="npu")
        self.allocations.append(tensor)
        return ct.c_void_p(tensor.data_ptr())


class KdaReference(OperatorReference):
    def __init__(self, vendor):
        self.runtime = _TorchStream()
        self.meta = ct.CDLL(str(CANN / "lib64/libnnopbase.so"), mode=ct.RTLD_GLOBAL)
        self.ops = ct.CDLL(str(Path(vendor) / "op_api/lib/libcust_opapi.so"), mode=ct.RTLD_LOCAL)
        self.tensors, self.scalars, self.arrays, self.outputs = [], [], [], []

    def prepare(self, operator, arguments, types):
        try:
            super().prepare(operator, arguments, types)
        except Exception as error:
            runtime = ct.CDLL(str(CANN / "lib64/libascendcl.so"))
            recent = runtime.aclGetRecentErrMsg
            recent.argtypes, recent.restype = [], ct.c_char_p
            message = recent()
            raise RuntimeError(
                f"{error}; CANN: {message.decode() if message else 'no detail'}"
            ) from error

    def descriptor(self, tensor):
        import torch

        require(tensor.is_contiguous(), "KDA benchmark expects contiguous tensors")
        dtype = {torch.bfloat16: 27, torch.float32: 0, torch.int64: 9}[tensor.dtype]
        return self.tensor(ct.c_void_p(tensor.data_ptr()), tuple(tensor.shape), dtype=dtype)

    def int_array(self, values):
        create = self.meta.aclCreateIntArray
        create.argtypes, create.restype = [ct.POINTER(ct.c_int64), ct.c_uint64], ct.c_void_p
        array = create((ct.c_int64 * len(values))(*values), len(values))
        require(array is not None, "aclCreateIntArray failed")
        self.arrays.append(array)
        return array

    @classmethod
    def recurrent(cls, vendor, device):
        import torch

        require(device["Q"].dtype == torch.bfloat16, "AscendC recurrent requires BF16 Q/K/V")
        result = cls(vendor)
        output = torch.empty_like(device["Value"])
        result.outputs = [output, device["State"]]
        arguments = [
            result.descriptor(device[n])
            for n in ("Q", "Key", "Value", "Gate", "Beta", "State", "Cu", "Indices")
        ]
        arguments += [None, None, None, b"TND", device["Q"].shape[-1] ** -0.5]
        arguments += [False, True, True, False, False, False, False, -5.0, True]
        arguments += [result.descriptor(output), result.descriptor(device["State"])]
        result.prepare(
            "aclnnRecurrentKda",
            arguments,
            [ct.c_void_p] * 11
            + [ct.c_char_p, ct.c_double]
            + [ct.c_bool] * 7
            + [ct.c_double, ct.c_bool]
            + [ct.c_void_p] * 2,
        )
        return result

    @classmethod
    def prefill(cls, vendor, device, cu, chunk_indices):
        import torch

        require(device["Q"].dtype == torch.bfloat16, "KDA comparison requires BF16 Q/K/V")
        result = cls(vendor)
        tokens, heads, _ = device["Q"].shape
        output, state = torch.empty_like(device["Value"]), torch.empty_like(device["Initial"])
        # Match the upstream Python forward's public auxiliary outputs.
        gate = torch.empty((heads, tokens, 128), dtype=torch.float32, device="npu")
        aqk = torch.empty((heads, tokens, 64), dtype=torch.bfloat16, device="npu")
        akk = torch.empty_like(aqk)
        result.outputs = [output, state, gate, aqk, akk]
        arguments = [result.descriptor(device[n]) for n in ("Q", "Key", "Value", "Gate", "Beta")]
        arguments += [
            None,
            None,
            result.descriptor(device["Initial"]),
            result.int_array(cu),
            result.int_array([x for pair in chunk_indices for x in pair]),
        ]
        arguments += [b"TND", 128**-0.5, 64, False, -5.0, False, True]
        arguments += [1e-6, True, False, False, True]
        arguments += [result.descriptor(t) for t in result.outputs] + [None] * 11
        result.prepare(
            "aclnnChunkKdaFwdV2",
            arguments,
            [ct.c_void_p] * 10
            + [ct.c_char_p, ct.c_double, ct.c_int64, ct.c_bool, ct.c_double]
            + [ct.c_bool] * 2
            + [ct.c_double]
            + [ct.c_bool] * 4
            + [ct.c_void_p] * 16,
        )
        return result

    def __call__(self):
        super().__call__()
        return tuple(self.outputs[:2])

    def close(self):
        super().close()
        destroy = self.meta.aclDestroyIntArray
        destroy.argtypes, destroy.restype = [ct.c_void_p], ct.c_int32
        for array in self.arrays:
            require(destroy(array) == 0, "aclDestroyIntArray failed")
