"""vllm-ascend custom ACLNN package, built with its official CANN workflow."""

import ctypes as ct
import hashlib
import json
import os
from pathlib import Path

from ascendflow.backend.cann import CANN
from ascendflow.diagnostics import require
from benchmarks.references.aclnn import OperatorReference


class SparseMlaReference(OperatorReference):
    interface = "aclnnSparseFlashMla TND/PA_BBND, shared BF16 KV, band4/causal3, topk1, sink"

    def __init__(self, runtime, pointers, policy, context, artifacts, *, query_lens):
        self.runtime = runtime
        self.vendor = Path(artifacts).resolve()
        require(
            str(self.vendor) in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(":"),
            "custom OPP path must be active before ACL initialization",
        )
        self.build_record = json.loads((self.vendor / "reference-build.json").read_text())
        for name, expected in self.build_record["runtime_files_sha256"].items():
            require(
                hashlib.sha256((self.vendor / name).read_bytes()).hexdigest() == expected,
                f"official reference artifact changed: {name}",
            )
        self.library = self.vendor / "op_api/lib/libcust_opapi.so"
        self.meta = ct.CDLL(str(CANN / "lib64/libnnopbase.so"), mode=ct.RTLD_GLOBAL)
        self.ops = ct.CDLL(str(self.library), mode=ct.RTLD_GLOBAL)
        self.tensors, self.scalars = [], []
        b, h, d, page, ratio, maximum, topk = (
            policy[key] for key in ("B", "H", "D", "PAGE", "RATIO", "MAX_SEQ", "TOPK")
        )
        t = policy["T"]
        require(len(query_lens) == b and sum(query_lens) == t, "reference query lengths differ")
        dtype = 27 if policy["DTYPE"] == "bf16" else 1
        self.metadata_pointer = runtime.allocate(bytes(1024 * 4))
        metadata = self.tensor(self.metadata_pointer, (1024,), dtype=3)
        cu = self.tensor(pointers["cu_seqlens_q"], (b + 1,), dtype=3)
        lengths = self.tensor(pointers["seqused_ori_kv"], (b,), dtype=3)
        cmp_lengths = self.tensor(pointers["seqused_cmp_kv"], (b,), dtype=3)
        residual = self.tensor(pointers["cmp_residual_kv"], (b,), dtype=3)
        ori_topk = self.tensor(pointers["ori_topk_length"], (t, 1), dtype=3)
        # Metadata generation belongs to scheduler preparation and is outside
        # the measured main operator, matching vLLM's separate metadata stage.
        self.prepare(
            "aclnnSparseFlashMlaMetadata",
            [
                cu,
                None,
                None,
                None,
                lengths,
                cmp_lengths,
                residual,
                ori_topk,
                None,
                h,
                1,
                d,
                b,
                int(max(query_lens)),
                context,
                context // ratio,
                0,
                topk,
                ratio,
                4,
                3,
                policy["WINDOW"] - 1,
                0,
                b"TND",
                b"PA_BBND",
                True,
                True,
                metadata,
            ],
            [ct.c_void_p] * 9
            + [ct.c_int64] * 14
            + [ct.c_char_p] * 2
            + [ct.c_bool] * 2
            + [ct.c_void_p],
        )
        self()
        runtime.synchronize()
        self.metadata_sha256 = hashlib.sha256(runtime.read(self.metadata_pointer, 4096)).hexdigest()
        # Retain tensor descriptors; only the completed metadata executor dies.
        destroy = self.meta.aclDestroyAclOpExecutor
        destroy.argtypes, destroy.restype = [ct.c_void_p], ct.c_int32
        if destroy(self.executor):
            raise RuntimeError("metadata executor destroy failed")
        self.prepare(
            "aclnnSparseFlashMla",
            [
                self.tensor(pointers["q"], (t, h, d), dtype=dtype),
                self.tensor(pointers["ori_kv"], (policy["ORI_BLOCKS"], page, 1, d), dtype=dtype),
                self.tensor(
                    pointers["cmp_kv"], (policy["CMP_BLOCKS"], page // ratio, 1, d), dtype=dtype
                ),
                None,
                self.tensor(pointers["cmp_sparse_indices"], (t, 1, topk), dtype=3),
                self.tensor(pointers["ori_block_table"], (b, maximum // page), dtype=3),
                self.tensor(pointers["cmp_block_table"], (b, maximum // page), dtype=3),
                cu,
                None,
                None,
                None,
                lengths,
                cmp_lengths,
                residual,
                ori_topk,
                None,
                self.tensor(pointers["sinks"], (h,)),
                metadata,
                policy["SCALE"],
                ratio,
                4,
                3,
                policy["WINDOW"] - 1,
                0,
                b"TND",
                b"PA_BBND",
                1,
                False,
                self.tensor(pointers["attention_out"], (t, h, d), dtype=dtype),
                None,
            ],
            [ct.c_void_p] * 18
            + [ct.c_double]
            + [ct.c_int64] * 5
            + [ct.c_char_p] * 2
            + [ct.c_int64, ct.c_bool]
            + [ct.c_void_p] * 2,
        )

    def identity(self):
        return {
            "interface": self.interface,
            "build": self.build_record,
            "library": str(self.library),
            "library_sha256": hashlib.sha256(self.library.read_bytes()).hexdigest(),
            "metadata_sha256": self.metadata_sha256,
            "workspace_bytes": self.workspace_size.value,
            "version_scope": "official vllm-ascend custom package on CANN 9.1; unmodified upstream host and device code",
        }
