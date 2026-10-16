"""Run the fixed FA example or independent ACLNN/AscendC baselines.

Reusing a compiled artifact never invokes the compiler. Numerical sampling is
explicit, and ACL event intervals include stream submission gaps. Set the
physical-to-logical device mapping before starting this process.
"""

import argparse
import ast
import ctypes as ct
import hashlib
import json
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path

from xdsl.dialects.builtin import f32

import benchmarks.validation.attention as attention_validation
from benchmarks.references.attention import NAME as REFERENCE_NAME
from benchmarks.references.attention import reference_source
from benchmarks.references.attention_aclnn import FlashAttentionReference
from benchmarks.validation.attention import decode, inputs_and_oracle
from xdsl_npu.build import build_file
from xdsl_npu.device import ACL, health_snapshot
from xdsl_npu.ir import BufferOp, MixedOp, RuntimeSizeOp, parse
from xdsl_npu.model import require
from xdsl_npu.performance import hardware_snapshot
from xdsl_npu.source_ir import AllocationOp, ElementOp, ProgramOp
from xdsl_npu.toolchain import CANN

ENTRY = "attention_target"
EXAMPLE = Path(__file__).resolve().parents[2] / "examples/flash_attention/attention.py"
POLICY = {"batch": 1, "heads": 8, "kv_heads": 8, "dim": 128, "groups": 20, "dtype": "f16"}
CONSTANTS = {
    "B": 1,
    "HQ": 8,
    "HKV": 8,
    "D": 128,
    "BQ": 128,
    "BK": 1024,
    "CK": 128,
    "VR": 8,
    "GROUPS": 20,
    "DTYPE": "f16",
    "SCALE": 0.08838834764831845,
}
SCALE = struct.unpack("<f", struct.pack("<f", CONSTANTS["SCALE"]))[0]
BUFFER_NAMES = {"Q": "q", "K": "k", "V": "v", "O": "o", "Score": "score", "P": "p", "PV": "pv"}
SHAPES = {
    "Q": ((1, 8, -1, 128), "f16", "Sq"),
    "K": ((1, 8, -1, 128), "f16", "Sk"),
    "V": ((1, 8, -1, 128), "f16", "Sk"),
    "O": ((1, 8, -1, 128), "f16", "Sq"),
    "Score": ((20, 2, 128, 1024), "f32", None),
    "P": ((20, 2, 128, 1024), "f16", None),
    "PV": ((20, 2, 128, 128), "f32", None),
}
GUARD = b"\x5a" * 512
TIMING_SCOPE = "ACL stream event interval including host submission gaps; not kernel-only"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_bytes(path):
    path = Path(path)
    require(path.is_file(), f"missing attention artifact: {path}")
    return path.read_bytes()


def read_json(path):
    return json.loads(read_bytes(path))


def example_policy(path=EXAMPLE):
    """Check the actual example defaults without importing or compiling it."""
    source = read_bytes(path)
    functions = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == ENTRY
    ]
    require(len(functions) == 1, "attention example entry changed")
    function = functions[0]
    require(
        len(function.args.defaults) == len(CONSTANTS),
        "attention example constexpr defaults changed",
    )
    defaults = {}
    for parameter, default in zip(
        function.args.args[-len(function.args.defaults) :], function.args.defaults, strict=True
    ):
        require(
            isinstance(parameter.annotation, ast.Attribute)
            and parameter.annotation.attr == "constexpr",
            "attention example policy must use explicit constexpr defaults",
        )
        defaults[parameter.arg] = (
            default.attr if isinstance(default, ast.Attribute) else ast.literal_eval(default)
        )
    require(
        defaults == CONSTANTS, "attention example policy differs from the validated fixed policy"
    )
    return {
        "path": str(Path(path).resolve()),
        "sha256": sha256(source),
        "entry": ENTRY,
        "constexprs": defaults,
    }


@dataclass(frozen=True)
class LauncherABI:
    symbol: str
    names: tuple[str, ...]
    keys: tuple[str, ...]

    @property
    def argtypes(self):
        return [ct.c_int64 if key in {"sq", "sk"} else ct.c_void_p for key in self.keys]

    def arguments(self, stream, pointers, sq, sk):
        values = {**pointers, "stream": stream, "sq": sq, "sk": sk}
        return tuple(values[key] for key in self.keys)


def launcher_abi(source, name, *, generated):
    """Bind named parameters from this artifact, never from DSL source order."""
    # Comments cannot supply a second apparent launcher or hide a parameter.
    source = re.sub(r"/\*.*?\*/|//[^\n]*", "", source, flags=re.DOTALL)
    signatures = re.findall(r'extern\s+"C"\s+int\s+(launch_\w+)\s*\(([^()]*)\)\s*\{', source)
    require(
        len(signatures) == 1 and signatures[0][0] == "launch_" + name,
        "attention artifact needs exactly its declared extern launcher",
    )
    pointers = (
        {"arg_" + key: value for key, value in BUFFER_NAMES.items()}
        if generated
        else {value: value for value in BUFFER_NAMES.values()}
    )
    sizes = {"npu_size_Sq": "sq", "npu_size_Sk": "sk"} if generated else {"sq": "sq", "sk": "sk"}
    names, keys = [], []
    for parameter in signatures[0][1].split(","):
        match = re.fullmatch(r"\s*(void\s*\*|int64_t)\s*(\w+)\s*", parameter)
        require(match is not None, "unsupported attention launcher parameter type")
        kind, argument = match.groups()
        if argument == "stream":
            require(
                not names and kind.replace(" ", "") == "void*",
                "attention launcher must start with its stream pointer",
            )
            key = "stream"
        elif argument in pointers:
            require(
                kind.replace(" ", "") == "void*",
                "attention buffer launcher parameter must be a pointer",
            )
            key = pointers[argument]
        else:
            require(argument in sizes and kind == "int64_t", "unknown attention launcher parameter")
            key = sizes[argument]
        names.append(argument)
        keys.append(key)
    require(
        len(keys) == len(set(keys)) == 10
        and set(keys) == {"stream", "sq", "sk", *BUFFER_NAMES.values()},
        "attention launcher has missing or duplicate named parameters",
    )
    require(keys[0] == "stream", "attention launcher must start with its stream pointer")
    return LauncherABI(signatures[0][0], tuple(names), tuple(keys))


def generated_contract(directory):
    """Read actual source shapes and physical GM extents; do not replan IR."""
    directory = Path(directory)
    before = read_bytes(directory / "before.xdsl")
    source = read_bytes(directory / "source.xdsl")
    physical, logical = parse(before.decode()), parse(source.decode())
    root, program = physical.body.block.first_op, logical.body.block.first_op
    require(
        isinstance(root, MixedOp)
        and root.sym_name.data == ENTRY
        and root.target.data == "dav-2201"
        and root.blocks.data == 20
        and root.disjoint_external.data == 1,
        "compiled attention physical policy differs from B1/H8/D128/groups20/f16",
    )
    require(
        isinstance(program, ProgramOp)
        and program.sym_name.data == ENTRY
        and program.kind.data == "mixed"
        and program.groups.data == 20
        and program.disjoint_external.data == 1,
        "compiled attention source policy differs from the validated example",
    )
    footprints = {
        (tuple(n.data for n in op.ref.type.shape), op.ref.type.dtype.data, op.ref.type.memory.data)
        for op in program.body.block.ops
        if isinstance(op, AllocationOp) and op.access.data == "local"
    }
    require(
        {
            ((2, 128, 128), "f16", "L1"),
            ((2, 8, 1024), "f32", "UB"),
            ((8, 1024), "f32", "UB"),
        }
        <= footprints,
        "compiled attention CK/VR local tensor policy differs",
    )
    scale_operations = [
        op
        for op in logical.walk()
        if isinstance(op, ElementOp)
        and op.kind.data == "mul_scalar"
        and op.scalar is not None
        and tuple(n.data for n in op.destination.type.shape) == (8, 1024)
    ]
    require(
        scale_operations
        and all(op.scalar.type == f32 for op in scale_operations)
        and {struct.pack("<f", op.scalar.value.data) for op in scale_operations}
        == {struct.pack("<f", SCALE)},
        "compiled attention F32 softmax scale differs",
    )
    allocations = {
        op.sym_name.data: op
        for op in program.body.block.ops
        if isinstance(op, AllocationOp) and op.access.data != "local"
    }
    buffers = {
        op.sym_name.data: op
        for op in physical.walk()
        if isinstance(op, BufferOp) and op.external.data
    }
    require(
        allocations.keys() == buffers.keys() == SHAPES.keys(),
        "compiled attention external allocation names differ",
    )
    sizes = {op.sym_name.data: op for op in root.body.block.ops if isinstance(op, RuntimeSizeOp)}
    require(
        sizes.keys() == {"Sq", "Sk"}
        and all(
            sizes[name].minimum.data == 1 and sizes[name].multiple.data == multiple
            for name, multiple in (("Sq", 128), ("Sk", 1024))
        ),
        "compiled attention runtime tile contracts differ",
    )
    contracts = {}
    for name, (shape, dtype, extent) in SHAPES.items():
        allocation, buffer = allocations[name], buffers[name]
        require(
            tuple(n.data for n in allocation.ref.type.shape) == shape
            and allocation.ref.type.dtype.data == dtype
            and allocation.ref.type.memory.data == "GM"
            and allocation.ref.type.layout.data == "linear",
            f"compiled attention tensor policy differs for {name}",
        )
        require(
            allocation.versions.data == 1
            and allocation.ring.data == 0
            and allocation.access.data == ("read" if name in {"Q", "K", "V"} else "write"),
            f"compiled attention access contract differs for {name}",
        )
        dimensions = tuple(
            value.owner.attributes.get("sym_name") for value in allocation.dimensions
        )
        require(
            len(dimensions) == int(extent is not None)
            and all(value is not None for value in dimensions)
            and tuple(value.data for value in dimensions) == (() if extent is None else (extent,)),
            f"compiled attention source extent differs for {name}",
        )
        nbytes = 0 if extent else math.prod(shape) * (4 if dtype == "f32" else 2)
        scale = 2048 if extent else 0
        require(
            buffer.parent_op() is root
            and buffer.ref.type.bank.data == "GM"
            and buffer.ref.type.dtype.data == dtype
            and buffer.nbytes.data == nbytes
            and buffer.count_scale is not None
            and buffer.count_scale.data == scale
            and buffer.initialized.data == int(name in {"Q", "K", "V"})
            and buffer.extent_count is (None if extent is None else sizes[extent].value),
            f"compiled attention physical extent/type differs for {name}",
        )
        contracts[BUFFER_NAMES[name]] = {
            "shape": shape,
            "dtype": dtype,
            "extent": extent,
            "fixed_bytes": nbytes,
            "bytes_per_extent": scale,
        }
    return {
        "before_ir_sha256": sha256(before),
        "source_ir_sha256": sha256(source),
        "buffers": contracts,
    }


@dataclass(frozen=True)
class Backend:
    name: str
    library: Path
    identity: dict
    abi: LauncherABI | None = None


def load_backend(output, backend, *, reference_output=None):
    """Validate only the selected backend's real binaries and provenance."""
    if backend == "aclnn":
        files = {name: CANN / "lib64" / name for name in ("libopapi.so", "libnnopbase.so")}
        identities = {
            name: {"path": str(path.resolve()), "sha256": sha256(read_bytes(path))}
            for name, path in files.items()
        }
        library = files["libopapi.so"].resolve()
        return Backend(
            "aclnn",
            library,
            {
                "backend": "aclnn",
                "kernel": FlashAttentionReference.interface,
                "library": str(library),
                "shared_sha256": identities["libopapi.so"]["sha256"],
                "libraries": identities,
                "adapter_sha256": sha256(
                    read_bytes(Path(__file__).parents[1] / "references/attention_aclnn.py")
                ),
            },
        )
    require(backend in {"generated", "ascendc"}, "unknown attention backend")
    require(
        backend != "ascendc" or reference_output is not None,
        "AscendC requires --reference-output with its own compiled artifact",
    )
    directory = Path(output if backend == "generated" else reference_output).resolve()
    name = ENTRY if backend == "generated" else REFERENCE_NAME
    report = read_json(directory / "verification.json")
    require(
        report.get("kernel") == name
        and (report.get("kind") == "asc_reference") == (backend == "ascendc"),
        "attention backend/artifact identity differs",
    )
    source_bytes = read_bytes(directory / (name + ".asc"))
    source = source_bytes.decode()
    library = directory / (name + ".so")
    manifest = read_json(directory / (name + ".compile.json"))
    shared_hash = sha256(read_bytes(library))
    require(
        manifest.get("target") == "dav-2201"
        and manifest.get("source_sha256") == sha256(source_bytes)
        and manifest.get("shared_sha256") == shared_hash,
        "attention source/library hash or target differs from its build manifest",
    )
    if backend == "generated":
        require(
            report.get("cross_query_verified") is True
            and report.get("query_verification", {}).get("verified") is True
            and report["query_verification"].get("resources", {}).get("verified") is True,
            "generated attention artifact lacks its completed query proof report",
        )
        contract = generated_contract(directory)
    else:
        require(
            source == reference_source(),
            "AscendC artifact differs from the validated fixed reference policy",
        )
        contract = {"reference_policy": POLICY}
    abi = launcher_abi(source, name, generated=backend == "generated")
    return Backend(
        backend,
        library,
        {
            "backend": backend,
            "kernel": name,
            "library": str(library),
            "shared_sha256": shared_hash,
            "compiled_source": str(directory / (name + ".asc")),
            "compiled_source_sha256": sha256(source_bytes),
            "compile_manifest": manifest,
            "launcher_symbol": abi.symbol,
            "launcher_parameters": abi.names,
            "contract": contract,
        },
        abi,
    )


def case_shape(value):
    match = re.fullmatch(r"([1-9]\d*):([1-9]\d*)", value)
    if match is None:
        raise argparse.ArgumentTypeError("cases must be positive Sq:Sk")
    sq, sk = map(int, match.groups())
    if sq % 128 or sk % 1024 or max(sq, sk) > (2**63 - 1) // 2048:
        raise argparse.ArgumentTypeError(
            "cases require complete BQ128/BK1024 tiles within int64 extents"
        )
    return sq, sk


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--physical", type=int)
    parser.add_argument("--backend", choices=("generated", "aclnn", "ascendc"), default="generated")
    parser.add_argument("--reference-output", type=Path)
    parser.add_argument(
        "--cases",
        type=case_shape,
        nargs="+",
        default=[(128, 1024), (384, 3072), (768, 1024), (512, 2048)],
    )
    parser.add_argument("--scope", choices=("full", "sampled"), default="full")
    parser.add_argument("--data-case", choices=("standard", "uniform", "sharp"), default="standard")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--record", type=Path)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--timeout", type=float, default=300)
    for key, value in POLICY.items():
        parser.add_argument(
            "--" + key.replace("_", "-"), type=str if key == "dtype" else int, default=value
        )
    args = parser.parse_args(argv)
    require(args.warmup >= 1 and args.repeats > 0, "warmup >=1 and repeats >0 are required")
    require(args.physical is None or args.physical >= 0, "physical device must be nonnegative")
    require(
        math.isfinite(args.timeout) and 0 < args.timeout <= 300,
        "one build must finish within 300 seconds",
    )
    require(
        {key: getattr(args, key) for key in POLICY} == POLICY,
        "runner supports only the validated fixed attention policy; variants are not qualified",
    )
    require(
        args.backend != "ascendc" or args.reference_output is not None,
        "AscendC requires --reference-output",
    )
    require(
        not args.build or args.backend == "generated", "--build compiles only the generated example"
    )
    require(
        args.reference_output is None or args.backend == "ascendc",
        "--reference-output belongs only to the AscendC backend",
    )
    return args


def expected_bytes(sq, sk):
    return {
        "q": sq * 2048,
        "k": sk * 2048,
        "v": sk * 2048,
        "o": sq * 2048,
        "score": 20971520,
        "p": 10485760,
        "pv": 2621440,
    }


def write_record(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def run(args):
    policy = example_policy()
    if args.build:
        build_file(EXAMPLE, args.output, entry=ENTRY, shared=True, timeout_s=args.timeout)
    backend = load_backend(args.output, args.backend, reference_output=args.reference_output)
    record_path = args.record or args.output / (
        "device-" + args.backend + "-" + args.data_case + ".json"
    )
    report = {
        "status": "artifact_validated",
        "backend": args.backend,
        "artifact": backend.identity,
        "policy": POLICY,
        "example": policy,
        "cases": [],
        "runner_sha256": sha256(read_bytes(__file__)),
        "oracle_sha256": sha256(read_bytes(attention_validation.__file__)),
        "timing_scope": TIMING_SCOPE,
        "comparison_scope": "fixed dense noncausal BNSD f16 policy; sampled checks remain sampled; no variant or kernel-only performance claim",
    }
    if args.physical is None:
        if args.record is not None:
            write_record(record_path, report)
        print(json.dumps(report, indent=2))
        return report
    import numpy as np

    prepared = []
    for sq, sk in args.cases:
        raw, expected, selected, oracle_s = inputs_and_oracle(
            POLICY, sq, sk, scope=args.scope, data_case=args.data_case
        )
        require(
            {name: len(data) for name, data in raw.items()} == expected_bytes(sq, sk),
            "attention oracle allocation contract differs from this compiled policy",
        )
        prepared.append((sq, sk, raw, expected, selected, oracle_s))
    # This inspects every card and enforces one explicitly mapped physical
    # card as logical zero, before this process initializes any ACL context.
    report["health"] = health_snapshot(args.physical, record_path.parent)
    report["hardware"] = hardware_snapshot(args.physical)
    runtime = ACL()
    reference = None
    report["status"] = "running"
    try:
        launch = None
        if backend.abi is not None:
            library = ct.CDLL(str(backend.library))
            launch = getattr(library, backend.abi.symbol)
            launch.argtypes, launch.restype = backend.abi.argtypes, ct.c_int
        for sq, sk, raw, expected, selected, oracle_s in prepared:
            pointers = {name: runtime.allocate(data + GUARD) for name, data in raw.items()}
            if args.backend == "aclnn":
                reference = FlashAttentionReference(
                    runtime,
                    pointers["q"],
                    pointers["k"],
                    pointers["v"],
                    pointers["o"],
                    (1, 8, 8, sq, sk, 128),
                    "f16",
                    SCALE,
                )

            arguments = (
                backend.abi.arguments(runtime.stream, pointers, sq, sk)
                if backend.abi is not None
                else ()
            )

            def invoke(reference=reference, arguments=arguments):
                if reference is not None:
                    reference()
                else:
                    require(
                        launch(*arguments) == 0,
                        f"{args.backend} launcher rejected the explicit input contract",
                    )

            for _ in range(args.warmup):
                invoke()
            runtime.synchronize()
            start, stop = runtime.event(), runtime.event()
            runtime.record(start)
            for _ in range(args.repeats):
                invoke()
            runtime.record(stop)
            runtime.synchronize()
            output = runtime.read(pointers["o"], len(raw["o"]) + len(GUARD))
            actual = decode(output[: len(raw["o"])], "f16", (1, 8, sq, 128))
            require(np.isfinite(actual).all(), "nonfinite or unwritten attention output")
            checked = actual[:, :, selected]
            error = float(np.max(np.abs(checked - expected)))
            rtol, atol = 0.002, 0.0002
            if not np.allclose(checked, expected, rtol=rtol, atol=atol):
                record_path.parent.mkdir(parents=True, exist_ok=True)
                (record_path.parent / f"mismatch-{args.backend}-{sq}-{sk}.bin").write_bytes(output)
                raise ValueError(
                    f"attention output differs from {args.scope} oracle; maximum absolute error={error}"
                )
            for name, pointer in pointers.items():
                observed = runtime.read(pointer, len(raw[name]) + len(GUARD))
                require(observed[-len(GUARD) :] == GUARD, f"{name} allocation guard changed")
                if name in {"q", "k", "v"}:
                    require(observed[: -len(GUARD)] == raw[name], f"{name} input changed")
            elapsed = ct.c_float()
            runtime.call(
                "aclrtEventElapsedTime",
                [ct.POINTER(ct.c_float), ct.c_void_p, ct.c_void_p],
                ct.byref(elapsed),
                start,
                stop,
            )
            require(
                math.isfinite(elapsed.value) and elapsed.value > 0,
                "invalid ACL stream event interval",
            )
            row = {
                "backend": args.backend,
                "kernel": reference.interface
                if reference is not None
                else backend.identity["kernel"],
                "library": str(backend.library),
                "shared_sha256": backend.identity["shared_sha256"],
                "compiled_source_sha256": backend.identity.get("compiled_source_sha256"),
                "sq": sq,
                "sk": sk,
                "shape_b_hq_hkv_sq_sk_d": [1, 8, 8, sq, sk, 128],
                "dtype": "f16",
                "layout": "BNSD",
                "softmax_scale_f32": SCALE,
                "softmax_scale_f32_bits": struct.pack("<f", SCALE).hex(),
                "causal": False,
                "query_start": 0,
                "data_case": args.data_case,
                "oracle_scope": args.scope,
                "query_rows_per_head": selected.tolist(),
                "checked_elements": int(checked.size),
                "finite_elements": int(actual.size),
                "max_abs_error": error,
                "rtol": rtol,
                "atol": atol,
                "oracle_s": oracle_s,
                "warmup": args.warmup,
                "repeats": args.repeats,
                "stream_interval_us": elapsed.value * 1000 / args.repeats,
                "timing_scope": TIMING_SCOPE,
                "cache_policy": "same allocations and inputs reused within the case; no cache flush",
                "input_contract": {
                    "allocation_bytes": expected_bytes(sq, sk),
                    "example_groups": 20,
                    "example_tile_bq_bk_ck_vr": [128, 1024, 128, 8],
                    "backend_buffer_arguments": ["q", "k", "v", "o"]
                    if args.backend == "aclnn"
                    else [key for key in backend.abi.keys if key not in {"stream", "sq", "sk"}],
                    "seed": 9102201,
                    "base_alignment": 512,
                },
                "argument_sha256": {name: sha256(data) for name, data in raw.items()},
                "output_sha256": sha256(output[: len(raw["o"])]),
                "inputs_unchanged": True,
                "guards_checked": True,
                "guard_bytes": len(GUARD),
                "backend_workspace_bytes": None
                if reference is None
                else reference.workspace_size.value,
                "backend_auxiliary_bytes": None if reference is None else reference.auxiliary_bytes,
            }
            if args.backend == "aclnn":
                row["reference_opapi_sha256"] = backend.identity["shared_sha256"]
            report["cases"].append(row)
            write_record(record_path, report)
            print(
                json.dumps(
                    {
                        key: row[key]
                        for key in (
                            "backend",
                            "sq",
                            "sk",
                            "oracle_scope",
                            "max_abs_error",
                            "stream_interval_us",
                        )
                    }
                ),
                flush=True,
            )
            if reference is not None:
                reference.close()
                reference = None
        report["status"] = "passed"
        write_record(record_path, report)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_record(record_path, report)
        raise
    finally:
        if reference is not None:
            reference.close()
        runtime.close()
    return report


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
