"""CPU-only artifact and runtime doubles; never load these fake libraries on a device."""

import ctypes as ct
import json
import math
from dataclasses import replace
from io import StringIO
from types import SimpleNamespace

import pytest
from xdsl.printer import Printer

from ascendflow import language as N
from ascendflow.dialects.source import ElementOp, SizeOp
from benchmarks.attention import run

GENERATED_ORDER = ("Q", "K", "Score", "P", "V", "PV", "O")


def generated_source(order=GENERATED_ORDER):
    parameters = ", ".join("void* arg_" + name for name in order)
    return (
        f'extern "C" int launch_attention_target(void* stream, {parameters}, '
        "int64_t npu_size_Sq, int64_t npu_size_Sk) { return 0; }\n"
    )


def metadata_ir(policy=run.DEFAULT_POLICY):
    physical = [
        '%sq = "npu.runtime_size"() {sym_name="Sq", minimum=#builtin.int<1>, multiple=#builtin.int<1>} : () -> index',
        '%sk = "npu.runtime_size"() {sym_name="Sk", minimum=#builtin.int<1>, multiple=#builtin.int<1>} : () -> index',
    ]
    logical = [
        '%sq = "npu_source.size"() {sym_name="Sq", minimum=#builtin.int<1>, multiple=#builtin.int<1>} : () -> index',
        '%sk = "npu_source.size"() {sym_name="Sk", minimum=#builtin.int<1>, multiple=#builtin.int<1>} : () -> index',
    ]
    for tensor in policy.tensors:
        name, shape, dtype, extent = tensor.name, tensor.shape, tensor.dtype, tensor.extent
        operands = "" if extent is None else "%" + extent.lower()
        types = "" if extent is None else "index"
        nbytes = tensor.fixed_bytes if extent is None else tensor.bytes_per_extent
        count_scale = (
            "" if extent is None else f", count_scale=#builtin.int<{tensor.bytes_per_extent}>"
        )
        physical.append(
            f'%{name} = "npu.buffer"({operands}) {{sym_name="{name}", nbytes=#builtin.int<{nbytes}>, '
            f"external=#builtin.int<1>, initialized=#builtin.int<{int(name in {'Q', 'K', 'V'})}>"
            f'{count_scale}}} : ({types}) -> !npu.ref<"GM", "{dtype}", "linear">'
        )
        dimensions = ", ".join(f"#builtin.int<{n}>" for n in shape)
        axes = "#builtin.int<2>" if extent else ""
        access = "read" if name in {"Q", "K", "V"} else "write"
        logical.append(
            f'%{name} = "npu_source.alloc"({operands}) {{sym_name="{name}", versions=#builtin.int<1>, '
            f'access="{access}", ring=#builtin.int<0>, dynamic_axes=[{axes}]}} : ({types}) -> '
            f'!npu_source.tensor<[{dimensions}], "{dtype}", "GM", "linear">'
        )
    names = (
        "q1",
        "chunk",
        "p1",
        "v1",
        "a0",
        "b0",
        "c0",
        "score_in",
        *(("score_work",) if policy.query_tile == 128 else ()),
        "p_out",
        "pv_in",
        "o_out",
        "m",
        "l",
        "o_state",
        "alpha",
        "partial",
        "next_m",
        "row_sum",
        "ones",
        "inverse",
        "soft_broadcast",
        "out_broadcast",
    )
    for name, (shape, dtype, memory, layout) in zip(names, policy.local_footprints, strict=True):
        dimensions = ", ".join(f"#builtin.int<{n}>" for n in shape)
        logical.append(
            f'%{name} = "npu_source.alloc"() {{sym_name="{name}", versions=#builtin.int<1>, '
            f'access="local", ring=#builtin.int<0>, dynamic_axes=[]}} : () -> '
            f'!npu_source.tensor<[{dimensions}], "{dtype}", "{memory}", "{layout}">'
        )
        nbytes = math.prod(shape) * (4 if dtype == "f32" else 2)
        physical.append(
            f'%{name} = "npu.buffer"() {{sym_name="{name}", nbytes=#builtin.int<{nbytes}>, '
            "external=#builtin.int<0>, initialized=#builtin.int<0>} "
            f': () -> !npu.ref<"{memory}", "{dtype}", "{layout}">'
        )
    for family in ("cube", "vector"):
        physical.append(
            f'"npu.actor_domain"() ({{\n^bb0(%{family}_sid: index):\n'
            f'"npu.end"() : () -> ()\n}}) {{kind="{family}"}} : () -> ()'
        )
    score_type = '!npu_source.tensor<[#builtin.int<8>, #builtin.int<1024>], "f32", "UB", "linear">'
    if policy.query_tile == 256:
        input_type = '!npu_source.tensor<[#builtin.int<2>, #builtin.int<8>, #builtin.int<1024>], "f32", "UB", "linear">'
        logical.extend(
            (
                '%zero = "arith.constant"() {value=0 : index} : () -> index',
                (
                    '%score_work = "npu_source.view"(%score_in, %zero, %zero, %zero) '
                    "{widths=[#builtin.int<1>, #builtin.int<8>, #builtin.int<1024>], "
                    "axes=[#builtin.int<1>, #builtin.int<2>], operandSegmentSizes=array<i32: 1, 3, 0>} "
                    f": ({input_type}, index, index, index) -> {score_type}"
                ),
            )
        )
    logical.append(
        f'%scaled = "npu_source.element"(%score_work, %score_work) {{kind="mul_scalar", '
        f"dependency_inputs=[], scalar={run.SCALE} : f32, operandSegmentSizes=array<i32: 1, 1, 0, 0, 0, 0, 0>}} "
        f': ({score_type}, {score_type}) -> !npu_source.tile<[#builtin.int<8>, #builtin.int<1024>], "f32">'
    )
    # Metadata-only masks come from real source syntax. They are not an
    # executable kernel or synchronization oracle for this fake artifact.
    probe = N.from_source(
        """
def metadata(Sk:N.int64):
    with N.Kernel():
        work=N.alloc_ub((8,1024))
        N.fill(work,NEG_INF,mask=N.arange(1024)[None,:] >= Sk)
""",
        constants={"NEG_INF": float("-inf")},
    ).to_ir()
    fill = next(op for op in probe.walk() if isinstance(op, ElementOp))
    next(op for op in probe.walk() if isinstance(op, SizeOp)).value.name_hint = "sk"
    fill.destination.name_hint = "score_work"
    suffix = StringIO()
    printer = Printer(stream=suffix)
    for _ in range(1 + int(policy.causal)):
        mask = fill.mask.owner.clone()
        update = fill.clone({fill.mask: mask.results[0]})
        printer.print_op(mask)
        suffix.write("\n")
        printer.print_op(update)
        suffix.write("\n")
    logical.append(suffix.getvalue())
    before = (
        'builtin.module { "npu.mixed"() ({\n'
        + "\n".join(physical)
        + '\n"npu.end"() : () -> ()\n}) {sym_name="attention_target", target="dav-2201", '
        "blocks=#builtin.int<20>, disjoint_external=#builtin.int<1>} : () -> () }"
    )
    source = (
        'builtin.module { "npu_source.program"() ({\n^bb0(%group: index):\n'
        + "\n".join(logical)
        + '\n"npu_source.yield"() : () -> ()\n}) {sym_name="attention_target", kind="mixed", '
        "groups=#builtin.int<20>, disjoint_external=#builtin.int<1>} : () -> () }"
    )
    return before, source


def publish(directory, *, backend="generated", source=None, policy=run.DEFAULT_POLICY):
    """Metadata-only test artifact; the fake library must never be loaded."""
    directory.mkdir(parents=True, exist_ok=True)
    name = run.ENTRY if backend == "generated" else run.REFERENCE_NAME
    source = source or (
        generated_source()
        if backend == "generated"
        else run.reference_source(**policy.reference_parameters())
    )
    library = b"not executable; CPU artifact-identity tests only"
    (directory / (name + ".asc")).write_text(source)
    (directory / (name + ".so")).write_bytes(library)
    manifest = {
        "target": "dav-2201",
        "source_sha256": run.sha256(source.encode()),
        "shared_sha256": run.sha256(library),
    }
    (directory / (name + ".compile.json")).write_text(json.dumps(manifest))
    report = {"kernel": name}
    if backend == "generated":
        compilation = {"assembly_s": 0.01, "emission_s": 0.01, "snapshot_s": 0.01, "total_s": 0.03}
        report.update(
            pipeline="structured-regions",
            verification_mode="generated-lifetimes",
            debug_structural_verification=False,
            compilation=compilation,
            cann=manifest,
        )
        (directory / "compilation.json").write_text(json.dumps(compilation))
        (directory / "build-result.json").write_text(
            json.dumps(
                {
                    "status": "passed",
                    "returncode": 0,
                    "error": None,
                    "seconds": 0.03,
                    "deadline_s": 300,
                }
            )
        )
        before, logical = metadata_ir(policy)
        (directory / "before.xdsl").write_text(before)
        (directory / "source.xdsl").write_text(logical)
    else:
        report["kind"] = "asc_reference"
    (directory / "verification.json").write_text(json.dumps(report))
    return directory


def mock_execution(monkeypatch, policy, *, actual_value=1.25, expected_value=1.25):
    """CPU doubles exercise wiring only; no kernel or device is executed."""
    np = pytest.importorskip("numpy")
    encode = run.attention_validation.encode
    original_decode = run.decode
    sq = policy.query_tile
    output = encode(np.full((1, 8, sq, 128), actual_value, dtype=np.float32), policy.dtype)
    selected = np.array([0, sq - 1], dtype=np.int64)
    expected = np.full((1, 8, len(selected), 128), expected_value, dtype=np.float32)
    state = {"oracle": [], "decode": [], "reference": [], "runtimes": [], "allocation_bytes": []}

    def oracle(selected_policy, sq, sk, **options):
        state["oracle"].append((selected_policy, sq, sk, options))
        raw = {
            key: b"\xcd" * size
            for key, size in replace(policy, query_tile=128).allocation_bytes(sq, sk).items()
        }
        raw["o"] = b"\xff\x7f" * (len(output) // 2)
        return raw, expected, selected, 0.0

    def decode(data, dtype, shape):
        state["decode"].append((dtype, shape))
        return original_decode(data, dtype, shape)

    class Runtime:
        stream = ct.c_void_p(99)

        def __init__(self):
            self.memory = {}
            self.closed = False
            state["runtimes"].append(self)

        def allocate(self, data):
            state["allocation_bytes"].append(len(data) - len(run.GUARD))
            pointer = (len(self.memory) + 1) * 512
            self.memory[pointer] = data
            return pointer

        def read(self, pointer, size):
            assert len(self.memory[pointer]) == size
            return self.memory[pointer]

        def synchronize(self):
            pass

        def event(self):
            return object()

        def record(self, event):
            pass

        def call(self, name, types, elapsed, start, stop):
            assert name == "aclrtEventElapsedTime"
            ct.cast(elapsed, ct.POINTER(ct.c_float))[0] = 1.0

        def close(self):
            self.closed = True
            self.memory.clear()

    class Reference:
        interface = "CPU mock ACLNN wiring only"

        def __init__(self, runtime, q, k, v, o, shape, dtype, scale, *, causal=False):
            state["reference"].append((shape, dtype, scale))
            self.runtime, self.output = runtime, o
            self.workspace_size = ct.c_uint64(0)
            self.auxiliary_bytes = 0

        def __call__(self):
            self.runtime.memory[self.output] = output + run.GUARD

        def close(self):
            pass

    class Launcher:
        def __init__(self, abi):
            self.index = abi.keys.index("o")

        def __call__(self, *arguments):
            state["runtimes"][-1].memory[arguments[self.index]] = output + run.GUARD
            return 0

    def library(path):
        return SimpleNamespace(
            launch_attention_target=Launcher(
                run.launcher_abi(generated_source(), run.ENTRY, generated=True)
            ),
            launch_attention_reference=Launcher(
                run.launcher_abi(
                    run.reference_source(**policy.reference_parameters()),
                    run.REFERENCE_NAME,
                    generated=False,
                )
            ),
        )

    monkeypatch.setattr(run, "inputs_and_oracle", oracle)
    monkeypatch.setattr(run, "decode", decode)
    monkeypatch.setattr(run, "ACL", Runtime)
    monkeypatch.setattr(run, "health_snapshot", lambda *args: {"mock": True})
    monkeypatch.setattr(run, "hardware_snapshot", lambda *args: {"mock": True})
    monkeypatch.setattr(run, "FlashAttentionReference", Reference)
    monkeypatch.setattr(run.ct, "CDLL", library)
    return state


def mock_cann(tmp_path, monkeypatch):
    cann = tmp_path / "cann"
    (cann / "lib64").mkdir(parents=True)
    for name in ("libopapi.so", "libnnopbase.so"):
        (cann / "lib64" / name).write_bytes(b"CPU mock library identity")
    monkeypatch.setattr(run, "CANN", cann)
