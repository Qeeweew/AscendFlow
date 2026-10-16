"""CPU-only parameter, actual named ABI and frozen-artifact runner checks."""

import ctypes as ct
import json
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
from xdsl.utils.exceptions import VerifyException

from benchmarks.attention import run
from benchmarks.attention.contract import AttentionPolicy
from xdsl_npu.model import CompileError

GENERATED_ORDER = ("Q", "K", "Score", "P", "V", "PV", "O")
POLICIES = tuple(AttentionPolicy(kv_heads=h, dtype=d) for h in (8, 2, 1) for d in ("f16", "bf16"))


def policy_arguments(policy):
    return ["--kv-heads", str(policy.kv_heads), "--dtype", policy.dtype]


def generated_source(order=GENERATED_ORDER):
    parameters = ", ".join("void* arg_" + name for name in order)
    return (
        f'extern "C" int launch_attention_target(void* stream, {parameters}, '
        "int64_t npu_size_Sq, int64_t npu_size_Sk) { return 0; }\n"
    )


def metadata_ir(policy=run.DEFAULT_POLICY):
    physical = [
        '%sq = "npu.runtime_size"() {sym_name="Sq", minimum=#builtin.int<1>, multiple=#builtin.int<128>} : () -> index',
        '%sk = "npu.runtime_size"() {sym_name="Sk", minimum=#builtin.int<1>, multiple=#builtin.int<1024>} : () -> index',
    ]
    logical = [
        '%sq = "npu_source.size"() {sym_name="Sq", minimum=#builtin.int<1>, multiple=#builtin.int<128>} : () -> index',
        '%sk = "npu_source.size"() {sym_name="Sk", minimum=#builtin.int<1>, multiple=#builtin.int<1024>} : () -> index',
    ]
    for tensor in policy.tensors:
        name, shape, dtype, extent = tensor.name, tensor.shape, tensor.dtype, tensor.extent
        operands = "" if extent is None else "%" + extent.lower()
        types = "" if extent is None else "index"
        nbytes = tensor.fixed_bytes
        physical.append(
            f'%{name} = "npu.buffer"({operands}) {{sym_name="{name}", nbytes=#builtin.int<{nbytes}>, '
            f"external=#builtin.int<1>, initialized=#builtin.int<{int(name in {'Q', 'K', 'V'})}>, "
            f'count_scale=#builtin.int<{tensor.bytes_per_extent}>}} : ({types}) -> !npu.ref<"GM", "{dtype}", "linear">'
        )
        dimensions = ", ".join(f"#builtin.int<{n}>" for n in shape)
        axes = "#builtin.int<2>" if extent else ""
        access = "read" if name in {"Q", "K", "V"} else "write"
        logical.append(
            f'%{name} = "npu_source.alloc"({operands}) {{sym_name="{name}", versions=#builtin.int<1>, '
            f'access="{access}", ring=#builtin.int<0>, dynamic_axes=[{axes}]}} : ({types}) -> '
            f'!npu_source.tensor<[{dimensions}], "{dtype}", "GM", "linear">'
        )
    for name, shape, dtype, memory, layout in (
        ("chunk", (2, 128, 128), policy.dtype, "L1", "NZ"),
        ("score_in", (2, 8, 1024), "f32", "UB", "linear"),
        ("score_work", (8, 1024), "f32", "UB", "linear"),
    ):
        dimensions = ", ".join(f"#builtin.int<{n}>" for n in shape)
        logical.append(
            f'%{name} = "npu_source.alloc"() {{sym_name="{name}", versions=#builtin.int<1>, '
            f'access="local", ring=#builtin.int<0>, dynamic_axes=[]}} : () -> '
            f'!npu_source.tensor<[{dimensions}], "{dtype}", "{memory}", "{layout}">'
        )
    score_type = '!npu_source.tensor<[#builtin.int<8>, #builtin.int<1024>], "f32", "UB", "linear">'
    logical.append(
        f'%scaled = "npu_source.element"(%score_work, %score_work) {{kind="mul_scalar", '
        f"dependency_inputs=[], scalar={run.SCALE} : f32, operandSegmentSizes=array<i32: 1, 1, 0, 0>}} "
        f': ({score_type}, {score_type}) -> !npu_source.tile<[#builtin.int<8>, #builtin.int<1024>], "f32">'
    )
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
        report.update(
            cross_query_verified=True,
            query_verification={"verified": True, "resources": {"verified": True}},
        )
        before, logical = metadata_ir(policy)
        (directory / "before.xdsl").write_text(before)
        (directory / "source.xdsl").write_text(logical)
    else:
        report["kind"] = "asc_reference"
    (directory / "verification.json").write_text(json.dumps(report))
    return directory


@pytest.fixture
def artifact(tmp_path):
    return publish(tmp_path / "generated")


def no_device(*args, **kwargs):
    pytest.fail("CPU runner checks must not initialize a device, CANN or compiler")


@pytest.fixture(autouse=True)
def forbid_device_and_build(monkeypatch):
    monkeypatch.setattr(run, "ACL", no_device)
    monkeypatch.setattr(run, "health_snapshot", no_device)
    monkeypatch.setattr(run, "hardware_snapshot", no_device)
    monkeypatch.setattr(run, "build_file", no_device)


def test_generated_launcher_uses_its_actual_order_instead_of_dsl_or_reference_order():
    abi = run.launcher_abi(generated_source(), "attention_target", generated=True)
    pointers = {name: object() for name in run.BUFFER_NAMES.values()}
    stream = object()
    assert abi.names == (
        "stream",
        *("arg_" + name for name in GENERATED_ORDER),
        "npu_size_Sq",
        "npu_size_Sk",
    )
    assert abi.arguments(stream, pointers, 128, 1024) == (
        stream,
        *(pointers[run.BUFFER_NAMES[name]] for name in GENERATED_ORDER),
        128,
        1024,
    )
    assert abi.argtypes == [ct.c_void_p] * 8 + [ct.c_int64] * 2
    reference = run.launcher_abi(run.reference_source(), run.REFERENCE_NAME, generated=False)
    assert reference.arguments(stream, pointers, 128, 1024) == (
        stream,
        *(pointers[name] for name in ("q", "k", "v", "o", "score", "p", "pv")),
        128,
        1024,
    )
    assert reference.keys != abi.keys


@pytest.mark.parametrize(
    "source",
    (
        generated_source(("Q", "K", "V", "O", "Score", "P", "P")),
        generated_source().replace("void* arg_PV", "void* arg_unknown"),
        generated_source().replace("void* arg_Q", "int64_t arg_Q"),
        generated_source().replace("npu_size_Sk", "sk"),
        generated_source().replace("void* stream", "int64_t stream"),
        generated_source() + generated_source(),
        run.reference_source(),
    ),
)
def test_ambiguous_missing_duplicate_or_foreign_launcher_abi_is_rejected(source):
    with pytest.raises(CompileError, match="launcher|parameter"):
        run.launcher_abi(source, "attention_target", generated=True)


@pytest.mark.parametrize(
    "case", ("0:1024", "128:0", "127:1024", "128:1023", "128", "-128:1024", f"{2**64}:1024")
)
def test_invalid_shape_is_rejected_before_loading_any_artifact(tmp_path, case):
    with pytest.raises(SystemExit):
        run.parse_args(["--output", str(tmp_path), "--cases", case])


@pytest.mark.parametrize(
    "arguments, reason",
    (
        (("--warmup", "0"), "warmup"),
        (("--repeats", "0"), "repeats"),
        (("--physical", "-1"), "physical"),
        (("--timeout", "301"), "300 seconds"),
        (("--timeout", "nan"), "300 seconds"),
        (("--heads", "4"), "fixed attention policy"),
        (("--dtype", "f32"), "fixed attention policy"),
        (("--kv-heads", "3"), "HQ divisible by HKV"),
        (("--kv-heads", "4"), "fixed attention policy"),
        (("--backend", "ascendc"), "reference-output"),
        (("--backend", "aclnn", "--build"), "only the generated"),
        (("--reference-output", "/unused"), "only to the AscendC"),
    ),
)
def test_invalid_launch_or_policy_options_are_rejected_before_artifact_loading(
    tmp_path, arguments, reason
):
    with pytest.raises(CompileError, match=reason):
        run.main(["--output", str(tmp_path), *arguments])


def test_real_example_defaults_match_the_declared_base_policy(tmp_path):
    assert run.example_policy()["constexprs"] == run.CONSTANTS
    changed = tmp_path / "changed.py"
    changed.write_text(
        run.EXAMPLE.read_text().replace("HQ: N.constexpr = 8", "HQ: N.constexpr = 4")
    )
    with pytest.raises(CompileError, match="example policy differs"):
        run.example_policy(changed)


def test_reusing_generated_artifact_never_builds_and_records_its_own_identity(artifact, tmp_path):
    record = tmp_path / "record.json"
    result = run.main(["--output", str(artifact), "--record", str(record)])
    assert result["status"] == "artifact_validated" and result["cases"] == []
    identity = result["artifact"]
    assert identity["library"] == str(artifact / "attention_target.so")
    assert identity["shared_sha256"] == run.sha256((artifact / "attention_target.so").read_bytes())
    assert identity["launcher_parameters"][3:5] == ("arg_Score", "arg_P")
    assert identity["contract"]["buffers"]["score"]["fixed_bytes"] == 20971520
    assert json.loads(record.read_text())["artifact"]["library"] == identity["library"]
    assert "not kernel-only" in result["timing_scope"]


@pytest.mark.parametrize("name", ("attention_target.asc", "attention_target.so"))
def test_changed_source_or_shared_object_is_rejected_before_any_device_initialization(
    artifact, name
):
    with (artifact / name).open("ab") as output:
        output.write(b"changed")
    with pytest.raises(CompileError, match="hash or target differs"):
        run.main(["--output", str(artifact), "--physical", "0"])


@pytest.mark.parametrize(
    "filename",
    ("attention_target.so", "attention_target.compile.json", "before.xdsl", "source.xdsl"),
)
def test_incomplete_generated_artifact_cannot_trigger_a_silent_rebuild(artifact, filename):
    (artifact / filename).unlink()
    with pytest.raises(CompileError, match="missing attention artifact"):
        run.main(["--output", str(artifact)])


@pytest.mark.parametrize(
    "field", ("source_shape", "physical_type", "blocks", "extent", "chunk", "vector_rows", "scale")
)
def test_compiled_tensor_and_physical_policy_are_explicitly_rechecked(artifact, field):
    path = artifact / (
        "source.xdsl"
        if field in {"source_shape", "chunk", "vector_rows", "scale"}
        else "before.xdsl"
    )
    original = path.read_text()
    if field == "source_shape":
        changed = original.replace("#builtin.int<8>", "#builtin.int<4>", 1)
    elif field == "physical_type":
        changed = original.replace('"f16"', '"bf16"', 1)
    elif field == "blocks":
        changed = original.replace("blocks=#builtin.int<20>", "blocks=#builtin.int<19>")
    elif field == "chunk":
        changed = original.replace(
            "#builtin.int<2>, #builtin.int<128>, #builtin.int<128>",
            "#builtin.int<2>, #builtin.int<128>, #builtin.int<64>",
        )
    elif field == "vector_rows":
        changed = original.replace(
            "#builtin.int<8>, #builtin.int<1024>", "#builtin.int<16>, #builtin.int<1024>"
        )
    elif field == "scale":
        changed = original.replace(f"scalar={run.SCALE}", "scalar=0.5")
    else:
        changed = original.replace('"npu.buffer"(%sq)', '"npu.buffer"(%sk)', 1)
    assert changed != original
    path.write_text(changed)
    with pytest.raises(
        CompileError, match="policy differs|extent/type differs|softmax scale differs"
    ):
        run.main(["--output", str(artifact)])


def test_aclnn_records_only_its_actual_cann_libraries_without_generated_artifacts(
    tmp_path, monkeypatch
):
    cann = tmp_path / "cann"
    (cann / "lib64").mkdir(parents=True)
    for name in ("libopapi.so", "libnnopbase.so"):
        (cann / "lib64" / name).write_bytes(name.encode())
    monkeypatch.setattr(run, "CANN", cann)
    result = run.main(["--output", str(tmp_path / "no-generated-artifact"), "--backend", "aclnn"])
    identity = result["artifact"]
    assert identity["library"] == str(cann / "lib64/libopapi.so")
    assert identity["shared_sha256"] == run.sha256(b"libopapi.so")
    assert set(identity["libraries"]) == {"libopapi.so", "libnnopbase.so"}
    assert "compiled_source_sha256" not in identity
    assert "attention_target" not in identity["kernel"]


def test_ascendc_uses_its_separate_artifact_and_reference_abi(artifact, tmp_path):
    reference = publish(tmp_path / "reference", backend="ascendc")
    result = run.main(
        ["--output", str(artifact), "--backend", "ascendc", "--reference-output", str(reference)]
    )
    assert result["artifact"]["library"] == str(reference / "attention_reference.so")
    assert result["artifact"]["launcher_parameters"] == (
        "stream",
        "q",
        "k",
        "v",
        "o",
        "score",
        "p",
        "pv",
        "sq",
        "sk",
    )
    with pytest.raises(CompileError, match="backend/artifact identity differs"):
        run.load_backend(artifact, "ascendc", reference_output=artifact)


def test_changed_reference_policy_is_rejected_even_with_a_matching_new_manifest(tmp_path):
    source = run.reference_source().replace("GROUPS = 20", "GROUPS = 19")
    assert source != run.reference_source()
    reference = publish(tmp_path / "reference", backend="ascendc", source=source)
    with pytest.raises(CompileError, match="selected reference policy"):
        run.load_backend(tmp_path, "ascendc", reference_output=reference)


def test_explicit_build_uses_the_new_example_and_one_total_deadline(artifact, monkeypatch):
    calls = []

    def build_file(*arguments, **keywords):
        calls.append((arguments, keywords))

    monkeypatch.setattr(run, "build_file", build_file)
    run.main(["--output", str(artifact), "--build"])
    assert calls == [
        (
            (run.EXAMPLE, artifact),
            {
                "entry": "attention_target",
                "parameters": run.DEFAULT_POLICY.parameters(),
                "shared": True,
                "timeout_s": 300,
            },
        )
    ]


@pytest.mark.parametrize("policy", POLICIES)
def test_immutable_policy_drives_seven_shapes_bytes_and_reference_parameters(policy):
    assert len(policy.parameters()) == 11
    assert policy.parameters() == {
        **run.CONSTANTS,
        "HKV": policy.kv_heads,
        "DTYPE": policy.dtype,
    }
    assert policy.allocation_bytes(384, 2048) == {
        "q": 384 * 2048,
        "k": 2048 * {8: 2048, 2: 512, 1: 256}[policy.kv_heads],
        "v": 2048 * {8: 2048, 2: 512, 1: 256}[policy.kv_heads],
        "o": 384 * 2048,
        "score": 20971520,
        "p": 10485760,
        "pv": 2621440,
    }
    tensors = {tensor.key: tensor for tensor in policy.tensors}
    assert tensors["k"].runtime_shape(384, 2048) == (1, policy.kv_heads, 2048, 128)
    assert tensors["q"].runtime_shape(384, 2048) == (1, 8, 384, 128)
    assert tensors["p"].dtype == policy.dtype
    assert tensors["score"].dtype == tensors["pv"].dtype == "f32"
    assert policy.shape(384, 2048) == (1, 8, policy.kv_heads, 384, 2048, 128)
    assert policy.tolerances == ((0.002, 0.0002) if policy.dtype == "f16" else (0.01, 0.001))
    assert policy.reference_parameters() == {
        "batch": 1,
        "heads": 8,
        "kv_heads": policy.kv_heads,
        "groups": 20,
        "dtype": policy.dtype,
    }
    with pytest.raises(FrozenInstanceError):
        policy.kv_heads = 4
    with pytest.raises(FrozenInstanceError):
        tensors["k"].shape = (1, 8, -1, 128)


@pytest.mark.parametrize(
    "changes",
    (
        {"batch": True},
        {"heads": 8.0},
        {"kv_heads": False},
        {"kv_heads": 0},
        {"kv_heads": -1},
        {"kv_heads": 3},
        {"kv_heads": 4},
        {"batch": 2},
        {"heads": 4},
        {"dim": 64},
        {"groups": 19},
        {"dtype": "f32"},
    ),
)
def test_undeclared_or_noninteger_policy_is_rejected(changes):
    with pytest.raises(CompileError, match="policy"):
        AttentionPolicy(**changes)


@pytest.mark.parametrize("policy", POLICIES)
def test_each_selected_policy_matches_actual_ir_and_reaches_public_specialize(
    tmp_path, monkeypatch, policy
):
    artifact = publish(tmp_path / "generated", policy=policy)
    calls = []
    monkeypatch.setattr(run, "build_file", lambda *args, **kwargs: calls.append((args, kwargs)))
    result = run.main(["--output", str(artifact), "--build", *policy_arguments(policy)])
    assert calls == [
        (
            (run.EXAMPLE, artifact),
            {
                "entry": run.ENTRY,
                "parameters": policy.parameters(),
                "shared": True,
                "timeout_s": 300,
            },
        )
    ]
    assert result["status"] == "artifact_validated" and result["cases"] == []
    assert result["policy"] == result["artifact"]["contract"]["policy"] == policy.as_dict()
    assert result["specialization"] == policy.parameters()
    assert result["buffer_contracts"]["k"]["bytes_per_extent"] == 256 * policy.kv_heads
    assert result["buffer_contracts"]["p"]["dtype"] == policy.dtype
    assert result["requested_cases"][0]["allocation_bytes"] == policy.allocation_bytes(128, 1024)
    assert "no device qualification" in result["comparison_scope"]


@pytest.mark.parametrize("policy", POLICIES[1:])
def test_default_artifact_cannot_be_relabelled_as_another_policy(artifact, policy):
    with pytest.raises(CompileError, match="policy differs"):
        run.main(["--output", str(artifact), "--physical", "0", *policy_arguments(policy)])


@pytest.mark.parametrize(
    "field", ("kv-scale", "kv-extent", "p-dtype", "local-dtype", "source-axis", "source-tile")
)
def test_selected_variant_rechecks_source_and_physical_contracts(tmp_path, field):
    policy = AttentionPolicy(kv_heads=2, dtype="bf16")
    artifact = publish(tmp_path / "variant", policy=policy)
    path = artifact / ("before.xdsl" if field in {"kv-scale", "kv-extent"} else "source.xdsl")
    text = path.read_text()
    if field == "kv-scale":
        text = text.replace("count_scale=#builtin.int<512>", "count_scale=#builtin.int<2048>", 1)
    elif field == "kv-extent":
        text = text.replace('"npu.buffer"(%sk)', '"npu.buffer"(%sq)', 1)
    elif field == "p-dtype":
        lines = text.splitlines()
        lines = [
            line.replace('"bf16"', '"f16"') if line.startswith("%P =") else line for line in lines
        ]
        text = "\n".join(lines)
    elif field == "local-dtype":
        lines = text.splitlines()
        lines = [
            line.replace('"bf16"', '"f16"') if line.startswith("%chunk =") else line
            for line in lines
        ]
        text = "\n".join(lines)
    elif field == "source-axis":
        text = text.replace("dynamic_axes=[#builtin.int<2>]", "dynamic_axes=[#builtin.int<0>]", 1)
    else:
        text = text.replace("multiple=#builtin.int<1024>", "multiple=#builtin.int<512>", 1)
    assert text != path.read_text()
    path.write_text(text)
    with pytest.raises(
        VerifyException if field == "source-axis" else CompileError,
        match="policy differs|extent/type differs|extent differs|tile contracts|dynamic shape axes",
    ):
        run.main(["--output", str(artifact), *policy_arguments(policy)])


@pytest.mark.parametrize("policy", POLICIES)
def test_ascendc_artifact_matches_its_own_selected_policy_source(tmp_path, policy):
    reference = publish(tmp_path / "reference", backend="ascendc", policy=policy)
    backend = run.load_backend(tmp_path, "ascendc", reference_output=reference, policy=policy)
    assert backend.identity["contract"]["reference_policy"] == policy.as_dict()
    other = AttentionPolicy(kv_heads=1 if policy.kv_heads != 1 else 2, dtype=policy.dtype)
    with pytest.raises(CompileError, match="selected reference policy"):
        run.load_backend(tmp_path, "ascendc", reference_output=reference, policy=other)


def test_selected_policy_checks_actual_allocation_integer_overflow(tmp_path):
    # Each Sk element needs 256 bytes for MQA, while Q/O retain 2048 bytes/Sq.
    safe_sk = ((2**63 - 1) // 256 // 1024) * 1024
    args = run.parse_args(
        ["--output", str(tmp_path), "--kv-heads", "1", "--cases", f"128:{safe_sk}"]
    )
    assert args.policy.kv_heads == 1
    with pytest.raises(CompileError, match="allocation bytes exceed"):
        run.parse_args(["--output", str(tmp_path), "--cases", f"128:{safe_sk}"])


def mock_execution(monkeypatch, policy, *, actual_value=1.25, expected_value=1.25):
    """CPU doubles exercise wiring only; no kernel or device is executed."""
    np = pytest.importorskip("numpy")
    encode = run.attention_validation.encode
    original_decode = run.decode
    output = encode(np.full((1, 8, 128, 128), actual_value, dtype=np.float32), policy.dtype)
    selected = np.array([0, 127], dtype=np.int64)
    expected = np.full((1, 8, len(selected), 128), expected_value, dtype=np.float32)
    state = {"oracle": [], "decode": [], "reference": [], "runtimes": []}

    def oracle(selected_policy, sq, sk, **options):
        state["oracle"].append((selected_policy, sq, sk, options))
        raw = {key: b"\xcd" * size for key, size in policy.allocation_bytes(sq, sk).items()}
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

        def __init__(self, runtime, q, k, v, o, shape, dtype, scale):
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


@pytest.mark.parametrize("policy", POLICIES)
def test_selected_policy_reaches_oracle_decode_aclnn_and_reports(tmp_path, monkeypatch, policy):
    state = mock_execution(monkeypatch, policy)
    mock_cann(tmp_path, monkeypatch)
    report = run.main(
        [
            "--output",
            str(tmp_path / "output"),
            "--backend",
            "aclnn",
            "--physical",
            "0",
            "--cases",
            "128:1024",
            "--scope",
            "sampled",
            *policy_arguments(policy),
        ]
    )
    assert state["oracle"] == [
        (policy.as_dict(), 128, 1024, {"scope": "sampled", "data_case": "standard"})
    ]
    assert state["decode"] == [(policy.dtype, (1, 8, 128, 128))]
    assert state["reference"] == [(policy.shape(128, 1024), policy.dtype, policy.scale)]
    assert report["artifact"]["runtime_policy"] == policy.as_dict()
    row = report["cases"][0]
    assert report["status"] == "passed" and row["dtype"] == policy.dtype
    assert (row["rtol"], row["atol"]) == policy.tolerances
    assert row["shape_b_hq_hkv_sq_sk_d"] == policy.shape(128, 1024)
    assert row["input_contract"]["allocation_bytes"] == policy.allocation_bytes(128, 1024)
    assert row["guards_checked"] is True and row["inputs_unchanged"] is True
    assert state["runtimes"][0].closed


@pytest.mark.parametrize("backend", ("generated", "ascendc"))
def test_selected_policy_uses_named_launcher_with_the_same_runtime_contract(
    tmp_path, monkeypatch, backend
):
    policy = AttentionPolicy(kv_heads=2, dtype="bf16")
    state = mock_execution(monkeypatch, policy)
    artifact = publish(tmp_path / "artifact", backend=backend, policy=policy)
    arguments = ["--output", str(artifact), "--backend", backend]
    if backend == "ascendc":
        arguments += ["--reference-output", str(artifact)]
    report = run.main(
        [
            *arguments,
            "--physical",
            "0",
            "--cases",
            "128:1024",
            "--scope",
            "sampled",
            *policy_arguments(policy),
        ]
    )
    assert report["status"] == "passed" and not state["reference"]
    assert state["oracle"][0][0] == policy.as_dict()
    assert state["decode"] == [("bf16", (1, 8, 128, 128))]


@pytest.mark.parametrize("dtype", ("f16", "bf16"))
def test_runtime_uses_original_dtype_tolerance_without_adaptation(tmp_path, monkeypatch, dtype):
    policy = AttentionPolicy(kv_heads=1, dtype=dtype)
    mock_execution(monkeypatch, policy, actual_value=1.005, expected_value=1.0)
    artifact = publish(tmp_path / "artifact", policy=policy)
    arguments = [
        "--output",
        str(artifact),
        "--physical",
        "0",
        "--cases",
        "128:1024",
        "--scope",
        "sampled",
        *policy_arguments(policy),
    ]
    if dtype == "f16":
        with pytest.raises(ValueError, match="output differs"):
            run.main(arguments)
    else:
        row = run.main(arguments)["cases"][0]
        assert (row["rtol"], row["atol"]) == (0.01, 0.001)


def test_wrong_oracle_allocation_is_rejected_before_health_or_acl(tmp_path, monkeypatch):
    policy = AttentionPolicy(kv_heads=2, dtype="bf16")
    artifact = publish(tmp_path / "artifact", policy=policy)
    monkeypatch.setattr(
        run, "inputs_and_oracle", lambda *args, **kwargs: ({"k": b"wrong-size"}, None, None, 0)
    )
    with pytest.raises(CompileError, match="oracle allocation contract differs"):
        run.main(
            [
                "--output",
                str(artifact),
                "--physical",
                "0",
                "--cases",
                "128:1024",
                *policy_arguments(policy),
            ]
        )
