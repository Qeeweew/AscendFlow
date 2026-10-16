"""CPU-only parameter, actual named ABI and frozen-artifact runner checks."""

import ctypes as ct
import json

import pytest

from benchmarks.attention import run
from xdsl_npu.model import CompileError

GENERATED_ORDER = ("Q", "K", "Score", "P", "V", "PV", "O")


def generated_source(order=GENERATED_ORDER):
    parameters = ", ".join("void* arg_" + name for name in order)
    return (
        f'extern "C" int launch_attention_target(void* stream, {parameters}, '
        "int64_t npu_size_Sq, int64_t npu_size_Sk) { return 0; }\n"
    )


def metadata_ir():
    physical = [
        '%sq = "npu.runtime_size"() {sym_name="Sq", minimum=#builtin.int<1>, multiple=#builtin.int<128>} : () -> index',
        '%sk = "npu.runtime_size"() {sym_name="Sk", minimum=#builtin.int<1>, multiple=#builtin.int<1024>} : () -> index',
    ]
    logical = [
        '%sq = "npu_source.size"() {sym_name="Sq", minimum=#builtin.int<1>, multiple=#builtin.int<128>} : () -> index',
        '%sk = "npu_source.size"() {sym_name="Sk", minimum=#builtin.int<1>, multiple=#builtin.int<1024>} : () -> index',
    ]
    for name, (shape, dtype, extent) in run.SHAPES.items():
        operands = "" if extent is None else "%" + extent.lower()
        types = "" if extent is None else "index"
        nbytes = 0 if extent else run.expected_bytes(128, 1024)[run.BUFFER_NAMES[name]]
        physical.append(
            f'%{name} = "npu.buffer"({operands}) {{sym_name="{name}", nbytes=#builtin.int<{nbytes}>, '
            f"external=#builtin.int<1>, initialized=#builtin.int<{int(name in {'Q', 'K', 'V'})}>, "
            f'count_scale=#builtin.int<{2048 if extent else 0}>}} : ({types}) -> !npu.ref<"GM", "{dtype}", "linear">'
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
        ("chunk", (2, 128, 128), "f16", "L1", "NZ"),
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


def publish(directory, *, backend="generated", source=None):
    """Metadata-only test artifact; the fake library must never be loaded."""
    directory.mkdir(parents=True, exist_ok=True)
    name = run.ENTRY if backend == "generated" else run.REFERENCE_NAME
    source = source or (generated_source() if backend == "generated" else run.reference_source())
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
        before, logical = metadata_ir()
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
        (("--dtype", "bf16"), "fixed attention policy"),
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


def test_real_example_defaults_match_the_single_supported_policy(tmp_path):
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
    with pytest.raises(CompileError, match="fixed reference policy"):
        run.load_backend(tmp_path, "ascendc", reference_output=reference)


def test_explicit_build_uses_the_new_example_and_one_total_deadline(artifact, monkeypatch):
    calls = []

    def build_file(*arguments, **keywords):
        calls.append((arguments, keywords))

    monkeypatch.setattr(run, "build_file", build_file)
    run.main(["--output", str(artifact), "--build"])
    assert calls == [
        ((run.EXAMPLE, artifact), {"entry": "attention_target", "shared": True, "timeout_s": 300})
    ]
