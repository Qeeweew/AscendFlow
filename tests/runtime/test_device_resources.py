"""Runtime hardware facts specialize one coherent launch before dependency analysis."""

import ctypes as ct
from dataclasses import replace

import pytest

from ascendflow import DeviceResources, compile
from ascendflow import language as N
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.device import ACL
from ascendflow.diagnostics import CompileError
from ascendflow.dialects.io import parse
from ascendflow.dialects.source import ProgramOp, SizeOp
from tests.models.structured_events import check_structured_events
from tests.runtime.test_workspace import host_library


def test_acl_stream_interval_conversion_keeps_acl_errors_visible():
    runtime = object.__new__(ACL)

    def call(name, types, output, start, stop):
        assert name == "aclrtEventElapsedTime" and (start, stop) == ("start", "stop")
        output._obj.value = 1.25

    runtime.call = call
    assert runtime.elapsed_us("start", "stop") == 1250.0


PROGRAM = """
def core_owned(Out:N.Tensor((n,128),N.f32,access="write",write_disjoint=True),n:N.i64):
    groups=N.get_core_num("mixed")
    with N.Kernel(groups,kind="mixed") as group:
        w=N.alloc_workspace((groups,2,128))
        x=N.alloc_ub((64,))
        for t in N.Dataflow(N.ceildiv(N.max(0,n-group),groups)):
            row=group+t*groups
            with N.Vector() as sid, N.Stage(0):
                N.fill(x,2.0)
                N.copy(x,w[group,t%2,sid*64:sid*64+64])
            with N.Vector() as sid, N.Stage(1):
                N.copy(w[group,t%2,sid*64:sid*64+64],x)
                N.copy(x,Out[row,sid*64:sid*64+64])
"""


def test_acl_queries_both_counts_on_the_mapped_logical_device_once():
    runtime = object.__new__(ACL)
    runtime._closed = False
    calls = []

    def call(name, types, device, attr, output):
        calls.append((name, device, attr))
        output._obj.value = {102: 24, 201: 40}[attr]

    runtime.call = call
    assert runtime.resources == DeviceResources(24, 40)
    assert runtime.resources.core_count("mixed") == 24
    assert calls == [("aclrtGetDeviceInfo", 0, 102), ("aclrtGetDeviceInfo", 0, 201)]


def test_runtime_device_query_failure_is_propagated_without_a_guessed_count():
    runtime = object.__new__(ACL)
    runtime._closed = False

    def fail(*args):
        raise RuntimeError("device query failed")

    runtime.call = fail
    with pytest.raises(RuntimeError, match="device query failed"):
        _ = runtime.resources


def test_core_kinds_keep_physical_cores_and_mixed_groups_distinct():
    resources = DeviceResources(24, 40)
    assert resources.core_count("cube") == 24
    assert resources.core_count("vector") == 40
    assert resources.core_count("mixed") == 24
    with pytest.raises(CompileError, match="core kind"):
        resources.core_count("all")


def test_source_requires_explicit_resource_binding_without_querying_a_device():
    with pytest.raises(CompileError, match="bind_device"):
        compile(N.from_source(PROGRAM))


@pytest.mark.parametrize(
    "resources,groups", [(DeviceResources(1, 2), 1), (DeviceResources(3, 6), 3)]
)
def test_launch_workspace_and_dependences_share_one_bound_configuration(resources, groups):
    definition = N.from_source(PROGRAM).bind_device(resources)
    artifact = compile(definition)
    source = parse(artifact.source_ir)
    program = source.body.block.first_op
    assert isinstance(program, ProgramOp) and program.groups.data == groups
    assert [op.sym_name.data for op in source.walk() if isinstance(op, SizeOp)] == ["n"]
    assert artifact.report["workspace"]["static_bytes"] == groups * 2 * 128 * 4
    assert artifact.report["device_resources"] == {
        "cube_cores": resources.cube_cores,
        "vector_cores": resources.vector_cores,
    }
    concrete = replace(definition, source=PROGRAM.replace('N.get_core_num("mixed")', str(groups)))
    explicit = compile(concrete)
    assert artifact.asc == explicit.asc and artifact.before_ir == explicit.before_ir
    before = collect_native_memory(parse(artifact.before_ir))
    after = collect_native_memory(artifact.plan)
    for count in (0, 1, 7):
        check_structured_events(
            after,
            {"n": count},
            logical_reference=before,
            groups=tuple(range(groups)),
            invocations=2,
        )


def test_runtime_rejects_a_different_device_configuration_before_workspace_allocation(tmp_path):
    artifact = compile(N.from_source(PROGRAM).bind_device(DeviceResources(3, 6)))
    library = host_library(artifact, tmp_path)
    runtime = object.__new__(ACL)
    runtime._closed, runtime._bound_kernels = False, False
    runtime.resources = DeviceResources(2, 4)
    runtime.stream = ct.c_void_p(99)
    runtime.allocate_uninitialized = lambda size: pytest.fail(
        "mismatched resources must not allocate"
    )
    with pytest.raises(CompileError, match="resources differ"):
        runtime.kernel(library, "core_owned", parameters=(7,))
