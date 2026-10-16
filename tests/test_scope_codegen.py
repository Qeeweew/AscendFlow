"""Retained parent/child ASC uses real calls and checked native insertion sites."""

import re
from dataclasses import replace

import pytest
from xdsl.dialects.builtin import ArrayAttr, IntAttr, StringAttr

from xdsl_npu import language as N
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.dialects.physical import CallOp, CaptureOp, ScopeChannelOp, ScopeSyncOp
from xdsl_npu.dialects.source import ProgramOp
from xdsl_npu.scope_codegen import emit_scopes, scope_boundary
from xdsl_npu.scope_lowering import lower_scopes

VECTOR = """
    def nested(X:N.Tensor((GROUPS,n,2,64),N.f32),
               Y:N.Tensor((GROUPS,n,2,64),N.f32,access="write"),
               n:N.int64,GROUPS:N.constexpr=2):
        with N.Kernel(GROUPS) as core:
            a=N.alloc_ub((2,64))
            b=N.alloc_ub((2,64))
            for outer in N.Pipelined(n,stages=3,
                    issue_order={"vector":("first","second")}):
                with N.Stage(2,name="second"):
                    N.fill(b[0,:],0.0)
                    for part in N.Pipelined(2,stages=2):
                        with N.Stage(0,name="fill"):
                            N.fill(b[part%2,:],1.0)
                        with N.Stage(1,name="store"):
                            N.copy(b[part%2,:],Y[core,outer,part,:])
                with N.Stage(0,name="first"):
                    for part in N.Pipelined(2,stages=2):
                        with N.Stage(0,name="load"):
                            N.copy(X[core,outer,part,:],a[part%2,:])
                        with N.Stage(1,name="scale"):
                            N.mul(a[part%2,:],2.0,out=a[part%2,:])
"""


CROSS = """
    def mixed(X:N.Tensor((n,16,16),N.f16),
              Score:N.Tensor((2,16,16),N.f32,access="write"),n:N.int64):
        with N.Kernel(kind="mixed"):
            q=N.alloc_l1((16,16))
            a=N.alloc_l0a((16,16))
            b=N.alloc_l0b((16,16))
            c=N.alloc_l0c((16,16),N.f32)
            u=N.alloc_ub((16,16))
            for outer in N.Pipelined(n,stages=2):
                with N.Stage(0,name="cube"),N.Cube():
                    for part in N.Pipelined(1,stages=1):
                        N.copy(X[outer,:,:],q)
                        N.copy(q,a)
                        N.copy(q,b,transpose=True)
                        N.matmul(a,b,accum=c,initialize=True)
                        N.copy(c,Score[outer%2,:,:])
                with N.Stage(1,name="vector"),N.Vector():
                    for part in N.Pipelined(1,stages=1):
                        N.copy(Score[outer%2,:,:],u)
                        N.mul(u,2.0,out=u)
"""


def checked_source(source, *, tie=False):
    from xdsl_npu.scope_sync import autosync_scopes, verify_scopes

    module = N.from_source(source, disjoint_external=True).to_ir()
    physical = lower_scopes(next(op for op in module.walk() if isinstance(op, ProgramOp)))
    if tie:
        calls = [op for op in physical.walk() if isinstance(op, CallOp) and op.stage.data == 2]
        assert len(calls) == 2
        calls[1].attributes["issue_order"] = calls[0].issue_order
    plan = autosync_scopes(physical)
    return verify_scopes(plan, _details=True)[1]


@pytest.fixture(scope="module")
def vector():
    return checked_source(VECTOR, tie=True)


@pytest.fixture(scope="module")
def mixed():
    return checked_source(CROSS)


def task_for_sync(checked, leaf, op):
    channel = checked.channels[op.channel.data]
    keys = channel.sources if op.action.data == "set" else channel.destinations
    sites = [
        site
        for key in keys
        for uid, site in [checked.access.endpoints[key]]
        if uid == leaf.call.uid.data
        and site.position.evaluate(leaf.stream.count.data) == op.position.data
    ]
    assert len(sites) == 1
    data = next(iter(checked.analyses[leaf.call.uid.data][0].actors.values()))
    return next(task for task in data.program.tasks if task.uid == sites[0].task[1])


def test_real_calls_use_parent_issue_order_and_separate_child_clocks(vector):
    asc = emit_scopes(vector)
    actor = next(iter(vector.program.parents))
    calls = [op for op in vector.program.parents[actor].body.block.ops if isinstance(op, CallOp)]
    assert len(calls) == 3
    assert [call.stage.data for call in calls] == [2, 2, 0]
    assert calls[0].issue_order == calls[1].issue_order
    # Lexical second-stage work is physically issued after the first-stage
    # child. Calls tied at one issue rank retain their original lexical order.
    expected = sorted(enumerate(calls), key=lambda item: (item[1].issue_order.data, item[0]))
    assert [call.uid.data for _, call in expected] == [
        calls[2].uid.data,
        calls[0].uid.data,
        calls[1].uid.data,
    ]
    assert [asc.index(f"// call {call.uid.data},") for _, call in expected] == sorted(
        asc.index(f"// call {call.uid.data},") for call in calls
    )
    assert "const int64_t npu_parent_n = n;" in asc
    assert "npu_parent_step < npu_parent_n + 2" in asc
    assert "const int64_t npu_parent_g = npu_parent_step - 2;" in asc
    assert asc.count("if (npu_parent_g >= 0 && npu_parent_g < npu_parent_n)") == 3
    assert asc.count("const int64_t n = 2;") == 2
    assert "const int64_t n = 1;" in asc
    assert "ordinary region: execute once" in asc
    assert "for (int64_t step = 0; step < n + 1; ++step)" in asc
    assert "if (g >= 0 && g < n)" in asc
    assert "scope_access" not in asc
    assert "npu_size_$" not in asc


def test_shared_header_launcher_and_real_root_storage_are_emitted_once(vector):
    asc = emit_scopes(vector)
    assert asc.count('#include "c_api/asc_simd.h"') == 1
    assert asc.count("asc_init();") == 1
    assert asc.count("asc_sync_pipe(PIPE_ALL)") == 1
    assert asc.count('extern "C" int launch_nested(') == 1
    assert "nested<<<2, nullptr, stream>>>" in asc
    assert f"> {(1 << 63) - 1 - 2}) return -1; // loop step overflow" in asc
    signature = next(line for line in asc.splitlines() if 'extern "C" __global__' in line)
    assert "__gm__ uint8_t* X" in signature and "__gm__ uint8_t* Y" in signature
    assert "int64_t npu_size_n" in signature
    for leaf in vector.program.leaves.values():
        for captured in leaf.captures:
            alias = captured.op.sym_name.data
            root = captured.root.sym_name.data
            assert alias not in signature
            line = next(line for line in asc.splitlines() if f"uint8_t* {alias} =" in line)
            assert f"= {root} + (" in line
            assert "cap_" not in line.split(" = ", 1)[1]
    assert "core_id" in asc and "npu_parent_g" in asc
    assert "launch_nested" not in emit_scopes(vector, host_launcher=False)


def test_scope_sync_uses_parent_epoch_and_exact_inner_native_position(vector):
    assert vector.channels
    rendered = 0
    for leaf in vector.program.leaves.values():
        boundary = scope_boundary(vector, leaf)
        for op in leaf.stream.walk():
            if not isinstance(op, ScopeSyncOp):
                continue
            rendered += 1
            task = task_for_sync(vector, leaf, op)
            code = boundary(task, "NATIVE();")
            channel = vector.channels[op.channel.data]
            marker = f"/* {channel.name} */"
            assert f"g == {op.position.data}" in code and marker in code
            assert (code.index(marker) > code.index("NATIVE();")) == (op.action.data == "set")
            if channel.distance:
                expected = (
                    f"npu_parent_g < npu_parent_n - {channel.distance}"
                    if op.action.data == "set"
                    else f"npu_parent_g >= {channel.distance}"
                )
                assert expected in code
            if channel.kind == "barrier":
                assert f"asc_sync_pipe(PIPE_{task.pipe})" in code
            else:
                function = "asc_sync_notify" if op.action.data == "set" else "asc_sync_wait"
                assert function in code
    assert rendered


def test_mixed_scope_uses_actual_cv_native_pipes_and_one_configuration(mixed):
    asc = emit_scopes(mixed)
    assert len(mixed.program.kernels) == 3
    assert asc.count("asc_init();") == 3
    assert asc.count("asc_sync_pipe(PIPE_ALL)") == 3
    assert asc.count("asc_set_l0c_copy_params(1, 0, 0);") == 1
    assert "asc_get_sub_block_id() == 0" in asc
    assert "asc_get_sub_block_id() == 1" in asc
    cross = [channel for channel in mixed.channels.values() if channel.kind == "cross"]
    assert cross
    for leaf in mixed.program.leaves.values():
        boundary = scope_boundary(mixed, leaf)
        for op in leaf.stream.walk():
            if not isinstance(op, ScopeSyncOp) or mixed.channels[op.channel.data].kind != "cross":
                continue
            task = task_for_sync(mixed, leaf, op)
            function = "asc_sync_block_arrive" if op.action.data == "set" else "asc_sync_block_wait"
            assert f"{function}(PIPE_{task.pipe}," in boundary(task, "NATIVE();")


def test_sparse_parent_residues_and_rotating_ids_use_parent_epoch_after_final_check(vector):
    from xdsl_npu.scope_sync import verify_scopes

    module = vector.program.module.clone()
    declaration = next(
        op for op in module.walk() if isinstance(op, ScopeChannelOp) and op.kind.data == "local"
    )
    name, other_name = declaration.sym_name.data, declaration.sym_name.data + "_rest"
    other = declaration.clone()
    other.attributes.update(
        sym_name=StringAttr(other_name),
        period=IntAttr(3),
        residues=ArrayAttr([IntAttr(1)]),
    )
    declaration.attributes.update(
        period=IntAttr(3),
        residues=ArrayAttr([IntAttr(0), IntAttr(2)]),
        event_ids=ArrayAttr([IntAttr(2), IntAttr(4), IntAttr(3)]),
    )
    declaration.parent.insert_op_after(other, declaration)
    for op in list(module.walk()):
        if isinstance(op, ScopeSyncOp) and op.channel.data == name:
            sync = op.clone()
            sync.attributes["channel"] = StringAttr(other_name)
            op.parent.insert_op_after(sync, op)
    # The partition keeps the original full activation domain. The actual
    # final verifier must prove coverage, resource separation and all three ID
    # lifetimes before these expert-edited operations may reach emission.
    _, checked = verify_scopes(module, _details=True)
    channel = checked.channels[name]
    assert channel.residues == (0, 2) and channel.ids == (2, 4, 3)
    for leaf in checked.program.leaves.values():
        for op in leaf.stream.walk():
            if not isinstance(op, ScopeSyncOp) or op.channel.data != name:
                continue
            code = scope_boundary(checked, leaf)(task_for_sync(checked, leaf, op), "NATIVE();")
            end = code.index(f"/* {name} */")
            fragment = code[code.rfind("if (", 0, end) : end]
            epoch = "npu_parent_g" if op.action.data == "set" else "(npu_parent_g - 1)"
            assert f"{epoch} % 3 == 0" in fragment and f"{epoch} % 3 == 2" in fragment
            body = fragment.split("{", 1)[1]
            assert "event_t(" in body and "npu_parent_g" in body
            assert not re.search(r"\bg\b", body)
            assert "@npu_scalar_" not in body


def test_capture_alias_named_like_its_root_keeps_the_original_pointer(vector):
    from xdsl_npu.scope_sync import verify_scopes

    module = vector.program.module.clone()
    for op in module.walk():
        if isinstance(op, CaptureOp):
            op.attributes["sym_name"] = op.source.owner.sym_name
    _, checked = verify_scopes(module, _details=True)
    asc = emit_scopes(checked)
    for leaf in checked.program.leaves.values():
        start = asc.index(f"// call {leaf.call.uid.data},")
        body = asc[start : asc.index("// asynchronous child exit", start)]
        for index, captured in enumerate(leaf.captures):
            name = captured.root.sym_name.data
            source = f"npu_capture_root_{index}"
            saved = body.index(f"uint8_t* {source} = {name};")
            alias = body.index(f"uint8_t* {name} = {source} + (")
            assert saved < alias
            assert f"uint8_t* {name} = {name} +" not in body


def test_unrecognized_scope_channel_and_wrong_native_position_are_rejected(vector):
    leaf, operation = next(
        (leaf, op)
        for leaf in vector.program.leaves.values()
        for op in leaf.stream.walk()
        if isinstance(op, ScopeSyncOp)
    )
    task = task_for_sync(vector, leaf, operation)
    name = operation.channel.data
    invalid = replace(
        vector,
        channels=vector.channels | {name: replace(vector.channels[name], kind="unsupported")},
    )
    with pytest.raises(CompileError, match="unsupported scope channel kind"):
        scope_boundary(invalid, leaf)(task, "NATIVE();")
    previous = operation.position
    operation.attributes["position"] = IntAttr(leaf.stream.count.data + 1)
    try:
        with pytest.raises(CompileError, match="not beside its checked native endpoint"):
            scope_boundary(vector, leaf)(task, "NATIVE();")
    finally:
        operation.attributes["position"] = previous
