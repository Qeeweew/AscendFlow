"""Render actual query clocks, guards and mixed native event order."""

import shutil
import subprocess
from dataclasses import replace

import pytest
from xdsl.dialects.builtin import ArrayAttr, IntAttr

from tests.fixtures.scoped_queries import scoped_query_recurrence
from tests.test_scope_serial_source import SOURCE
from xdsl_npu import language as N
from xdsl_npu.budget import limit
from xdsl_npu.completion import CountDomain, Position
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.dialects.physical import QueryChannelOp, QuerySyncOp, ScopeSyncOp
from xdsl_npu.lowering import lower_source
from xdsl_npu.scalar import bindings
from xdsl_npu.scope_codegen import emit_scope_actor, emit_scopes, scope_boundary
from xdsl_npu.scope_model import extract_scopes
from xdsl_npu.scope_query_codegen import query_loop_lines, render_query_sync
from xdsl_npu.scope_query_completion import QuerySite
from xdsl_npu.scope_query_protocols import (
    QueryProtocol,
    install_query_protocols,
    read_query_protocols,
)
from xdsl_npu.scope_sync import _plan_scope_parent, _verify_scope_parent

FIRST, LAST = Position(False, 0), Position(True, 0)
ODD = CountDomain(2, 1, minimum=3)


@N.prim_func(disjoint_external=True)
def composite_query_bound(
    Y: N.Tensor((2, 2, 8), N.f32, access="write"),
    queries: N.int64(minimum=0),
    count: N.int64(minimum=1),
):
    with N.Kernel(2) as group:
        state = N.alloc_ub((2, 8), N.f32, name="state")
        for query in N.serial(N.ceildiv(N.max(0, queries * 3 - group), 2)):
            for kv in N.Pipelined(count, stages=1):
                with N.Stage(0, name="compute"):
                    for row in N.Pipelined(2, stages=2):
                        with N.Stage(0, name="fill"):
                            N.fill(state[row, :], 1.0)
                        with N.Stage(1, name="store"):
                            if kv == count - 1:
                                N.copy(state[row, :], Y[group, row, :])


def endpoint(checked, pipe, *, query, parent, point):
    for uid, site in checked.access.endpoints.values():
        if site.task[0] != "aic" or site.position.offset:
            continue
        if any(
            task.uid == site.task[1] and task.pipe == pipe
            for data in checked.program.leaves[uid].loop.actors.values()
            for task in data.program.tasks
        ):
            return QuerySite(query, uid, replace(site, point=point), parent)
    pytest.fail("fixture has no matching actual native")


@pytest.fixture(scope="module")
def installed():
    """Invocation-local proof, intentionally insufficient for public emission."""
    with limit(60):
        module = lower_source(scoped_query_recurrence)
        _plan_scope_parent(extract_scopes(module))
        _, checked, _ = _verify_scope_parent(extract_scopes(module))
        protocol = QueryProtocol(
            "query_release",
            endpoint(checked, "FIX", query=0, parent=LAST, point="end"),
            endpoint(checked, "M", query=1, parent=FIRST, point="begin"),
            CountDomain(1, 0, minimum=1),
            (7,),
        )
        install_query_protocols(checked.program, checked.access, (protocol,))
        _, checked, _ = _verify_scope_parent(extract_scopes(module))
    return checked, protocol


def native(checked, endpoint):
    return next(
        task
        for data in checked.program.leaves[endpoint.call].loop.actors.values()
        for task in data.program.tasks
        if task.uid == endpoint.site.task[1]
    )


def operation(checked, action):
    return next(
        op
        for op in checked.program.module.walk()
        if isinstance(op, QuerySyncOp) and op.action.data == action
    )


def render(checked, protocol, action):
    point = protocol.source if action == "set" else protocol.target
    return render_query_sync(
        checked,
        checked.program.leaves[point.call],
        operation(checked, action),
        native(checked, point),
        protocol,
    )


def changed(installed, *, distance=2, ids=(7, 2, 5), domain=ODD):
    checked, _ = installed
    module = checked.program.module.clone()
    declaration = next(op for op in module.walk() if isinstance(op, QueryChannelOp))
    declaration.attributes["distance"] = IntAttr(distance)
    declaration.attributes["event_ids"] = ArrayAttr(IntAttr(i) for i in ids)
    from xdsl_npu.scope_query_protocols import _count

    declaration.attributes["count"] = _count(domain)
    _, checked, _ = _verify_scope_parent(extract_scopes(module))
    return checked, read_query_protocols(checked.program, checked.access)[0]


def test_actual_actor_has_distinct_query_kv_child_loops_and_keeps_query_captures(installed):
    checked, _ = installed
    code = "\n".join(emit_scope_actor(checked, "aic", {}))
    assert "for (int64_t npu_query_g = 0; npu_query_g < npu_query_n; ++npu_query_g)" in code
    assert "for (int64_t npu_parent_step = 0;" in code
    assert "for (int64_t step = 0; step < n +" in code
    assert (
        code.index("npu_query_g = 0")
        < code.index("npu_parent_step = 0")
        < code.index("for (int64_t step = 0")
    )
    assert "const int64_t npu_parent_n = n;" in code
    assert "const int64_t n = 2;" in code
    capture_lines = [line for line in code.splitlines() if "uint8_t* cap_" in line]
    assert any("= Q +" in line and "npu_query_g" in line for line in capture_lines)
    assert any(
        "= Score +" in line and "npu_query_g" in line and "npu_parent_g" in line
        for line in capture_lines
    )
    assert "npu_size_$" not in code and "$parent1" not in code
    assert "PIPE_ALL" not in code


def test_single_id_is_constant_and_activation_uses_independent_query_kv_and_child_clocks(installed):
    checked, protocol = installed
    set_code, wait_code = (render(checked, protocol, action) for action in ("set", "wait"))
    for code in (set_code, wait_code):
        assert "g == 0" in code and "npu_parent_n >= 1" in code
        assert "event_t(7)" in code and "npu_query_g %" not in code
    assert "npu_parent_g == (npu_parent_n - 1)" in set_code
    assert "npu_query_n > 1 && npu_query_g < (npu_query_n - 1)" in set_code
    assert "npu_parent_g == 0 && npu_query_g >= 1" in wait_code
    assert "asc_sync_notify(PIPE_FIX, PIPE_M" in set_code
    assert "asc_sync_wait(PIPE_FIX, PIPE_M" in wait_code


def test_count_parity_and_query_distance_two_are_never_mixed_into_kv_distance(installed):
    checked, protocol = changed(installed)
    set_code, wait_code = (render(checked, protocol, action) for action in ("set", "wait"))
    for code in (set_code, wait_code):
        assert "npu_parent_n >= 3" in code and "(npu_parent_n % 2) == 1" in code
    assert "npu_query_n > 2 && npu_query_g < (npu_query_n - 2)" in set_code
    assert "npu_query_g >= 2" in wait_code
    assert "(npu_query_g % 3)" in set_code
    assert "((npu_query_g - 2) % 3)" in wait_code
    assert "npu_parent_g == (npu_parent_n - 1)" in set_code
    assert "npu_parent_g == 0" in wait_code


def test_mixed_parent_query_order_matches_actual_native_cluster(installed):
    checked, protocol = installed
    for action, point in (("set", protocol.source), ("wait", protocol.target)):
        leaf = checked.program.leaves[point.call]
        op = operation(checked, action)
        scope_ops = []
        cursor = op.prev_op
        while isinstance(cursor, (ScopeSyncOp, QuerySyncOp)):
            scope_ops.append(cursor)
            cursor = cursor.prev_op
        assert any(isinstance(sync, ScopeSyncOp) for sync in scope_ops)
        code = scope_boundary(checked, leaf)(native(checked, point), "NATIVE();")
        marker = f"/* {protocol.name} */"
        for scope in scope_ops:
            assert code.index(f"/* {scope.channel.data} */") < code.index(marker)
        assert (code.index("NATIVE();") < code.index(marker)) == (action == "set")


def test_invocation_only_report_cannot_open_public_asc_gate(installed):
    checked, _ = installed
    assert checked.query_report is None
    with pytest.raises(CompileError, match="verified cross-query protocols"):
        emit_scopes(checked)


def test_query_notification_precedes_the_actual_following_child_command(installed):
    checked, _ = installed
    module = checked.program.module.clone()
    for op in tuple(module.walk()):
        if isinstance(op, (QueryChannelOp, QuerySyncOp)):
            op.parent.erase_op(op)
    _, checked, _ = _verify_scope_parent(extract_scopes(module))
    protocol = QueryProtocol(
        "query_load_release",
        endpoint(checked, "MTE2", query=0, parent=LAST, point="end"),
        endpoint(checked, "MTE1", query=1, parent=FIRST, point="begin"),
        CountDomain(1, 0, minimum=1),
        (7,),
    )
    install_query_protocols(checked.program, checked.access, (protocol,))
    _, checked, _ = _verify_scope_parent(extract_scopes(module))
    uid = protocol.source.call
    inner = next(iter(checked.analyses[uid][0].actors))
    assert any(
        command.kind == "set" and command.task.uid == protocol.source.site.task[1]
        for command in checked.analyses[uid][2][inner]
    )
    code = "\n".join(emit_scope_actor(checked, "aic", {}))
    assert code.index(render(checked, protocol, "set")) < code.index(
        f"// aic/{protocol.source.site.task[1]}: set,"
    )


def test_actual_uniform_serial_bound_is_rendered_without_a_cyclic_owner_rewrite():
    text = (
        SOURCE.replace("N.ceildiv(N.max(0,total_queries-group),GROUPS)", "total_queries")
        .replace("((total_queries,2,16,16)", "((GROUPS,total_queries,2,16,16)")
        .replace("((total_queries,4,64)", "((GROUPS,total_queries,4,64)")
        .replace("Q[flat,part,:,:]", "Q[group,query,part,:,:]")
        .replace("O[flat,lane*2+part,:]", "O[group,query,lane*2+part,:]")
    )
    program = extract_scopes(lower_source(N.from_source(text, disjoint_external=True)))
    invocation = next(iter(program.invocations.values()))
    assert invocation.cyclic is None
    names, _ = bindings([invocation.bound])
    assert query_loop_lines(invocation, names) == [
        "  const int64_t npu_query_n = npu_size_total_queries;"
    ]


def test_public_compilation_emits_only_after_actual_query_verification():
    from tests.test_scope_query_planning import query_state_store
    from xdsl_npu import compile
    from xdsl_npu.scope_sync import verify_scopes

    artifact = compile(query_state_store, timeout_s=40)
    report, checked = verify_scopes(artifact.plan, _details=True)
    assert report["cross_query_verified"] is True
    assert checked.query_report["verified"] is True
    assert emit_scopes(checked) == artifact.asc
    assert "for (int64_t npu_query_g = 0;" in artifact.asc
    assert "npu_query_g >= 1" in artifact.asc
    assert artifact.asc.count("asc_sync_pipe(PIPE_ALL)") == 1
    assert artifact.asc.index("} // serial query:") < artifact.asc.index("asc_sync_pipe(PIPE_ALL)")


def test_composite_query_total_is_checked_before_launch_and_computed_once_per_owner():
    from xdsl_npu import compile

    artifact = compile(composite_query_bound, timeout_s=40)
    assert artifact.report["cross_query_verified"] is True
    assert "uniform product overflow" in artifact.asc
    assert "npu_size_queries * 3" in artifact.asc
    assert "npu_size_$core" not in artifact.asc
    assert artifact.asc.index("const int64_t core_id") < artifact.asc.index("npu_query_n =")
    assert "npu_query_g" in artifact.asc


def test_generated_query_bounds_and_event_guards_execute_literal_integer_oracle(
    installed, tmp_path
):
    """Host C++ executes scalar snippets only; no CANN or device is involved."""
    compiler = shutil.which("c++")
    assert compiler is not None
    checked, protocol = changed(installed)
    set_code, wait_code = (render(checked, protocol, action) for action in ("set", "wait"))
    invocation = checked.program.invocations["aic"]
    names, _ = bindings([invocation.cyclic.total])
    bound = "\n".join(query_loop_lines(invocation, names))
    maximum = (1 << 63) - 1
    bound_cases = [
        (total, core, (max(0, total - core) + checked.program.groups - 1) // checked.program.groups)
        for total in (0, 1, 2, 3, 7, maximum)
        for core in range(checked.program.groups)
    ]
    source = tmp_path / "query_scalars.cpp"
    source.write_text(
        "#include <cassert>\n#include <cstdint>\n#include <initializer_list>\n"
        "enum {PIPE_FIX=1, PIPE_M=2}; using event_t=int;\n"
        "int sets, waits, set_id, wait_id;\n"
        "void asc_sync_notify(int,int,int id) { ++sets; set_id=id; }\n"
        "void asc_sync_wait(int,int,int id) { ++waits; wait_id=id; }\n"
        "int main() {\nstruct Bound {int64_t total, core, expected;};\n"
        + "for (auto item : {"
        + ",".join(
            f"Bound{{{total}LL,{core},{expected}LL}}" for total, core, expected in bound_cases
        )
        + "}) {\nconst int64_t npu_size_total_queries=item.total, core_id=item.core;\n"
        + bound
        + "\nassert(npu_query_n == item.expected);\n}\n"
        + "const int ids[] = {7,2,5};\n"
        + "for (int64_t npu_parent_n : {0LL,1LL,2LL,3LL,7LL,9223372036854775807LL})\n"
        + "for (int64_t npu_query_n : {0LL,1LL,2LL,3LL,7LL,9223372036854775807LL})\n"
        + "for (int64_t npu_query_g : {0LL,1LL,2LL,npu_query_n ? npu_query_n-1 : 0LL})\n"
        + "for (int64_t npu_parent_g : {0LL,1LL,npu_parent_n ? npu_parent_n-1 : 0LL})\n"
        + "for (int64_t g : {0LL,1LL}) {\nsets=waits=0;\n"
        + "if (npu_query_g < npu_query_n) {\n"
        + set_code
        + "\n"
        + wait_code
        + "\n}\n"
        + "bool active = npu_query_g < npu_query_n && g == 0 && npu_parent_n >= 3 && npu_parent_n%2 == 1;\n"
        + "bool expected_set = active && npu_parent_g == npu_parent_n-1 && __int128(npu_query_g)+2 < npu_query_n;\n"
        + "bool expected_wait = active && npu_parent_g == 0 && npu_query_g >= 2;\n"
        + "assert(sets == int(expected_set) && waits == int(expected_wait));\n"
        + "if (sets) assert(set_id == ids[npu_query_g%3]);\n"
        + "if (waits) assert(wait_id == ids[(npu_query_g-2)%3]);\n"
        + "}\n}\n"
    )
    executable = tmp_path / "query_scalars"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-fsanitize=undefined",
            "-fno-sanitize-recover=undefined",
            str(source),
            "-o",
            str(executable),
        ],
        capture_output=True,
        check=True,
        timeout=20,
    )
    subprocess.run([str(executable)], capture_output=True, check=True, timeout=5)
