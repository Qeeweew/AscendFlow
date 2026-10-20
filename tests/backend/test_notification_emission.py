"""notification emission contracts."""

import subprocess

import pytest
from xdsl.dialects import arith, scf

from ascendflow import language as N
from ascendflow.analysis.structured_schedule import (
    ActionGuard,
    Comparison,
    Placement,
    SyncAction,
    SyncKind,
)
from ascendflow.backend.structured import EmissionWork, emit_structure_debug
from ascendflow.dialects.structured import NotificationOp, PipelineLoopOp
from ascendflow.passes.executable import ExecutablePlan, NativeExecution
from ascendflow.passes.materialize_structure import materialize_notifications
from ascendflow.passes.simplify_scalars import simplify_scalars
from ascendflow.scalar import expression, literal
from ascendflow.target.ascend910b import Actor, Engine
from examples.gemm.batched import batched_matmul
from examples.vector.scale import scale
from tests.fixtures.control import SHARED
from tests.support.asc_trace import TRACE_SHIM, compile_cpp, memory


def test_shared_aiv_abi_family_order_and_singleton_actions():
    _, facts = memory(N.from_source(SHARED, disjoint_external=True))
    vector = next(template for template in facts.templates if template.spec.native == "muls")
    action = SyncAction(
        vector.operation,
        Placement.BEFORE,
        (Actor.AIV1,),
        SyncKind.LOCAL_WAIT,
        Engine.MTE2,
        Engine.V,
        literal(3),
    )
    domains = tuple(op for op in facts.module.walk() if op.name == "npu.actor_domain")
    materialize_notifications(facts, (action,))
    facts.module.verify()
    assert tuple(op for op in facts.module.walk() if op.name == "npu.actor_domain") == domains
    asc = emit_structure_debug(facts)
    assert asc.count("asc_mul_scalar(") == 1
    assert "const int64_t npu_group = asc_get_block_idx();" in asc
    assert "const int64_t npu_sid = asc_get_sub_block_id();" in asc
    assert asc.count("npu_sid == 1") == 1 and "npu_sid == 0" not in asc
    assert "asc_sync_wait(PIPE_MTE2, PIPE_V, event_t(3));" in asc
    assert asc.count("npu_sid") > 2
    assert "auto npu_stage_" not in asc and "operator()" not in asc
    _, cube = memory(batched_matmul)
    cube_asc = emit_structure_debug(cube)
    assert cube_asc.index("asc_copy_l0c2gm(") < cube_asc.index("asc_copy_gm2l1_nd2nz(")


def test_actions_surround_complete_packet_and_loop_invocation_with_ordered_guards(tmp_path):
    from tests.models.structured_events import _Interpret

    _, facts = memory(scale)
    loop = next(op for op in facts.control.loops if isinstance(op, PipelineLoopOp))
    logical = facts.control.loops[loop]
    task = facts.templates[1].operation
    last = expression("sub", logical.count, literal(1))
    event = expression("rem", expression("floordiv", last, literal(2)), literal(8))
    guard = ActionGuard((Comparison(logical.count, ">", literal(0)),))
    actions = (
        SyncAction(
            loop,
            Placement.ENTRY,
            (Actor.AIV0,),
            SyncKind.LOCAL_WAIT,
            Engine.MTE2,
            Engine.V,
            event,
            guard,
        ),
        SyncAction(task, Placement.BEFORE, (Actor.AIV0,), SyncKind.FENCE, Engine.V, Engine.V),
        SyncAction(
            task,
            Placement.AFTER,
            (Actor.AIV0,),
            SyncKind.LOCAL_SET,
            Engine.V,
            Engine.MTE3,
            literal(4),
        ),
        SyncAction(
            loop,
            Placement.EXIT,
            (Actor.AIV0,),
            SyncKind.CROSS_SET,
            Engine.MTE3,
            Engine.FIX,
            literal(9),
            guard,
        ),
    )
    materialize_notifications(facts, actions)
    facts.module.verify()
    asc = emit_structure_debug(facts)
    entry = asc.index("asc_sync_wait(")
    for_start = asc.index("for (int64_t")
    fence, native, after = map(
        asc.index,
        (
            "asc_sync_pipe(PIPE_V)",
            "asc_mul_scalar(",
            "asc_sync_notify(PIPE_V, PIPE_MTE3",
        ),
    )
    exit_action = asc.index("asc_sync_block_arrive(")
    assert fence < native < after
    assert entry < for_start < exit_action
    assert asc[entry:for_start].count("event_t(") == 1
    notifications = [op for op in facts.module.walk() if isinstance(op, NotificationOp)]
    for notification in (notifications[0], notifications[-1]):
        assert isinstance(notification.parent_op(), scf.IfOp)
        assert notification.parent_op().parent is facts.module.body.block.first_op.body.block
    assert asc.count("asc_sync_pipe(") == 1
    shim = TRACE_SHIM.replace(
        "void asc_sync_notify(int,int,int){} void asc_sync_wait(int,int,int){}",
        'void asc_sync_notify(int,int,int id){std::cout<<"set "<<id<<"\\n";} '
        'void asc_sync_wait(int,int,int id){std::cout<<"wait "<<id<<"\\n";}',
    ).replace(
        "void asc_sync_pipe(int){}",
        'void asc_sync_pipe(int){std::cout<<"fence\\n";} '
        'void asc_sync_block_arrive(int,int id){std::cout<<"arrive "<<id<<"\\n";}',
    )
    code = "\n".join(
        line for line in asc.splitlines() if not line.startswith(("#include", "#error"))
    )
    binary = compile_cpp(
        tmp_path,
        shim
        + code
        + "int main(int argc,char**argv){scale((uint8_t*)1000000,(uint8_t*)2000000,std::atoll(argv[1]));}",
        sanitize=True,
    )
    kinds = {"local_set": "set", "local_wait": "wait", "cross_set": "arrive", "fence": "fence"}
    for count in (0, 1, 2, 3, 7):
        interpreter = _Interpret(facts, {"tiles": count}, physical=True, maximum=1000)
        interpreter.block(facts.module.body.block.first_op.body.block, {}, 0, 0, Actor.AIV0)
        expected = [
            (command.occurrence.operation.native.data, None)
            if command.kind == "native"
            else (kinds[command.kind], command.event)
            for command in interpreter.commands
        ]
        result = subprocess.run(
            [str(binary), str(count)], check=True, capture_output=True, text=True, timeout=5
        )
        actual = [
            (parts[0], int(parts[1]) if parts[0] in {"set", "wait", "arrive"} else None)
            for parts in (line.split() for line in result.stdout.splitlines())
        ]
        assert actual == expected


@pytest.mark.parametrize("simplified", [False, True])
def test_single_aiv_guard_precedes_an_id_undefined_on_the_other_lane(tmp_path, simplified):
    _, facts = memory(N.from_source(SHARED, disjoint_external=True))
    vector = next(template for template in facts.templates if template.spec.native == "muls")
    materialize_notifications(
        facts,
        (
            SyncAction(
                vector.operation,
                Placement.BEFORE,
                (Actor.AIV1,),
                SyncKind.LOCAL_WAIT,
                Engine.MTE2,
                Engine.V,
                expression("floordiv", literal(3), facts.control.sid),
            ),
        ),
    )
    if simplified:
        simplify_scalars(facts.module)
    asc = emit_structure_debug(facts)
    assert asc.count("npu_sid == 1") == 1
    asc = "\n".join(
        line for line in asc.splitlines() if not line.startswith(("#include", "#error"))
    )
    shim = TRACE_SHIM.replace(
        "void asc_sync_notify(int,int,int){} void asc_sync_wait(int,int,int){}",
        'void asc_sync_notify(int,int,int){} void asc_sync_wait(int,int,int id){std::cout<<"wait "<<trace_actor<<" "<<id<<"\\n";}',
    )
    binary = compile_cpp(
        tmp_path,
        shim
        + asc
        + "int main(){for(trace_actor=1;trace_actor<=2;++trace_actor)"
        + "shared((uint8_t*)1000000,(uint8_t*)2000000,(uint8_t*)3000000,"
        + "(uint8_t*)4000000,1);}",
        sanitize=True,
    )
    result = subprocess.run([str(binary)], check=True, capture_output=True, text=True, timeout=5)
    assert [line for line in result.stdout.splitlines() if line.startswith("wait ")] == ["wait 2 3"]


@pytest.mark.parametrize("simplified", [False, True])
def test_guarded_event_and_first_last_arithmetic_stay_inside_short_circuit(tmp_path, simplified):
    from ascendflow.backend.structured import _Context, _Emitter

    _, facts = memory(
        N.from_source("""
        def guarded(n:N.int64):
            with N.Kernel():
                pass
    """)
    )
    root = facts.module.body.block.first_op
    parameter = next(op for op in facts.module.walk() if op.name == "npu.runtime_size")
    count = facts.control.environment.expression(parameter.value)
    last = expression("sub", count, literal(1))
    guard = ActionGuard(
        (
            Comparison(count, ">", literal(0)),
            Comparison(expression("rem", last, literal(2)), "==", literal(0)),
        )
    )
    action = SyncAction(
        root,
        Placement.ENTRY,
        (Actor.AIV0,),
        SyncKind.LOCAL_WAIT,
        Engine.MTE2,
        Engine.V,
        expression("rem", literal(7), count),
        guard,
    )
    materialize_notifications(facts, (action,))
    facts.module.verify()
    if simplified:
        simplify_scalars(facts.module)
    emitter = _Emitter(
        ExecutablePlan(
            facts.module,
            tuple(NativeExecution.from_template(t, t.operation) for t in facts.templates),
            {},
        ),
        EmissionWork(),
    )
    code = "\n".join(emitter.block(root.body.block, _Context({}, "vector")))
    assert "emit_actions" not in dir(emitter)
    division = next(
        op for op in root.walk() if isinstance(op, arith.RemUIOp) and op.rhs is parameter.value
    )
    assert isinstance(division.parent_op(), scf.IfOp)
    if simplified:
        condition = division.parent_op().cond.owner
        assert isinstance(condition, arith.CmpiOp) and condition.predicate.value.data == 4
        assert condition.lhs is parameter.value
        assert condition.rhs.owner.value.value.data == 0
    else:
        assert isinstance(division.parent_op().parent_op(), scf.IfOp)
    source = (
        """
#include <cstdint>
#include <climits>
#include <cassert>
#include <initializer_list>
enum {PIPE_MTE2,PIPE_V};using event_t=int;
int calls=0,id=-1;
void asc_sync_wait(int,int,int event){++calls;id=event;}
int main(){for(int64_t n:std::initializer_list<int64_t>{0,1,2,3,INT64_MAX}){calls=0;id=-1;
const int64_t npu_size_n=n;
"""
        + code
        + """
 assert(calls==int(n>0 && (n-1)%2==0));
 if(calls)assert(id==7%n);
}}
"""
    )
    binary = compile_cpp(tmp_path, source, sanitize=True)
    subprocess.run([str(binary)], check=True, capture_output=True, text=True, timeout=5)


def test_singleton_participant_guard_precedes_event_arithmetic_in_actual_actor_scope(tmp_path):
    from ascendflow.dialects.structured import ActorDomainOp

    _, facts = memory(
        N.from_source("""
        def singleton():
            with N.Kernel(kind="mixed"):
                pass
    """)
    )
    root = facts.module.body.block.first_op
    divisor = expression("sub", literal(1), facts.control.sid)
    action = SyncAction(
        root,
        Placement.ENTRY,
        (Actor.AIV0,),
        SyncKind.LOCAL_WAIT,
        Engine.MTE2,
        Engine.V,
        expression("rem", literal(7), divisor),
    )
    original = tuple(facts.module.walk())
    materialize_notifications(facts, (action,))
    facts.module.verify()
    retained = set(original)
    assert tuple(op for op in facts.module.walk() if op in retained) == original
    domain = next(op for op in root.walk() if isinstance(op, ActorDomainOp))
    remainder = next(op for op in root.walk() if isinstance(op, arith.RemUIOp))
    assert domain not in facts.control.frames
    assert isinstance(remainder.parent_op(), scf.IfOp)
    assert remainder.parent_op().parent_op() is domain
    comparison = remainder.parent_op().cond.owner
    assert comparison.lhs is domain.body.block.args[0]
    asc = emit_structure_debug(facts)
    asc = "\n".join(
        line for line in asc.splitlines() if not line.startswith(("#include", "#error"))
    )
    shim = TRACE_SHIM.replace(
        "void asc_sync_notify(int,int,int){} void asc_sync_wait(int,int,int){}",
        "int calls=0; void asc_sync_notify(int,int,int){} void asc_sync_wait(int,int,int){++calls;}",
    )
    source = (
        shim
        + asc
        + """
int main(){
 trace_actor=1; singleton(); if(calls!=1)return 1;
 trace_actor=2; singleton(); if(calls!=1)return 2;
 trace_actor=0; singleton(); if(calls!=1)return 3;
}
"""
    )
    binary = compile_cpp(tmp_path, source, sanitize=True)
    subprocess.run([str(binary)], check=True, capture_output=True, text=True, timeout=5)


def test_notification_ssa_binds_actual_nested_serial_and_pipeline_coordinates(tmp_path):
    source = N.from_source("""
        def coordinates(n:N.int64,k:N.int64):
            with N.Kernel():
                x=N.alloc_ub((64,))
                for i in N.serial(1,n,2):
                    for j in N.Pipelined(k):
                        with N.Stage(1,name="work"):
                            N.fill(x,1.0)
""")
    _, facts = memory(source)
    serial, pipeline = facts.control.loops.values()
    event = expression(
        "rem",
        expression("add", expression("mul", literal(4), serial.ordinal), pipeline.ordinal),
        literal(8),
    )
    materialize_notifications(
        facts,
        (
            SyncAction(
                facts.templates[0].operation,
                Placement.BEFORE,
                (Actor.AIV0,),
                SyncKind.LOCAL_WAIT,
                Engine.MTE2,
                Engine.V,
                event,
            ),
        ),
    )
    facts.module.verify()
    notification = next(op for op in facts.module.walk() if isinstance(op, NotificationOp))
    assert notification.parent is facts.templates[0].operation.parent
    asc = emit_structure_debug(facts)
    asc = "\n".join(
        line for line in asc.splitlines() if not line.startswith(("#include", "#error"))
    )
    shim = TRACE_SHIM.replace(
        "void asc_sync_notify(int,int,int){} void asc_sync_wait(int,int,int){}",
        'void asc_sync_notify(int,int,int){} void asc_sync_wait(int,int,int id){std::cout<<"event "<<id<<"\\n";}',
    )
    binary = compile_cpp(
        tmp_path,
        shim
        + asc
        + "int main(int argc,char** argv){coordinates(std::atoll(argv[1]),std::atoll(argv[2]));}",
        sanitize=True,
    )
    for n, k in ((0, 0), (1, 3), (2, 0), (2, 1), (5, 3)):
        result = subprocess.run(
            [str(binary), str(n), str(k)],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        events = [int(line[6:]) for line in result.stdout.splitlines() if line.startswith("event ")]
        assert events == [
            (4 * ordinal + j) % 8 for ordinal, _ in enumerate(range(1, n, 2)) for j in range(k)
        ]
