"""Retained-tree ASC formatting and finite host-only scalar/address traces.

The C++ shim records calls and addresses. It executes no tensor arithmetic,
CANN code, FA body, asynchronous queue or device operation, and proves no
hardware completion relation.
"""

import re
import shutil
import subprocess
from dataclasses import replace
from importlib import import_module

import pytest
from xdsl.dialects import arith, scf

from ascendflow import language as N
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.analysis.structured_schedule import (
    ActionGuard,
    Comparison,
    Placement,
    SyncAction,
    SyncKind,
)
from ascendflow.backend.structured import EmissionWork, _emit_owned, emit_structure_debug
from ascendflow.diagnostics import CompileError
from ascendflow.dialects.io import dump
from ascendflow.dialects.structured import AddressOp, NotificationOp, PipelineLoopOp
from ascendflow.passes.lower_native import lower_native_structure
from ascendflow.passes.materialize_structure import materialize_notifications
from ascendflow.passes.structured_sync import synchronize_structure
from ascendflow.scalar import expression, literal
from ascendflow.target.ascend910b import Actor, Engine
from examples.flash_attention.attention import attention_target
from examples.gemm.batched import batched_matmul
from examples.gemm.reduction import reduction_gemm
from examples.vector.scale import scale
from tests.control.test_recursive_native import NESTED, dispatch
from tests.control.test_shared_aiv import SHARED
from tests.fixtures.nested import nested_cv

CARRY = """
def carry(n:N.int64):
    with N.Kernel():
        x=N.alloc_ub((64,),name="x")
        total=7
        for i in N.serial(1,n,2):
            if i<3:
                total=total+i
            else:
                total=total+2
        for j in N.Pipelined(total,stages=1):
            N.fill(x,1.0)
"""


def memory(source):
    structure = lower_native_structure(source)
    return structure, collect_native_memory(structure.module)


def forbid_analysis(monkeypatch):

    def forbidden(*args, **kwargs):
        raise AssertionError("formatting must consume the existing owned facts")

    targets = (
        ("ascendflow.target.native", ("native_contract",)),
        ("ascendflow.passes.lower_native", ("lower_native_structure",)),
        ("ascendflow.analysis.structured_memory", ("collect_native_memory",)),
        ("ascendflow.analysis.structured_bounds", ("admit_memory_bounds",)),
        ("ascendflow.analysis.structured_dependencies", ("build_structured_dependencies",)),
    )
    modules = [(import_module(name), attributes) for name, attributes in targets]
    for module, attributes in modules:
        for attribute in attributes:
            if hasattr(module, attribute):
                monkeypatch.setattr(module, attribute, forbidden)


@pytest.mark.parametrize("source", [scale, batched_matmul])
def test_owned_emission_only_consumes_final_ir_and_preserves_it(source, monkeypatch):
    plan = synchronize_structure(source)
    before = dump(plan.module)
    forbid_analysis(monkeypatch)
    work = EmissionWork()
    asc = _emit_owned(plan, work=work)
    assert dump(plan.module) == before
    assert work.native_templates == work.native_prints == len(plan.memory.templates)
    assert all(count == 1 for count in work.task_prints.values())
    assert work.actions_printed == sum(isinstance(op, NotificationOp) for op in plan.module.walk())
    assert work.output_bytes == len(asc.encode())
    assert "#error" not in asc and 'extern "C" int launch_' in asc
    assert "asc_sync_notify(" in asc and "asc_sync_wait(" in asc
    assert "PIPE_ALL" not in asc  # Final drains are already explicit root EXIT actions.
    assert "npu_size_$" not in asc


def test_fa_structure_debug_prints_each_template_once_and_is_not_executable(monkeypatch):
    structure, facts = memory(attention_target)
    forbid_analysis(monkeypatch)
    work = EmissionWork()
    asc = emit_structure_debug(facts, work=work)
    assert work.native_templates == work.native_prints == structure.work["native_templates"]
    assert work.native_templates == 44
    assert all(count == 1 for count in work.task_prints.values())
    assert work.output_bytes == len(asc.encode())
    assert asc.count("if ASC_IS_AIV") == 1 and asc.count("if ASC_IS_AIC") == 1
    assert '#error "Unsynchronized structured intermediate: debug output only"' in asc
    assert "launch_" not in asc and "PIPE_ALL" not in asc and "$" not in asc
    assert asc.count("asc_set_l0c_copy_params(") == 1
    assert asc.index("asc_set_l0c_copy_params(") < asc.index("if ASC_IS_AIV")


def test_nested_print_work_depends_on_templates_instead_of_iteration_counts():
    outputs = []
    for count in (1024, 1_048_576):
        structure, facts = memory(N.from_source(NESTED).specialize(PARTS=count))
        work = EmissionWork()
        asc = emit_structure_debug(facts, work=work)
        assert work.native_templates == work.native_prints == 7
        assert asc.count("for (int64_t npu_ordinal_") == len(structure.loops) == 3
        assert all(prints == 1 for prints in work.task_prints.values())
        outputs.append(work.output_bytes)
    assert abs(outputs[1] - outputs[0]) < 100


@pytest.mark.parametrize("count,mode,prints", [(0, None, 0), (1, "short", 1), (8, "long", 1)])
def test_static_zero_short_and_long_pipeline_counts(count, mode, prints):
    _, facts = memory(
        N.from_source(f"""
        def fixed():
            with N.Kernel():
                x=N.alloc_ub((64,))
                for t in N.Pipelined({count},stages=4):
                    with N.Stage(3,name="work"):
                        N.fill(x,1.0)
    """)
    )
    work = EmissionWork()
    asc = emit_structure_debug(facts, work=work)
    assert work.native_templates == 1 and work.native_prints == prints
    if mode is None:
        assert "for (int64_t" not in asc
    else:
        assert f"// {mode} pipeline phases" in asc
        assert asc.count("for (int64_t") == 1
    assert "asc_sync_" not in asc


def test_shared_aiv_abi_family_order_and_singleton_actions():
    _, facts = memory(N.from_source(SHARED, disjoint_external=True))
    vector = next(template for template in facts.templates if template.contract.native == "muls")
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
    assert "npu_sid == 1" in asc and "npu_sid == 0" not in asc
    assert "if (npu_sid == 1) { asc_sync_wait(PIPE_MTE2, PIPE_V, event_t(3)); }" in asc
    assert asc.count("npu_sid") > 2 and "npu_group *" in asc
    _, cube = memory(batched_matmul)
    cube_asc = emit_structure_debug(cube)
    assert [label for label in re.findall(r"// stage (\w+)", cube_asc)] == [
        "export",
        "load",
        "operands",
        "compute",
    ]


def test_actions_surround_complete_packet_and_loop_invocation_with_ordered_guards():
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
    assert entry < for_start < fence < native < after < exit_action
    assert asc[entry:for_start].count("event_t(") == 1
    notifications = [op for op in facts.module.walk() if isinstance(op, NotificationOp)]
    for notification in (notifications[0], notifications[-1]):
        assert isinstance(notification.parent_op(), scf.IfOp)
        assert notification.parent_op().parent is loop.parent
    assert asc.count("asc_sync_pipe(") == 1


def test_foreign_plan_module_and_unbound_address_symbols_are_rejected():
    _, facts = memory(scale)
    with pytest.raises(CompileError, match="owned synchronization"):
        _emit_owned(facts)
    plan = synchronize_structure(scale)
    forged = object.__new__(type(plan))
    from dataclasses import fields

    for name in (field.name for field in fields(plan)):
        object.__setattr__(forged, name, getattr(plan, name))
    object.__setattr__(forged, "structure", lower_native_structure(scale))
    with pytest.raises(CompileError, match="different owned structures"):
        _emit_owned(forged)
    invalid = SyncAction(
        facts.templates[0].operation,
        Placement.ENTRY,
        (Actor.AIV0,),
        SyncKind.LOCAL_WAIT,
        Engine.MTE2,
        Engine.V,
        literal(0),
    )
    with pytest.raises(CompileError, match="invocation boundary placement"):
        materialize_notifications(facts, (invalid,))
    template = facts.templates[0]
    address = replace(
        template.accesses[0].address,
        offset=facts.control.loops[next(iter(facts.control.loops))].ordinal,
    )
    # Retained proof expressions cannot choose the executable address.
    changed = replace(
        template,
        accesses=(replace(template.accesses[0], address=address), *template.accesses[1:]),
    )
    altered = replace(facts, templates=(changed, *facts.templates[1:]))
    assert emit_structure_debug(altered) == emit_structure_debug(facts)
    # A packet moved out of its real SSA scope is rejected at that actual use.
    root = facts.module.body.block.first_op
    template.operation.parent.detach_op(template.operation)
    root.body.block.insert_op_before(template.operation, root.body.block.last_op)
    with pytest.raises(CompileError, match="unbound structured SSA scalar"):
        emit_structure_debug(facts)


def test_final_address_ssa_and_notification_operations_are_the_executable_input(monkeypatch):
    from xdsl.dialects.builtin import FloatAttr, IndexType, StringAttr, f32

    _, facts = memory(scale)
    task = facts.templates[0].operation
    action = SyncAction(
        task,
        Placement.BEFORE,
        (Actor.AIV0,),
        SyncKind.LOCAL_WAIT,
        Engine.MTE2,
        Engine.V,
        literal(3),
    )
    materialize_notifications(facts, (action,))
    notification = next(op for op in facts.module.walk() if isinstance(op, NotificationOp))
    address = task.refs[0].owner
    assert isinstance(address, AddressOp)
    constant = arith.ConstantOp.from_int_and_width(128, IndexType())
    address.parent.insert_op_before(constant, address)
    address.operands = (address.source, constant.result, *address.origins, *address.extents)
    event = arith.ConstantOp.from_int_and_width(6, IndexType())
    notification.parent.insert_op_before(event, notification)
    notification.operands = (event.result,)
    notification.kind = StringAttr(SyncKind.LOCAL_SET.value)
    compute = next(t for t in facts.templates if t.contract.native == "muls")
    compute.operation.scalar = FloatAttr(3.0, f32)
    assert compute.contract.scalar == 2.0  # Retained analysis is deliberately unchanged.
    facts.module.verify()

    def forbidden(*args, **kwargs):
        pytest.fail("backend interpreted a retained scalar proof expression")

    monkeypatch.setattr(facts.control.environment, "expression", forbidden)
    asc = emit_structure_debug(facts)
    assert "npu_buffer_0 + (128)" in asc
    assert "float(3.0)" in asc
    assert "asc_sync_notify(PIPE_MTE2, PIPE_V, event_t(6));" in asc
    assert "asc_sync_wait(" not in asc
    notification.parent.erase_op(notification)
    assert "asc_sync_notify(" not in emit_structure_debug(facts)


def test_replaced_native_address_operand_is_read_from_final_ir():
    from xdsl.dialects.builtin import IndexType

    _, facts = memory(scale)
    task = facts.templates[0].operation
    original = task.refs[0].owner
    constant = arith.ConstantOp.from_int_and_width(256, IndexType())
    replacement = original.clone()
    replacement.operands = (
        replacement.source,
        constant.result,
        *replacement.origins,
        *replacement.extents,
    )
    original.parent.insert_ops_before((constant, replacement), original)
    task.operands = (replacement.results[0], *tuple(task.refs)[1:])
    facts.module.verify()
    # This is debug printing of deliberately changed IR, not proof reuse for
    # execution. The backend must not print the detached retained operand.
    asc = emit_structure_debug(facts)
    assert "npu_buffer_0 + (256)" in asc


def test_controller_bounds_come_from_actual_ssa_instead_of_retained_logical_expressions(
    monkeypatch,
):
    from xdsl.dialects.builtin import IndexType

    _, facts = memory(scale)
    loop = next(iter(facts.control.loops))
    original_count = facts.control.loops[loop].count
    two = arith.ConstantOp.from_int_and_width(2, IndexType())
    loop.parent.insert_op_before(two, loop)
    loop.operands = (loop.lower, two.result, loop.step)
    assert facts.control.loops[loop].count is original_count

    def forbidden(*args, **kwargs):
        pytest.fail("backend interpreted a retained control proof expression")

    monkeypatch.setattr(facts.control.environment, "expression", forbidden)
    asc = emit_structure_debug(facts)
    assert re.search(r"npu_count_\d+ = \(2 > 0", asc)
    assert "// long pipeline phases" in asc


TRACE_SHIM = r"""
#include <cstdint>
#include <iostream>
#include <cstdlib>
#define __global__
#define __vector__
#define __cube__
#define __mix__(a,b)
#define __gm__
#define __ubuf__
#define __cbuf__
#define __ca__
#define __cb__
#define __cc__
#define ASC_IS_AIC (trace_actor == 0)
#define ASC_IS_AIV (trace_actor != 0)
using half=uint16_t; using event_t=int;
enum {PIPE_S,PIPE_MTE2,PIPE_MTE1,PIPE_M,PIPE_FIX,PIPE_V,PIPE_MTE3,inc};
int64_t trace_group=0,trace_actor=1;
void asc_init(){} uint64_t asc_get_ctrl(){return 0;} void asc_set_ctrl(uint64_t){}
int64_t asc_get_block_idx(){return trace_group;}
int64_t asc_get_sub_block_id(){return trace_actor-1;}
void asc_set_l0c_copy_params(int,int,int){}
void asc_sync_notify(int,int,int){} void asc_sync_wait(int,int,int){}
void asc_sync_pipe(int){}
template<class... P> void record(const char* name,float scalar,P... p){
 std::cout<<name<<" "<<scalar; ((std::cout<<" "<<uintptr_t(p)),...); std::cout<<"\n";
}
template<class D,class S,class... P> void asc_copy_gm2ub(D d,S s,P...){record("copy",0,s,d);}
template<class D,class S,class... P> void asc_copy_ub2gm(D d,S s,P...){record("copy",0,s,d);}
template<class D,class S,class... P> void asc_copy_gm2l1_nd2nz(D d,S s,P...){record("nd2nz",0,s,d);}
template<class D,class S,class... P> void asc_copy_l12l0a(D d,S s,P...){record("load2d_a",0,s,d);}
template<class D,class S,class... P> void asc_copy_l12l0b(D d,S s,P...){record("load2d_b_nt",0,s,d);}
template<class D,class S,class... P> void load_cbuf_to_cb(D d,S s,P...){record("load2d_b_nn",0,s,d);}
template<class D,class A,class B,class... P> void asc_mmad(D d,A a,B b,P...){record("mad",0,a,b,d);}
template<class D,class S,class... P> void asc_copy_l0c2gm(D d,S s,P...){record("fix",0,s,d);}
template<class D,class S> void asc_mul_scalar(D d,S s,float f,uint32_t){record("muls",f,s,d);}
template<class D> void asc_duplicate(D d,float f,uint32_t){record("fill",f,d);}
"""


def compile_cpp(tmp_path, source, *, sanitize=False):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler required for scalar/address trace")
    path = tmp_path / "trace.cpp"
    path.write_text(source)
    binary = tmp_path / "trace"
    flags = ["-fsanitize=undefined", "-fno-sanitize-recover=undefined"] if sanitize else []
    subprocess.run(
        [compiler, "-std=c++17", "-O0", *flags, str(path), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    return binary


@pytest.mark.parametrize(
    "case", ["scale", "nested", "carry", "shared", "gemm", "cv_cube", "cv_vector"]
)
def test_actual_generated_control_and_gm_addresses_match_independent_ir_dispatch(case, tmp_path):
    sources = {
        "scale": scale,
        "nested": N.from_source(NESTED),
        "carry": N.from_source(CARRY),
        "shared": N.from_source(SHARED, disjoint_external=True),
        "cv_cube": nested_cv,
        "cv_vector": nested_cv,
        "gemm": batched_matmul.specialize(
            M=16,
            N=16,
            K=16,
            GROUPS=3,
            L1_SLOTS=2,
            L0_SLOTS=2,
            C_SLOTS=2,
        ),
    }
    structure, facts = memory(sources[case])
    root = facts.module.body.block.first_op
    groups = 1 if root.blocks is None else root.blocks.data
    actor = "cube" if case in {"gemm", "cv_cube"} else "vector"
    cases = [
        (n, k, group % groups, sid)
        for n, k, group, sid in [(0, 0, 0, 0), (1, 0, 0, 0), (2, 1, 1, 1), (5, 3, 2, 0)]
    ]
    expected_dispatch = []
    for n, k, group, sid in cases:
        runtime = {"tiles": n} if case in {"scale", "gemm"} else {"n": n, "k": k}
        _, expected = dispatch(structure, **runtime, group=group, actor=actor, sid=sid)
        expected_dispatch.append(expected)
    # The original scalar oracle has already run. Only the single owned tree
    # is subsequently materialized and formatted; no executable snapshot copy.
    original_operations = tuple(facts.module.walk())
    materialize_notifications(
        facts,
        tuple(
            SyncAction(
                template.operation,
                placement,
                template.participants,
                SyncKind.FENCE,
                template.contract.engine,
                template.contract.engine,
            )
            for template in facts.templates
            for placement in (Placement.BEFORE, Placement.AFTER)
        ),
    )
    facts.module.verify()
    retained = set(original_operations)
    assert tuple(op for op in facts.module.walk() if op in retained) == original_operations
    asc = emit_structure_debug(facts)
    assert asc.count("asc_sync_pipe(") == 2 * len(facts.templates)
    # This test-only shim deliberately removes the debug artifact's hard error.
    asc = "\n".join(
        line for line in asc.splitlines() if not line.startswith(("#include", "#error"))
    )
    calls = {
        "scale": "scale((uint8_t*)1000000,(uint8_t*)2000000,n);",
        "nested": "nested((uint8_t*)1000000,n,k);",
        "carry": "carry(n);",
        "shared": "shared((uint8_t*)1000000,(uint8_t*)2000000,(uint8_t*)3000000,(uint8_t*)4000000,n);",
        "gemm": "batched_matmul((uint8_t*)1000000,(uint8_t*)2000000,(uint8_t*)3000000,n);",
        "cv_cube": "nested_cv((uint8_t*)1000000,(uint8_t*)2000000,(uint8_t*)3000000,n);",
        "cv_vector": "nested_cv((uint8_t*)1000000,(uint8_t*)2000000,(uint8_t*)3000000,n);",
    }
    binary = compile_cpp(
        tmp_path,
        TRACE_SHIM
        + asc
        + """
int main(int argc,char** argv){
 int64_t n=std::atoll(argv[1]),k=std::atoll(argv[2]);
 trace_group=std::atoll(argv[3]);trace_actor=std::atoll(argv[4]);
 """
        + calls[case]
        + "\n}",
    )
    bases = {
        "scale": {"X": 1_000_000, "Y": 2_000_000, "inputs": 0, "outputs": 512},
        "nested": {"X": 1_000_000, "x": 0},
        "carry": {"x": 0},
        "shared": {"Scratch": 3_000_000, "Y": 4_000_000, "x": 0, "y": 1024},
        "gemm": {
            "A": 1_000_000,
            "B": 2_000_000,
            "C": 3_000_000,
            "a1": 0,
            "b1": 1024,
            "a0": 0,
            "b0": 0,
            "c0": 0,
        },
        "cv_cube": {
            "X": 1_000_000,
            "Score": 2_000_000,
            "O": 3_000_000,
            "x1": 0,
            "a": 0,
            "b": 0,
            "c": 0,
            "v": 0,
        },
        "cv_vector": {
            "X": 1_000_000,
            "Score": 2_000_000,
            "O": 3_000_000,
            "x1": 0,
            "a": 0,
            "b": 0,
            "c": 0,
            "v": 0,
        },
    }[case]
    if case in {"cv_cube", "cv_vector"}:
        allocations = [
            op for op in facts.module.walk() if op.name == "npu.buffer" and not op.external.data
        ]
        assert len({op.ref.type.bank.data for op in allocations}) == len(allocations)
        bases.update({op.sym_name.data: 0 for op in allocations})
    for (n, k, group, sid), expected in zip(cases, expected_dispatch, strict=True):
        result = subprocess.run(
            [str(binary), str(n), str(k), str(group), str(0 if actor == "cube" else sid + 1)],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        actual = [
            (parts[0], float(parts[1]), tuple(map(int, parts[2:])))
            for parts in (line.split() for line in result.stdout.splitlines())
        ]
        assert actual == [
            (native, scalar, tuple(bases[root] + offset for root, offset in refs))
            for native, scalar, refs in expected
        ]


def test_checked_launcher_rejects_overflow_alias_and_bad_pointer_before_dispatch(tmp_path):
    plan = synchronize_structure(scale)
    asc = _emit_owned(plan)
    launcher = asc[asc.index('extern "C" int launch_') :]
    launcher = re.sub(r"  scale<<<[^\n]+", "  ++launches;", launcher)
    source = "#include <cstdint>\n#include <climits>\n#include <cassert>\nint launches=0;\n"
    source += (
        launcher
        + r"""
int main(){
 void* x=(void*)uintptr_t(4096); void* y=(void*)uintptr_t(8192);
 assert(launch_scale(nullptr,nullptr,nullptr,0)==0);
 assert(launch_scale(nullptr,x,y,1)==0 && launches==2);
 assert(launch_scale(nullptr,x,y,-1)==-1);
 assert(launch_scale(nullptr,x,y,INT64_MAX)==-1);
 assert(launch_scale(nullptr,x,y,INT64_MAX/256+1)==-1);
 assert(launch_scale(nullptr,x,x,1)==-1);
 assert(launch_scale(nullptr,(void*)uintptr_t(4097),y,1)==-1);
 assert(launch_scale(nullptr,nullptr,y,1)==-1);
 assert(launch_scale(nullptr,(void*)(UINTPTR_MAX-31),y,1)==-1);
 assert(launches==2);
}
"""
    )
    binary = compile_cpp(tmp_path, source, sanitize=True)
    subprocess.run([str(binary)], check=True, capture_output=True, text=True, timeout=5)


def test_guarded_event_and_first_last_arithmetic_stay_inside_short_circuit(tmp_path):
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
    emitter = _Emitter(facts, EmissionWork())
    code = "\n".join(emitter.block(root.body.block, _Context({}, "vector")))
    assert "emit_actions" not in dir(emitter)
    division = next(
        op for op in root.walk() if isinstance(op, arith.RemUIOp) and op.rhs is parameter.value
    )
    assert isinstance(division.parent_op(), scf.IfOp)
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
                    for j in N.Pipelined(k,stages=2):
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


def test_async_native_and_notification_operations_survive_generic_scalar_passes():
    from xdsl.context import Context
    from xdsl.transforms.canonicalize import CanonicalizePass
    from xdsl.transforms.common_subexpression_elimination import CommonSubexpressionElimination

    from ascendflow.dialects.physical import TaskOp

    _, facts = memory(
        N.from_source("""
        def effects():
            with N.Kernel():
                x=N.alloc_ub((64,))
                N.fill(x,1.0)
                N.fill(x,1.0)
    """)
    )
    actions = tuple(
        SyncAction(
            template.operation,
            Placement.BEFORE,
            template.participants,
            SyncKind.FENCE,
            Engine.V,
            Engine.V,
        )
        for template in facts.templates
        for _ in range(2)
    )
    materialize_notifications(facts, actions)
    effects = tuple(op for op in facts.module.walk() if isinstance(op, (TaskOp, NotificationOp)))
    context = Context()
    CanonicalizePass().apply(context, facts.module)
    CommonSubexpressionElimination().apply(context, facts.module)
    CanonicalizePass().apply(context, facts.module)
    facts.module.verify()
    assert (
        tuple(op for op in facts.module.walk() if isinstance(op, (TaskOp, NotificationOp)))
        == effects
    )


def test_real_reduction_first_chunk_clear_uses_actual_stage_ordinal_ssa(tmp_path):
    _, facts = memory(reduction_gemm.specialize(M=16, N=16, BK=16, GROUPS=1))
    asc = emit_structure_debug(facts)
    assert re.search(r"asc_mmad\([^\n]*\(npu_generation_\d+ == 0\)", asc)
    assert not re.search(r"\bg\b", asc)
    asc = "\n".join(
        line for line in asc.splitlines() if not line.startswith(("#include", "#error"))
    )
    shim = TRACE_SHIM.replace(
        'template<class D,class A,class B,class... P> void asc_mmad(D d,A a,B b,P...){record("mad",0,a,b,d);}',
        'template<class D,class A,class B> void asc_mmad(D,A,B,int,int,int,int,bool,bool,bool clear){std::cout<<"clear "<<clear<<"\\n";}',
    )
    binary = compile_cpp(
        tmp_path,
        shim
        + asc
        + "int main(int argc,char** argv){reduction_gemm((uint8_t*)1000000,(uint8_t*)2000000,(uint8_t*)3000000,std::atoll(argv[1]));}",
    )
    for count in (0, 1, 3):
        result = subprocess.run(
            [str(binary), str(count)], check=True, capture_output=True, text=True, timeout=5
        )
        flags = [int(line[6:]) for line in result.stdout.splitlines() if line.startswith("clear ")]
        assert flags == [int(index == 0) for index in range(count)]

    from ascendflow.backend.structured_native import format_native

    template = next(item for item in facts.templates if item.contract.native == "mad_first")
    from xdsl.dialects.builtin import ArrayAttr, IntAttr, StringAttr

    template.operation.native = StringAttr("mad_cycle")
    template.operation.params = ArrayAttr(IntAttr(v) for v in (16, 16, 16, 2))
    code = format_native(
        template,
        render=lambda value: "actual_offset",
        roots={address.root: "root" for address in facts.addresses.values()},
        task_names={template.operation: "mad"},
        generation="actual_stage_ordinal",
    )
    assert "(actual_stage_ordinal % 2 == 0)" in code
