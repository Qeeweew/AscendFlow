"""control emission contracts."""

import re
import subprocess
from dataclasses import replace

import pytest
from xdsl.dialects import arith

from ascendflow import language as N
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.analysis.structured_schedule import (
    Placement,
    SyncAction,
    SyncKind,
)
from ascendflow.backend.structured import EmissionWork, _emit_owned, emit_structure_debug
from ascendflow.passes.executable import NativeExecution
from ascendflow.passes.materialize_structure import materialize_notifications
from ascendflow.passes.simplify_scalars import simplify_scalars
from ascendflow.passes.structured_sync import synchronize_structure
from ascendflow.target.ascend910b import Actor
from examples.gemm.batched import batched_matmul
from examples.gemm.reduction import reduction_gemm
from examples.vector.scale import scale
from tests.fixtures.control import CARRY, NESTED, SHARED
from tests.fixtures.nested import nested_cv
from tests.models.native_control import dispatch
from tests.support.asc_trace import TRACE_SHIM, compile_cpp, memory


@pytest.mark.parametrize("depth", [1, 2, 4])
def test_nested_pipelines_have_one_clock_loop_and_no_phase_controllers(depth):
    source = ["def nested(n:N.i64):", "    with N.Kernel():", "        x=N.alloc_ub((64,))"]
    for level in range(depth):
        indent = " " * (8 + 8 * level)
        source += [
            indent + f"for i{level} in N.Pipelined(n):",
            indent + "    with N.Stage(2,name='work'):",
        ]
    source.append(" " * (8 + 8 * depth) + "N.fill(x,1.0)")
    _, facts = memory(N.from_source("\n".join(source)))
    work = EmissionWork()
    asc = emit_structure_debug(facts, work=work)
    assert work.native_templates == work.native_prints == 1
    assert asc.count("for (int64_t npu_step_") == depth
    assert not re.search(r"NpuPipelinePhase|prologue|steady|epilogue|short|auto npu_stage_", asc)


@pytest.mark.parametrize("count,prints", [(0, 0), (1, 1), (8, 1)])
def test_static_empty_and_nonempty_pipelines_use_one_dispatch_rule(count, prints):
    _, facts = memory(
        N.from_source(f"""
        def fixed():
            with N.Kernel():
                x=N.alloc_ub((64,))
                for t in N.Pipelined({count}):
                    with N.Stage(3,name="work"):
                        N.fill(x,1.0)
""")
    )
    work = EmissionWork()
    asc = emit_structure_debug(facts, work=work)
    assert work.native_templates == 1 and work.native_prints == prints
    if count == 0:
        assert "for (int64_t" not in asc
    else:
        assert asc.count("for (int64_t npu_step_") == 1
    assert not re.search(r"NpuPipelinePhase|prologue|steady|epilogue|short|auto npu_stage_", asc)
    assert "asc_sync_" not in asc


def test_controller_bounds_come_from_actual_ssa_instead_of_retained_logical_expressions(
    monkeypatch,
):
    from xdsl.dialects.builtin import IndexType

    _, facts = memory(scale)
    loop = next(iter(facts.control.loops))
    original_count = facts.control.loops[loop].count
    two = arith.ConstantOp.from_int_and_width(2, IndexType())
    loop.parent.insert_op_before(two, loop)
    loop.operands = (two.result,)
    assert facts.control.loops[loop].count is original_count

    def forbidden(*args, **kwargs):
        pytest.fail("backend interpreted a retained control proof expression")

    monkeypatch.setattr(facts.control.environment, "expression", forbidden)
    asc = emit_structure_debug(facts)
    assert re.search(r"npu_step_\d+ < 4;", asc)
    assert asc.count("for (int64_t npu_step_") == 1
    assert "prologue" not in asc and "epilogue" not in asc


@pytest.mark.parametrize(
    "case", ["scale", "nested", "carry", "shared", "gemm", "cv_cube", "cv_vector"]
)
@pytest.mark.parametrize("simplified", [False, True])
def test_actual_generated_control_and_gm_addresses_match_independent_ir_dispatch(
    case, simplified, tmp_path
):
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
        for n, k, group, sid in [
            (0, 0, 0, 0),
            (1, 0, 0, 0),
            (2, 1, 1, 1),
            (3, 2, 1, 0),
            (5, 3, 2, 0),
            (8, 5, 1, 1),
        ]
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
                template.spec.engine,
                template.spec.engine,
            )
            for template in facts.templates
            for placement in (Placement.BEFORE, Placement.AFTER)
        ),
    )
    facts.module.verify()
    retained = set(original_operations)
    assert tuple(op for op in facts.module.walk() if op in retained) == original_operations
    if simplified:
        simplify_scalars(facts.module)
        facts.module.verify()
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


@pytest.mark.parametrize("case", ["scale", "nested"])
def test_compact_native_and_sync_traces_match_the_actual_ir(case, tmp_path):
    from tests.models.structured_events import _Interpret

    plan = synchronize_structure(scale if case == "scale" else N.from_source(NESTED))
    assert plan.condition_work.input_notifications > plan.condition_work.output_notifications
    asc = _emit_owned(plan, host_launcher=False)
    assert not re.search(r"NpuPipelinePhase|prologue|steady|epilogue|short|auto npu_stage_", asc)
    code = "\n".join(line for line in asc.splitlines() if not line.startswith("#include"))
    route = 'const char* engines[]={"S","MTE2","MTE1","M","FIX","V","MTE3"};'
    shim = TRACE_SHIM.replace(
        "void asc_sync_notify(int,int,int){} void asc_sync_wait(int,int,int){}",
        route
        + 'void asc_sync_notify(int a,int b,int id){std::cout<<"local_set "<<engines[a]<<" "<<engines[b]<<" "<<id<<"\\n";} '
        + 'void asc_sync_wait(int a,int b,int id){std::cout<<"local_wait "<<engines[a]<<" "<<engines[b]<<" "<<id<<"\\n";}',
    ).replace(
        "void asc_sync_pipe(int){}",
        'void asc_sync_pipe(int a){std::cout<<"fence "<<engines[a]<<" "<<engines[a]<<" -1\\n";}',
    )
    call = (
        "scale((uint8_t*)1000000,(uint8_t*)2000000,n);"
        if case == "scale"
        else "nested((uint8_t*)1000000,n,k);"
    )
    binary = compile_cpp(
        tmp_path,
        shim
        + code
        + "int main(int argc,char**argv){int64_t n=std::atoll(argv[1]),k=std::atoll(argv[2]);"
        + call
        + "}",
        sanitize=True,
    )
    for n, k in [(0, 0), (1, 1), (2, 1), (3, 2), (4, 3), (8, 5)]:
        parameters = {"tiles": n} if case == "scale" else {"n": n, "k": k}
        interpreter = _Interpret(
            collect_native_memory(plan.module), parameters, physical=True, maximum=10000
        )
        interpreter.block(plan.module.body.block.first_op.body.block, {}, 0, 0, Actor.AIV0)
        expected = [
            (
                command.occurrence.operation.native.data,
                command.occurrence.operation.scalar.value.data,
            )
            if command.kind == "native"
            else (
                command.kind,
                command.source.value,
                command.destination.value,
                -1 if command.event is None else command.event,
            )
            for command in interpreter.commands
        ]
        result = subprocess.run(
            [str(binary), str(n), str(k)], check=True, capture_output=True, text=True, timeout=5
        )
        actual = [
            (parts[0], parts[1], parts[2], int(parts[3]))
            if parts[0] in {"local_set", "local_wait", "fence"}
            else (parts[0], float(parts[1]))
            for parts in (line.split() for line in result.stdout.splitlines())
        ]
        assert actual == expected, (case, n, k)


def test_real_reduction_first_chunk_clear_uses_actual_stage_ordinal_ssa(tmp_path):
    _, facts = memory(reduction_gemm.specialize(M=16, N=16, BK=16, GROUPS=1))
    asc = emit_structure_debug(facts)
    assert re.search(r"asc_mmad\([^\n]*== 0\)", asc)
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

    template = next(item for item in facts.templates if item.spec.native == "mad_first")
    from xdsl.dialects.builtin import ArrayAttr, IntAttr, StringAttr

    template.operation.native = StringAttr("mad_cycle")
    template.operation.params = ArrayAttr(IntAttr(v) for v in (16, 16, 16, 2))
    # A different primitive needs its own static contract, even in a formatter
    # microtest; the original mad_first contract cannot authorize mad_cycle.
    template = replace(
        template, spec=replace(template.spec, native="mad_cycle", params=(16, 16, 16, 2))
    )
    code = format_native(
        NativeExecution.from_template(template, template.operation),
        render=lambda value: "actual_offset",
        roots={address.root: "root" for address in facts.addresses.values()},
        cache_aliases={},
        task_names={template.operation: "mad"},
        generation="actual_stage_ordinal",
    )
    assert "(actual_stage_ordinal % 2 == 0)" in code
