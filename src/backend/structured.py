"""ASC formatting of the final physical SSA control and notification tree.

No native admission, address decoding, dependency scan or synchronization
selection occurs here. One shared AIV body uses the C API group/sid ABI.
Each pipeline has one physical clock loop. Adjacent stages at the same offset
share one existence guard. Nested loop coordinates remain separate SSA bindings.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from xdsl.dialects import arith, scf
from xdsl.dialects.builtin import FloatAttr, IndexType, IntegerAttr, IntegerType

from ..analysis.scalar_uses import ScalarUses
from ..analysis.structured_schedule import SyncKind
from ..diagnostics import ErrorKind, require
from ..dialects.physical import (
    BufferOp,
    CoreIdOp,
    EndOp,
    FixConfigOp,
    KernelOp,
    Load3DConfigOp,
    MixedOp,
    RuntimeSizeOp,
    TaskOp,
)
from ..dialects.structured import (
    ActorDomainOp,
    AddressOp,
    NotificationOp,
)
from ..passes.executable import ExecutablePlan, NativeExecution
from ..target.ascend910b import Actor
from .native import QUALIFIER, load3d_init
from .structured_native import format_native

_BINARY = {
    arith.AddiOp: "+",
    arith.SubiOp: "-",
    arith.MuliOp: "*",
    arith.DivUIOp: "/",
    arith.RemUIOp: "%",
    arith.AndIOp: "&&",
    arith.OrIOp: "||",
}


def _integer_constant(value):
    owner = value.owner
    if isinstance(owner, arith.ConstantOp) and isinstance(owner.value, IntegerAttr):
        return owner.value.value.data
    return None


def _empty_region(region):
    return all(
        isinstance(op, scf.YieldOp) and not op.arguments
        for block in region.blocks
        for op in block.ops
    )


@dataclass(slots=True)
class EmissionWork:
    native_templates: int = 0
    native_prints: int = 0
    actions_printed: int = 0
    output_bytes: int = 0
    task_prints: dict = field(default_factory=dict)


@dataclass
class _Context:
    values: dict
    family: str

    def child(self):
        return _Context(dict(self.values), self.family)


class _Emitter:
    def __init__(self, plan, work):
        require(isinstance(plan, ExecutablePlan), "structured emission needs terminal SSA")
        self.plan, self.work = plan, work
        self.expression_cost = {}
        self.root = plan.module.body.block.first_op
        require(isinstance(self.root, (KernelOp, MixedOp)), "unsupported structured kernel root")
        self.mixed = isinstance(self.root, MixedOp)
        self.templates = {item.operation: item for item in plan.natives}
        self.task_names = {operation: f"task_{i}" for i, operation in enumerate(self.templates)}
        self.buffers = tuple(op for op in plan.module.walk() if isinstance(op, BufferOp))
        self.roots = {op.ref: f"npu_buffer_{i}" for i, op in enumerate(self.buffers)}
        self.externals = tuple(op for op in self.buffers if op.external.data)
        self.parameters = tuple(op for op in plan.module.walk() if isinstance(op, RuntimeSizeOp))
        uses = ScalarUses(plan.module, plan.natives)
        self.relevant = uses.operations
        self.scalar_uses = uses.values
        self.scalar_reads = uses.reads
        self.serial = 0
        self.work.native_templates = len(plan.natives)

    def fresh(self, prefix):
        result = f"npu_{prefix}_{self.serial}"
        self.serial += 1
        return result

    def scalar(self, value, context):
        if value in context.values:
            return context.values[value]
        owner = value.owner
        if isinstance(owner, arith.ConstantOp):
            if isinstance(owner.value, FloatAttr):
                return repr(owner.value.value.data) + "f"
            constant = owner.value.value.data
            if isinstance(value.type, IntegerType) and value.type.width.data == 1:
                return "true" if constant else "false"
            return "INT64_MIN" if constant == -(1 << 63) else str(constant)
        require(
            False,
            "unbound structured SSA scalar at emission",
            kind=ErrorKind.UNSUPPORTED,
            phase="structured-ASC",
            witness=(str(value), str(getattr(owner, "name", type(owner).__name__))),
        )

    def arithmetic(self, operation, context):
        operands = tuple(self.scalar(operand, context) for operand in operation.operands)
        a, b = operands[:2]
        if isinstance(operation, arith.SelectOp):
            code = f"({a} ? {b} : {operands[2]})"
        elif isinstance(operation, arith.CmpiOp):
            predicates = ("==", "!=", "<", "<=", ">", ">=", "<", "<=", ">", ">=")
            predicate = operation.predicate.value.data
            if predicate >= 6:
                a, b = f"uint64_t({a})", f"uint64_t({b})"
            code = f"({a} {predicates[predicate]} {b})"
        elif type(operation) in _BINARY:
            if isinstance(operation, (arith.DivUIOp, arith.RemUIOp)):
                ranges = self.plan.scalar_ranges
                if (
                    ranges.get(operation.lhs, (-1, 0))[0] < 0
                    or ranges.get(operation.rhs, (0, 0))[0] <= 0
                ):
                    a, b = f"uint64_t({a})", f"uint64_t({b})"
            operator = _BINARY[type(operation)]
            if isinstance(operation, (arith.AndIOp, arith.OrIOp)) and not (
                isinstance(operation.result.type, IntegerType)
                and operation.result.type.width.data == 1
            ):
                operator = "&" if isinstance(operation, arith.AndIOp) else "|"
            constant = _integer_constant(operation.rhs)
            if isinstance(operation, arith.AddiOp) and constant is not None and constant < 0:
                code = f"({a} - {-constant})"
            else:
                code = f"({a} {operator} {b})"
        elif isinstance(operation, (arith.MinSIOp, arith.MaxSIOp)):
            predicate = "<" if isinstance(operation, arith.MinSIOp) else ">"
            code = f"({a} {predicate} {b} ? {a} : {b})"
        else:
            assert isinstance(operation, arith.CeilDivUIOp)
            a, b = f"uint64_t({a})", f"uint64_t({b})"
            code = f"({a} / {b} + ({a} % {b} != 0))"
        # Inline single-use integer SSA and small predicates at their actual
        # use. Their operands are immutable SSA bindings in this lexical
        # invocation; no load, floating arithmetic or effect is rematerialized.
        result = operation.results[0]
        own_cost = (
            6
            if isinstance(operation, (arith.DivUIOp, arith.RemUIOp, arith.CeilDivUIOp))
            else 2
            if isinstance(operation, (arith.SelectOp, arith.MinSIOp, arith.MaxSIOp))
            else 1
        )
        cost = own_cost + sum(
            self.expression_cost.get((value, context.family), 0) for value in operation.operands
        )
        uses = self.scalar_reads[result, context.family]
        if (
            (uses == 1 and cost <= 12)
            or (uses == 2 and cost == 1)
            or (isinstance(operation, arith.CmpiOp) and cost == 1)
        ):
            self.expression_cost[result, context.family] = cost
            self.bind(operation.results[0], code, context)
            return []
        name = self.fresh("scalar")
        self.bind(operation.results[0], name, context)
        return [f"const {self.ctype(operation.results[0])} {name} = {code};"]

    @staticmethod
    def ctype(value):
        if isinstance(value.type, IntegerType) and value.type.width.data == 1:
            return "bool"
        return "int64_t" if isinstance(value.type, (IndexType, IntegerType)) else "float"

    def bind(self, value, name, context):
        context.values[value] = name

    def notification(self, operation, context):
        actors = tuple(
            actor
            for value in operation.participants
            if ((actor := Actor(value.data)) is Actor.AIC) == (context.family == "cube")
        )
        if not actors:
            return []
        event = None if operation.event is None else self.scalar(operation.event, context)
        source, destination = operation.source.data, operation.destination.data
        kind = SyncKind(operation.kind.data)
        if kind is SyncKind.FENCE:
            code = f"asc_sync_pipe(PIPE_{source});"
        elif kind in {SyncKind.LOCAL_SET, SyncKind.LOCAL_WAIT}:
            function = "asc_sync_notify" if kind is SyncKind.LOCAL_SET else "asc_sync_wait"
            code = f"{function}(PIPE_{source}, PIPE_{destination}, event_t({event}));"
        else:
            is_set = kind is SyncKind.CROSS_SET
            function = "asc_sync_block_arrive" if is_set else "asc_sync_block_wait"
            code = f"{function}(PIPE_{source if is_set else destination}, uint16_t({event}));"
        if self.mixed and context.family == "vector" and len(actors) == 1:
            code = f"if (npu_sid == {0 if actors[0] is Actor.AIV0 else 1}) {{ {code} }}"
        self.work.actions_printed += 1
        return [code]

    def task(self, operation, context):
        template = self.templates[operation]
        if context.family not in self.relevant[operation]:
            return []
        packet = template.spec.packet
        require(
            packet is not None and packet.closed_mask, "ASC needs an audited closed native packet"
        )
        local = context.child()
        generation = (
            self.scalar(operation.refs[-1], local)
            if template.spec.native in {"mad_first", "mad_cycle"}
            else "0"
        )
        code = format_native(
            template,
            render=lambda value: self.scalar(value, local),
            roots=self.roots,
            task_names=self.task_names,
            generation=generation,
        )
        lines = code.splitlines()
        if operation.value is not None:
            self.bind(operation.value, f"npu_value_{self.task_names[operation]}", context)
        self.work.native_prints += 1
        self.work.task_prints[operation] = self.work.task_prints.get(operation, 0) + 1
        return lines

    def loop(self, operation, context):
        local = context.child()
        clock = "npu.pipeline_clock" in operation.attributes
        induction = self.fresh("step" if clock else "ordinal")
        lower, upper, step = (
            self.scalar(value, context) for value in (operation.lb, operation.ub, operation.step)
        )
        carries, lines = [], []
        for initial, argument, result in zip(
            operation.iter_args, operation.body.block.args[1:], operation.results, strict=True
        ):
            name = self.fresh("carry")
            lines.append(f"{self.ctype(result)} {name} = {self.scalar(initial, context)};")
            self.bind(argument, name, local)
            self.bind(result, name, context)
            carries.append(name)
        low, high = _integer_constant(operation.lb), _integer_constant(operation.ub)
        if low is not None and high is not None and low >= high:
            return lines
        update = (
            f"++{induction}" if _integer_constant(operation.step) == 1 else f"{induction} += {step}"
        )
        lines.append(f"for (int64_t {induction} = {lower}; {induction} < {upper}; {update}) {{")
        self.bind(operation.body.block.args[0], induction, local)
        body = self.block(operation.body.block, local)
        if not body and not carries:
            return lines[:-1]
        lines.extend("  " + line for line in body)
        temporaries = []
        for value, carry in zip(operation.body.block.last_op.arguments, carries, strict=True):
            temporary = self.fresh("next")
            lines.append(f"  const {self.ctype(value)} {temporary} = {self.scalar(value, local)};")
            temporaries.append((carry, temporary))
        lines.extend(f"  {carry} = {temporary};" for carry, temporary in temporaries)
        return lines + ["}"]

    def conditional(self, operation, context):
        condition = _integer_constant(operation.cond)
        code = None if condition is not None else self.scalar(operation.cond, context)
        if condition is not None:
            region = operation.true_region if condition else operation.false_region
            if not region.blocks:
                return []
            nested = context.child()
            lines = self.block(region.block, nested)
            for result, value in zip(
                operation.results, region.block.last_op.arguments, strict=True
            ):
                self.bind(result, self.scalar(value, nested), context)
            return lines
        # A guarded chain has one ordered short-circuit condition. Prefix
        # arithmetic can join it only when it emits no declaration/effect;
        # its expression remains on the protected right side of &&.
        while not operation.results and _empty_region(operation.false_region):
            nested = context.child()
            body = tuple(operation.true_region.block.ops)
            inner = next((op for op in body if isinstance(op, scf.IfOp)), None)
            if inner is None or inner.results or not _empty_region(inner.false_region):
                break
            if any(
                not isinstance(op, (scf.YieldOp, arith.ConstantOp))
                for op in body[body.index(inner) + 1 :]
            ):
                break
            prefix = []
            for op in body[: body.index(inner)]:
                if isinstance(op, arith.ConstantOp):
                    continue
                if not isinstance(
                    op,
                    (
                        *tuple(_BINARY),
                        arith.CmpiOp,
                        arith.SelectOp,
                        arith.MinSIOp,
                        arith.MaxSIOp,
                        arith.CeilDivUIOp,
                    ),
                ):
                    prefix.append("effect")
                    break
                prefix.extend(self.arithmetic(op, nested))
            if prefix:
                break
            code = f"({code} && {self.scalar(inner.cond, nested)})"
            operation, context = inner, nested
        results, lines = [], []
        for result in operation.results:
            name = self.fresh("choice")
            lines.append(f"{self.ctype(result)} {name};")
            self.bind(result, name, context)
            results.append(name)
        lines.append(f"if ({code}) {{")
        regions = (operation.true_region, operation.false_region)
        for index, region in enumerate(regions):
            if not region.blocks:
                continue
            if index and not results and _empty_region(region):
                continue
            if index:
                lines.append("} else {")
            nested = context.child()
            lines.extend("  " + line for line in self.block(region.block, nested))
            if results:
                for name, value in zip(results, region.block.last_op.arguments, strict=True):
                    lines.append(f"  {name} = {self.scalar(value, nested)};")
        lines.append("}")
        return lines

    def block(self, block, context):
        lines = []
        for operation in block.ops:
            if isinstance(operation, TaskOp):
                lines.extend(self.task(operation, context))
            elif isinstance(operation, NotificationOp):
                lines.extend(self.notification(operation, context))
            elif isinstance(operation, ActorDomainOp):
                if operation.kind.data == context.family:
                    nested = context.child()
                    self.bind(
                        operation.body.block.args[0],
                        "npu_sid" if context.family == "vector" else "0",
                        nested,
                    )
                    lines.extend(self.block(operation.body.block, nested))
            elif isinstance(operation, scf.ForOp):
                if context.family in self.relevant[operation] or operation.results:
                    lines.extend(self.loop(operation, context))
            elif isinstance(operation, scf.IfOp):
                if context.family in self.relevant[operation] or operation.results:
                    lines.extend(self.conditional(operation, context))
            elif isinstance(operation, FixConfigOp):
                if context.family == "cube":
                    lines += [
                        "asc_set_l0c_copy_params(1, 0, 0);",
                        "asc_sync_notify(PIPE_S, PIPE_FIX, 0);",
                        "asc_sync_wait(PIPE_S, PIPE_FIX, 0);",
                    ]
            elif isinstance(operation, Load3DConfigOp):
                if context.family == "cube":
                    if operation.fp32 is not None:
                        lines += [
                            "asc_set_fp32_mode();",
                            "asc_sync_notify(PIPE_S, PIPE_M, event_t(0));",
                            "asc_sync_wait(PIPE_S, PIPE_M, event_t(0));",
                        ]
                    lines.extend(
                        line.strip()
                        for line in load3d_init((operation.a_width.data, operation.b_width.data))
                    )
            elif isinstance(operation, RuntimeSizeOp):
                self.bind(operation.value, f"npu_size_{operation.sym_name.data}", context)
            elif isinstance(operation, CoreIdOp):
                self.bind(operation.value, "npu_group", context)
            elif isinstance(
                operation,
                (
                    *tuple(_BINARY),
                    arith.CmpiOp,
                    arith.MinSIOp,
                    arith.MaxSIOp,
                    arith.CeilDivUIOp,
                    arith.SelectOp,
                ),
            ):
                if context.family in self.relevant[operation]:
                    lines.extend(self.arithmetic(operation, context))
            elif isinstance(
                operation,
                (
                    BufferOp,
                    AddressOp,
                    EndOp,
                    scf.YieldOp,
                    arith.ConstantOp,
                ),
            ):
                continue
            else:
                require(
                    False,
                    f"unmodeled structured ASC operation {operation.name}",
                    kind=ErrorKind.UNSUPPORTED,
                    phase="structured-ASC",
                )
        return lines

    def render(self, *, executable=False, host_launcher=False):
        name = self.root.sym_name.data
        require(name.isascii() and name.isidentifier(), "kernel name is not a C identifier")
        signature = [f"__gm__ uint8_t* {self.roots[op.ref]}" for op in self.externals]
        signature.extend(f"int64_t npu_size_{op.sym_name.data}" for op in self.parameters)
        kind = (
            "__mix__(1, 2)"
            if self.mixed
            else "__cube__"
            if self.root.core.data == "aic"
            else "__vector__"
        )
        lines = [
            "// AscendFlow ASC: retained SSA control and actor issue packets",
            '#include "c_api/asc_simd.h"',
            "#include <cstdint>",
            "#include <climits>",
        ]
        if not executable:
            lines.append('#error "Unsynchronized structured intermediate: debug output only"')
        from .vector_mask import helper_lines, needs_helpers

        if needs_helpers(self.plan.natives):
            lines.extend([""] + helper_lines())
        lines.extend(
            [
                "",
                f'extern "C" __global__ {kind} void {name}({", ".join(signature)}) {{',
                "  asc_init();",
                "  asc_set_ctrl(asc_get_ctrl() | (uint64_t(1) << 48));",
                "  const int64_t npu_group = asc_get_block_idx();",
                "  const int64_t npu_sid = " + ("asc_get_sub_block_id();" if self.mixed else "0;"),
            ]
        )
        families = (
            ("cube", "vector")
            if self.mixed
            else ("cube" if self.root.core.data == "aic" else "vector",)
        )
        for family in families:
            indent = "    " if self.mixed else "  "
            if self.mixed:
                lines.append(f"  if ASC_IS_{'AIC' if family == 'cube' else 'AIV'} {{")
            peaks = {}
            for allocation in self.buffers:
                if allocation.external.data:
                    continue
                bank = allocation.ref.type.bank.data
                if (bank == "UB") != (family == "vector"):
                    continue
                alignment = 32 if bank == "UB" else 512
                offset = (peaks.get(bank, 0) + alignment - 1) // alignment * alignment
                peaks[bank] = offset + allocation.nbytes.data
                lines.append(
                    f"{indent}{QUALIFIER[bank]} uint8_t* {self.roots[allocation.ref]} = ({QUALIFIER[bank]} uint8_t*){offset};"
                )
            for template in self.plan.natives:
                if (
                    template.spec.native in {"scalar_load", "index_load"}
                    and family in self.relevant[template.operation]
                ):
                    lines.append(
                        f"{indent}{'int64_t' if template.spec.native == 'index_load' else 'float'} npu_value_{self.task_names[template.operation]} = 0;"
                    )
            context = _Context({}, family)
            body = self.block(self.root.body.block, context)
            lines.extend(indent + line for line in body)
            if self.mixed:
                lines.append("  }")
        lines.append("}")
        if executable and host_launcher:
            lines.extend(self.launcher())
        result = "\n".join(lines) + "\n"
        self.work.output_bytes = len(result.encode())
        return result

    def launcher(self):
        return _launcher(self)


def emit_structure_debug(memory, *, work=None):
    """Explicit non-executable inspection; no launcher or completion claim."""
    from ..analysis.local_schedule import bind_local_schedules
    from ..passes.lower_execution import lower_execution
    from ..passes.simplify_scalars import ScalarFacts, simplify_scalars

    facts = ScalarFacts()
    natives = tuple(NativeExecution.from_template(t, t.operation) for t in memory.templates)
    lower_execution(memory.module, natives, facts, bind_local_schedules(memory))
    simplify_scalars(memory.module, facts=facts)
    work = EmissionWork() if work is None else work
    return _Emitter(ExecutablePlan(memory.module, natives, facts.ranges), work).render()


def _emit_owned(plan, *, host_launcher=True, work=None):
    from ..passes.structured_sync import StructuredSynchronization

    require(
        isinstance(plan, StructuredSynchronization), "executable ASC needs owned synchronization"
    )
    work = EmissionWork() if work is None else work
    return _Emitter(plan.executable, work).render(
        executable=True,
        host_launcher=host_launcher,
    )


def _launcher(emitter):
    """Bind cache aliases of the same GM allocations, then launch one kernel."""
    root = emitter.root
    buffer_names = [buffer.sym_name.data for buffer in emitter.externals]
    require(
        all(name.isascii() and name.isidentifier() for name in buffer_names)
        and len(set(buffer_names)) == len(buffer_names),
        "launcher buffer names must be unique C identifiers",
    )
    signature = ["void* stream"] + [f"void* arg_{name}" for name in buffer_names]
    signature += [f"int64_t npu_size_{op.sym_name.data}" for op in emitter.parameters]
    arguments = [f"(uint8_t*)arg_{name}" for name in buffer_names]
    bypass = {
        i
        for i, buffer in enumerate(emitter.externals)
        if buffer.cache is not None and buffer.cache.data == "bypass"
    }
    cache_setup, declarations = [], []
    if bypass:
        declarations = [
            'extern "C" int32_t rtGetDevice(int32_t*);',
            'extern "C" int32_t rtGetL2CacheOffset(uint32_t, uint64_t*);',
        ]
        cache_setup = [
            "  int32_t npu_device = 0;",
            "  int32_t npu_status = rtGetDevice(&npu_device);",
            "  if (npu_status != 0) return npu_status;",
            "  uint64_t npu_l2_offset = 0;",
            "  npu_status = rtGetL2CacheOffset(uint32_t(npu_device), &npu_l2_offset);",
            "  if (npu_status != 0) return npu_status;",
            "  if (npu_l2_offset == 0) return -1;",
        ]
        for i in bypass:
            arguments[i] = f"(uint8_t*)(uintptr_t(arg_{buffer_names[i]}) + npu_l2_offset)"
    arguments += [f"npu_size_{op.sym_name.data}" for op in emitter.parameters]
    groups = 1 if root.blocks is None else root.blocks.data
    return [
        "",
        *declarations,
        f'extern "C" int launch_{root.sym_name.data}({", ".join(signature)}) {{',
        *cache_setup,
        f"  {root.sym_name.data}<<<{groups}, nullptr, stream>>>({', '.join(arguments)});",
        "  return 0;",
        "}",
    ]
