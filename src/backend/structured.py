"""ASC formatting of the final physical SSA control and notification tree.

No native admission, address decoding, dependency scan or synchronization
selection occurs here. One shared AIV body uses the C API group/sid ABI.
Pipeline templates print once with edge guards, avoiding exponential phase
body duplication under nesting. This does not claim optimized steady code.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from xdsl.dialects import arith, scf
from xdsl.dialects.builtin import FloatAttr, IndexType, IntegerType

from ..analysis.structured_memory import NativeMemoryFacts
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
    PipelineLoopOp,
    PipelineStageOp,
)
from ..scalar import expression, literal
from ..target.ascend910b import Actor
from .native import QUALIFIER, load3d_init
from .structured_native import format_native

_BINARY = {
    arith.AddiOp: "+",
    arith.SubiOp: "-",
    arith.MuliOp: "*",
    arith.DivUIOp: "/",
    arith.RemUIOp: "%",
}


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
    def __init__(self, memory, work):
        require(
            isinstance(memory, NativeMemoryFacts), "structured emission needs owned memory facts"
        )
        require(
            memory.control.module is memory.module, "emission control belongs to another module"
        )
        self.memory, self.control, self.work = memory, memory.control, work
        self.root = memory.module.body.block.first_op
        require(isinstance(self.root, (KernelOp, MixedOp)), "unsupported structured kernel root")
        self.mixed = isinstance(self.root, MixedOp)
        self.templates = {item.operation: item for item in memory.templates}
        self.task_names = {operation: f"task_{i}" for i, operation in enumerate(self.templates)}
        self.buffers = tuple(op for op in memory.module.walk() if isinstance(op, BufferOp))
        self.roots = {op.ref: f"npu_buffer_{i}" for i, op in enumerate(self.buffers)}
        self.externals = tuple(op for op in self.buffers if op.external.data)
        self.parameters = tuple(op for op in memory.module.walk() if isinstance(op, RuntimeSizeOp))
        self.relevant = defaultdict(set)
        for template in memory.templates:
            for actor in template.participants:
                self._mark(template.operation, "cube" if actor is Actor.AIC else "vector")
        for operation in memory.module.walk():
            if isinstance(operation, NotificationOp):
                for participant in operation.participants:
                    actor = Actor(participant.data)
                    self._mark(operation, "cube" if actor is Actor.AIC else "vector")
        self.serial = 0
        self.work.native_templates = len(memory.templates)

    def _mark(self, operation, family):
        while operation is not None:
            self.relevant[operation].add(family)
            operation = operation.parent_op()

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
            return "INT64_MIN" if constant == -(1 << 63) else str(constant)
        require(
            False,
            "unbound structured SSA scalar at emission",
            kind=ErrorKind.UNSUPPORTED,
            phase="structured-ASC",
        )

    def arithmetic(self, operation, context):
        a, b = (self.scalar(operand, context) for operand in operation.operands)
        if isinstance(operation, arith.CmpiOp):
            predicates = ("==", "!=", "<", "<=", ">", ">=", "<", "<=", ">", ">=")
            predicate = operation.predicate.value.data
            if predicate >= 6:
                a, b = f"uint64_t({a})", f"uint64_t({b})"
            code = f"({a} {predicates[predicate]} {b})"
        elif type(operation) in _BINARY:
            if isinstance(operation, (arith.DivUIOp, arith.RemUIOp)):
                a, b = f"uint64_t({a})", f"uint64_t({b})"
            code = f"({a} {_BINARY[type(operation)]} {b})"
        elif isinstance(operation, (arith.MinSIOp, arith.MaxSIOp)):
            predicate = "<" if isinstance(operation, arith.MinSIOp) else ">"
            code = f"({a} {predicate} {b} ? {a} : {b})"
        else:
            assert isinstance(operation, arith.CeilDivUIOp)
            a, b = f"uint64_t({a})", f"uint64_t({b})"
            code = f"({a} / {b} + ({a} % {b} != 0))"
        name = self.fresh("scalar")
        self.bind(operation.results[0], name, context)
        return [f"const {self.ctype(operation.results[0])} {name} = {code};"]

    @staticmethod
    def ctype(value):
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
        packet = template.contract.packet
        require(
            packet is not None and packet.closed_mask, "ASC needs an audited closed native packet"
        )
        local = context.child()
        lines = ["{"]
        stage = operation.parent_op()
        while stage is not None and not isinstance(stage, PipelineStageOp):
            stage = stage.parent_op()
        require(
            stage is not None or template.contract.native not in {"mad_first", "mad_cycle"},
            "native initialization policy needs its actual pipeline ordinal SSA",
        )
        generation = "0" if stage is None else self.scalar(stage.body.block.args[0], local)
        code = format_native(
            template,
            render=lambda value: self.scalar(value, local),
            roots=self.roots,
            task_names=self.task_names,
            generation=generation,
        )
        lines.extend("  " + line for line in code.splitlines())
        lines.append("}")
        if operation.value is not None:
            self.bind(operation.value, f"npu_value_{self.task_names[operation]}", context)
        self.work.native_prints += 1
        self.work.task_prints[operation] = self.work.task_prints.get(operation, 0) + 1
        return lines

    def loop(self, operation, context):
        local = context.child()
        count, ordinal = self.fresh("count"), self.fresh("ordinal")
        lower_value, upper_value = (
            (operation.lower, operation.upper)
            if isinstance(operation, PipelineLoopOp)
            else (operation.lb, operation.ub)
        )
        lower, upper, step = (
            self.scalar(value, context) for value in (lower_value, upper_value, operation.step)
        )
        extent = f"({upper} - {lower})"
        lines = [
            f"const int64_t {count} = ({upper} > {lower} ? {extent} / {step} + ({extent} % {step} != 0) : 0);"
        ]
        carries = []
        if isinstance(operation, scf.ForOp):
            for initial, argument, result in zip(
                operation.iter_args, operation.body.block.args[1:], operation.results, strict=True
            ):
                name = self.fresh("carry")
                lines.append(f"{self.ctype(result)} {name} = {self.scalar(initial, context)};")
                self.bind(argument, name, local)
                self.bind(result, name, context)
                carries.append(name)
        static_lower, static_upper, static_step = (
            value.owner for value in (lower_value, upper_value, operation.step)
        )
        static_count = None
        if all(
            isinstance(value, arith.ConstantOp)
            for value in (static_lower, static_upper, static_step)
        ):
            difference = static_upper.value.value.data - static_lower.value.value.data
            stride = static_step.value.value.data
            static_count = max(0, difference) // stride + (max(0, difference) % stride != 0)
        if static_count == 0:
            return lines
        horizon = operation.schedule.horizon.data if isinstance(operation, PipelineLoopOp) else 0
        if horizon:
            mode = (
                "dynamic" if static_count is None else "short" if static_count < horizon else "long"
            )
            lines.append(f"// {mode} pipeline phases; one body, stage edge guards")
        lines.append(
            f"for (int64_t {ordinal} = 0; {ordinal} < {count} + {horizon}; ++{ordinal}) {{"
        )
        if isinstance(operation, PipelineLoopOp):
            stages = [
                stage
                for stage in operation.body.block.ops
                if isinstance(stage, PipelineStageOp) and context.family in self.relevant[stage]
            ]
            if operation.issue_order is not None and context.family in operation.issue_order.data:
                labels = [label.data for label in operation.issue_order.data[context.family]]
                ranks = {label: rank for rank, label in enumerate(labels)}
                stages.sort(key=lambda stage: ranks[stage.label.data])
            for stage in stages:
                nested = local.child()
                g = self.fresh("generation")
                lines += [
                    f"  {{ // stage {stage.label.data}",
                    f"    const int64_t {g} = {ordinal} - {stage.offset.data};",
                    f"    if ({g} >= 0 && {g} < {count}) {{",
                ]
                iv = f"({lower} + {g} * {step})"
                self.bind(stage.body.block.args[0], g, nested)
                self.bind(stage.body.block.args[1], iv, nested)
                lines.extend("      " + line for line in self.block(stage.body.block, nested))
                lines += ["    }", "  }"]
        else:
            iv = f"({lower} + {ordinal} * {step})"
            self.bind(operation.body.block.args[0], iv, local)
            lines.extend("  " + line for line in self.block(operation.body.block, local))
            yielded = operation.body.block.last_op
            if carries:
                temporaries = []
                for value, carry in zip(yielded.arguments, carries, strict=True):
                    temporary = self.fresh("next")
                    lines.append(
                        f"  const {self.ctype(value)} {temporary} = {self.scalar(value, local)};"
                    )
                    temporaries.append((carry, temporary))
                lines.extend(f"  {carry} = {temporary};" for carry, temporary in temporaries)
        lines.append("}")
        return lines

    def conditional(self, operation, context):
        results, lines = [], []
        for result in operation.results:
            name = self.fresh("choice")
            lines.append(f"{self.ctype(result)} {name};")
            self.bind(result, name, context)
            results.append(name)
        lines.append(f"if ({self.scalar(operation.cond, context)}) {{")
        regions = (operation.true_region, operation.false_region)
        for index, region in enumerate(regions):
            if not region.blocks:
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
            elif isinstance(operation, (PipelineLoopOp, scf.ForOp)):
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
                (*tuple(_BINARY), arith.CmpiOp, arith.MinSIOp, arith.MaxSIOp, arith.CeilDivUIOp),
            ):
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

    def render(self, *, executable=False, host_launcher=False, host_expressions=()):
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
            for template in self.memory.templates:
                if (
                    template.contract.native == "scalar_load"
                    and family in self.relevant[template.operation]
                ):
                    lines.append(
                        f"{indent}float npu_value_{self.task_names[template.operation]} = 0.0f;"
                    )
            context = _Context({}, family)
            body = self.block(self.root.body.block, context)
            lines.extend(indent + line for line in body)
            if self.mixed:
                lines.append("  }")
        lines.append("}")
        if executable and host_launcher:
            lines.extend(self.launcher(host_expressions))
        result = "\n".join(lines) + "\n"
        self.work.output_bytes = len(result.encode())
        return result

    def launcher(self, host_expressions):
        return _launcher(self, host_expressions)


def emit_structure_debug(memory, *, work=None):
    """Explicit non-executable inspection; no launcher or completion claim."""
    work = EmissionWork() if work is None else work
    return _Emitter(memory, work).render()


def _emit_owned(plan, *, host_launcher=True, work=None):
    from ..passes.structured_sync import StructuredSynchronization

    require(
        isinstance(plan, StructuredSynchronization), "executable ASC needs owned synchronization"
    )
    require(
        plan.structure.module is plan.module is plan.memory.module
        and plan.admission.structure is plan.structure
        and plan.dependencies.memory is plan.memory
        and plan.selection.dependencies is plan.dependencies
        and plan.resources.selection is plan.selection,
        "synchronization and memory belong to different owned structures",
    )
    work = EmissionWork() if work is None else work
    return _Emitter(plan.memory, work).render(
        executable=True,
        host_launcher=host_launcher,
        host_expressions=plan.admission.host_expressions,
    )


def _launcher(emitter, host_expressions):
    # Defined below the recursive printer so host checks cannot participate in
    # address/control admission or change selected instruction placement.
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
    arguments += [f"npu_size_{op.sym_name.data}" for op in emitter.parameters]
    lines = ["", f'extern "C" int launch_{root.sym_name.data}({", ".join(signature)}) {{']
    names = {}
    for parameter in emitter.parameters:
        name = f"npu_size_{parameter.sym_name.data}"
        names[emitter.control.environment.expression(parameter.value)] = name
        lines.append(
            f"  if ({name} < {parameter.minimum.data} || {name} % {parameter.multiple.data} != 0) return -1;"
        )

    def checked(value):
        if value in names:
            return names[value]
        if value.constant is not None:
            require(-(1 << 63) <= value.constant < (1 << 63), "host scalar constant exceeds int64")
            return "INT64_MIN" if value.constant == -(1 << 63) else str(value.constant)
        require(value.kind != "argument", "host check contains an unbound varying coordinate")
        a, b = (checked(child) for child in value.args)
        if value.kind in {"floordiv", "rem", "ceildiv"}:
            lines.append(f"  if ({a} < 0 || {b} <= 0) return -1;")
        symbol = f"npu_host_{len(names)}"
        if value.kind in {"add", "mul"}:
            code = f"(__int128)({a}) {'+' if value.kind == 'add' else '*'} (__int128)({b})"
        else:
            code = value.c(
                names
                | {child: rendered for child, rendered in zip(value.args, (a, b), strict=True)}
            )
        lines.append(f"  const __int128 {symbol}_wide = {code};")
        lines.append(
            f"  if ({symbol}_wide < INT64_MIN || {symbol}_wide > INT64_MAX) return -1; // scalar overflow"
        )
        lines.append(f"  const int64_t {symbol} = int64_t({symbol}_wide);")
        names[value] = symbol
        return symbol

    for value in host_expressions:
        checked(value)
    for index, allocation in enumerate(emitter.externals):
        size = literal(allocation.nbytes.data)
        if allocation.extent_count is not None:
            size = expression(
                "mul",
                emitter.control.environment.expression(allocation.extent_count),
                literal(allocation.count_scale.data),
            )
        nbytes = checked(size)
        lines += [
            f"  const uint64_t npu_bytes_{index} = uint64_t({nbytes});",
            f"  const uintptr_t npu_ptr_{index} = (uintptr_t)arg_{buffer_names[index]};",
            f"  if (npu_bytes_{index} && (!npu_ptr_{index} || npu_ptr_{index} % 32 || npu_ptr_{index} > UINTPTR_MAX - npu_bytes_{index})) return -1;",
        ]
    if root.disjoint_external.data:
        for i in range(len(emitter.externals)):
            for j in range(i + 1, len(emitter.externals)):
                lines.append(
                    f"  if (npu_bytes_{i} && npu_bytes_{j} && npu_ptr_{i} < npu_ptr_{j} + npu_bytes_{j} && npu_ptr_{j} < npu_ptr_{i} + npu_bytes_{i}) return -1;"
                )
    groups = 1 if root.blocks is None else root.blocks.data
    lines += [
        f"  {root.sym_name.data}<<<{groups}, nullptr, stream>>>({', '.join(arguments)});",
        "  return 0;",
        "}",
    ]
    return lines
