"""Native tensor-memory facts on the retained recursive physical structure.

Each actual native calls the target rule once; each AddressOp is decoded once.
One Vector template has two participant identities, not two copied tasks.
These are memory facts only. Axis/byte bounds, native overlap/alignment,
ownership, register/controller effects and completion remain separate duties
before this intermediate structure may become an executable plan.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from types import MappingProxyType

from xdsl.dialects import scf
from xdsl.dialects.builtin import IndexType, ModuleOp, f32
from xdsl.ir import SSAValue

from ..diagnostics import require
from ..dialects.physical import TaskOp
from ..dialects.source import _dominates
from ..dialects.structured import PipelineStageOp, ReadyGroupOp
from ..dialects.types import RefType
from ..mask import (
    ITERATION_VALUES,
    MASK_REPEAT_POLICY,
    Program,
    iteration_strategy,
    word_strategy,
)
from ..scalar import expression, literal
from ..target import native
from ..target.ascend910b import Actor, Engine, supports_engine
from ..target.packets import audited_packet
from ..target.scalars import round_f32
from .native import AccessMode, ConditionalRead
from .structured_addresses import AddressFacts, StructuredAddresses
from .structured_control import ControlFacts, ControlFrame, build_control_facts
from .structured_schedule import Comparison


@dataclass(frozen=True, slots=True)
class MemoryAccess:
    operand: SSAValue
    operand_index: int
    address: AddressFacts
    mode: AccessMode
    read_condition: ConditionalRead | None = None


@dataclass(frozen=True, slots=True)
class MemoryTemplate:
    operation: TaskOp
    frame: ControlFrame
    spec: native.NativeSpec
    accesses: tuple[MemoryAccess, ...]
    scalar_source: TaskOp | None

    @property
    def participants(self):
        return self.frame.participants


@dataclass(frozen=True)
class NativeMemoryFacts:
    module: ModuleOp
    control: ControlFacts
    templates: tuple[MemoryTemplate, ...]
    addresses: Mapping[SSAValue, AddressFacts]
    native_contract_calls: int
    ready_groups: tuple[tuple[MemoryTemplate, ...], ...] = ()

    @cached_property
    def by_operation(self):
        return MappingProxyType({template.operation: template for template in self.templates})


def _operand_count(operation):
    """Leading reference operands; later operands are scalar/predicate atoms."""
    count = operation.attributes.get("address_count")
    if count is None:
        return len(operation.refs)
    require(
        0 <= count.data <= len(operation.refs),
        "native address operand count exceeds its actual operands",
    )
    return count.data


def scalar_operands(operation):
    return tuple(operation.refs[_operand_count(operation) :])


def _conditional_reads(operation, contract, control):
    if not contract.read_resets:
        return {}
    operands = scalar_operands(operation)
    require(
        len(operands) == 1 and isinstance(operands[0].type, IndexType),
        "native reset needs an SSA ordinal",
    )
    ordinal = control.environment.expression(operands[0])
    loops = control.frames[operation].loops()
    stage = operation.parent_op()
    while stage is not None and not isinstance(stage, PipelineStageOp):
        stage = stage.parent_op()
    if stage is not None:
        loop = stage.parent_op()
        require(
            operands[0] is stage.body.block.args[0]
            and any(item.operation is loop for item in loops),
            "native reset needs its actual pipeline stage ordinal",
        )
    else:
        # Terminal SCF retains the actual ordinal operand and clock origin;
        # a nested serial loop must never replace its pipeline's reset clock.
        clocks = [
            item
            for item in loops
            if isinstance(item.operation, scf.ForOp)
            and "npu.pipeline_clock" in item.operation.attributes
        ]
        matching = [item for item in clocks if item.ordinal.variables & ordinal.variables]
        require(
            clocks and (matching or ordinal.constant is not None),
            "native reset needs its actual pipeline stage ordinal",
        )
        loop = (matching or clocks)[-1].operation
    result = {}
    for reset in contract.read_resets:
        value = (
            ordinal if reset.period is None else expression("rem", ordinal, literal(reset.period))
        )
        result[reset.operand] = ConditionalRead(
            loop,
            ordinal,
            Comparison(value, "!=", literal(0)),
            Comparison(value, "==", literal(0)),
        )
    return result


def _check_mask_program(operation, refs, params):
    """Consume the same lane-predicate facts the emitter will replay."""
    attributes = operation.attributes
    if "nodes" not in attributes:
        return
    require(
        {"shape", "node_shapes", "nodes"} <= set(attributes),
        "masked native lost part of its lane-predicate program",
    )
    program = Program.decode({name: attributes[name] for name in ("shape", "node_shapes", "nodes")})
    require(
        program.lanes == program.lanes // 64 * 64,
        "lane predicate must cover whole 64-lane blocks",
    )
    chunks = program.lanes // 64
    require(
        operation.native.data in {"mask_bits"} or operation.native.data.startswith("mask_"),
        "lane-predicate program belongs to another native",
    )
    word = attributes.get("mask_word")
    require(
        word is not None and word.data == word_strategy(program),
        "masked native word recipe differs from the admitted predicate program",
    )
    if operation.native.data == "mask_bits":
        require(
            attributes.get("mask_iteration") is None,
            "predicate materialization has no element iteration recipe",
        )
        require(
            params == (chunks,) and refs[0].geometry.total_nbytes == chunks * 32,
            "predicate materialization must write exactly its packed blocks",
        )
        require(
            attributes.get("mask_repeat") is None,
            "predicate materialization emits no repeated native calls",
        )
    else:
        require(
            refs[-1].geometry.total_nbytes == chunks * 256,
            "masked native destination must match its predicate lane domain",
        )
        if operation.native.data != "mask_fill":
            require(
                all(ref.geometry.total_nbytes == chunks * 256 for ref in refs),
                "masked native operand must match its predicate lane domain",
            )
        repeat = attributes.get("mask_repeat")
        require(
            repeat is not None and repeat.data == MASK_REPEAT_POLICY,
            "masked native needs the admitted full-run repeat policy",
        )
        iteration = attributes.get("mask_iteration")
        require(
            iteration is not None
            and iteration.data in ITERATION_VALUES
            and iteration.data == iteration_strategy(program),
            "masked native iteration recipe differs from the admitted predicate program",
        )


def _check_mask_layout(operation):
    """Packed predicate byte contracts selected before memory/sync analysis."""
    native, attributes = operation.native.data, operation.attributes
    expected = (
        "packed32_zeroed"
        if native.startswith("compare_")
        else "packed32_block_read"
        if native in {"select_masked", "select_masked_update"}
        else None
    )
    recorded = attributes.get("mask_layout")
    require(
        (expected is None) == (recorded is None)
        and (recorded is None or recorded.data == expected),
        "packed predicate native lost its selected byte-layout recipe",
    )


def collect_native_memory(module: ModuleOp, *, control: ControlFacts | None = None):
    """Consume one owned module and retain original operations and roots.

    Supplying already-owned control facts avoids another control traversal.
    Caller ownership is local to this invocation, never an external verified
    marker or a cache keyed by source/module text.
    """
    control = build_control_facts(module) if control is None else control
    require(control.module is module, "control facts belong to another physical module")
    decoder = StructuredAddresses(control.environment)
    templates, addresses = [], {}
    scalar_values = {}
    scalar_positions, scalar_ancestors = {}, {}
    for operation in control.frames:
        if not isinstance(operation, TaskOp):
            continue
        frame = control.frames[operation]
        require(frame.participants, "mixed native lacks an actual execution domain")
        current, stage = frame, 0
        while current is not None:
            if isinstance(current.operation, PipelineStageOp):
                stage = current.operation.offset.data
                break
            if current.loop is not None:
                break
            current = current.parent
        require(operation.stage.data == stage, "native stage disagrees with its actual loop scope")
        atoms = scalar_operands(operation)
        if operation.native.data in {"mad_first", "mad_cycle"}:
            require(
                len(atoms) == 1 and isinstance(atoms[0].type, IndexType),
                "native reset needs an explicit index ordinal after its references",
            )
        scalar_source = None
        if operation.native.data in {"scale_value", "sub_value", "scalar_store"}:
            require(
                len(atoms) == 1 and atoms[0] in scalar_values,
                "native scalar operand requires a preceding actual Scalar load",
            )
            scalar_source = scalar_values[atoms[0]]
            _dominates(atoms[0], operation, scalar_positions, scalar_ancestors)
        if scalar_source is None:
            require(
                all(isinstance(value.type, IndexType) for value in atoms),
                "native scalar/predicate operands must retain index types",
            )
        operands = tuple(operation.refs[: _operand_count(operation)])
        require(
            all(isinstance(value.type, RefType) for value in operands),
            "native memory operands must retain reference types",
        )
        refs = tuple(decoder.decode(value) for value in operands)
        prefix_indices = (
            (0,)
            if operation.native.data == "nd2nz_rows"
            else (0, 1)
            if operation.native.data in {"copy_rows_prefix", "nd2nz_overlay"}
            else ()
        )
        require(
            tuple(i for i, ref in enumerate(refs) if ref.valid_rows is not None) == prefix_indices,
            "runtime row-prefix addresses require their matching native instruction",
        )
        if prefix_indices:
            require(len(atoms) == 1, "runtime row copy needs one actual row count")
            count = control.environment.expression(atoms[0])
            require(
                all(refs[i].valid_rows == count for i in prefix_indices),
                "runtime row address and instruction counts disagree",
            )
        addresses.update(zip(operands, refs, strict=True))
        params = tuple(value.data for value in operation.params)
        geometries = tuple(ref.geometry for ref in refs)
        pipe, modes = native.native_contract(
            operation.native.data,
            geometries,
            params,
            disjoint=lambda refs=refs: len(refs) == 2 and decoder.disjoint(*refs),
        )
        engine = Engine(pipe)
        require(
            all(supports_engine(actor, engine) for actor in frame.participants),
            "native engine belongs to another actor participant",
        )
        for ref in refs:
            bank = ref.geometry.bank
            require(
                bank == "GM"
                or all(
                    actor is not Actor.AIC if bank == "UB" else actor is Actor.AIC
                    for actor in frame.participants
                ),
                "local storage bank belongs to another actor participant",
            )
        require(operation.scalar.type == f32, "native Vector scalar literals must be f32")
        round_f32(
            operation.scalar.value.data,
            allow_infinity=operation.native.data in {"fill", "mask_fill"},
        )
        require(
            (operation.value is not None)
            == (operation.native.data in {"scalar_load", "index_load"}),
            "only synchronous Scalar load may return a numerical SSA value",
        )
        if operation.value is not None:
            if operation.native.data == "index_load":
                require(
                    len(atoms) == 2 and isinstance(operation.value.type, IndexType),
                    "index load needs its two bounds and an index result",
                )
                require(
                    refs[0].geometry.bank == "UB"
                    or refs[0].root.owner.access is not None
                    and refs[0].root.owner.access.data == "read",
                    "GM index metadata must be immutable",
                )
            else:
                require(operation.value.type == f32, "floating Scalar load must produce f32")
                scalar_values[operation.value] = operation
        _check_mask_program(operation, refs, params)
        _check_mask_layout(operation)
        spec = native.NativeSpec(operation.native.data, engine, modes, params, geometries)
        require(
            not spec.registers or frame.participants == (Actor.AIC,),
            "Load3D and FP32 register state belongs to the AIC",
            phase="structured-memory",
            location=str(operation.location),
        )
        require(
            spec.packet is not None
            and spec.packet is audited_packet(spec.native)
            and spec.packet.closed_mask,
            "native requires an audited closed instruction packet",
            phase="structured-memory",
            location=str(operation.location),
        )
        conditions = _conditional_reads(operation, spec, control)
        accesses = tuple(
            MemoryAccess(operand, index, ref, AccessMode(mode), conditions.get(index))
            for index, (operand, ref, mode) in enumerate(zip(operands, refs, modes, strict=True))
        )
        for access in accesses:
            buffer = access.address.root.owner
            require(
                buffer.access is None or buffer.access.data != "read" or not access.mode.writes,
                "native writes a read-only tensor",
                phase="structured-memory",
                location=str(operation.location),
            )
            require(
                buffer.write_disjoint is None
                or buffer.write_disjoint.data == 1
                and buffer.external.data == 1
                and buffer.access is not None
                and buffer.access.data in {"write", "read_write"},
                "write_disjoint requires a writable external tensor",
                phase="structured-memory",
                location=str(operation.location),
            )
        templates.append(MemoryTemplate(operation, frame, spec, accesses, scalar_source))
    by_operation = {template.operation: template for template in templates}
    groups, grouped = [], set()
    for operation in control.frames:
        if not isinstance(operation, ReadyGroupOp):
            continue
        operation.verify()
        members = tuple(by_operation[value.owner] for value in operation.members)
        require(
            len({member.spec.engine for member in members}) == 1
            and all(member.frame is members[0].frame for member in members),
            "CopyGroup needs one actual source queue and activation; split the group",
            phase="structured-memory",
            location=str(operation.location),
        )
        require(
            not any(member.operation in grouped for member in members),
            "CopyGroup native cannot belong to multiple groups",
        )
        grouped.update(member.operation for member in members)
        groups.append(members)
    return NativeMemoryFacts(
        module,
        control,
        tuple(templates),
        MappingProxyType(addresses),
        len(templates),
        tuple(groups),
    )
