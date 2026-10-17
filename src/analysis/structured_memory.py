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
from types import MappingProxyType

from xdsl.dialects import scf
from xdsl.dialects.builtin import IndexType, ModuleOp, f32
from xdsl.ir import SSAValue

from ..diagnostics import require
from ..dialects.physical import TaskOp
from ..dialects.structured import PipelineStageOp
from ..dialects.types import RefType
from ..scalar import expression, literal
from ..target import native
from ..target.ascend910b import Actor, Engine, supports_engine
from ..target.packets import audited_packet
from ..target.scalars import round_f32
from .native import AccessMode, ConditionalRead, NativeContract
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
    contract: NativeContract
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


def _conditional_reads(operation, contract, control):
    if not contract.read_resets:
        return {}
    require(isinstance(operation.refs[-1].type, IndexType), "native reset needs an SSA ordinal")
    ordinal = control.environment.expression(operation.refs[-1])
    loops = control.frames[operation].loops()
    stage = operation.parent_op()
    while stage is not None and not isinstance(stage, PipelineStageOp):
        stage = stage.parent_op()
    if stage is not None:
        loop = stage.parent_op()
        require(
            operation.refs[-1] is stage.body.block.args[0]
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
        operands = tuple(operation.refs)
        if operation.native.data in {"mad_first", "mad_cycle"}:
            require(
                len(operands) == 4 and isinstance(operands[-1].type, IndexType),
                "native reset needs three references and an explicit index ordinal",
            )
            operands = operands[:-1]
        scalar_source = None
        if operation.native.data in {"scale_value", "sub_value"}:
            require(
                len(operands) == 3 and operands[-1] in scalar_values,
                "native scalar operand requires a preceding actual Scalar load",
            )
            scalar_source = scalar_values[operands[-1]]
            require(
                scalar_source.parent is operation.parent,
                "Scalar capture needs the same physical scope; delayed values need storage",
            )
            operands = operands[:-1]
        require(
            all(isinstance(value.type, RefType) for value in operands),
            "native memory operands must retain reference types",
        )
        refs = tuple(decoder.decode(value) for value in operands)
        addresses.update(zip(operands, refs, strict=True))
        params = tuple(value.data for value in operation.params)
        pipe, modes = native.native_contract(
            operation.native.data,
            tuple(ref.geometry for ref in refs),
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
        scalar = round_f32(
            operation.scalar.value.data, allow_infinity=operation.native.data == "fill"
        )
        require(
            (operation.value is not None) == (operation.native.data == "scalar_load"),
            "only synchronous Scalar load may return a numerical SSA value",
        )
        if operation.value is not None:
            scalar_values[operation.value] = operation
        contract = NativeContract(operation.native.data, engine, modes, params, scalar)
        require(
            contract.packet is not None
            and contract.packet is audited_packet(contract.native)
            and contract.packet.closed_mask,
            "native requires an audited closed instruction packet",
            phase="structured-memory",
            location=str(operation.location),
        )
        conditions = _conditional_reads(operation, contract, control)
        accesses = tuple(
            MemoryAccess(operand, index, ref, AccessMode(mode), conditions.get(index))
            for index, (operand, ref, mode) in enumerate(zip(operands, refs, modes, strict=True))
        )
        templates.append(MemoryTemplate(operation, frame, contract, accesses, scalar_source))
    return NativeMemoryFacts(
        module, control, tuple(templates), MappingProxyType(addresses), len(templates)
    )
