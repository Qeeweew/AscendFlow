"""Actual native accesses and physical cuts of one decoded private loop.

References come from the existing physical view decoder, never finite task
instances or a completion planner. Operation and allocation identities stay
attached to every access. Descriptors must be rebuilt after IR mutation.

A task rank is relative to that native's logical generation g: (g + stage,
lexical). A caller walking physical step j must explicitly substitute
g = j - consumer.stage before comparing a producer at g - distance.
An end cut is an anchor for a real capture, not a cumulative retirement fact.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING

from xdsl.dialects.builtin import f32
from xdsl.ir import Operation, SSAValue

from ..diagnostics import ErrorKind, require
from ..dialects.physical import BufferOp, FixConfigOp, KernelOp, Load3DConfigOp, StreamOp, TaskOp
from ..target.ascend910b import Actor, Engine, Milestone, supports_engine
from ..target.native import implicit_completion as native_implicit_completion
from ..target.native import native_contract
from ..target.packets import NativePacket, audited_packet
from ..target.scalars import round_f32
from .frontier import Clock, PrefixRank

if TYPE_CHECKING:
    from ..loop_model import Loop
    from ..model import Ref


class AccessMode(str, Enum):
    READ = "R"
    WRITE = "W"
    READ_WRITE = "RW"

    @property
    def reads(self):
        return "R" in self.value

    @property
    def writes(self):
        return "W" in self.value


@dataclass(frozen=True, slots=True)
class NativeContract:
    """One actual instruction's audited route, operands, and scalar literal."""

    native: str
    engine: Engine
    modes: tuple[str, ...]
    params: tuple[int, ...]
    scalar: float
    packet: NativePacket | None = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "packet", audited_packet(self.native))

    @property
    def pipe(self):
        return self.engine.value

    @property
    def implicit_completion(self):
        return native_implicit_completion(self)


def native_task_contract(operation: TaskOp, refs: tuple[Ref, ...]) -> NativeContract:
    """Shared single-task contract for legacy decoding and structured effects."""
    require(isinstance(operation, TaskOp), "native effects require an actual TaskOp")
    params = tuple(value.data for value in operation.params)
    pipe, modes = native_contract(operation.native.data, refs, params)
    require(operation.scalar.type == f32, "native Vector scalar literals must be f32")
    scalar = round_f32(operation.scalar.value.data, allow_infinity=operation.native.data == "fill")
    return NativeContract(operation.native.data, Engine(pipe), modes, params, scalar)


@dataclass(frozen=True, slots=True)
class NativeAccess:
    operand_index: int
    operand: SSAValue
    root: SSAValue
    ref: Ref
    mode: AccessMode

    def __post_init__(self):
        require(
            type(self.operand_index) is int and self.operand_index >= 0,
            "native access needs its actual operand index",
        )
        require(
            isinstance(self.operand, SSAValue) and isinstance(self.root, SSAValue),
            "native access needs actual operand and allocation SSA identities",
        )
        require(
            isinstance(self.root.owner, BufferOp) and self.root is self.root.owner.ref,
            "native access root must be its actual allocation",
        )
        require(isinstance(self.mode, AccessMode), "native access needs a typed memory mode")


class Register(str, Enum):
    ND_PARA = "ND_PARA"
    FMATRIX_A = "FMATRIX_A"
    FMATRIX_B = "FMATRIX_B"
    L13D_REPEAT = "L13D_REPEAT"
    L13D_PADDING = "L13D_PADDING"
    L13D_SIZE = "L13D_SIZE"
    VECTOR_MASK_MODE = "VECTOR_MASK_MODE"
    VECTOR_MASK = "VECTOR_MASK"


@dataclass(frozen=True, slots=True)
class NativeRegisterAccess:
    """A whole named SPR resource, scoped to one actor, never tensor bytes.

    A setup overwrite belongs to the actual native recipe. Invariant reads
    retain the real initializer as a witness; they do not invent its readiness.
    """

    operation: TaskOp
    actor: Actor
    engine: Engine
    resource: Register
    mode: AccessMode
    configuration: Operation | None = None


@dataclass(frozen=True, slots=True)
class NativeCut:
    operation: TaskOp
    actor: Actor
    engine: Engine | None
    rank: PrefixRank
    milestone: Milestone

    def __post_init__(self):
        require(isinstance(self.operation, TaskOp), "native cut needs an actual TaskOp")
        require(
            isinstance(self.actor, Actor) and isinstance(self.rank, PrefixRank),
            "native cut needs a declared actor and clock rank",
        )
        require(isinstance(self.milestone, Milestone), "native cut needs a typed milestone")
        require(
            self.milestone in {Milestone.ISSUE, Milestone.ADMISSION, Milestone.NATIVE_END},
            "a native end anchor cannot establish a retirement capture",
            kind=ErrorKind.UNSUPPORTED,
            phase="native effects",
        )
        require(
            self.engine is None
            if self.milestone is Milestone.ISSUE
            else supports_engine(self.actor, self.engine),
            "native cut has an invalid engine",
        )

    @property
    def clock(self):
        return self.rank.clock


@dataclass(frozen=True, slots=True)
class NativeTask:
    operation: TaskOp
    actor: Actor
    engine: Engine
    clock: Clock
    coordinate: SSAValue
    native: str
    params: tuple[int, ...]
    stage: int
    lexical: int
    issue_order: int
    accesses: tuple[NativeAccess, ...]
    registers: tuple[NativeRegisterAccess, ...]
    implicit_completion: bool
    last: bool = False
    scalar_source: TaskOp | None = None
    packet: NativePacket | None = None

    @property
    def rank(self):
        """Affine physical rank relative to this native's logical generation."""
        return PrefixRank(self.clock, self.stage, self.lexical)

    def cut(self, milestone: Milestone, *, generation_offset=0):
        require(type(generation_offset) is int, "native generation offset must be an integer")
        return NativeCut(
            self.operation,
            self.actor,
            None if milestone is Milestone.ISSUE else self.engine,
            PrefixRank(self.clock, self.stage + generation_offset, self.lexical),
            milestone,
        )

    @property
    def consumer_cut(self):
        return self.cut(Milestone.ADMISSION)

    @property
    def capture_cut(self):
        return self.cut(Milestone.NATIVE_END)

    def modes_for(self, generation: int):
        """Read-before-write modes retain the native accumulator clear policy."""
        require(type(generation) is int and generation >= 0, "invalid native generation")
        modes = tuple(access.mode for access in self.accesses)
        if self.native == "mad_first":
            return (*modes[:2], AccessMode.WRITE if generation == 0 else AccessMode.READ_WRITE)
        if self.native == "mad_cycle":
            return (
                *modes[:2],
                AccessMode.WRITE if generation % self.params[3] == 0 else AccessMode.READ_WRITE,
            )
        return modes


def _registers(operation, actor, engine, configurations):
    native = operation.native.data
    if native in {"fix", "fix16"}:
        configuration = configurations.get(FixConfigOp)
        require(configuration is not None, "FIX needs its actual ND_PARA initializer")
        return (
            NativeRegisterAccess(
                operation, actor, engine, Register.ND_PARA, AccessMode.READ, configuration
            ),
        )
    if native in {"load3d_a", "load3d_b_nn"}:
        configuration = configurations.get(Load3DConfigOp)
        require(configuration is not None, "Load3D needs its actual SPR initializer")
        resource = Register.FMATRIX_A if native == "load3d_a" else Register.FMATRIX_B
        width = configuration.a_width if native == "load3d_a" else configuration.b_width
        require(
            width.data == operation.params.data[0].data, "Load3D has a stale FMATRIX configuration"
        )
        return tuple(
            NativeRegisterAccess(operation, actor, engine, resource, AccessMode.READ, configuration)
            for resource in (
                resource,
                Register.L13D_REPEAT,
                Register.L13D_PADDING,
                Register.L13D_SIZE,
            )
        )
    return ()


@dataclass(frozen=True, slots=True)
class DecodedNativeTask:
    """Owned native/access invariant established once during physical admission."""

    operation: TaskOp
    contract: NativeContract
    stage: int
    issue_order: int
    accesses: tuple[NativeAccess, ...]
    registers: tuple[NativeRegisterAccess, ...]
    last: bool
    scalar_source: TaskOp | None
    lexical: int = -1

    @property
    def packet(self):
        return self.contract.packet


@dataclass(frozen=True, slots=True)
class DecodedNativeEffects:
    kernel: KernelOp
    stream: StreamOp
    actor: Actor
    coordinate: SSAValue
    tasks: tuple[DecodedNativeTask, ...]


def decode_native_task(
    operation: TaskOp,
    contract: NativeContract,
    refs: tuple[Ref, ...],
    operands: tuple[SSAValue, ...],
    roots: dict[SSAValue, SSAValue],
    *,
    actor: Actor,
    issue_order: int,
    configurations: dict[type[Operation], Operation],
    last: bool,
) -> DecodedNativeTask:
    """Attach the decoder's already resolved roots and one native contract.

    The physical decoder owns domination, view geometry, bounds, and native
    validation. This function neither walks view chains nor revalidates a
    native. Registers are kept separate from tensor-byte accesses.
    """
    require(supports_engine(actor, contract.engine), "native engine belongs to another actor")
    accesses = tuple(
        NativeAccess(number, operand, roots[operand], ref, AccessMode(mode))
        for number, (operand, ref, mode) in enumerate(
            zip(operands, refs, contract.modes, strict=True)
        )
    )
    scalar_source = (
        operation.refs[-1].owner if contract.native in {"scale_value", "sub_value"} else None
    )
    require(
        scalar_source is None or isinstance(scalar_source, TaskOp),
        "native scalar capture lost its actual source operation",
    )
    return DecodedNativeTask(
        operation,
        contract,
        operation.stage.data,
        issue_order,
        accesses,
        _registers(operation, actor, contract.engine, configurations),
        last,
        scalar_source,
    )


def own_native_effects(kernel, stream, actor, tasks) -> DecodedNativeEffects:
    """Finish admission with the actual physical issue order, once per body."""
    tasks = tuple(tasks)
    ordered = sorted(range(len(tasks)), key=lambda i: (tasks[i].issue_order, i))
    lexical = {index: rank for rank, index in enumerate(ordered)}
    return DecodedNativeEffects(
        kernel,
        stream,
        actor,
        stream.body.block.args[0],
        tuple(replace(task, lexical=lexical[index]) for index, task in enumerate(tasks)),
    )


def extract_native_tasks(loop: Loop, *, clock: Clock) -> tuple[NativeTask, ...]:
    """Bind the admitted private loop's owned descriptors to a declared clock.

    No views, contracts, or finite instances are rebuilt here. An external
    physical module enters the decoder once; an owned IR transformation must
    replace invalidated effects rather than reuse them after mutation.
    Mixed actors, invocation frames and nested regions are unsupported by this
    first adapter and cannot silently select another analysis.
    """
    require(isinstance(clock, Clock), "native effects need an explicitly declared clock")
    kernel = getattr(loop, "op", None)
    actors = getattr(loop, "actors", {})
    require(
        isinstance(kernel, KernelOp)
        and len(actors) == 1
        and getattr(loop, "invocation", None) is None
        and getattr(loop, "region", None) is None
        and getattr(loop, "cyclic", None) is None,
        "native effects currently require one decoded private local loop",
        kind=ErrorKind.UNSUPPORTED,
        phase="native effects",
    )
    data = next(iter(actors.values()))
    owned = data.owned_effects
    require(
        isinstance(owned, DecodedNativeEffects)
        and owned.kernel is kernel
        and owned.stream is data.stream,
        "native effects need the decoder's actual owned local descriptors",
        kind=ErrorKind.UNSUPPORTED,
        phase="native effects",
    )
    return tuple(
        NativeTask(
            task.operation,
            owned.actor,
            task.contract.engine,
            clock,
            owned.coordinate,
            task.contract.native,
            task.contract.params,
            task.stage,
            task.lexical,
            task.issue_order,
            task.accesses,
            task.registers,
            task.contract.implicit_completion,
            task.last,
            task.scalar_source,
            packet=task.packet,
        )
        for task in owned.tasks
    )
