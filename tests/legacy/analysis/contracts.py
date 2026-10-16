"""Historical M2 direct conditional memory contracts used by the test oracle.

A real source retirement capture carries facts established at its native's
admission. This index retains those admission obligations as conditions, not
as an acquired Frontier. A direct witness is valid only after its entire bound
admission contract and the actual capture/acquire protocol are established.
"""

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from math import gcd
from types import MappingProxyType

from xdsl.ir import SSAValue

from xdsl_npu.analysis.hazards import Activation, MemoryWitness, NativeHazard
from xdsl_npu.analysis.native import NativeCut, NativeTask
from xdsl_npu.analysis.regions import Output
from xdsl_npu.diagnostics import require
from xdsl_npu.dialects.physical import TaskOp
from xdsl_npu.target.ascend910b import Milestone


@dataclass(slots=True)
class ContractWork:
    tasks_indexed: int = 0
    accesses_indexed: int = 0
    hazards_indexed: int = 0
    admission_lookups: int = 0
    capture_lookups: int = 0
    witness_queries: int = 0
    candidate_visits: int = 0
    activation_checks: int = 0
    residue_visits: int = 0
    range_checks: int = 0


@dataclass(frozen=True, slots=True)
class AdmissionContract:
    task: NativeTask
    obligations: tuple[NativeHazard, ...]


@dataclass(frozen=True, slots=True)
class BoundAdmission:
    """All obligations, with this native at g + generation_offset.

    The original obligations keep their own consumer-relative activations.
    Binding does not copy them or resolve any requirement recursively.
    """

    contract: AdmissionContract
    generation_offset: int


@dataclass(frozen=True, slots=True)
class ConditionalCapture:
    admission: AdmissionContract
    passthrough: Output | None = None

    @property
    def producer(self) -> NativeTask:
        return self.admission.task

    @property
    def anchor(self) -> NativeCut:
        """Only the native-end anchor; this object is not a real Capture."""
        return self.producer.capture_cut


class WitnessKind(str, Enum):
    OWN_RETIREMENT = "own-retirement"
    ADMISSION_PREMISE = "admission-premise"


@dataclass(frozen=True, slots=True)
class CaptureWitness:
    """A conditional direct proof, never an already acquired completion fact.

    ``via`` binds the proposed source capture to the target's actual consumer.
    All ``assumptions`` remain outstanding, even when only ``premise`` supplies
    the particular fact needed by ``target``. A real selected retirement
    capture, matching acquire, and activation proof remain required as well.
    """

    kind: WitnessKind
    capture: ConditionalCapture
    via: NativeHazard
    target: NativeHazard
    assumptions: BoundAdmission
    premise: NativeHazard | None = None

    @property
    def milestone(self) -> Milestone:
        return Milestone.RETIREMENT_CAPTURE


def _activation_implies(
    required: Activation,
    available: Activation,
    offset: int,
    work: ContractWork | None,
) -> bool:
    """required(g,N) implies available(g+offset,N), without iterating N.

    Residue implication checks complete gcd cosets, not a product/LCM of
    periods. Only the existing first/last endpoint vocabulary is interpreted.
    """
    if work is not None:
        work.activation_checks += 1
    if required.source_last and required.distance:
        return False  # The requested pair never exists; manufacture no witness.
    if offset > 0 or required.distance + offset < available.distance:
        return False
    last = required.consumer_last or required.source_last
    if available.consumer_last and not (last and offset == 0):
        return False
    if available.source_last and not (last and offset - available.distance == 0):
        return False
    divisor = gcd(required.period, available.period)
    counts = {}
    for residue in available.residues:
        if work is not None:
            work.residue_visits += 1
        key = residue % divisor
        counts[key] = counts.get(key, 0) + 1
    needed = available.period // divisor
    for residue in required.residues:
        if work is not None:
            work.residue_visits += 1
        if counts.get((residue + offset) % divisor, 0) != needed:
            return False
    return True


def _contains_memory(
    available: MemoryWitness,
    required: MemoryWitness,
    offset: int,
    work: ContractWork | None,
) -> bool:
    if work is not None:
        work.range_checks += 1
    return (
        available.root is required.root
        and available.frame_stride == required.frame_stride
        # Frame-relative rebinding needs its own range contract. Do not infer
        # it from an equal storage spelling or discard the translation.
        and (not available.frame_stride or offset == 0)
        and available.start <= required.start < required.stop <= available.stop
    )


@dataclass(frozen=True, slots=True)
class MemoryContracts:
    _admissions: Mapping[TaskOp, AdmissionContract]
    _inputs: Mapping[tuple[TaskOp, TaskOp], tuple[NativeHazard, ...]]
    _roots: Mapping[TaskOp, frozenset[SSAValue]]
    access_count: int
    hazard_count: int

    def _owned(self, task: NativeTask) -> AdmissionContract:
        require(isinstance(task, NativeTask), "memory contract needs an actual indexed NativeTask")
        contract = self._admissions.get(task.operation)
        require(
            contract is not None and contract.task is task,
            "memory contract uses a different native task snapshot",
        )
        return contract

    def _hazard(self, hazard: NativeHazard) -> None:
        require(isinstance(hazard, NativeHazard), "memory contract needs an actual native hazard")
        self._owned(hazard.producer)
        self._owned(hazard.consumer)
        require(
            hazard.memory.root in self._roots[hazard.producer.operation]
            and hazard.memory.root in self._roots[hazard.consumer.operation],
            "memory contract hazard changed its actual allocation root",
        )

    def admission(self, task: NativeTask, *, work: ContractWork | None = None) -> AdmissionContract:
        if work is not None:
            work.admission_lookups += 1
        return self._owned(task)

    def capture(
        self,
        task: NativeTask,
        *,
        passthrough: Output | None = None,
        work: ContractWork | None = None,
    ) -> ConditionalCapture:
        if work is not None:
            work.capture_lookups += 1
        require(
            passthrough is None or isinstance(passthrough, Output),
            "capture passthrough must retain an actual Output expression",
        )
        return ConditionalCapture(self._owned(task), passthrough)

    def direct_witness(
        self,
        via: NativeHazard,
        target: NativeHazard,
        *,
        passthrough: Output | None = None,
        work: ContractWork | None = None,
    ) -> CaptureWitness | None:
        """Inspect only this capture's bucket for the target's actual producer.

        The caller still selects and proves the real source Set/fence and its
        consumer acquire. No other capture, history, or completion graph is
        searched. Different native identities never match just by engine/rank.
        """
        if work is not None:
            work.witness_queries += 1
        self._hazard(via)
        self._hazard(target)
        if via.consumer is not target.consumer:
            return None
        if not _activation_implies(target.activation, via.activation, 0, work):
            return None
        capture = self.capture(via.producer, passthrough=passthrough, work=work)
        offset = -via.activation.distance
        assumptions = BoundAdmission(capture.admission, offset)
        if (
            via.producer is target.producer
            and via.activation.distance == target.activation.distance
        ):
            return CaptureWitness(WitnessKind.OWN_RETIREMENT, capture, via, target, assumptions)
        if target.reader_prefix or via.reader_prefix:
            return None  # An individual admission premise cannot retire a reader family.
        if not _contains_memory(via.memory, target.memory, 0, work):
            return None
        for premise in self._inputs.get((via.producer.operation, target.producer.operation), ()):
            if work is not None:
                work.candidate_visits += 1
            if premise.reader_prefix:
                continue
            if target.activation.distance != via.activation.distance + premise.activation.distance:
                continue
            if not _contains_memory(premise.memory, target.memory, offset, work):
                continue
            if not _activation_implies(target.activation, premise.activation, offset, work):
                continue
            return CaptureWitness(
                WitnessKind.ADMISSION_PREMISE, capture, via, target, assumptions, premise
            )
        return None

    @property
    def statistics(self) -> dict[str, int]:
        return {
            "native_tasks": len(self._admissions),
            "native_accesses": self.access_count,
            "admission_obligations": self.hazard_count,
            "producer_buckets": len(self._inputs),
            "recursive_queries": 0,
        }


def build_memory_contracts(
    tasks: tuple[NativeTask, ...],
    hazards: tuple[NativeHazard, ...],
    *,
    work: ContractWork | None = None,
) -> MemoryContracts:
    """Index existing effects and obligations in O(T + A + H), without decoding."""
    native, roots = {}, {}
    accesses = 0
    clock = None
    for task in tasks:
        require(isinstance(task, NativeTask), "memory contracts need actual native tasks")
        if clock is None:
            clock = task.clock
        require(task.clock is clock, "memory contracts need one explicitly bound clock")
        require(task.operation not in native, "memory contracts repeat a native operation")
        native[task.operation] = task
        roots[task.operation] = frozenset(access.root for access in task.accesses)
        accesses += len(task.accesses)
        if work is not None:
            work.tasks_indexed += 1
            work.accesses_indexed += len(task.accesses)
    obligations, inputs = defaultdict(list), defaultdict(list)
    for hazard in hazards:
        require(isinstance(hazard, NativeHazard), "memory contracts need actual native hazards")
        require(
            isinstance(hazard.producer, NativeTask) and isinstance(hazard.consumer, NativeTask),
            "memory hazard needs actual native endpoints",
        )
        require(
            native.get(hazard.producer.operation) is hazard.producer
            and native.get(hazard.consumer.operation) is hazard.consumer,
            "memory hazard endpoints are not the indexed native task snapshots",
        )
        require(
            hazard.memory.root in roots[hazard.producer.operation]
            and hazard.memory.root in roots[hazard.consumer.operation],
            "memory hazard does not retain the actual allocation root",
        )
        obligations[hazard.consumer.operation].append(hazard)
        inputs[hazard.consumer.operation, hazard.producer.operation].append(hazard)
        if work is not None:
            work.hazards_indexed += 1
    admissions = {
        op: AdmissionContract(task, tuple(obligations[op])) for op, task in native.items()
    }
    return MemoryContracts(
        MappingProxyType(admissions),
        MappingProxyType({key: tuple(bucket) for key, bucket in inputs.items()}),
        MappingProxyType(roots),
        accesses,
        len(hazards),
    )
