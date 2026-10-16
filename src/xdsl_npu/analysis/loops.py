"""One-step preservation of explicit structured-loop frontier invariants.

This is a conditional induction contract, not a loop exit summary. The owner
must establish the candidate at the first executed iteration and account for
its exact activation domain. We check one body application against I(g+1),
expressed as g+offset+1 in the one explicitly declared loop clock. External
clock parameters keep their identity and rank; no clock relation is inferred.

There is no fixed-point search and no trip-count input. For P fixed boundary
ports, K declared prefix domains and H body requirements, stored proof data is
O(P*K+H). Work is one finite body application plus O(P*K+H*P) checks, in
addition to that application's explicit wiring cost. Guards must be one exact
declared partition. Different guard implications and clock bindings belong to
separate, explicit contracts.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from ..diagnostics import CompileError, ErrorKind, require
from .frontier import Clock, Frontier, FrontierWork, KnownPrefix, PrefixRank
from .regions import BoundaryLayout, BoundaryTransfer, Port, Requirement


@dataclass(frozen=True, slots=True)
class CarriedRequirement:
    port: Port
    prefix: KnownPrefix


class LoopInvariantError(CompileError):
    """An unproved candidate, retaining the original body and consumer cuts."""

    def __init__(self, message, body, *, missing_requirements=(), missing_carry=()):
        self.body = body
        self.missing_requirements = tuple(missing_requirements)
        self.missing_carry = tuple(missing_carry)
        witnesses = tuple(
            f"{item.consumer.name} requires {item.prefix}" for item in self.missing_requirements
        ) + tuple(f"{item.port} needs {item.prefix}" for item in self.missing_carry)
        super().__init__(
            message, kind=ErrorKind.UNSUPPORTED, phase="loop-invariant", witness=witnesses
        )


@dataclass(slots=True)
class LoopWork:
    body_applications: int = 0
    rebound_prefixes: int = 0
    preservation_checks: int = 0
    entry_checks: int = 0


def _bindings(layout, bindings):
    require(isinstance(layout, BoundaryLayout), "loop needs a declared boundary layout")
    require(
        isinstance(bindings, Mapping) and set(bindings) == layout.ports,
        "loop frontier bindings must cover exactly its boundary ports",
    )
    require(
        all(
            isinstance(state, Frontier) and state.layout == layout.frontier
            for state in bindings.values()
        ),
        "loop frontier layout mismatch",
    )
    return MappingProxyType(dict(bindings))


def _guard(prefix, guard):
    require(
        prefix.domain.partition.guard == guard,
        "loop invariant requires one exact guard partition; bind guards explicitly",
        kind=ErrorKind.UNSUPPORTED,
        phase="loop-invariant",
    )


def _missing(required, available, *, work=None, entry=False):
    missing = []
    for port, state in required.items():
        for domain, rank in state.known.items():
            if work is not None:
                if entry:
                    work.entry_checks += 1
                else:
                    work.preservation_checks += 1
            prefix = KnownPrefix(domain, rank)
            if not available[port].covers(prefix):
                missing.append(CarriedRequirement(port, prefix))
    return tuple(missing)


@dataclass(frozen=True, slots=True)
class LoopInvariant:
    """I(g) satisfies the body requirements and body(I(g)) implies I(g+1).

    ``check_entry`` discharges the initial-state precondition separately. The
    body and its requirements are original immutable contract objects. Native
    IR referenced by consumer witnesses remains owned by the compilation.
    ``after_one`` is only that single symbolic step, never the final exit.
    """

    body: BoundaryTransfer
    clock: Clock
    guard: str
    candidate: Mapping[Port, Frontier]
    next_candidate: Mapping[Port, Frontier]
    after_one: Mapping[Port, Frontier]

    @property
    def requirements(self) -> tuple[Requirement, ...]:
        return self.body.requirements

    def check_entry(self, inputs, *, work: LoopWork | None = None):
        """Check startup's already-bound facts; this does not insert a Wait."""
        inputs = _bindings(self.body.layout, inputs)
        missing = _missing(self.candidate, inputs, work=work, entry=True)
        if missing:
            raise LoopInvariantError(
                "loop initial state does not establish its candidate",
                self.body,
                missing_carry=missing,
            )
        return inputs

    @property
    def statistics(self):
        return {
            "ports": len(self.body.layout.ports),
            "frontier_capacity": self.body.layout.frontier.capacity,
            "candidate_prefixes": sum(len(state) for state in self.candidate.values()),
            "next_prefixes": sum(len(state) for state in self.next_candidate.values()),
            "step_prefixes": sum(len(state) for state in self.after_one.values()),
            "body_requirements": len(self.body.requirements),
            "body_applications": 1,
            "fixed_point_rounds": 0,
            "expanded_iterations": 0,
        }


def check_loop_invariant(
    body: BoundaryTransfer,
    clock: Clock,
    candidate: Mapping[Port, Frontier],
    *,
    guard: str = "all",
    work: LoopWork | None = None,
    frontier_work: FrontierWork | None = None,
) -> LoopInvariant:
    """Validate an explicit candidate once, without finding or strengthening it.

    At current iteration g, a candidate rank g+d is required at next entry as
    (g+1)+d. Only ranks in ``clock`` receive this +1 rebinding. Another clock's
    rank is an unchanged external parameter, compared only in its own domain.
    Empty/short execution does not get to assume this contract: the owner must
    establish the base case on the exact domain before consuming the result.
    """
    require(isinstance(body, BoundaryTransfer), "loop induction needs an actual body transfer")
    require(isinstance(clock, Clock), "loop induction needs one declared clock")
    require(type(guard) is str and bool(guard), "loop induction needs an exact guard partition")
    require(
        not body.layout.frontier.domains
        or any(domain.clock is clock for domain in body.layout.frontier.domains),
        "loop clock is not the boundary's declared clock; bind clocks explicitly",
        kind=ErrorKind.UNSUPPORTED,
        phase="loop-invariant",
    )
    candidate = _bindings(body.layout, candidate)
    for state in candidate.values():
        for domain, rank in state.known.items():
            _guard(KnownPrefix(domain, rank), guard)
    for output in body.outputs.values():
        for domain, rank in output.guarantees.known.items():
            _guard(KnownPrefix(domain, rank), guard)
    for requirement in body.requirements:
        _guard(requirement.prefix, guard)

    missing = body.missing(candidate)
    if missing:
        raise LoopInvariantError(
            "loop candidate does not satisfy actual body consumers",
            body,
            missing_requirements=missing,
        )
    after = body._apply_established(candidate, work=frontier_work)
    if work is not None:
        work.body_applications += 1
    following = {}
    for port, state in candidate.items():
        prefixes = []
        for domain, rank in state.known.items():
            if domain.clock == clock:
                rank = PrefixRank(clock, rank.offset + 1, rank.lexical)
                if work is not None:
                    work.rebound_prefixes += 1
            prefixes.append(KnownPrefix(domain, rank))
        following[port] = Frontier(body.layout.frontier, prefixes)
    following = MappingProxyType(following)
    missing = _missing(following, after, work=work)
    if missing:
        raise LoopInvariantError(
            "loop body does not establish next-generation carried prefixes",
            body,
            missing_carry=missing,
        )
    return LoopInvariant(body, clock, guard, candidate, following, after)


def zero_iterations(layout: BoundaryLayout, inputs: Mapping[Port, Frontier]):
    """Zero repetitions are identity, independent of any body preconditions.

    Independent resource-lifetime seed/cleanup is not part of this zero body
    transfer. No native completion, body guarantee or token is manufactured.
    """
    return _bindings(layout, inputs)
