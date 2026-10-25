"""Actor-private MAD/FIX unitflag words on exact, complete L0C planes.

Unitflag orders the participating L0C accesses, not native/pipe completion.
The admitted word alternates an accumulation's final update and its one FIX
consume in the same dispatch stage. Intermediate MADs only check the flag.
No notification, global barrier, extra event ID or speculative SSA is created.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..diagnostics import ErrorKind, require
from ..scalar import literal
from ..target.ascend910b import Engine
from .hazards import HazardKind
from .integer_facts import IntegerFacts
from .smt import all_of
from .structured_memory import scalar_operands
from .structured_schedule import Comparison

if TYPE_CHECKING:
    from .structured_dispatch import StructuredDispatch
    from .structured_memory import MemoryTemplate, NativeMemoryFacts
    from .structured_protocols import RequirementBinding


def _require(condition, message):
    require(condition, message, kind=ErrorKind.UNSUPPORTED, phase="unitflag")


def _context(template):
    return (
        *(
            test
            for value, low, high in template.frame.domains()
            for test in (Comparison(value, ">=", low), Comparison(value, "<=", high))
        ),
        *template.frame.predicates(),
    )


def _plane(template):
    return next(a.address for a in template.accesses if a.address.geometry.bank == "L0C")


def _same_plane(left, right, arithmetic, assumptions=()):
    return (
        left.root is right.root
        and left.geometry.contiguous
        and right.geometry.contiguous
        and left.geometry.total_nbytes == right.geometry.total_nbytes
        and arithmetic.queries.equivalent(left.offset, right.offset, assumptions).proved
    )


@dataclass(frozen=True, slots=True)
class UnitFlagPair:
    publisher: MemoryTemplate
    consumer: MemoryTemplate
    checks: tuple[MemoryTemplate, ...]
    publication: tuple[Comparison, ...]
    consumption: tuple[Comparison, ...]


@dataclass(frozen=True, slots=True)
class UnitFlagCoverage:
    binding: RequirementBinding
    pair: UnitFlagPair
    following: UnitFlagPair | None = None


@dataclass(frozen=True)
class UnitFlagProtocols:
    memory: NativeMemoryFacts
    dispatch: StructuredDispatch
    pairs: tuple[UnitFlagPair, ...]

    def cover(self, binding):
        need = binding.requirement
        memory = self.memory
        source = memory.by_operation.get(binding.source.operation)
        consumer = memory.by_operation.get(binding.consumer.operation)
        if source is None or consumer is None or binding.source.actor is not binding.consumer.actor:
            return None
        if need.kind is HazardKind.RAW:
            for pair in self.pairs:
                if source in pair.checks and consumer is pair.consumer:
                    return UnitFlagCoverage(binding, pair)
        if need.kind is HazardKind.WAR:
            arithmetic = IntegerFacts(memory.control.environment.facts)
            for pair in self.pairs:
                if source is not pair.consumer:
                    continue
                for following in self.pairs:
                    if consumer in following.checks and _same_plane(
                        _plane(source),
                        _plane(consumer),
                        arithmetic,
                    ):
                        return UnitFlagCoverage(binding, pair, following)
        return None


def admit_unit_flags(memory, dispatch):
    """Qualify actual native words once, retaining the original memory facts.

    Current support deliberately requires one complete contiguous plane per word,
    and one possible updating MAD followed by one coexecuting FIX in the same
    scope/stage. Partial reads, mismatched activation and interleaved planes are
    diagnosed; raw flag bits never authorize dropping a dependence by themselves.
    """
    arithmetic = IntegerFacts(memory.control.environment.facts)
    pairs = []
    for tasks in dispatch.bodies.values():
        checked, publisher, stage = [], None, None
        for task in sorted(tasks, key=lambda t: (t.stage, t.lexical)):
            template = task.template
            if not template.spec.unit_flag:
                continue
            _require(stage is None or task.stage == stage, "unitflag word crosses dispatch stages")
            stage = task.stage
            plane = _plane(template)
            _require(plane.geometry.contiguous, "unitflag needs a complete contiguous L0C plane")
            _require(
                plane.offset.constant is not None,
                "unitflag currently requires a fixed L0C plane, without rotating slots",
            )
            context = _context(template)
            if template.spec.engine is Engine.M:
                atoms = scalar_operands(template.operation)
                _require(
                    len(atoms) == 1 + bool(template.spec.read_resets),
                    "unitflag MAD lost its actual flag operand",
                )
                flag = memory.control.environment.expression(atoms[0])
                _require(
                    arithmetic.queries.implies(
                        all_of(
                            Comparison(flag, ">=", literal(2)),
                            Comparison(flag, "<=", literal(3)),
                        ),
                        context,
                    ).proved,
                    "MAD unit_flag must be 2 or 3 on its actual domain",
                )
                _require(publisher is None, "another flagged MAD intervenes before FIX consumption")
                if checked:
                    _require(
                        _same_plane(_plane(checked[0]), plane, arithmetic, context),
                        "unitflag word interleaves different L0C planes",
                    )
                checked.append(template)
                update = Comparison(flag, "==", literal(3))
                if not arithmetic.queries.satisfies((*context, update)).impossible:
                    publisher = template, (*template.frame.predicates(), update)
            else:
                _require(
                    template.spec.engine is Engine.FIX and publisher is not None,
                    "unitflag FIX has no preceding updating MAD",
                )
                final, publication = publisher
                consumption = template.frame.predicates()
                _require(
                    _same_plane(_plane(final), plane, arithmetic, context),
                    "unitflag FIX must consume the exact published L0C plane",
                )
                _require(
                    all(t.spec.params[:2] == template.spec.params[:2] for t in checked),
                    "unitflag MAD/FIX matrix dimensions must agree",
                )
                _require(
                    template.spec.params[0] % 16 == template.spec.params[1] % 16 == 0,
                    "unitflag FIX requires complete output fractals",
                )
                if plane.valid_rows is not None:
                    _require(
                        arithmetic.queries.equivalent(
                            plane.valid_rows,
                            literal(template.spec.params[0]),
                            context,
                        ).proved,
                        "unitflag FIX cannot leave unconsumed runtime rows",
                    )
                ranges = tuple(t for t in context if t not in consumption)
                _require(
                    arithmetic.queries.implies(all_of(*consumption), (*ranges, *publication)).proved
                    and arithmetic.queries.implies(
                        all_of(*publication), (*ranges, *consumption)
                    ).proved,
                    "unitflag publication and consumption must have identical activation",
                )
                for other in tasks:
                    candidate = other.template
                    if candidate.spec.unit_flag or not any(
                        a.address.root is plane.root for a in candidate.accesses
                    ):
                        continue
                    _require(
                        arithmetic.queries.satisfies(
                            (*context, *publication, *_context(candidate))
                        ).impossible,
                        "unflagged access interferes with a unitflag L0C word",
                    )
                pairs.append(
                    UnitFlagPair(final, template, tuple(checked), publication, consumption)
                )
                checked, publisher, stage = [], None, None
        _require(not checked and publisher is None, "unitflag word has no final FIX consumption")
    return UnitFlagProtocols(memory, dispatch, tuple(pairs))
