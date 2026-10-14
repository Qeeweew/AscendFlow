"""Small exact residue operations; never enumerate a runtime loop length.

Storage aliases and protocol predicates have separate periods. Only predicates
which really share a command are unioned. Completion queries project a storage
family onto the protocol clock by CRT, retaining its earliest real occurrence.
"""

from math import gcd, lcm

from .model import require

MAX_PERIOD = 840
MAX_PERIOD_TEMPLATES = 65536
MAX_ACCESS_SEGMENTS = 1_000_000
MAX_PROOF_STATES = 1_000_000


def bounded_period(period, context):
    require(
        0 < period <= MAX_PERIOD,
        f"{context}: coupled period {period} exceeds analysis budget {MAX_PERIOD}; "
        "no expansion or serialization fallback",
    )
    return period


def normalize(period, residues):
    bounded_period(period, "residue family")
    residues = set(residues)
    require(residues and all(0 <= r < period for r in residues), "invalid residue family")
    for divisor in range(1, period + 1):
        if period % divisor == 0:
            chosen = tuple(sorted({r % divisor for r in residues}))
            if len(chosen) * (period // divisor) == len(residues):
                return divisor, chosen
    raise AssertionError("period cannot normalize")


def union(families, context):
    # Accumulate all cells of EACH period before taking any LCM. A ring's
    # per-cell residues normally unite to period 1 at a native endpoint.
    reduced = [normalize(p, rs) for p, rs in sorted(families.items())]
    if (1, (0,)) in reduced:
        return 1, (0,)
    period = bounded_period(lcm(*(p for p, _ in reduced)), context)
    return normalize(period, {r for p, rs in reduced for a in rs for r in range(a, period, p)})


def project(period, residues, clock, minimum):
    """Yield (clock residue, earliest real generation >= minimum).

    For each clock residue a path starting at the earliest intersection covers
    all later translations with that residue. No product-period graph is needed.
    """
    common = gcd(period, clock)
    quotient = clock // common
    inverse = pow(period // common, -1, quotient) if quotient > 1 else 0
    repeat = period * quotient
    first = {}
    for a in residues:
        for b in range(a % common, clock, common):
            generation = a + period * (((b - a) // common * inverse) % quotient)
            generation += max(0, (minimum - generation + repeat - 1) // repeat) * repeat
            first[b] = min(first.get(b, generation), generation)
    return tuple(sorted(first.items()))
