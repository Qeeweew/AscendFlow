"""Disjoint finite affine frames, including row-pitched column slices.

Each byte address is q * stride + residue. A constant child shifts q by
0..count-1; interval unions in q prove separation without enumerating count.
This proves physical ownership only, never completion or initialization.
"""

from itertools import pairwise

from .budget import poll
from .diagnostics import require

MAX_AFFINE_VISITS = 1_000_000


def _union(intervals):
    merged = []
    for lo, hi in sorted(intervals):
        if merged and lo <= merged[-1][1]:
            merged[-1] = merged[-1][0], max(merged[-1][1], hi)
        else:
            merged.append((lo, hi))
    return merged


def _pieces(lo, hi, stride):
    """At most three residue slabs, with half-open quotient intervals."""
    q, r = divmod(lo, stride)
    end, tail = divmod(hi, stride)
    if q == end:
        yield r, tail, q, q + 1
        return
    if r:
        yield r, stride, q, q + 1
        q += 1
    if q < end:
        yield 0, stride, q, end
    if tail:
        yield 0, tail, end, end + 1


def verify_affine_frames(refs, count, groups, stride, *, root):
    """Check actual fixed-allocation GM frames and every owner's translation.

    The caller proves every reference has the same positive affine slope and no
    periodic term. Same-generation aliases are united; distinct generations and
    owners must have disjoint full footprints, including holes between rows.
    """
    require(count >= 0 and stride > 0, "invalid finite affine frame domain")
    require(
        groups * sum(ref.rows for ref in refs) <= MAX_AFFINE_VISITS,
        f"{root}: affine ownership fragment budget exceeded",
    )
    shifts = {
        tuple(ref.address(0, core) - ref.address(0) for core in range(groups)) for ref in refs
    }
    require(len(shifts) == 1, f"{root}: written stream aliases change between groups")
    if count == 0:
        return
    base = _union(part for ref in refs for part in ref.segments(0))
    pieces = [
        (a, b, lo, hi, core)
        for core, shift in enumerate(next(iter(shifts)))
        for start, stop in base
        for a, b, lo, hi in _pieces(start + shift, stop + shift, stride)
    ]
    cuts = sorted({point for a, b, *_ in pieces for point in (a, b)})
    require(
        (len(cuts) - 1) * len(pieces) <= MAX_AFFINE_VISITS,
        f"{root}: affine ownership residue-sweep budget exceeded",
    )
    for a, b in pairwise(cuts):
        poll("finite affine ownership", root=root, pieces=len(pieces))
        owners = {}
        for left, right, lo, hi, core in pieces:
            if left <= a and b <= right:
                owners.setdefault(core, []).append((lo, hi))
        expanded = []
        for core, spans in owners.items():
            spans = _union(spans)
            require(
                count == 1 or all(hi - lo == 1 for lo, hi in spans),
                f"{root}: access crosses its task frame (overlapping affine generations)",
            )
            for lo, hi in spans:
                expanded.append((lo, hi + count - 1, core))
        end, owner = None, None
        for lo, hi, core in sorted(expanded):
            require(
                end is None or lo >= end,
                f"{root}: "
                + (
                    "access crosses its task frame (overlapping affine generations)"
                    if owner == core
                    else "cross-group stream ownership overlap"
                ),
            )
            end, owner = hi, core
