"""Logical rectangles in one L1 layout; no completion or event evidence."""

from dataclasses import dataclass, replace

from ..indexing import arguments, domain_nonnegative, exact_constant, normalize_or_identity
from ..scalar import Expr, expression, literal, substitute
from ..target.layout import NZLayout


@dataclass(frozen=True, slots=True)
class MatrixRegion:
    layout: NZLayout
    starts: tuple[Expr, ...]
    stops: tuple[Expr, ...]

    @classmethod
    def whole(cls, layout):
        return cls(
            layout, tuple(literal(0) for _ in layout.shape), tuple(map(literal, layout.shape))
        )

    def bind(self, bindings):
        return replace(
            self,
            starts=tuple(substitute(v, bindings) for v in self.starts),
            stops=tuple(substitute(v, bindings) for v in self.stops),
        )

    def canonical(self, facts):
        return replace(
            self,
            starts=tuple(normalize_or_identity(v, facts) for v in self.starts),
            stops=tuple(normalize_or_identity(v, facts) for v in self.stops),
        )

    def sweep(self, ordinal, count, facts):
        """Union a nonempty, adjacent rectangular traversal without expansion."""
        axes = [
            axis
            for axis, values in enumerate(zip(self.starts, self.stops, strict=True))
            if any(ordinal.args[0] in arguments(value) for value in values)
        ]
        if not axes:
            return self
        if len(axes) != 1:
            return None  # A diagonal traversal is not its rectangular hull.
        axis = axes[0]
        start, stop = self.starts[axis], self.stops[axis]
        width = exact_constant(expression("sub", stop, start), facts)
        first = substitute(start, {ordinal: literal(0)})
        shifted = substitute(start, {ordinal: expression("add", ordinal, literal(1))})
        stride = exact_constant(expression("sub", shifted, start), facts)
        if width is None or stride is None or not 0 <= stride <= width:
            return None
        # Verify a true translation, not merely a constant first difference at
        # one point. The real source ordinal is retained until this summary.
        linear = expression("add", first, expression("mul", ordinal, literal(stride)))
        if exact_constant(expression("sub", start, linear), facts) != 0:
            return None
        starts, stops = list(self.starts), list(self.stops)
        starts[axis] = first
        stops[axis] = expression(
            "add",
            expression("add", first, literal(width)),
            expression("mul", expression("sub", count, literal(1)), literal(stride)),
        )
        return replace(self, starts=tuple(starts), stops=tuple(stops))

    def contains(self, other, domains, facts):
        return self.layout == other.layout and all(
            domain_nonnegative(expression("sub", start, left), domains, facts)
            and domain_nonnegative(expression("sub", right, stop), domains, facts)
            for left, right, start, stop in zip(
                self.starts, self.stops, other.starts, other.stops, strict=True
            )
        )

    def disjoint(self, other, domains, facts):
        return self.layout == other.layout and any(
            domain_nonnegative(expression("sub", b, c), domains, facts)
            or domain_nonnegative(expression("sub", a, d), domains, facts)
            for a, c, b, d in zip(self.starts, self.stops, other.starts, other.stops, strict=True)
        )


class RegionDefinitions:
    """Exact rectangle unions with writer provenance, never bounding boxes."""

    def __init__(self):
        self.regions = {}

    def add(self, region, writers, facts):
        region = region.canonical(facts)
        if any(a == b for a, b in zip(region.starts, region.stops, strict=True)):
            return
        writers = tuple(dict.fromkeys(writers))
        while True:
            for prior, origins in tuple(self.regions.items()):
                if prior.layout != region.layout:
                    continue
                differing = [
                    axis
                    for axis, pair in enumerate(zip(prior.starts, prior.stops, strict=True))
                    if pair != (region.starts[axis], region.stops[axis])
                ]
                if not differing:
                    self.regions[prior] = tuple(dict.fromkeys((*origins, *writers)))
                    return
                if len(differing) != 1:
                    continue
                axis = differing[0]
                a, b = prior.starts[axis], prior.stops[axis]
                c, d = region.starts[axis], region.stops[axis]
                if b == c:
                    lo, hi = a, d
                elif d == a:
                    lo, hi = c, b
                elif all(v.constant is not None for v in (a, b, c, d)) and (
                    a.constant <= d.constant and c.constant <= b.constant
                ):
                    lo, hi = (
                        literal(min(a.constant, c.constant)),
                        literal(max(b.constant, d.constant)),
                    )
                else:
                    continue
                starts, stops = list(region.starts), list(region.stops)
                starts[axis], stops[axis] = lo, hi
                region = replace(region, starts=tuple(starts), stops=tuple(stops))
                writers = tuple(dict.fromkeys((*origins, *writers)))
                del self.regions[prior]
                break
            else:
                self.regions[region] = writers
                return

    def covering(self, region, domains, facts):
        whole = MatrixRegion.whole(region.layout)
        for definition, writers in self.regions.items():
            # Legal source views stay in the allocation. Complete definition
            # therefore needs no dynamic containment proof for each reader.
            if definition == whole or definition.contains(region, domains, facts):
                return writers
        return None
