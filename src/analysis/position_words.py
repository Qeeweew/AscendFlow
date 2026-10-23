"""Immutable bands of real hierarchical positions, without causal facts.

Only the last lexical rank varies within one word. A target splits that rank
into before/tie/after bands; every member of a band has the same complete
clock comparison formula. Queues, guards, captures and consumption belong to
the caller. This index supplies none of those permissions.
"""

from bisect import bisect_left, bisect_right
from itertools import pairwise

from .access_frontier import QueueEndpoint


def actor_points(points):
    return tuple(sorted(points, key=lambda point: point.actor.value))


def position_word(points, order):
    points = actor_points(points)
    paths = tuple(order.position(point) for point in points)
    base = paths[0][-1][2]
    return (
        tuple(
            (
                point.actor,
                point.placement,
                path[:-1],
                path[-1][:2],
                path[-1][2] - base,
                # Distinct zero-width region ports have no implicit tie order.
                point.operation if isinstance(point.occurrence.endpoint, QueueEndpoint) else None,
                order.dispatch.site(point).stage,
            )
            for point, path in zip(points, paths, strict=True)
        ),
        base,
    )


class PositionWord:
    def __init__(self, entries, order):
        self.order = order
        self.examples = dict(entries)
        self.keys = sorted(self.examples)
        example = self.examples[self.keys[0]]
        self.paths = tuple(order.position(point) for point in actor_points(example))
        self.base = self.paths[0][-1][2]

    def bands(self, target):
        """Yield rank intervals and actual representatives, retaining ties."""
        target = actor_points(target)
        cuts = {0, len(self.keys)}
        for path, point in zip(self.paths, target, strict=True):
            other = self.order.position(point)
            depth = len(path) - 1
            if len(other) > depth and path[-1][0] is other[depth][0]:
                threshold = other[depth][2] - (path[-1][2] - self.base)
                cuts.add(bisect_left(self.keys, threshold))
                cuts.add(bisect_right(self.keys, threshold))
        for start, end in pairwise(sorted(cuts)):
            yield start, end, actor_points(self.examples[self.keys[start]])

    def ranges(self, target, requirements, *, before, work):
        target = actor_points(target)
        result = []
        for start, end, representative in self.bands(target):
            work.order_bands += 1
            relative = {
                self.order.compare(a, b, need)
                for a, b in zip(
                    representative if before else target,
                    target if before else representative,
                    strict=True,
                )
                for need in requirements
            }
            if relative <= {-1, 0}:
                if result and result[-1][1] == start:
                    result[-1] = result[-1][0], end
                else:
                    result.append((start, end))
        return result
