"""Index already-required capture/acquire pairs by their real issue words.

An entry is always one actual notification; independent captures and acquires
are never mixed. Within a word only the final lexical position varies. A query
partitions those positions at the target's actual lexical cuts, checks each
distinct order condition, then searches a two-dimensional range index. No old
notification history is scanned within a word and no unknown order is assumed.
This is completion coverage only, not event consumption or ID-reuse evidence.
"""

from bisect import bisect_left

from .position_words import PositionWord, position_word
from .structured_protocols import NotificationKind


def _capture(item):
    return item.consumers if item.mechanism is NotificationKind.PIPE_FENCE else item.sources


class _AcquiredWord:
    def __init__(self, items, order, work):
        self.work = work
        self.representative = items[0]
        self.captures = PositionWord(
            ((position_word(_capture(n), order)[1], _capture(n)) for n in items), order
        )
        self.acquires = PositionWord(
            ((position_word(n.consumers, order)[1], n.consumers) for n in items), order
        )
        self.size = 1 << (len(self.captures.keys) - 1).bit_length()
        # Each capture-tree vertex stores the actual acquired lexical ranks
        # and their monotonically increasing retained-notification numbers.
        # Insertion follows this word's real acquire order. A rectangle query
        # visits logarithmically many vertices and binary-searches each list.
        self.ranks = [[] for _ in range(2 * self.size)]
        self.numbers = [[] for _ in range(2 * self.size)]
        self.first = None
        self.capture_low = len(self.captures.keys)
        self.capture_high = 0

    def add(self, item, number, order):
        capture = bisect_left(self.captures.keys, position_word(_capture(item), order)[1])
        acquire = bisect_left(self.acquires.keys, position_word(item.consumers, order)[1])
        self.capture_low = min(self.capture_low, capture)
        self.capture_high = max(self.capture_high, capture + 1)
        vertex = self.size + capture
        while vertex:
            assert not self.ranks[vertex] or self.ranks[vertex][-1] <= acquire
            self.ranks[vertex].append(acquire)
            self.numbers[vertex].append(number)
            vertex //= 2
        if self.first is None:
            self.first = number

    def find(self, item):
        captures = tuple(
            (max(start, self.capture_low), min(end, self.capture_high))
            for start, end in self.captures.ranges(
                item.sources, item.requirements, before=False, work=self.work
            )
            if start < self.capture_high and self.capture_low < end
        )
        if not captures:
            return None
        acquires = self.acquires.ranges(
            item.consumers, item.requirements, before=True, work=self.work
        )
        result = None
        for start, end in captures:
            vertices = []
            start, end = start + self.size, end + self.size
            while start < end:
                if start & 1:
                    vertices.append(start)
                    start += 1
                if end & 1:
                    end -= 1
                    vertices.append(end)
                start, end = start // 2, end // 2
            for vertex in vertices:
                self.work.range_vertices += 1
                for low, high in acquires:
                    position = bisect_left(self.ranks[vertex], low)
                    if position < len(self.ranks[vertex]) and self.ranks[vertex][position] < high:
                        number = self.numbers[vertex][position]
                        result = number if result is None else min(result, number)
        return result


class AcquiredPrefixes:
    """Mutable coverage frontier, owned by one notification-binding pass.

    Word keys retain the whole actual coordinate path and activation class.
    Different words are checked separately; this is no promise of linear
    work for arbitrarily many distinct conditional invocation domains.
    """

    def __init__(self, items, order, route, activation, work):
        self.order, self.route = order, route
        self.work = work
        groups, self.keys = {}, {}
        for item in items:
            key = (
                route(item),
                activation(item),
                position_word(_capture(item), order)[0],
                position_word(item.consumers, order)[0],
            )
            self.keys[item] = key
            groups.setdefault(key, []).append(item)
        self.words = {key: _AcquiredWord(group, order, work) for key, group in groups.items()}
        self.work.acquisition_words = len(self.words)
        self.routes = {}

    def find(self, item, contains_activation):
        result = None
        for word in self.routes.get(self.route(item), ()):
            if result is not None and word.first >= result:
                break
            self.work.word_queries += 1
            if not contains_activation(word.representative, item, self.order):
                continue
            number = word.find(item)
            if number is not None:
                result = number if result is None else min(result, number)
        return result

    def add(self, item, number):
        word = self.words[self.keys[item]]
        if word.first is None:
            self.routes.setdefault(self.route(item), []).append(word)
        word.add(item, number, self.order)
        self.work.retained_acquires += 1
