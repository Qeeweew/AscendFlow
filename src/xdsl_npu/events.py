"""Finite local binary flag allocation from real consumption paths.

Each chain is old Wait -> next Set effect in the unchanged command graph.
Maximum matching supplies a minimum chain cover of THESE modeled paths. This
is neither permission to invent ordering nor a hardware impossibility proof.
The final verifier rechecks actual IDs and does not trust this allocator.
"""

from collections import defaultdict, deque

from xdsl.dialects.builtin import ArrayAttr, IntAttr

from .ir import ChannelOp
from .model import require


def bits(mask):
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


def minimum_chains(successors):
    """Hopcroft-Karp, bitset edges and iterative DFS (no recursion limit)."""
    n = len(successors)
    left, right = [-1] * n, [-1] * n
    infinity = n + 1
    while True:
        distance = [infinity] * n
        queue = deque()
        for u in range(n):
            if left[u] == -1:
                distance[u] = 0
                queue.append(u)
        shortest = infinity
        while queue:
            u = queue.popleft()
            if distance[u] >= shortest:
                continue
            for v in bits(successors[u]):
                matched = right[v]
                if matched == -1:
                    shortest = distance[u] + 1
                elif distance[matched] == infinity:
                    distance[matched] = distance[u] + 1
                    queue.append(matched)
        if shortest == infinity:
            break
        for root in range(n):
            if left[root] != -1:
                continue
            stack = [(root, iter(bits(successors[root])))]
            path = []
            while stack:
                u, targets = stack[-1]
                for v in targets:
                    matched = right[v]
                    if matched == -1 and distance[u] + 1 == shortest:
                        for (node, _), target in zip(stack, (*path, v)):
                            left[node], right[target] = target, node
                        stack.clear()
                        break
                    if matched != -1 and distance[matched] == distance[u] + 1:
                        stack.append((matched, iter(bits(successors[matched]))))
                        path.append(v)
                        break
                else:
                    distance[u] = infinity
                    stack.pop()
                    if path:
                        path.pop()
    chains = []
    for start in range(n):
        if right[start] != -1:
            continue
        chain, u = [], start
        while u != -1:
            chain.append(u)
            u = left[u]
        chains.append(chain)
    # A maximum antichain from the matching's minimum vertex cover.
    reachable_left = {u for u in range(n) if left[u] == -1}
    reachable_right = set()
    queue = deque(sorted(reachable_left))
    while queue:
        u = queue.popleft()
        for v in bits(successors[u]):
            if left[u] == v or v in reachable_right:
                continue
            reachable_right.add(v)
            if right[v] != -1 and right[v] not in reachable_left:
                reachable_left.add(right[v])
                queue.append(right[v])
    antichain = sorted(reachable_left - reachable_right)
    require(len(chains) == len(antichain), "internal event chain-cover certificate mismatch")
    return chains, antichain


def lifetime_chains(ex, graph, waits, physical_sets):
    """No ID-induced ordering: the graph contains only real command contracts."""
    arrivals = {
        (name, epoch): node for items in physical_sets.values() for node, name, epoch in items
    }
    banks = defaultdict(list)
    for cmd in ex.commands:
        if cmd.kind == "set":
            banks[cmd.event[:2]].append((cmd.channel, cmd.epoch))
    result = {}
    for bank, keys in banks.items():
        successors = [
            sum(
                1 << j
                for j in range(i + 1, len(keys))
                if graph.precedes(waits[key], arrivals[keys[j]])
            )
            for i, key in enumerate(keys)
        ]
        chains, antichain = minimum_chains(successors)
        result[bank] = (
            [[keys[i] for i in chain] for chain in chains],
            [keys[i] for i in antichain],
        )
    return result


def pack_event_lifetimes(module):
    from .sync import completion_graph, execution

    ex = execution(module)
    graph, _, _, waits, sets = completion_graph(ex)
    allocation = lifetime_chains(ex, graph, waits, sets)
    selected = {}
    for bank, (chains, antichain) in allocation.items():
        require(
            len(chains) <= 8,
            f"{bank[0]}->{bank[1]}: local binary event bank has 8 IDs; "
            f"verified lifetime packing requires {len(chains)}. "
            "Incomparable notifications under the current completion contracts: "
            + ", ".join(f"{name}[{epoch}]" for name, epoch in antichain),
        )
        for flag, chain in enumerate(chains):
            for key in chain:
                selected[key] = flag
    for op in module.body.block.first_op.body.block.ops:
        if not isinstance(op, ChannelOp):
            continue
        ch = ex.channels[op.sym_name.data]
        ids = tuple(selected[(ch.name, g)] for g in range(ch.lower, ch.upper, ch.stride))
        op.attributes.pop("event_ids", None)
        for width in range(1, min(8, len(ids)) + 1):
            base = ids[0]
            if base + width <= 8 and all(flag == base + i % width for i, flag in enumerate(ids)):
                op.attributes.update(base=IntAttr(base), width=IntAttr(width))
                break
        else:
            op.attributes.update(
                base=IntAttr(0),
                width=IntAttr(1),
                event_ids=ArrayAttr(IntAttr(flag) for flag in ids),
            )


def selection_expression(ids, ordinal):
    """Encode the exact finite ID map as piecewise short-period selections.

    Lifetime packing can change its rotation at a prologue, query boundary or
    epilogue. A whole-map period test misses these runs and emits one branch
    per generation. Greedily find the longest prefix with period <= 8 instead.
    Each of the eight scans is bounded by the consumed prefix, so discovering
    runs takes O(8 * len(ids)) time. This does not recolor events or move waits;
    the original verified map remains the authority for every valid ordinal.
    No constant table memory, DMA or out-of-range ordinal contract is added.
    """
    require(bool(ids), "event selection requires a nonempty ID map")

    def select(values, index, lo, hi):
        if all(flag == values[lo] for flag in values[lo:hi]):
            return str(values[lo])
        middle = (lo + hi) // 2
        return (
            f"(({index}) < {middle} ? {select(values, index, lo, middle)}"
            f" : {select(values, index, middle, hi)})"
        )

    runs = []
    start = 0
    while start < len(ids):
        stop, width = start, 1
        for period in range(1, min(8, len(ids) - start) + 1):
            end = start + period
            while end < len(ids) and ids[end] == ids[end - period]:
                end += 1
            if end > stop:
                stop, width = end, period
        values = ids[start : start + width]
        index = f"(({ordinal}) - {start})" if start else f"({ordinal})"
        index = f"({index} % {width})"
        if all(flag == values[0] + i for i, flag in enumerate(values)):
            code = str(values[0]) if width == 1 else f"({values[0]} + {index})"
        else:
            code = select(values, index, 0, width)
        runs.append((start, code))
        start = stop

    def choose(lo, hi):
        if hi - lo == 1:
            return runs[lo][1]
        middle = (lo + hi) // 2
        return f"(({ordinal}) < {runs[middle][0]} ? {choose(lo, middle)} : {choose(middle, hi)})"

    return choose(0, len(runs))
