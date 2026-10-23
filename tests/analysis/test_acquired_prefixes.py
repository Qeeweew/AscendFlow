"""Real notification coverage against an independent exhaustive reference.

These comparisons cover binding choices, not event consumption or ID sharing.
The retained event/byte interpreter tests those independent obligations.
"""

from dataclasses import replace

import pytest

from ascendflow import language as N
from ascendflow.analysis.structured_protocols import NotificationKind
from ascendflow.passes import notification_binding as binding
from examples.gemm.batched import batched_matmul
from examples.kda.prefill_prepare import prefill_prepare
from examples.vector.scale import scale
from tests.fixtures.nested import nested_copy, nested_cv, partitioned_cv, shared_l0c_qk_pv
from tests.support.structured import selected


def exhaustive(items, order):
    def key(item):
        task = order.dispatch.site(binding._actor_points(item.consumers)[0])
        steps, scope = [(task.stage, task.lexical)], task.scope
        while scope.parent is not None:
            steps.append((scope.parent_stage, scope.parent_lexical * 3 + 1))
            scope = scope.parent
        return task.actor.value, tuple(reversed(steps)), item.source_engine.value

    kept = []
    for item in sorted(items, key=key):
        for i, previous in enumerate(kept):
            if binding._route_key(previous) != binding._route_key(item):
                continue
            if not binding._contains_activation(previous, item, order):
                continue
            acquire = binding._capture_order(
                previous.consumers, item.consumers, item.requirements, order
            )
            capture = binding._capture_order(
                item.sources,
                previous.consumers
                if previous.mechanism is NotificationKind.PIPE_FENCE
                else previous.sources,
                item.requirements,
                order,
            )
            if acquire <= {-1, 0} and capture <= {-1, 0}:
                kept[i] = replace(previous, requirements=previous.requirements + item.requirements)
                break
        else:
            kept.append(item)
    return tuple(kept)


@pytest.mark.parametrize(
    "source",
    [
        scale,
        batched_matmul,
        nested_copy,
        nested_cv,
        partitioned_cv,
        shared_l0c_qk_pv,
        prefill_prepare.specialize(CHUNKS=4, H=8, DTYPE="bf16", SEQUENCES=1, NC=4),
    ],
)
def test_index_keeps_the_exact_exhaustive_capture_acquire_and_ownership_choice(monkeypatch, source):
    current = binding._reuse_acquisitions

    def compare(items, order):
        expected = exhaustive(items, order)
        actual = current(items, order)
        assert actual == expected
        return actual

    monkeypatch.setattr(binding, "_reuse_acquisitions", compare)
    assert selected(source).closed


def test_sequential_binding_queries_scale_with_issue_words_not_old_acquire_history(monkeypatch):
    current = binding._reuse_acquisitions
    measurements = []

    def compare(items, order):
        original, counts = order.compare, [0]

        def counted(*args):
            counts[0] += 1
            return original(*args)

        order.compare = counted
        expected = exhaustive(items, order)
        dense = counts[0]
        counts[0] = 0
        actual = current(items, order)
        indexed = counts[0]
        order.compare = original
        assert actual == expected
        assert indexed < 12 * len(items)
        measurements.append((len(items), dense, indexed))
        return actual

    monkeypatch.setattr(binding, "_reuse_acquisitions", compare)
    for count in (16, 64, 256):
        body = "\n".join(
            f"        N.fill(x[{panel},:],1.0)\n        N.load_scalar(x[{panel},:1])"
            for panel in range(count)
        )
        source = N.from_source(f"""
def sequential():
    with N.Kernel():
        x=N.alloc_ub(({count},64),N.f32)
{body}
""")
        assert selected(source).closed
    assert measurements[2][1] > 10 * measurements[1][1]
    assert measurements[2][2] < 5 * measurements[1][2]
