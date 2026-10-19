"""A finite strided stream preserves byte holes without per-iteration cells."""

from types import SimpleNamespace

import pytest

from ascendflow.analysis.native import AccessMode
from ascendflow.analysis.structured_access import _Partitioner
from ascendflow.scalar import Expr, Fact, expression, literal
from ascendflow.target.ascend910b import Actor
from ascendflow.target.native import NativeGeometry


@pytest.mark.parametrize("count", (1, 2, 17, 64))
@pytest.mark.parametrize("mode", (AccessMode.READ, AccessMode.WRITE))
def test_strided_dense_loop_is_one_exact_rectangle(count, mode):
    ordinal = Expr("argument", ("item",))
    loop = SimpleNamespace(operation=object(), ordinal=ordinal, count=literal(count))
    offset = expression("add", literal(32), expression("mul", ordinal, literal(256)))
    memory = SimpleNamespace(
        templates=(),
        control=SimpleNamespace(environment=SimpleNamespace(facts={"item": Fact(0, 1, count - 1)})),
    )
    access = SimpleNamespace(
        mode=mode,
        address=SimpleNamespace(geometry=NativeGeometry("GM", "f32", "linear", 64)),
    )
    template = SimpleNamespace(frame=SimpleNamespace(loops=lambda: (loop,)))
    projections = _Partitioner(memory, None)._slots_plain(
        template, access, Actor.AIV0, offset, (), ()
    )
    assert len(projections) == 1
    (projection,) = projections
    assert projection.contracted == (loop.operation,)
    assert not projection.slots
    represented = {
        projection.origin.constant + projection.start + row * projection.row_stride + byte
        for row in range(projection.rows)
        for byte in range(projection.stop - projection.start)
    }
    actual = {32 + item * 256 + byte for item in range(count) for byte in range(64)}
    assert represented == actual
