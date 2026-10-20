"""Actor clock translations preserve real bytes, waits, and two-AIV quorum."""

import pytest

from ascendflow import language as N
from ascendflow.passes.structured_sync import synchronize_structure
from tests.fixtures.nested import partitioned_cv
from tests.models.structured_events import check_compiled_events


@pytest.fixture(scope="module", params=[(0, -1), (1, -1), (3, -1), (0, 3), (3, 0)])
def translated(request):
    cube, vector = request.param
    source = partitioned_cv.source
    source = source.replace('N.Stage(0, name="produce")', f'N.Stage({cube}, name="produce")')
    source = source.replace('N.Stage(2, name="consume")', f'N.Stage({cube + 2}, name="consume")')
    source = source.replace(
        'N.Stage(1, name="transform")', f'N.Stage({vector + 1}, name="transform")'
    )
    return synchronize_structure(N.from_source(source, disjoint_external=True))


@pytest.mark.parametrize("count", (0, 1, 2, 3, 5))
def test_each_actor_may_choose_its_own_clock_origin(translated, count):
    # This oracle executes actual queues/Set/Wait/addresses. It does not use
    # the production origin potentials or phase ranking as expected edges.
    execution = check_compiled_events(translated, {"n": count})
    assert bool(execution.natives) == bool(count)
    assert bool(execution.bytes_read) == bool(count)
