"""Immutable compiled artifacts shared by generation and mutation regressions.

Compilation.plan returns a fresh mutable IR each time; deliberately corrupted
plans cannot contaminate another test. No compilation cache is installed in the
product, and tests of compilation itself do not use these fixtures.
"""

import pytest

from ascendflow import compile


@pytest.fixture(scope="session")
def state_artifact():
    from examples.state.recurrence import state_sum

    return compile(state_sum.specialize(B=16, GROUPS=1))


@pytest.fixture(scope="session")
def grouped_artifact():
    from examples.state.grouped import grouped_state_sum

    return compile(grouped_state_sum.specialize(B=16, GROUPS=1))


@pytest.fixture(scope="session")
def frame_artifact():
    from examples.state.frame_snapshot import frame_state_sum

    return compile(frame_state_sum.specialize(B=16, GROUPS=4))


@pytest.fixture(scope="session")
def shared_artifact():
    from examples.state.shared_inputs import shared_state_sum

    return compile(shared_state_sum.specialize(B=16, GROUPS=4))
