"""The final scope guard must rebuild count and epoch predicates from actual IR."""

from dataclasses import replace

import pytest
from xdsl.dialects.builtin import IntAttr

from tests.fixtures.nested import shared_l0c_qk_pv
from xdsl_npu.compilation import compile
from xdsl_npu.completion import CountDomain, Position
from xdsl_npu.ir import ScopeChannelOp
from xdsl_npu.model import CompileError
from xdsl_npu.scope_domains import Activation, decode_activation, encode_activation
from xdsl_npu.scope_sync import verify_scopes


@pytest.fixture(scope="module")
def shared_compilation():
    return compile(shared_l0c_qk_pv, timeout_s=25)


def _channel(module, distance, first, last):
    expected = Activation(CountDomain(1, 0, minimum=2), first, last)
    matches = []
    for op in module.walk():
        if not isinstance(op, ScopeChannelOp) or op.activation is None:
            continue
        actual = decode_activation(
            op.activation,
            distance=op.distance.data,
            period=op.period.data,
            residues=tuple(item.data for item in op.residues),
        )
        if op.kind.data == "local" and op.distance.data == distance and actual == expected:
            matches.append((op, actual))
    assert len(matches) == 1
    return matches[0]


def test_final_guard_rejects_startup_predicate_missing_count_two(shared_compilation):
    # Each plan access parses fresh IR, so no planner or previous verifier state is reused.
    module = shared_compilation.plan
    op, active = _channel(module, 1, Position(False, 0), Position(False, 0))
    op.attributes["activation"] = encode_activation(
        replace(active, count=replace(active.count, minimum=3))
    )
    with pytest.raises(CompileError, match="missing scoped (WAR|WAW)"):
        verify_scopes(module)


def test_final_guard_rejects_tail_predicate_moved_to_count_minus_three(shared_compilation):
    module = shared_compilation.plan
    op, active = _channel(module, 1, Position(True, 1), Position(True, 1))
    # N>=3 keeps both endpoints valid; rejection must come from rebuilt memory obligations.
    op.attributes["activation"] = encode_activation(
        replace(
            active,
            count=replace(active.count, minimum=3),
            first=Position(True, 2),
            last=Position(True, 2),
        )
    )
    with pytest.raises(CompileError, match="missing scoped (WAR|WAW)"):
        verify_scopes(module)


def test_final_guard_rejects_signed_epoch_changed_to_same_epoch(shared_compilation):
    module = shared_compilation.plan
    op, _ = _channel(module, -1, Position(False, 1), Position(True, 0))
    op.attributes["distance"] = IntAttr(0)
    with pytest.raises(CompileError, match="missing scoped (WAR|WAW)"):
        verify_scopes(module)
