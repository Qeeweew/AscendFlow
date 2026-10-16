"""Standard-pass integration and independently executed emitted scalar code."""

from xdsl_npu.asc_scalar import ScalarScope


def test_standard_passes_fold_identities_and_share_real_ssa_nodes():
    scope = ScalarScope()
    a = scope.div(scope.sub(scope.add("g", 0), 0), 1)
    b = scope.mul("g", 1)
    left, right = scope.rem(a, 2), scope.rem(b, 2)
    code = "\n".join(scope.finish([f"consume({scope.use(left)}, {scope.use(right)});"]))
    assert " - 0" not in code and " / 1" not in code and " * 1" not in code
    assert code.count("g % 2") == 1
    assert "consume(npu_scalar_0, npu_scalar_0)" in code
