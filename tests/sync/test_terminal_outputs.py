"""Final-only writes in compute retain empty, LAST and repeated-call semantics."""

import pytest

from ascendflow import compile
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.dialects.io import parse
from examples.gemm.hierarchical import hierarchical_gemm
from examples.gemm.reduction import reduction_gemm
from tests.models.structured_events import check_structured_events


@pytest.fixture(scope="module", params=["nt", "nn", "hierarchical"])
def terminal_program(request):
    hierarchy = request.param == "hierarchical"
    definition = hierarchical_gemm if hierarchy else reduction_gemm
    policy = {"M": 16, "N": 16, "BK": 16, "GROUPS": 2}
    if not hierarchy:
        policy["TRANSPOSE_B"] = request.param == "nt"
    result = compile(definition.specialize(**policy), timeout_s=20)
    return (
        hierarchy,
        collect_native_memory(parse(result.before_ir)),
        collect_native_memory(parse(result.plan_ir)),
    )


@pytest.mark.parametrize(
    "outputs,panels", [(0, 0), (0, 5), (1, 0), (3, 0), (1, 1), (3, 1), (1, 2), (3, 5)]
)
def test_terminal_write_count_and_real_retirement_across_reused_accumulators(
    terminal_program, outputs, panels
):
    hierarchy, before, after = terminal_program
    parameters = {"panels": panels, **({"outputs": outputs} if hierarchy else {})}
    trace = check_structured_events(
        after, parameters, logical_reference=before, groups=(0, 1), invocations=2
    )
    exports = [cmd for cmd in trace.natives if cmd.occurrence.operation.native.data == "fix"]
    expected = (outputs if hierarchy else 1) if panels else 0
    assert len(exports) == 2 * 2 * expected
    if expected:
        assert trace.bytes_read and trace.bytes_written and trace.consumptions
