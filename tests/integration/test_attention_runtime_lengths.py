"""Small-BK causal policies, arbitrary runtime lengths and actual tail bytes."""

from collections import Counter

import pytest

from ascendflow.dialects.physical import RuntimeSizeOp
from benchmarks.attention.contract import AttentionPolicy
from tests.models.structured_events import check_structured_events
from tests.support.attention_compilation import compile_attention


@pytest.fixture(scope="module", params=[128, 256])
def attention(tmp_path_factory, request):
    return compile_attention(
        tmp_path_factory,
        AttentionPolicy(causal=True, query_tile=request.param, kv_tile=256, vector_rows=16),
    )


@pytest.mark.parametrize(
    "sq,sk,groups",
    [
        (256, 1024, (0, 19)),
        (768, 3072, (0, 7, 19)),
        (2048, 1024, (0,)),
        (1, 1, (0, 19)),
        (127, 129, (0, 7, 19)),
        (129, 127, (0, 7, 19)),
        (257, 513, (0, 7, 19)),
        (769, 385, (0, 7, 19)),
    ],
)
def test_unaligned_queries_and_keys_keep_real_bounds_and_event_consumption(
    attention, sq, sk, groups
):
    final, original, asc, _, policy = attention
    checked = check_structured_events(
        final, {"Sq": sq, "Sk": sk}, groups=groups, logical_reference=original, max_commands=100000
    )
    assert checked.bytes_read and checked.bytes_written and checked.consumptions
    if (sq, sk) == (256, 1024):
        loads = Counter()
        for command in checked.natives:
            for span in command.spans:
                if span.root.sym_name.data in {"K", "V"}:
                    loads[span.root.sym_name.data] += span.end - span.start
        assert loads == {"K": policy.kv_tile * 128 * 2, "V": policy.query_tile * 128 * 2}
    assert asc.count("if ASC_IS_AIV") == 1
    assert sum(t.spec.native == "mask_fill" for t in final.templates) == 2
    assert "PIPE_ALL" not in asc


def test_one_compiled_policy_declares_only_positive_runtime_lengths(attention):
    final, _, _, _, _ = attention
    sizes = {
        op.sym_name.data: (op.minimum.data, op.multiple.data)
        for op in final.module.walk()
        if isinstance(op, RuntimeSizeOp)
    }
    assert sizes == {"Sq": (1, 1), "Sk": (1, 1)}
