"""Expanded address algebra must retain already-proved cyclic owner ranges."""

import pytest

from xdsl_npu.cyclic import CyclicDomain
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.indexing import check_axis, interval
from xdsl_npu.scalar import Expr, Fact
from xdsl_npu.scalar import expression as e
from xdsl_npu.scalar import literal as c

CORE, QUERY, TOTAL, BIAS = (Expr("argument", (name,)) for name in ("$core", "$q", "n", "bias"))
FACTS = {"n": Fact(), "bias": Fact()}
FLAT = e("add", CORE, e("mul", QUERY, c(4)))
DOMAINS = [
    (CORE, c(0), c(3)),
    (QUERY, c(0), e("sub", CyclicDomain(TOTAL, 4).local_bound, c(1))),
    (FLAT, c(0), e("sub", TOTAL, c(1))),
]
ADDRESS = e("add", e("mul", CORE, c(1024)), e("mul", QUERY, c(4096)))


def test_expanded_joint_address_keeps_uniform_bias_and_real_allocation_bound():
    address = e("add", ADDRESS, BIAS)
    lower, upper = interval(address, DOMAINS, FACTS)
    for total in range(20):
        for bias in (0, 32, 1024):
            values = {"n": total, "bias": bias}
            assert lower.evaluate(values) == bias
            assert upper.evaluate(values) == (total - 1) * 1024 + bias
            for core in range(4):
                for query in range(max(0, (total - core + 3) // 4)):
                    actual = address.evaluate(values | {"$core": core, "$q": query})
                    assert lower.evaluate(values) <= actual <= upper.evaluate(values)
    # A partial tile has 32 bytes of bias and only 992 remaining bytes.
    check_axis(e("add", ADDRESS, c(32)), 992, e("mul", TOTAL, c(1024)), DOMAINS, FACTS)


@pytest.mark.parametrize("fault", ("frame_overrun", "negative_bias", "wrong_stride", "other_clock"))
def test_a_joint_domain_cannot_authorize_a_changed_address(fault):
    address, width = ADDRESS, 1024
    if fault == "frame_overrun":
        address = e("add", address, c(32))
    elif fault == "negative_bias":
        address = e("sub", address, c(32))
    elif fault == "wrong_stride":
        address = e("add", e("mul", CORE, c(1024)), e("mul", QUERY, c(5120)))
    else:
        address = e("add", address, Expr("argument", ("$unknown",)))
    with pytest.raises(CompileError, match="negative|bounded|unproven"):
        check_axis(address, width, e("mul", TOTAL, c(1024)), DOMAINS, FACTS)


def test_independent_owner_query_intervals_do_not_invent_the_joint_domain():
    with pytest.raises(CompileError, match="bounded|unproven"):
        check_axis(ADDRESS, 1024, e("mul", TOTAL, c(1024)), DOMAINS[:-1], FACTS)
