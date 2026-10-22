"""Case predicates and union coverage consume the actual mathematical context."""

import pytest
from xdsl.dialects import arith
from xdsl.dialects.builtin import IndexType
from xdsl.ir import Block

from ascendflow.analysis.smt import UNKNOWN, Answer, Exists
from ascendflow.passes.simplify_conditions import _Domains, _Ranges
from ascendflow.passes.simplify_scalars import _Scope

EVEN = _Ranges(((0, 0), (2, 2)))
MIDDLE = _Ranges(((1, 2),))
UNIT = _Ranges(((0, 1),))


def domain():
    block = Block(arg_types=[IndexType()] * 3)
    x, y, z = block.args
    two = arith.ConstantOp.from_int_and_width(2, IndexType())
    phase = arith.RemUIOp(x, two.result)
    # The complete mathematical context can learn the admitted phase range
    # only after this SSA's modulo link is encountered. It restricts x to 0/2.
    scope = _Scope(ranges={x: (0, 3), y: (0, 3), z: (0, 1), phase.result: (0, 0)})
    return _Domains(scope, {}), ((x, 1),), ((y, 1),), ((z, 1),), phase.result


def test_case_normalization_keeps_predicate_values_and_input_order():
    domains, x, y, z, _ = domain()
    normal = domains.normalize({x: EVEN, y: MIDDLE, z: UNIT})
    assert normal == {x: EVEN, y: MIDDLE}
    reversed_input = dict(reversed(tuple(normal.items())))
    assert domains.normalize(reversed_input) == normal
    assert list(domains.normalize(reversed_input)) == [y, x]
    assert domains.union([normal]) == [normal]


def test_union_coverage_consumes_a_later_modulo_atom_and_its_bounds():
    domains, x, _, z, phase = domain()
    assert domains.normalize({x: EVEN, z: UNIT}) == {x: EVEN}
    assert domains.bounds(((phase, 1),), {}) == _Ranges(((0, 0),))
    # New bounds plus phase == x % 2 now imply the entire old guard.
    assert domains.union([{x: _Ranges(((0, 0),))}, {x: _Ranges(((2, 2),))}]) == [{}]


def test_case_bounds_consume_a_new_base_relation_before_union():
    domains, x, _, z, _ = domain()
    assert domains.normalize({x: EVEN, z: UNIT}) == {x: EVEN}
    domains.base[x] = EVEN
    assert domains.normalize({x: EVEN}) == {}


@pytest.mark.parametrize("unknown", ("existence", "coverage"))
def test_unknown_never_substitutes_for_the_actual_union_proof(monkeypatch, unknown):
    domains, x, _, z, phase = domain()
    assert domains.bounds(((phase, 1),), {}) == _Ranges(((0, 0),))
    satisfy, implies = domains.queries.satisfies, domains.queries.implies

    def existence(*args):
        if unknown == "existence":
            return Exists(UNKNOWN, "satisfies", reason="controlled unknown")
        return satisfy(*args)

    def coverage(*args):
        if unknown == "coverage":
            return Answer(UNKNOWN, "implies", reason="controlled unknown")
        return implies(*args)

    monkeypatch.setattr(domains.queries, "satisfies", existence)
    monkeypatch.setattr(domains.queries, "implies", coverage)
    assert domains.normalize({x: EVEN, z: UNIT}) == {x: EVEN}
    # A real universal proof may simplify the union even when nonemptiness
    # was unknown. An unknown universal answer keeps the original guard.
    alternatives = [{x: _Ranges(((0, 0),))}, {x: _Ranges(((2, 2),))}]
    assert domains.union(alternatives) == ([{}] if unknown == "existence" else [{x: EVEN}])
