"""Generic logical predicates, masked updates and packed compare/select.

Coordinate predicate values are evaluated independently from the source IR;
the emitted 64-lane mask-word algorithm is executed on the host C++ compiler
with exact 128-bit arithmetic. Event ordering and mask-storage reuse are
checked by the independent finite physical event oracle.
"""

from __future__ import annotations

import subprocess
from itertools import product
from math import prod

import pytest
from xdsl.dialects import arith
from xdsl.dialects.builtin import IntegerAttr

from ascendflow import compile
from ascendflow import language as N
from ascendflow.analysis.structured_admission import admit_structured_memory
from ascendflow.analysis.structured_bounds import admit_memory_bounds
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.backend.vector_mask import helper_lines
from ascendflow.diagnostics import CompileError
from ascendflow.dialects.io import parse
from ascendflow.dialects.physical import TaskOp
from ascendflow.dialects.source import MaskOp, SizeOp
from ascendflow.passes.lower_native import lower_native_structure
from ascendflow.target.ascend910b import Engine
from ascendflow.target.native import NativeGeometry, NativeSpec
from ascendflow.target.packets import MaskStep
from tests.models.structured_events import check_structured_events


def _scalar(value, environment, memo):
    if value in memo:
        return memo[value]
    owner = value.owner
    assert isinstance(
        owner,
        (
            arith.ConstantOp,
            arith.AddiOp,
            arith.SubiOp,
            arith.MuliOp,
            arith.DivUIOp,
            arith.RemUIOp,
            arith.MinSIOp,
            arith.MaxSIOp,
            arith.CeilDivUIOp,
            arith.CmpiOp,
            SizeOp,
        ),
    ), type(owner)
    if isinstance(owner, SizeOp):
        result = environment[owner.sym_name.data]
    elif isinstance(owner, arith.ConstantOp):
        assert isinstance(owner.value, IntegerAttr)
        result = owner.value.value.data
    else:
        operands = [_scalar(item, environment, memo) for item in owner.operands]
        if isinstance(owner, arith.AddiOp):
            result = operands[0] + operands[1]
        elif isinstance(owner, arith.SubiOp):
            result = operands[0] - operands[1]
        elif isinstance(owner, arith.MuliOp):
            result = operands[0] * operands[1]
        elif isinstance(owner, arith.DivUIOp):
            result = operands[0] // operands[1]
        elif isinstance(owner, arith.RemUIOp):
            result = operands[0] % operands[1]
        elif isinstance(owner, arith.MinSIOp):
            result = min(operands)
        elif isinstance(owner, arith.MaxSIOp):
            result = max(operands)
        elif isinstance(owner, arith.CeilDivUIOp):
            result = -(-operands[0] // operands[1])
        else:
            assert isinstance(owner, arith.CmpiOp)
            predicates = (
                lambda a, b: a == b,
                lambda a, b: a != b,
                lambda a, b: a < b,
                lambda a, b: a <= b,
                lambda a, b: a > b,
                lambda a, b: a >= b,
            )
            result = predicates[owner.predicate.value.data](*operands)
    memo[value] = result
    return result


def _evaluate(program, mask, index, environment, memo):
    scalar_memo = {}
    values = []
    for node, stored in zip(program.nodes, program.node_shapes, strict=True):
        args = node.args
        if node.op == 0:  # const
            values.append(args[0])
        elif node.op == 1:  # scalar
            values.append(_scalar(mask.scalars[args[0]], environment, scalar_memo))
        elif node.op == 2:  # lane
            target = len(program.shape) - len(stored) + args[0]
            values.append(index[target] if stored[args[0]] > 1 else 0)
        elif node.op == 3:
            values.append(values[args[0]] + values[args[1]])
        elif node.op == 4:
            values.append(values[args[0]] - values[args[1]])
        elif node.op == 5:
            values.append(values[args[0]] * args[1])
        elif node.op == 6:
            values.append(values[args[0]] // args[1])
        elif node.op == 7:
            values.append(values[args[0]] % args[1])
        elif node.op == 8:
            values.append(-values[args[0]])
        elif node.op == 9:
            left, right, kind = values[args[0]], values[args[1]], args[2]
            values.append(
                (
                    left == right,
                    left != right,
                    left < right,
                    left <= right,
                    left > right,
                    left >= right,
                )[kind]
            )
        elif node.op == 10:
            values.append(values[args[0]] and values[args[1]])
        elif node.op == 11:
            values.append(values[args[0]] or values[args[1]])
        elif node.op == 12:
            values.append(not values[args[0]])
        else:
            assert node.op == 13
            values.append(values[args[0]])
    return values[-1]


def mask_values(result, environment):
    """All source predicate programs with their exact per-lane truth values."""
    module = parse(result.source_ir)
    truth = []
    for operation in module.walk():
        if isinstance(operation, MaskOp):
            program = operation.program()
            lanes = {
                index: _evaluate(program, operation, index, environment, {})
                for index in product(*[range(size) for size in program.shape])
            }
            truth.append((program.shape, lanes))
    return truth


def _task(memory, native):
    return next(
        template.operation for template in memory.templates if template.spec.native == native
    )


def _ordered(execution):
    indexes = {
        command.occurrence.operation.uid.data: index
        for index, command in enumerate(execution.commands)
        if command.kind == "native"
    }

    def ordered(source, consumer):
        first = execution.commands[indexes[source.uid.data]]
        second = execution.commands[indexes[consumer.uid.data]]
        return execution.happens_before(first.end, second.begin)

    return ordered


def _masked_fill(mask_expression):
    return N.from_source(
        f"""
        def kernel(X:N.Tensor((16,256),N.f32), Y:N.Tensor((16,256),N.f32,access="write"),
                   q:N.int64, lo:N.int64, hi:N.int64):
            with N.Kernel(1, kind="vector"):
                work=N.alloc_ub((16,256),N.f32)
                N.copy(X,work)
                rows=N.arange(16)[:, None]
                cols=N.arange(256)[None, :]
                mask={mask_expression}
                N.fill(work,-7.0,mask=mask)
                N.copy(work,Y)
        """,
        disjoint_external=True,
    )


@pytest.mark.parametrize(
    "expression,environment,expected",
    [
        (
            "(cols + q) > rows",
            {"q": 5, "lo": 0, "hi": 0},
            lambda r, c, e: (c + 5) > r,
        ),
        (
            "((cols >= lo) & (cols < hi)) & ((rows % 3) == 1)",
            {"q": 0, "lo": 70, "hi": 190},
            lambda r, c, e: 70 <= c < 190 and r % 3 == 1,
        ),
        (
            "((cols + 17) % 5) == 2",
            {"q": 0, "lo": 0, "hi": 0},
            lambda r, c, e: (c + 17) % 5 == 2,
        ),
        (
            "((cols % 2) == 0) | ((rows + q) < cols)",
            {"q": -1000, "lo": 0, "hi": 0},
            lambda r, c, e: c % 2 == 0 or (r - 1000) < c,
        ),
        (
            "~((cols + q) > rows)",
            {"q": 0, "lo": 0, "hi": 0},
            lambda r, c, e: not (c > r),
        ),
    ],
)
def test_coordinate_predicate_values_broadcast_and_boolean(expression, environment, expected):
    result = compile(_masked_fill(expression), timeout_s=120)
    truth = mask_values(result, environment)
    assert truth
    for shape, lanes in truth[-1:]:
        assert shape == (16, 256)
        for (r, c), value in lanes.items():
            assert value == expected(r, c, environment), (r, c)


@pytest.mark.parametrize(
    "expression,environment,expected",
    [
        ("(cols + q) > rows", {"q": 0, "lo": 0, "hi": 0}, lambda r, c: c > r),
        (
            "((cols >= lo) & (cols < hi))",
            {"q": 0, "lo": 0, "hi": 0},
            lambda r, c: False,
        ),
        (
            "((cols >= lo) & (cols < hi))",
            {"q": 0, "lo": 0, "hi": 256},
            lambda r, c: True,
        ),
        (
            "(cols + 9223372036854775807) > (rows + (-9223372036854775807 - 1))",
            {"q": 0, "lo": 0, "hi": 0},
            lambda r, c: (c + (1 << 63) - 1) > (r - (1 << 63)),
        ),
    ],
)
def test_predicate_empty_all_and_extreme_signed_origins(expression, environment, expected):
    result = compile(_masked_fill(expression), timeout_s=120)
    _shape, lanes = mask_values(result, environment)[-1]
    assert tuple(lanes) == tuple(product(range(16), range(256)))
    assert all(value == expected(r, c) for (r, c), value in lanes.items())


HOST_SOURCE = r"""
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#define __aicore__
%(helpers)s
static __int128 npu_parse(const char* text) {
  bool negative = *text == '-';
  if (negative) ++text;
  __int128 value = 0;
  while (*text) { value = value * 10 + (*text - '0'); ++text; }
  return negative ? -value : value;
}
int main() {
  long long a;
  char b[64];
  int kind;
  while (std::scanf("%lld %63s %d", &a, b, &kind) == 3) {
    std::printf("%llu\n", static_cast<unsigned long long>(
        npu_mask_monotone(a, npu_parse(b), kind)));
  }
  return 0;
}
"""


def _run_mask_helper(cases, tmp_path):
    source = tmp_path / "mask_helper.cpp"
    binary = tmp_path / "mask_helper"
    source.write_text(HOST_SOURCE.replace("%(helpers)s", "\n".join(helper_lines())))
    subprocess.run(
        ["g++", "-O1", "-std=c++17", str(source), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = "\n".join(f"{a} {b} {kind}" for a, b, kind in cases)
    completed = subprocess.run(
        [str(binary)], input=lines, check=True, capture_output=True, text=True
    )
    return [int(value) for value in completed.stdout.split()]


def _expected_word(a, b, kind):
    ops = (
        lambda value: value == 0,
        lambda value: value != 0,
        lambda value: value < 0,
        lambda value: value <= 0,
        lambda value: value > 0,
        lambda value: value >= 0,
    )[kind]
    word = 0
    for lane in range(64):
        if ops(a * lane + b):
            word |= 1 << lane
    return word


def test_emitted_mask_word_algorithm_is_exact(tmp_path):
    values = [
        -(1 << 120),
        -(1 << 64),
        -(1 << 63),
        -65,
        -64,
        -63,
        -1,
        0,
        1,
        63,
        64,
        65,
        (1 << 63) - 1,
        (1 << 120),
    ]
    cases = [
        (a, b, kind)
        for a in (-64, -7, -1, 0, 1, 3, 64, (1 << 62))
        for b in values
        for kind in range(6)
    ]
    words = _run_mask_helper(cases, tmp_path)
    assert len(words) == len(cases)
    for (a, b, kind), word in zip(cases, words, strict=True):
        assert word == _expected_word(a, b, kind), (a, b, kind)


HOST_UNIT_SOURCE = r"""
#include <cstdint>
#include <cstdio>
#define __aicore__
%(helpers)s
static __int128 npu_parse(const char* text) {
  bool negative = *text == '-';
  if (negative) ++text;
  __int128 value = 0;
  while (*text) { value = value * 10 + (*text - '0'); ++text; }
  return negative ? -value : value;
}
int main() {
  char b[64];
  int kind;
  while (std::scanf("%63s %d", b, &kind) == 2) {
    std::printf("%llu\n", static_cast<unsigned long long>(
        npu_mask_unit(npu_parse(b), kind)));
  }
  return 0;
}
"""


def test_unit_coefficient_word_is_exact_without_a_search(tmp_path):
    """The O(1) unit-coefficient word is exact, including int64 extremes."""
    source = tmp_path / "mask_unit.cpp"
    binary = tmp_path / "mask_unit"
    source.write_text(HOST_UNIT_SOURCE.replace("%(helpers)s", "\n".join(helper_lines())))
    subprocess.run(
        ["g++", "-O1", "-std=c++17", str(source), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
    )
    extremes = [
        -(1 << 125),
        -(1 << 120),
        -(1 << 64),
        -(1 << 63),
        -(1 << 63) + 1,
        -(1 << 63) + 64,
        (1 << 63) - 65,
        (1 << 63) - 1,
        (1 << 64),
        (1 << 120),
        (1 << 125),
        (1 << 126) - 1,
        -(1 << 126) + 1,
    ]
    values = [*range(-200, 201), *extremes]
    cases = [(b, kind) for b in values for kind in range(6)]
    lines = "\n".join(f"{b} {kind}" for b, kind in cases)
    completed = subprocess.run(
        [str(binary)], input=lines, check=True, capture_output=True, text=True
    )
    words = [int(value) for value in completed.stdout.split()]
    assert len(words) == len(cases)
    for (b, kind), word in zip(cases, words, strict=True):
        assert word == _expected_word(1, b, kind), (b, kind)


def test_unit_lane_coefficients_emit_the_interval_recipe():
    from ascendflow.mask import ITERATION_CHUNKS, ITERATION_ROWS, WORD_AFFINE

    # A single unit comparison selects the selected row traversal; a Boolean
    # composition keeps chunk iteration, where the +1 unit word (and the -1
    # negate-and-flip form) are still emitted per 64-lane block.
    for expression, call, searched in (
        ("((cols + q) > rows) & ((rows * 2) <= 30)", "npu_mask_unit(", True),
        ("((cols + q) > rows) & ((rows - cols) >= q)", "npu_mask_unit(-(", False),
    ):
        result = compile(_masked_fill(expression), timeout_s=120)
        task = next(
            template
            for template in collect_native_memory(result.plan).templates
            if template.spec.native == "mask_fill"
        )
        attributes = task.operation.attributes
        assert attributes["mask_word"].data == WORD_AFFINE
        assert attributes["mask_iteration"].data == ITERATION_CHUNKS
        assert call in result.asc
        # A zero-coefficient comparison keeps the exact point test; unit-only
        # compositions emit no search at all.
        assert ("npu_mask_monotone((__int128)(" in result.asc) == searched
    rows = compile(_masked_fill("(cols + q) > rows"), timeout_s=120)
    row_task = next(
        template
        for template in collect_native_memory(rows.plan).templates
        if template.spec.native == "mask_fill"
    )
    assert row_task.operation.attributes["mask_iteration"].data == ITERATION_ROWS
    assert "for (int64_t npu_row = 0" in rows.asc
    assert "npu_mask_clamp(" in rows.asc
    # A non-unit coefficient keeps the general constant-coefficient fallback.
    general = compile(_masked_fill("(cols * 3) > rows"), timeout_s=120)
    assert "npu_mask_monotone((__int128)(3LL)" in general.asc


def _mask_spec(expression, shape=(16, 256)):
    """The filled tile's predicate program and its admitted replay facts."""
    rank = len(shape)
    coords = []
    for axis, extent in enumerate(shape):
        parts = ["None"] * rank
        parts[axis] = ":"
        coords.append(f"i{axis}=N.arange({extent})[{', '.join(parts)}]")
    body = [
        f'def kernel(X:N.Tensor({shape},N.f32), Y:N.Tensor({shape},N.f32,access="write"),',
        "           q:N.int64, lo:N.int64, hi:N.int64):",
        '    with N.Kernel(1, kind="vector"):',
        f"        work=N.alloc_ub({shape},N.f32)",
        "        N.copy(X,work)",
        *(f"        {coordinate}" for coordinate in coords),
        f"        N.fill(work,-7.0,mask={expression})",
        "        N.copy(work,Y)",
    ]
    source = N.from_source("\n".join(body), disjoint_external=True)
    result = compile(source, timeout_s=120)
    module = parse(result.source_ir)
    fill = next(
        operation
        for operation in module.walk()
        if operation.name == "npu_source.element" and operation.kind.data == "fill"
    )
    op = fill.mask.owner
    task = next(
        template
        for template in collect_native_memory(result.plan).templates
        if template.spec.native == "mask_fill"
    )
    attributes = task.operation.attributes
    return (
        op,
        op.program(),
        [value.owner.sym_name.data for value in op.scalars],
        attributes["mask_word"].data,
        attributes["mask_repeat"].data,
        attributes["mask_iteration"].data,
    )


HOST_EMITTER_SOURCE = r"""
#include <cstdint>
#include <cstdio>
#define __aicore__
#define __ubuf__
@@HELPERS@@
static uint64_t gn_mask = UINT64_MAX;
static long gn_calls = 0;
static float gn_buffer[@@MAXLANES@@];
static void asc_set_mask_norm(void) { gn_mask = UINT64_MAX; }
static void asc_set_vector_mask(int64_t, uint64_t mask) { gn_mask = mask; }
static void asc_duplicate(float* dst, float value, uint8_t repeat, int, int) {
  ++gn_calls;
  for (int group = 0; group < static_cast<int>(repeat); ++group) {
    float* at = dst + group * 64;
    for (int lane = 0; lane < 64; ++lane)
      if ((gn_mask >> lane) & UINT64_C(1)) at[lane] = value;
  }
}
static void reset_buffer(int lanes) {
  gn_calls = 0;
  for (int index = 0; index < lanes; ++index) gn_buffer[index] = 991.0f;
}
static void report(int chunks) {
  std::printf("calls %ld\n", gn_calls);
  for (int chunk = 0; chunk < chunks; ++chunk) {
    unsigned long long word = 0;
    for (int lane = 0; lane < 64; ++lane)
      if (gn_buffer[chunk * 64 + lane] == -7.0f) word |= (1ULL << lane);
    std::printf("%llu\n", word);
  }
}
@@CASES@@
int main() {
@@CALLS@@
  return 0;
}
"""


def _run_emitter_cases(cases, tmp_path):
    """Execute the real emitted masked-fill replay on the host C++ compiler."""
    from types import SimpleNamespace

    from ascendflow.backend.vector_mask import masked_element_lines

    bodies = []
    shapes = []
    expectations = []
    for spec, environment in cases:
        op, program, names, word, repeat, iteration = spec
        task = SimpleNamespace(
            spec=NativeSpec(
                "mask_fill",
                Engine.V,
                ("RW",),
                (program.lanes // 64,),
                (NativeGeometry("UB", "f32", "linear", program.lanes * 4),),
            ),
            scalar=-7.0,
            operands=tuple(f"(int64_t)({environment[name]})" for name in names),
            mask=program,
            mask_word=word,
            mask_repeat=repeat,
            mask_iteration=iteration,
        )
        bodies.append("\n".join(masked_element_lines(task, ["gn_buffer"])))
        shapes.append(program.shape)
        expectations.append(
            {
                index: _evaluate(program, op, index, environment, {})
                for index in product(*[range(extent) for extent in program.shape])
            }
        )
    maxlanes = max(prod(shape) for shape in shapes)
    source_text = HOST_EMITTER_SOURCE
    for token, text in (
        ("@@HELPERS@@", "\n".join(helper_lines())),
        ("@@MAXLANES@@", str(maxlanes)),
        (
            "@@CASES@@",
            "\n".join(
                f"static void case_{number}(void) {{\n"
                f"  reset_buffer({prod(shape)});\n{body}\n"
                f"  report({prod(shape) // 64});\n}}"
                for number, (body, shape) in enumerate(zip(bodies, shapes, strict=True))
            ),
        ),
        ("@@CALLS@@", "\n".join(f"  case_{number}();" for number in range(len(bodies)))),
    ):
        source_text = source_text.replace(token, text)
    source = tmp_path / "mask_emit.cpp"
    binary = tmp_path / "mask_emit"
    source.write_text(source_text)
    completed = subprocess.run(
        ["g++", "-O1", "-std=c++17", str(source), "-o", str(binary)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    run = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    rows = run.stdout.split()
    calls = []
    offset = 0
    for number, (shape, expected) in enumerate(zip(shapes, expectations, strict=True)):
        chunks = prod(shape) // 64
        assert rows[offset] == "calls"
        calls.append(int(rows[offset + 1]))
        for chunk in range(chunks):
            truth = 0
            for lane in range(64):
                flat = chunk * 64 + lane
                index = []
                rest = flat
                for extent in reversed(shape):
                    index.append(rest % extent)
                    rest //= extent
                if expected[tuple(reversed(index))]:
                    truth |= 1 << lane
            assert int(rows[offset + 2 + chunk]) == truth, (number, chunk)
        offset += chunks + 2
    assert offset == len(rows)
    return calls


def test_runtime_tile_bounds_replay_is_exact(tmp_path):
    """The emitted replays match independent per-lane integer truth."""
    cases = [
        # single unit comparison: row-interval traversal at several widths
        (_mask_spec("(i1 + q) > i0", (4, 64)), {"q": 3}),
        (_mask_spec("(i1 + q) > i0", (4, 128)), {"q": 3}),
        (_mask_spec("(i1 + q) > i0", (16, 256)), {"q": 3}),
        (_mask_spec("(i1 + q) > i0", (16, 256)), {"q": 300}),
        (_mask_spec("(i1 + q) > i0", (16, 256)), {"q": -1000}),
        (_mask_spec("(i1 + q) > i0", (16, 1024)), {"q": 3}),
        (_mask_spec("(i1 + q) > i0", (2, 1024)), {"q": 1000}),
        # all six kinds, unit and negated-unit, incl. eq/ne two-span points
        (_mask_spec("(i1 + q) >= i0"), {"q": 3}),
        (_mask_spec("(i1 + q) < i0"), {"q": 3}),
        (_mask_spec("(i1 + q) <= i0"), {"q": 3}),
        (_mask_spec("(i1 + q) == i0"), {"q": 3}),
        (_mask_spec("(i1 + q) == i0"), {"q": 1000}),
        (_mask_spec("(i1 + q) != i0"), {"q": 3}),
        (_mask_spec("(i1 + q) != i0"), {"q": 1000}),
        (_mask_spec("(i0 >= i1 + q)"), {"q": -5}),
        (_mask_spec("(i0 >= i1 + q)"), {"q": 5}),
        # non-unit / zero coefficient / Boolean / periodic keep the chunk replay
        (_mask_spec("((i1 * 3) - q) > (i0 * 2)"), {"q": 2}),
        (_mask_spec("((i1 - i1) + q) > i0"), {"q": 3}),
        (_mask_spec("((i1 + q) > i0) & ((i0 * 2) <= 30)"), {"q": 300}),
        (_mask_spec("((i1 + q) > i0) & ((i0 * 2) <= 30)"), {"q": 3}),
        (_mask_spec("~((i1 + q) > i0)"), {"q": 3}),
        (_mask_spec("((i0 >= lo) & (i1 < hi))"), {"lo": 0, "hi": 190}),
        (_mask_spec("((i0 >= lo) & (i1 < hi))"), {"lo": 0, "hi": 1000}),
        (_mask_spec("((i0 >= lo) & (i1 < hi))"), {"lo": 5, "hi": 0}),
        (_mask_spec("((i1 // 64) % 2) == 0"), {}),
        # opposite signed uses of one runtime atom: true difference is 2*q
        (_mask_spec("(i1 + q) > (i1 - q)"), {"q": 5}),
        (_mask_spec("(i1 + q) > (i1 - q)"), {"q": 0}),
        (_mask_spec("(i1 + q) > (i1 - q)"), {"q": -5}),
        # exact int64 extremes through the row threshold
        (_mask_spec("(i1 + q) > (i0 + (-9223372036854775807 - 1))"), {"q": (1 << 63) - 1}),
        (_mask_spec("(i1 + q) > (i0 + (-9223372036854775807 - 1))"), {"q": -(1 << 63)}),
        (_mask_spec("((i1 + q) + (-9223372036854775807 - 1)) == (i0 + q)"), {"q": 0}),
        # Full-run + trailing/leading recomputation: a run that advances past
        # whole blocks must not reuse the previous chunk/from/span.
        *[(_mask_spec("i1 < hi", (4, 256)), {"hi": hi}) for hi in (65, 127, 129, 191, 255)],
        (_mask_spec("i1 <= hi", (4, 256)), {"hi": 65}),
        (_mask_spec("i1 <= hi", (4, 256)), {"hi": 127}),
        (_mask_spec("i1 <= hi", (4, 256)), {"hi": 255}),
        # `ne` point past one whole chunk: first span has a full run + trailing
        # partial, second span has a leading partial + full run.
        (_mask_spec("(i1 + q) != i0", (4, 256)), {"q": -100}),
        # One row longer than 255 whole blocks plus a trailing partial: the
        # bounded 255 split plus one 1-block repeat and a trailing 63-lane
        # bitmap, i.e. exactly three element calls.
        (_mask_spec("i1 < hi", (1, 16448)), {"hi": 16447}),
    ]
    calls = _run_emitter_cases(cases, tmp_path)
    # Row intervals on a fully mixed large tile: 16 rows x <= 3 spans each, far
    # below the 1024/64 = 256 per-chunk visits it replaces.
    large_mixed = calls[5]
    assert large_mixed <= 16 * 4, calls
    # Same-chunk span (C=64) is exactly one bitmap; the empty/full tiles are the
    # retained whole-tile runtime shortcut.
    assert calls[0] <= 4, calls
    assert calls[6] == 1, calls
    assert calls[4] == 0, calls
    tail = calls[len(cases) - 10 :]
    # Every `i1 < hi` tail case is exactly one full-run call plus one trailing
    # partial block per row; the `<=` variants cover the whole row in one run
    # except hi=65/127/191-style tails handled above.
    assert tail[0:5] == [8, 8, 8, 8, 8], tail
    assert tail[5:8] == [8, 4, 1], tail  # hi=255 proves the whole tile true
    ne_point = tail[8]
    assert 8 < ne_point <= 4 * 6, tail
    large = tail[9]
    assert large == 3, tail


def test_runtime_tile_bounds_prescribe_the_affine_dispatch_only():
    from ascendflow.mask import ITERATION_CHUNKS, ITERATION_ROWS, WORD_AFFINE, WORD_LANE

    affine = compile(_masked_fill("(cols + q) > rows"), timeout_s=120).asc
    assert "if (npu_m_all" in affine and "else if (!(npu_m_none" in affine
    # The selected row traversal is the unproven fallback of the same replay.
    assert "for (int64_t npu_row = 0" in affine
    assert "for (int64_t npu_chunk = 0" not in affine
    assert "asc_set_vector_mask(0, 0)" not in affine
    assert "PIPE_ALL" not in affine
    spec = _mask_spec("(i1 + q) > i0")
    assert spec[3] == WORD_AFFINE and spec[5] == ITERATION_ROWS
    # A Boolean composition keeps the tested runtime bounds + chunk replay.
    composed = compile(_masked_fill("((cols + q) > rows) & ((rows * 2) <= 30)"), timeout_s=120).asc
    assert "if (npu_m_all" in composed and "for (int64_t npu_chunk = 0" in composed
    assert "for (int64_t npu_row = 0" not in composed
    assert _mask_spec("((i1 + q) > i0) & ((i0 * 2) <= 30)")[5] == ITERATION_CHUNKS
    general = compile(_masked_fill("(cols * 3) > rows"), timeout_s=120).asc
    assert "npu_mask_monotone((__int128)(3LL)" in general
    # A periodic predicate is not proven affine: no bounds, no row traversal.
    periodic = compile(_masked_fill("((cols // 64) % 2) == 0"), timeout_s=120).asc
    assert "npu_m_all" not in periodic and "npu_m_lo" not in periodic
    assert "npu_mask_word |= (UINT64_C(1) << npu_lane)" in periodic
    assert _mask_spec("((i1 // 64) % 2) == 0")[3] == WORD_LANE


def _row_fill_asc(shape, expression):
    """Compile the masked fill at an arbitrary shape and return its ASC."""
    rank = len(shape)
    coords = []
    for axis, extent in enumerate(shape):
        parts = ["None"] * rank
        parts[axis] = ":"
        coords.append(f"i{axis}=N.arange({extent})[{', '.join(parts)}]")
    source = N.from_source(
        "\n".join(
            [
                f'def kernel(X:N.Tensor({shape},N.f32), Y:N.Tensor({shape},N.f32,access="write"),',
                "           q:N.int64, lo:N.int64, hi:N.int64):",
                '    with N.Kernel(1, kind="vector"):',
                f"        work=N.alloc_ub({shape},N.f32)",
                "        N.copy(X,work)",
                *(f"        {coordinate}" for coordinate in coords),
                f"        N.fill(work,-7.0,mask={expression})",
                "        N.copy(work,Y)",
            ]
        ),
        disjoint_external=True,
    )
    return compile(source, timeout_s=120).asc


def test_row_full_run_keeps_the_static_255_repeat_boundary():
    """A row bounds a full run statically: split only above 255 whole blocks."""

    def row_body(asc):
        # The whole-tile all-true shortcut keeps its own bounded run replay;
        # only the per-row spans are under test here.
        at = asc.index("for (int64_t npu_row = 0")
        return asc[at : asc.index("asc_sync_notify(PIPE_V, PIPE_MTE3", at)]

    small = row_body(_row_fill_asc((16, 256), "(i1 + q) > i0"))
    assert "int64_t npu_rep = (npu_end - npu_at) / 64;" in small
    assert "npu_left" not in small and "? 255 :" not in small
    # Exactly 255 whole blocks per row still fits one ISA repeat ...
    edge = row_body(_row_fill_asc((1, 16320), "i1 < hi"))
    assert "int64_t npu_rep = (npu_end - npu_at) / 64;" in edge
    assert "npu_left" not in edge and "? 255 :" not in edge
    # ... while one block more needs the bounded 255 split.
    large = row_body(_row_fill_asc((1, 16384), "i1 < hi"))
    assert "int64_t npu_left = (npu_end - npu_at) / 64;" in large
    assert "npu_rep = npu_left > 255 ? 255 : npu_left;" in large


def test_memory_collection_rejects_a_tampered_iteration_recipe():
    from xdsl.dialects.builtin import StringAttr

    from ascendflow.mask import ITERATION_CHUNKS

    result = compile(_masked_fill("(cols + q) > rows"), timeout_s=120)
    plan = parse(result.plan_ir)
    task = next(
        operation
        for operation in plan.walk()
        if isinstance(operation, TaskOp) and operation.native.data == "mask_fill"
    )
    assert task.attributes["mask_iteration"].data == "rows"
    task.attributes["mask_iteration"] = StringAttr(ITERATION_CHUNKS)
    with pytest.raises(CompileError, match="iteration recipe"):
        collect_native_memory(plan)


def test_partial_update_preserves_initialized_lanes_and_rejects_uninitialized():
    result = compile(_masked_fill("(cols + q) > rows"), timeout_s=120)
    memory = collect_native_memory(result.plan)
    (masked,) = [t for t in memory.templates if t.spec.native == "mask_fill"]
    assert masked.spec.modes == ("RW",)
    packet = masked.spec.packet
    assert packet is not None and packet.closed_mask and packet.each_mask_call
    assert packet.clobbers == {"VECTOR_MASK", "VECTOR_MASK_MODE"}
    asc = result.asc
    # Full chunks reuse one all-ones bitmap; no zero bitmap is ever issued.
    assert "asc_set_vector_mask(0, UINT64_MAX)" in asc
    assert "asc_set_vector_mask(0, 0)" not in asc
    assert "PIPE_ALL" not in asc
    uninitialized = N.from_source(
        """
        def kernel(Y:N.Tensor((16,256),N.f32,access="write"), q:N.int64):
            with N.Kernel(1, kind="vector"):
                work=N.alloc_ub((16,256),N.f32)
                rows=N.arange(16)[:, None]
                cols=N.arange(256)[None, :]
                N.fill(work,-7.0,mask=(cols + q) > rows)
                N.copy(work,Y)
        """,
        disjoint_external=True,
    )
    with pytest.raises(CompileError, match="initialized writer coverage"):
        compile(uninitialized, timeout_s=120)


def _data_predicate_program():
    return N.from_source(
        """
        def kernel(X:N.Tensor((8,128),N.f32), T:N.Tensor((8,128),N.f32),
                   Y:N.Tensor((8,128),N.f32,access="write")):
            with N.Kernel(1, kind="vector"):
                work=N.alloc_ub((8,128),N.f32)
                other=N.alloc_ub((8,128),N.f32)
                temp=N.alloc_ub((8,128),N.f32)
                bits=N.alloc_mask((8,128))
                bits2=N.alloc_mask((8,128))
                bits3=N.alloc_mask((8,128))
                N.copy(X,work)
                N.copy(T,other)
                p=N.compare(work,0.5,kind="gt",out=bits)
                N.mul(work,2.0,out=work)
                q=N.compare(work,other,kind="lt",out=bits2)
                both=N.mask_and(p,q,out=bits3)
                N.fill(other,3.0)
                N.where(both,work,other,out=work)
                N.copy(work,Y)
        """,
        disjoint_external=True,
    )


def test_data_compare_where_and_mask_logic_keep_actual_packed_reads():
    result = compile(_data_predicate_program(), timeout_s=180)
    memory = collect_native_memory(result.plan)
    natives = [template.spec.native for template in memory.templates]
    assert "compare_scalar_gt" in natives
    assert "compare_lt" in natives
    assert "mask_and" in natives
    assert "select_masked" in natives
    compare = next(t for t in memory.templates if t.spec.native == "compare_scalar_gt")
    select = next(t for t in memory.templates if t.spec.native == "mask_and")
    assert compare.spec.modes == ("R", "RW")
    assert select.spec.modes == ("R", "R", "W")
    packet = next(
        t for t in memory.templates if t.spec.native == "select_masked"
    ).spec.packet
    assert packet is not None and packet.closed_mask
    assert "CMPMASK" in packet.clobbers and packet.internal_fences == ("V",)
    where = next(t.spec for t in memory.templates if t.spec.native == "select_masked")
    assert where.modes == ("R", "R", "R", "W")
    # Packed predicate storage is byte-addressed, never retyped as f32.
    for template in memory.templates:
        if template.spec.native in {"compare_scalar_gt", "compare_lt"}:
            assert template.accesses[-1].address.geometry.dtype == "u8"
    original = collect_native_memory(parse(result.before_ir))
    execution = check_structured_events(memory, logical_reference=original)
    assert execution.bytes_read and execution.bytes_written
    assert execution.hazards
    ordered = _ordered(execution)
    compare_gt = _task(memory, "compare_scalar_gt")
    compare_lt = _task(memory, "compare_lt")
    combine = _task(memory, "mask_and")
    select = _task(memory, "select_masked")
    assert ordered(compare_gt, combine) and ordered(compare_lt, combine)
    assert ordered(combine, select)
    assert ordered(compare_gt, select)


def test_source_overwrite_does_not_change_an_earlier_compare():
    result = compile(_data_predicate_program(), timeout_s=180)
    memory = collect_native_memory(result.plan)
    execution = check_structured_events(
        memory, logical_reference=collect_native_memory(parse(result.before_ir))
    )
    ordered = _ordered(execution)
    first = _task(memory, "compare_scalar_gt")
    overwrite = next(
        template.operation for template in memory.templates if template.spec.native == "muls"
    )
    assert ordered(first, overwrite)
    assert ordered(first, _task(memory, "mask_and"))


def _reuse_program(*, consume_before_reuse):
    first = "N.fill(work,-9.0,mask=p,scratch=temp)" if consume_before_reuse else ""
    return N.from_source(
        f"""
        def kernel(X:N.Tensor((8,128),N.f32), Y:N.Tensor((8,128),N.f32,access="write")):
            with N.Kernel(1, kind="vector"):
                work=N.alloc_ub((8,128),N.f32)
                temp=N.alloc_ub((8,128),N.f32)
                bits=N.alloc_mask((8,128))
                N.copy(X,work)
                p=N.compare(work,0.5,kind="gt",out=bits)
                {first}
                q=N.compare(work,1.5,kind="gt",out=bits)
                N.fill(work,-1.0,mask=q,scratch=temp)
                {"N.fill(work,-9.0,mask=p,scratch=temp)" if not consume_before_reuse else ""}
                N.copy(work,Y)
        """,
        disjoint_external=True,
    )


def test_predicate_storage_reuse_requires_the_old_mask_to_be_dead():
    with pytest.raises(CompileError, match="possible overwrite before its actual native reader"):
        compile(_reuse_program(consume_before_reuse=False), timeout_s=120)
    result = compile(_reuse_program(consume_before_reuse=True), timeout_s=180)
    memory = collect_native_memory(result.plan)
    execution = check_structured_events(
        memory, logical_reference=collect_native_memory(parse(result.before_ir))
    )
    ordered = _ordered(execution)
    compares = [
        template.operation
        for template in memory.templates
        if template.spec.native.startswith("compare_scalar_")
    ]
    selects = [
        template.operation
        for template in memory.templates
        if template.spec.native == "select_masked_update"
    ]
    assert len(compares) == 2 and len(selects) == 2
    assert ordered(compares[0], selects[0])
    assert ordered(selects[0], compares[1])
    assert ordered(compares[1], selects[1])


def test_materialized_coordinate_predicate_combines_coherently():
    program = N.from_source(
        """
        def kernel(X:N.Tensor((8,128),N.f32), Y:N.Tensor((8,128),N.f32,access="write")):
            with N.Kernel(1, kind="vector"):
                work=N.alloc_ub((8,128),N.f32)
                temp=N.alloc_ub((8,128),N.f32)
                bits=N.alloc_mask((8,128))
                cstore=N.alloc_mask((8,128))
                mstore=N.alloc_mask((8,128))
                N.copy(X,work)
                rows=N.arange(8)[:, None]
                cols=N.arange(128)[None, :]
                coord=((cols % 5) == 0) & (cols > rows)
                cmask=N.materialize(coord,out=cstore)
                p=N.compare(work,0.0,kind="gt",out=bits)
                both=N.mask_and(p,cmask,out=mstore)
                N.fill(temp,2.0)
                N.fill(work,-2.0,mask=both,scratch=temp)
                N.copy(work,Y)
        """,
        disjoint_external=True,
    )
    result = compile(program, timeout_s=180)
    memory = collect_native_memory(result.plan)
    natives = [template.spec.native for template in memory.templates]
    assert "mask_bits" in natives and "mask_and" in natives
    bits = next(t for t in memory.templates if t.spec.native == "mask_bits")
    assert bits.spec.engine.value == "S" and bits.spec.modes == ("W",)
    assert bits.spec.packet is not None and bits.spec.packet.closed_mask
    for template in memory.templates:
        for access in template.accesses:
            if access.address.geometry.dtype == "u8":
                assert access.address.geometry.nbytes % 32 == 0
    execution = check_structured_events(
        memory, logical_reference=collect_native_memory(parse(result.before_ir))
    )
    ordered = _ordered(execution)
    assert ordered(_task(memory, "mask_bits"), _task(memory, "mask_and"))
    assert ordered(_task(memory, "compare_scalar_gt"), _task(memory, "mask_and"))


def test_mask_alternates_with_count_packets_without_incoming_state():
    program = N.from_source(
        """
        def kernel(X:N.Tensor((8,128),N.f32), Y:N.Tensor((8,128),N.f32,access="write"), q:N.int64):
            with N.Kernel(1, kind="vector"):
                work=N.alloc_ub((8,128),N.f32)
                N.copy(X,work)
                rows=N.arange(8)[:, None]
                cols=N.arange(128)[None, :]
                N.fill(work,-1.0,mask=(cols + q) > rows)
                N.exp(work,out=work)
                N.fill(work,2.0,mask=(cols < rows))
                N.copy(work,Y)
        """,
        disjoint_external=True,
    )
    result = compile(program, timeout_s=120)
    memory = collect_native_memory(result.plan)
    masked = [t for t in memory.templates if t.spec.native == "mask_fill"]
    exp = [t for t in memory.templates if t.spec.native == "exp"]
    assert len(masked) == 2 and len(exp) == 1
    for template in masked:
        assert template.spec.packet is not None
        assert template.spec.packet.mask_steps[0].value == "set-mode-normal"
        assert template.spec.packet.each_mask_call
    assert exp[0].spec.packet.each_count_call
    # Each packet configures its own state; no packet consumes an incoming mask.
    assert "PIPE_ALL" not in result.asc


@pytest.mark.parametrize(
    "shape,message",
    [((8, 100), "whole 64-lane"), ((3, 40), "whole 64-lane")],
)
def test_non_whole_lane_domains_are_rejected(shape, message):
    rows, cols = shape
    program = N.from_source(
        f"""
        def kernel(X:N.Tensor(({rows},{cols}),N.f32), Y:N.Tensor(({rows},{cols}),N.f32,access="write")):
            with N.Kernel(1, kind="vector"):
                work=N.alloc_ub(({rows},{cols}),N.f32)
                bits=N.alloc_mask(({rows},{cols}))
                N.copy(X,work)
                p=N.compare(work,0.0,kind="gt",out=bits)
                N.copy(work,Y)
        """,
        disjoint_external=True,
    )
    with pytest.raises((CompileError, Exception), match=message):
        compile(program, timeout_s=60)


def test_masked_conversion_and_row_broadcast_are_rejected():
    program = N.from_source(
        """
        def kernel(X:N.Tensor((8,128),N.f32), Y:N.Tensor((8,128),N.f32,access="write")):
            with N.Kernel(1, kind="vector"):
                work=N.alloc_ub((8,128),N.f32)
                other=N.alloc_ub((8,128),N.f16)
                rows=N.arange(8)[:, None]
                cols=N.arange(128)[None, :]
                N.copy(X,work)
                N.cast(work,out=other,mask=(cols + 1) > rows)
                N.copy(work,Y)
        """,
        disjoint_external=True,
    )
    with pytest.raises(CompileError, match="masked row broadcast and conversion"):
        compile(program, timeout_s=60)


def test_where_requires_materialized_predicate_and_defined_branches():
    symbolic = N.from_source(
        """
        def kernel(X:N.Tensor((8,128),N.f32), Y:N.Tensor((8,128),N.f32,access="write")):
            with N.Kernel(1, kind="vector"):
                work=N.alloc_ub((8,128),N.f32)
                other=N.alloc_ub((8,128),N.f32)
                rows=N.arange(8)[:, None]
                cols=N.arange(128)[None, :]
                N.copy(X,work)
                N.copy(X,other)
                N.where((cols > rows),work,other,out=work)
                N.copy(work,Y)
        """,
        disjoint_external=True,
    )
    with pytest.raises(CompileError, match="materialized data predicate"):
        compile(symbolic, timeout_s=60)
    undefined = N.from_source(
        """
        def kernel(X:N.Tensor((8,128),N.f32), Y:N.Tensor((8,128),N.f32,access="write")):
            with N.Kernel(1, kind="vector"):
                work=N.alloc_ub((8,128),N.f32)
                other=N.alloc_ub((8,128),N.f32)
                bits=N.alloc_mask((8,128))
                N.copy(X,work)
                p=N.compare(work,0.0,kind="gt",out=bits)
                N.where(p,work,other,out=work)
                N.copy(work,Y)
        """,
        disjoint_external=True,
    )
    with pytest.raises(CompileError, match="initialized writer coverage"):
        compile(undefined, timeout_s=60)


def test_packed_compare_and_not_are_declared_vector_packets():
    result = compile(
        N.from_source(
            """
            def kernel(X:N.Tensor((8,128),N.f32), Y:N.Tensor((8,128),N.f32,access="write")):
                with N.Kernel(1, kind="vector"):
                    work=N.alloc_ub((8,128),N.f32)
                    temp=N.alloc_ub((8,128),N.f32)
                    bits=N.alloc_mask((8,128))
                    nbits=N.alloc_mask((8,128))
                    N.copy(X,work)
                    p=N.compare(work,0.5,kind="gt",out=bits)
                    n=N.mask_not(p,out=nbits)
                    N.fill(work,-1.0,mask=n,scratch=temp)
                    N.copy(work,Y)
            """,
            disjoint_external=True,
        ),
        timeout_s=120,
    )
    memory = collect_native_memory(result.plan)
    zero = _task(memory, "mask_zero")
    compare = _task(memory, "compare_scalar_gt")
    negative = _task(memory, "mask_not")
    selection = _task(memory, "select_masked_update")
    templates = {t.operation: t for t in memory.templates}
    zero_template, compare_template = templates[zero], templates[compare]
    negative_template = templates[negative]
    # Padding is defined by a real count-mode Vector Dup, not a hidden store.
    assert zero_template.spec.engine is Engine.V
    assert zero_template.spec.modes == ("W",)
    assert zero_template.spec.packet.closed_mask
    assert zero_template.spec.packet.each_count_call
    assert zero_template.spec.packet.mask_steps == (
        MaskStep.SET_COUNTER,
        MaskStep.SET_COUNT,
        MaskStep.VECTOR_READ,
        MaskStep.RESTORE_NORMAL,
    )
    (out,) = zero_template.accesses
    assert out.address.geometry.total_nbytes == 16 * 32
    # Packed compare is still the audited mask-free Vector packet.
    assert compare_template.spec.engine is Engine.V
    assert compare_template.spec.modes == ("R", "RW")
    assert compare_template.spec.packet.mask_steps == ()
    assert compare_template.spec.packet.internal_fences == ()
    # Packed NOT is a real count-mode Vector native, matching its packet.
    assert negative_template.spec.engine is Engine.V
    assert negative_template.spec.modes == ("R", "W")
    assert negative_template.spec.packet.each_count_call
    assert negative_template.spec.packet.mask_steps == zero_template.spec.packet.mask_steps
    for template in (compare_template, negative_template):
        (last,) = template.accesses[-1:]
        assert last.address.geometry.total_nbytes == 16 * 32
    asc = result.asc
    # No scalar UB stores or hidden word building survive in the emitted code.
    assert "*(__ubuf__ uint64_t*)" not in asc
    assert "npu_bits" not in asc
    assert "UINT64_C(0)" not in asc
    zero_call = asc.index("asc_duplicate((__ubuf__ uint16_t*)")
    compare_call = asc.index("asc_gt_scalar(")
    fence = asc.index("asc_sync_pipe(PIPE_V)", zero_call)
    not_call = asc.index("asc_not(")
    assert zero_call < fence < compare_call < not_call
    # An independent finite event model proves the needed orderings.
    execution = check_structured_events(
        memory, logical_reference=collect_native_memory(parse(result.before_ir))
    )
    ordered = _ordered(execution)
    assert ordered(zero, compare)
    assert ordered(compare, negative)
    assert ordered(negative, selection)
    hazards = {
        (hazard.kind, hazard.source.operation, hazard.consumer.operation)
        for hazard in execution.hazards
    }
    assert ("WAW", zero, compare) in hazards
    assert ("RAW", compare, negative) in hazards
    assert ("RAW", negative, selection) in hazards


def test_materialized_not_has_unary_dependency_and_distinct_storage():
    program = N.from_source(
        """
        def kernel(X:N.Tensor((8,128),N.f32), Y:N.Tensor((8,128),N.f32,access="write")):
            with N.Kernel(1, kind="vector"):
                work=N.alloc_ub((8,128),N.f32)
                temp=N.alloc_ub((8,128),N.f32)
                bits=N.alloc_mask((8,128))
                nbits=N.alloc_mask((8,128))
                N.copy(X,work)
                p=N.compare(work,0.0,kind="gt",out=bits)
                n=N.mask_not(p,out=nbits)
                N.fill(work,7.0,mask=n,scratch=temp)
                N.copy(work,Y)
        """,
        disjoint_external=True,
    )
    result = compile(program, timeout_s=120)
    memory = collect_native_memory(result.plan)
    negative = _task(memory, "mask_not")
    template = next(t for t in memory.templates if t.operation is negative)
    assert len(template.accesses) == 2
    assert template.accesses[0].mode.reads and template.accesses[1].mode.writes
    result_ir = parse(result.source_ir)
    combine = [op for op in result_ir.walk() if op.name == "npu_source.mask_combine"]
    assert len(combine) == 1 and combine[0].kind.data == "not"
    assert combine[0].rhs is None and combine[0].rhs_storage is None
    assert len(combine[0].dependency_inputs) == 1
    with pytest.raises(CompileError, match="distinct predicate storage"):
        compile(
            N.from_source(
                """
                def kernel(X:N.Tensor((8,128),N.f32), Y:N.Tensor((8,128),N.f32,access="write")):
                    with N.Kernel(1, kind="vector"):
                        work=N.alloc_ub((8,128),N.f32)
                        bits=N.alloc_mask((8,128))
                        N.copy(X,work)
                        p=N.compare(work,0.0,kind="gt",out=bits)
                        N.mask_not(p,out=bits)
                        N.copy(work,Y)
                """,
                disjoint_external=True,
            ),
            timeout_s=60,
        )


def test_exact_integer_limits_reject_unsafe_forms_and_emit_floor_division():
    def fill_expression(expression):
        return N.from_source(
            f"""
            def kernel(X:N.Tensor((8,128),N.f32), Y:N.Tensor((8,128),N.f32,access="write"),
                       q:N.int64):
                with N.Kernel(1, kind="vector"):
                    work=N.alloc_ub((8,128),N.f32)
                    N.copy(X,work)
                    rows=N.arange(8)[:, None]
                    cols=N.arange(128)[None, :]
                    N.fill(work,-1.0,mask={expression})
                    N.copy(work,Y)
            """,
            disjoint_external=True,
        )

    extreme = compile(
        fill_expression("(cols + 9223372036854775807) > (rows + (-9223372036854775807 - 1))"),
        timeout_s=120,
    )
    # These exact extremes prove every lane true, so the admitted whole-tile
    # coverage replays one repeated full-mask run and evaluates no word.
    assert "npu_mask_monotone((__int128)(" not in extreme.asc
    assert all(mask_values(extreme, {})[-1][1].values())
    # A runtime atom keeps the tile mixed, so the extreme int64 literal is
    # still emitted inside the exact 128-bit word arithmetic.
    mixed_extreme = compile(
        fill_expression(
            "((cols + q) + (-9223372036854775807 - 1)) > (rows + (-9223372036854775807 - 1))"
        ),
        timeout_s=120,
    )
    assert "(-9223372036854775807LL - 1)" in mixed_extreme.asc
    assert "npu_mask_monotone(" in mixed_extreme.asc
    with pytest.raises(CompileError, match="int64"):
        compile(fill_expression("(cols * 9223372036854775808) > rows"), timeout_s=60)
    with pytest.raises(CompileError, match="lane coefficient"):
        compile(
            fill_expression("(cols * 4611686018427387904) > rows"),
            timeout_s=60,
        )
    with pytest.raises(CompileError, match="positive int64 divisor"):
        compile(fill_expression("(cols // 9223372036854775808) == 0"), timeout_s=60)
    negative = compile(
        fill_expression("((rows - 5) % 2) == 1"),
        timeout_s=120,
    )
    assert "npu_mask_floormod(" in negative.asc
    truth = mask_values(negative, {})[-1][1]
    assert all(value == ((r - 5) % 2 == 1) for (r, _c), value in truth.items())
    floor = compile(
        fill_expression("((cols - 7) // 5) == -1"),
        timeout_s=120,
    )
    assert "npu_mask_floordiv(" in floor.asc
    truth = mask_values(floor, {})[-1][1]
    assert all(value == ((c - 7) // 5 == -1) for (_r, c), value in truth.items())


def test_scalar_and_periodic_combinations_accept_both_operand_orders(tmp_path):
    for expression in (
        "((q + rows) > 0) & (((cols // 8) % 2) == 0)",
        "(((cols // 8) % 2) == 0) & ((q + rows) > 0)",
        "(((-cols) % 5) == 0) & (rows >= 0)",
    ):
        result = compile(_masked_fill(expression), timeout_s=120)
        truth = mask_values(result, {"q": 3, "lo": 0, "hi": 0})[-1][1]
        for (r, c), value in truth.items():
            if "// 8" in expression:
                scalar_first = "(q + rows) > 0" in expression.split("&")[0]
                expected = (
                    (3 + r) > 0 and (c // 8) % 2 == 0
                    if scalar_first
                    else (c // 8) % 2 == 0 and (3 + r) > 0
                )
            elif "% 5" in expression:
                expected = (-c) % 5 == 0 and r >= 0
            else:
                expected = False
            assert value == expected, (expression, r, c)


HOST_FLOOR = r"""
#include <cstdint>
#include <cstdio>
#define __aicore__
%(helpers)s
static __int128 npu_parse(const char* text) {
  bool negative = *text == '-';
  if (negative) ++text;
  __int128 value = 0;
  while (*text) { value = value * 10 + (*text - '0'); ++text; }
  return negative ? -value : value;
}
int main() {
  char a[64], b[64];
  int op;
  while (std::scanf("%63s %63s %d", a, b, &op) == 3) {
    __int128 va = npu_parse(a), vb = npu_parse(b);
    __int128 value = op == 0 ? npu_mask_floordiv(va, vb) : npu_mask_floormod(va, vb);
    std::printf("%lld %llu\n", (long long)(value >> 64), (unsigned long long)value);
  }
  return 0;
}
"""


def test_emitted_floor_division_matches_python_for_negative_numerators(tmp_path):
    source = tmp_path / "floor_helper.cpp"
    binary = tmp_path / "floor_helper"
    source.write_text(HOST_FLOOR.replace("%(helpers)s", "\n".join(helper_lines())))
    subprocess.run(
        ["g++", "-O1", "-std=c++17", str(source), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
    )
    values = [-(1 << 100), -(1 << 63), -101, -7, -1, 0, 1, 7, 101, (1 << 63) - 1]
    divisors = [1, 2, 3, 5, 7, 64, (1 << 62)]
    cases = [(a, b, op) for a in values for b in divisors for op in (0, 1)]
    lines = "\n".join(f"{a} {b} {op}" for a, b, op in cases)
    completed = subprocess.run(
        [str(binary)], input=lines, check=True, capture_output=True, text=True
    )
    rows = [line.split() for line in completed.stdout.splitlines()]
    assert len(rows) == len(cases)
    for (a, b, op), (hi, lo) in zip(cases, rows, strict=True):
        unsigned = (int(hi) << 64) | int(lo)
        value = unsigned - (1 << 128) if unsigned >= (1 << 127) else unsigned
        expected = a // b if op == 0 else a % b
        assert value == expected, (a, b, op)


def test_masked_scratch_must_not_alias_destination_allocation():
    alias = N.from_source(
        """
        def kernel(X:N.Tensor((8,128),N.f32), Y:N.Tensor((8,128),N.f32,access="write")):
            with N.Kernel(1, kind="vector"):
                work=N.alloc_ub((8,128),N.f32)
                bits=N.alloc_mask((8,128))
                N.copy(X,work)
                p=N.compare(work,0.0,kind="gt",out=bits)
                N.fill(work,-1.0,mask=p,scratch=work)
                N.copy(work,Y)
        """,
        disjoint_external=True,
    )
    with pytest.raises(CompileError, match="must not alias"):
        compile(alias, timeout_s=60)
    aliasing_view = N.from_source(
        """
        def kernel(X:N.Tensor((8,128),N.f32), Y:N.Tensor((8,128),N.f32,access="write")):
            with N.Kernel(1, kind="vector"):
                work=N.alloc_ub((8,128),N.f32)
                bits=N.alloc_mask((8,128))
                N.copy(X,work)
                p=N.compare(work,0.0,kind="gt",out=bits)
                N.fill(work,-1.0,mask=p,scratch=work[0:8,0:128])
                N.copy(work,Y)
        """,
        disjoint_external=True,
    )
    with pytest.raises(CompileError, match="must not alias"):
        compile(aliasing_view, timeout_s=60)


def test_selected_word_recipe_and_repeat_policy_are_recorded_before_sync():
    affine = compile(_masked_fill("(cols + q) > rows"), timeout_s=120)
    lane = compile(_masked_fill("((cols // 8) % 2) == 0"), timeout_s=120)
    for result, word in ((affine, "affine64"), (lane, "lane64")):
        memory = collect_native_memory(result.plan)
        tasks = [t for t in memory.templates if t.spec.native == "mask_fill"]
        assert len(tasks) == 1
        attributes = tasks[0].operation.attributes
        assert attributes["mask_word"].data == word
        assert attributes["mask_repeat"].data == "full_runs_255"
    assert "npu_mask_monotone(" in affine.asc
    assert "npu_mask_word |= (UINT64_C(1) << npu_lane)" in lane.asc


def test_generic_mask_example_oracle_matches_source_predicates():
    import numpy as np

    from benchmarks.vector import run_masks
    from examples.vector.masks import masks

    result = compile(masks.specialize(ROWS=8, COLUMNS=128), timeout_s=180)
    rows, columns = 8, 128
    x, z = run_masks.field_tensors(rows, columns)
    for q, lo, hi in ((0, 0, 0), (7, 32, 96), ((1 << 63) - 1, 0, 0)):
        expected0, expected1, expected2 = run_masks.oracle(x, z, q, lo, hi)
        # Pure per-lane Python/NumPy logical reference, not the compiler.
        for r in range(rows):
            for c in range(columns):
                interval = lo <= c < hi
                periodic = (c // 64) % 2 == 0
                origin = (c + q) > r
                fill = (interval != periodic) and origin
                update = (x[r, c] * np.float32(2) if interval else x[r, c]) + np.float32(1)
                assert expected0[r, c] == (np.float32(-3) if fill else update)
                p = x[r, c] > np.float32(0.25)
                data = x[r, c] * np.float32(5)
                q_pred = data > z[r, c]
                both = bool(p) and bool(q_pred)
                either = both or not p
                selected = data if either else z[r, c]
                assert expected1[r, c] == selected
                assert expected2[r, c] == (np.float32(-9) if both else selected)
        module = parse(result.source_ir)
        fill_op = next(
            op for op in module.walk() if op.name == "npu_source.element" and op.kind.data == "fill"
        )
        program = fill_op.mask.owner.program()
        assert program.shape == (rows, columns)
        fill_mask = {
            index: _evaluate(program, fill_op.mask.owner, index, {"q": q, "lo": lo, "hi": hi}, {})
            for index in product(range(rows), range(columns))
        }
        for (r, c), value in fill_mask.items():
            interval = lo <= c < hi
            periodic = (c // 64) % 2 == 0
            origin = (c + q) > r
            assert value == ((interval != periodic) and origin)
    assert run_masks._parse_cases(["0:0:0", f"{(1 << 63) - 1}:3:9"])
    with pytest.raises(CompileError, match="nonnegative signed int64"):
        run_masks._parse_cases(["-1:0:0"])
    with pytest.raises(CompileError, match="q:lo:hi"):
        run_masks._parse_cases(["1:2"])


def test_compare_padding_comes_from_a_declared_vector_task_and_pipe_fence():
    result = compile(_data_predicate_program(), timeout_s=180)
    asc = result.asc
    zero = asc.index("asc_duplicate((__ubuf__ uint16_t*)")
    compare = asc.index("asc_gt_scalar(")
    fence = asc.index("asc_sync_pipe(PIPE_V)", zero)
    assert zero < fence < compare
    assert "*(__ubuf__ uint64_t*)" not in asc
    memory = collect_native_memory(result.plan)
    zero_template = next(t for t in memory.templates if t.spec.native == "mask_zero")
    template = next(t for t in memory.templates if t.spec.native == "compare_scalar_gt")
    assert zero_template.spec.engine is Engine.V
    assert zero_template.spec.modes == ("W",)
    assert zero_template.accesses[0].address.geometry.total_nbytes == 16 * 32
    # The padded logical Tile is declared as a complete write by the recipe.
    assert template.accesses[-1].mode.writes
    assert template.accesses[-1].address.geometry.total_nbytes == 16 * 32


def test_standalone_compare_cannot_authorize_uninitialized_packed_reads():
    source = N.from_source(
        """
        def kernel(X:N.Tensor((8,128),N.f32), Y:N.Tensor((8,128),N.f32,access="write")):
            with N.Kernel(1, kind="vector"):
                work=N.alloc_ub((8,128),N.f32)
                other=N.alloc_ub((8,128),N.f32)
                temp=N.alloc_ub((8,128),N.f32)
                bits=N.alloc_mask((8,128))
                N.copy(X,work)
                N.copy(X,other)
                p=N.compare(work,0.0,kind="gt",out=bits)
                N.where(p,work,other,out=temp)
                N.copy(temp,Y)
        """,
        disjoint_external=True,
    )
    structure = lower_native_structure(source)
    # The declared mask_zero writer satisfies ordinary read admission.
    memory = collect_native_memory(structure.module)
    admit_structured_memory(structure, memory, admit_memory_bounds(memory))
    # Remove only that initializer; the partial compare's RW destination read
    # must not become a fresh whole-block initialization fact by itself.
    zero = next(
        operation
        for operation in structure.module.walk()
        if isinstance(operation, TaskOp) and operation.native.data == "mask_zero"
    )
    zero.detach()
    partial = collect_native_memory(structure.module)
    partial_template = next(
        template
        for template in partial.templates
        if template.spec.native == "compare_scalar_gt"
    )
    assert partial_template.spec.modes == ("R", "RW")
    with pytest.raises(CompileError, match="initialized writer coverage"):
        admit_structured_memory(structure, partial, admit_memory_bounds(partial))


def test_memory_collection_rejects_a_tampered_word_recipe():
    from xdsl.dialects.builtin import StringAttr

    from ascendflow.mask import WORD_AFFINE, WORD_LANE

    result = compile(_masked_fill("(cols + q) > rows"), timeout_s=120)
    plan = parse(result.plan_ir)
    task = next(
        operation
        for operation in plan.walk()
        if isinstance(operation, TaskOp) and operation.native.data == "mask_fill"
    )
    # Tamper with the value the admitted predicate did not select.
    recorded = task.attributes["mask_word"].data
    assert recorded in {WORD_AFFINE, WORD_LANE}
    task.attributes["mask_word"] = StringAttr(WORD_LANE if recorded == WORD_AFFINE else WORD_AFFINE)
    with pytest.raises(CompileError, match="word recipe"):
        collect_native_memory(plan)
