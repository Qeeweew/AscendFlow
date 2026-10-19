"""Host-only C++ execution of emitted mask arithmetic, without CANN or a device."""

from __future__ import annotations

import subprocess
from itertools import product
from math import prod

from ascendflow.backend.vector_mask import helper_lines
from ascendflow.target.ascend910b import Engine
from ascendflow.target.native import NativeGeometry, NativeSpec
from tests.models.mask_predicates import _evaluate

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
