"""ASC emission of lane-predicate programs and packed predicate recipes.

The emitter replays the exact predicate program and packed-block facts that
native admission consumed; it selects no synchronization, allocates no
storage and never rewrites a mask into a global barrier. Scalar packed word
building (`mask_bits`) is its own declared S native; packed zeroing, packed
compare and packed NOT/AND/OR are declared Vector natives. No emitter
path hides one engine's work inside another engine's task.

A whole 64-lane block is either empty (skip), full (one repeat bitmap reused
across consecutive full blocks) or partial (its own 64-bit bitmap). A zero
bitmap is never issued to a vector instruction.

Two admitted facts keep the generated scalar work small. A proven `full` or
`empty` whole-tile coverage replays one repeated full-mask run or no element
call at all; only `mixed` tiles replay the per-block word loop. A unit lane
coefficient (+-1) builds its word from one clamped 128-bit interval instead of
a search; other constant coefficients keep the general monotone fallback.
"""

from __future__ import annotations

from math import isinf, prod

from ..diagnostics import require
from ..mask import (
    ADD,
    AND,
    BROADCAST,
    CMP,
    CMP_KINDS,
    CONST,
    DIVC,
    ITERATION_ROWS,
    ITERATION_VALUES,
    LANE,
    MASK_REPEAT_POLICY,
    MODC,
    MULC,
    NEG,
    NOT,
    OR,
    SCALAR,
    SUB,
    VALUE_OPS,
    WORD_AFFINE,
    WORD_LANE,
    affine_axes,
    comparison_root,
)

# Comparison kind after negating both sides of a unit-coefficient predicate.
_FLIP_KIND = {0: 0, 1: 1, 2: 4, 3: 5, 4: 2, 5: 3}


def _int_literal(value):
    """An exact C++ signed 64-bit literal, including INT64_MIN."""
    if value == -(1 << 63):
        return "(-9223372036854775807LL - 1)"
    require(
        -(1 << 63) < value < (1 << 63),
        "emitted integer literal exceeds the exact signed int64 range",
    )
    return f"{value}LL"


def needs_helpers(templates):
    return any("nodes" in template.operation.attributes for template in templates)


def helper_lines():
    """Exact 128-bit predicate helpers shared by the emitted kernel."""
    return [
        "static __aicore__ inline uint64_t npu_mask_interval(int64_t lo, int64_t hi) {",
        "  if (lo < 0) lo = 0;",
        "  if (hi > 64) hi = 64;",
        "  if (lo >= hi) return UINT64_C(0);",
        "  uint64_t below = lo == 0 ? UINT64_C(0) : ((UINT64_C(1) << lo) - 1);",
        "  uint64_t upto = hi >= 64 ? UINT64_MAX : ((UINT64_C(1) << hi) - 1);",
        "  return upto & ~below;",
        "}",
        "static __aicore__ inline uint64_t npu_mask_unit(__int128 b, int32_t kind) {",
        "  // Unit lane coefficient: value(lane) = lane + b. The threshold is",
        "  // clamped inside signed 128 bits, before any int64 conversion or",
        "  // shift, so INT64_MIN/INT64_MAX operands stay exact. Values below -2",
        "  // and above 65 are indistinguishable for every comparison kind.",
        "  __int128 t = -b;",
        "  if (t < -2) t = -2;",
        "  if (t > 65) t = 65;",
        "  const int64_t lane = static_cast<int64_t>(t);",
        "  const bool inside = lane >= 0 && lane < 64;",
        "  if (kind == 0) return inside ? (UINT64_C(1) << lane) : UINT64_C(0);",
        "  if (kind == 1) return inside ? (UINT64_MAX & ~(UINT64_C(1) << lane)) : UINT64_MAX;",
        "  if (kind == 2) return npu_mask_interval(0, lane);",
        "  if (kind == 3) return npu_mask_interval(0, lane + 1);",
        "  if (kind == 4) return npu_mask_interval(lane + 1, 64);",
        "  return npu_mask_interval(lane, 64);",
        "}",
        "static __aicore__ inline __int128 npu_mask_floordiv(__int128 a, __int128 b) {",
        "  __int128 q = a / b;",
        "  __int128 r = a % b;",
        "  return q - ((r != 0) && ((r < 0) != (b < 0)));",
        "}",
        "static __aicore__ inline __int128 npu_mask_floormod(__int128 a, __int128 b) {",
        "  __int128 r = a % b;",
        "  return r == 0 ? r : (((r < 0) != (b < 0)) ? r + b : r);",
        "}",
        "static __aicore__ inline int64_t npu_mask_clamp(__int128 value, int64_t extent) {",
        "  if (value < 0) return 0;",
        "  if (value > (__int128)(extent)) return extent;",
        "  return static_cast<int64_t>(value);",
        "}",
        "static __aicore__ inline uint64_t npu_mask_monotone(__int128 a, __int128 b, int32_t kind) {",
        "  if (a == 0) {",
        "    bool truth = kind == 0 ? b == 0 : kind == 1 ? b != 0 : kind == 2 ? b < 0 :",
        "        kind == 3 ? b <= 0 : kind == 4 ? b > 0 : b >= 0;",
        "    return truth ? UINT64_MAX : UINT64_C(0);",
        "  }",
        "  if (a < 0) {",
        "    a = -a;",
        "    b = -b;",
        "    kind = kind == 2 ? 4 : kind == 4 ? 2 : kind == 3 ? 5 : kind == 5 ? 3 : kind;",
        "  }",
        "  if (a == 1) return npu_mask_unit(b, kind);",
        "  if (kind == 0 || kind == 1) {",
        "    if (b % a == 0) {",
        "      __int128 lane = -b / a;",
        "      if (lane >= 0 && lane < 64) {",
        "        uint64_t bit = UINT64_C(1) << static_cast<int64_t>(lane);",
        "        return kind == 0 ? bit : (UINT64_MAX & ~bit);",
        "      }",
        "    }",
        "    return kind == 0 ? UINT64_C(0) : UINT64_MAX;",
        "  }",
        "  int64_t lo = 0, hi = 64;",
        "  while (lo < hi) {",
        "    int64_t mid = (lo + hi) / 2;",
        "    __int128 value = a * mid + b;",
        "    bool truth = kind == 2 ? value < 0 : kind == 3 ? value <= 0 :",
        "        kind == 4 ? value > 0 : value >= 0;",
        "    if (truth == (kind == 2 || kind == 3)) lo = mid + 1; else hi = mid;",
        "  }",
        "  return (kind == 2 || kind == 3) ? npu_mask_interval(0, lo) : npu_mask_interval(lo, 64);",
        "}",
    ]


def _axis_index(program, axis, flat):
    extent = program.shape[axis]
    if extent == 1:
        return "0"
    stride = prod(program.shape[axis + 1 :])
    return f"((({flat}) / {stride}) % {extent})"


def _chunk_aligned(program):
    return all(prod(program.shape[axis + 1 :]) % 64 == 0 for axis in range(len(program.shape) - 1))


def _linear(program, prefix, chunk):
    """Per-node lane coefficient plus a 128-bit base expression."""
    coefficients, bases, values = {}, {}, {}
    for index, node in enumerate(program.nodes):
        if node.op not in VALUE_OPS:
            continue
        if node.op == CONST:
            coefficients[index] = 0
            bases[index] = f"((__int128)({_int_literal(node.args[0])}))"
        elif node.op == SCALAR:
            coefficients[index] = 0
            bases[index] = f"((__int128)({prefix}scalar_{node.args[0]}))"
        elif node.op == LANE:
            stored = program.node_shapes[index]
            target = len(program.shape) - len(stored) + node.args[0]
            if target == len(program.shape) - 1 and stored[node.args[0]] > 1:
                coefficients[index] = 1
                extent = program.shape[target]
                bases[index] = f"((__int128)((({chunk}) * 64) % {extent}))"
            else:
                coefficients[index] = 0
                bases[index] = f"((__int128)({_axis_index(program, target, f'({chunk}) * 64')}))"
        elif node.op == NEG:
            coefficients[index] = -coefficients[node.args[0]]
            bases[index] = f"(-{bases[node.args[0]]})"
        elif node.op == MULC:
            factor = node.args[1]
            coefficients[index] = coefficients[node.args[0]] * factor
            bases[index] = f"({bases[node.args[0]]} * (__int128)({_int_literal(factor)}))"
        elif node.op == ADD:
            left, right = node.args
            coefficients[index] = coefficients[left] + coefficients[right]
            bases[index] = f"({bases[left]} + {bases[right]})"
        else:
            assert node.op == SUB
            left, right = node.args
            coefficients[index] = coefficients[left] - coefficients[right]
            bases[index] = f"({bases[left]} - {bases[right]})"
        values[index] = index
    return coefficients, bases, values


def _scalar_lines(operands):
    """Loop-invariant uniform index atoms, declared once per native."""
    return [
        f"const int64_t npu_m_scalar_{index} = (int64_t)({text});"
        for index, text in enumerate(operands)
    ]


def _word_lines(program, chunk, strategy):
    """Replay the selected word recipe for one 64-lane block.

    The caller declares the uniform index atoms once per native; this body is
    replayed for every 64-lane block (`chunk` is the block index expression).
    """
    require(
        strategy in {WORD_AFFINE, WORD_LANE},
        "masked native lost its selected word recipe",
    )
    prefix = "npu_m_"
    lines = []
    if strategy == WORD_AFFINE:
        require(
            not program.is_periodic and _chunk_aligned(program),
            "affine mask-word recipe cannot represent this predicate",
        )
        coefficients, bases, values = _linear(program, prefix, chunk)
        for index in values:
            lines.append(f"const __int128 {prefix}v{index} = {bases[index]};")
        words = {}
        for index, node in enumerate(program.nodes):
            if node.op == CMP:
                left, right, kind = node.args
                difference = coefficients[left] - coefficients[right]
                value = f"{prefix}v{values[left]} - {prefix}v{values[right]}"
                if difference in {1, -1}:
                    # Unit lane coefficient: exact O(1) interval word, no
                    # binary search. A -1 coefficient normalizes by negating
                    # the exact 128-bit difference and flipping the kind.
                    if difference == 1:
                        call = f"npu_mask_unit({value}, {kind})"
                    else:
                        call = f"npu_mask_unit(-({value}), {_FLIP_KIND[kind]})"
                else:
                    call = (
                        f"npu_mask_monotone((__int128)({_int_literal(difference)}), "
                        f"{value}, {kind})"
                    )
                words[index] = f"const uint64_t {prefix}b{index} = {call};"
            elif node.op in {AND, OR}:
                operator = "&" if node.op == AND else "|"
                words[index] = (
                    f"const uint64_t {prefix}b{index} = "
                    f"{prefix}b{node.args[0]} {operator} {prefix}b{node.args[1]};"
                )
            elif node.op == NOT:
                words[index] = (
                    f"const uint64_t {prefix}b{index} = UINT64_MAX & ~{prefix}b{node.args[0]};"
                )
            elif node.op == BROADCAST:
                words[index] = f"const uint64_t {prefix}b{index} = {prefix}b{node.args[0]};"
        lines.extend(words[index] for index in sorted(words))
        lines.append(f"const uint64_t npu_mask_word = {prefix}b{len(program.nodes) - 1};")
        return lines
    lines.append("uint64_t npu_mask_word = 0;")
    lines.append("for (int64_t npu_lane = 0; npu_lane < 64; ++npu_lane) {")
    lines.append(f"  const int64_t {prefix}flat = ({chunk}) * 64 + npu_lane;")
    for axis in range(len(program.shape) - 1):
        lines.append(
            f"  const int64_t {prefix}ax{axis} = {_axis_index(program, axis, prefix + 'flat')};"
        )
    values, booleans = {}, {}
    for index, node in enumerate(program.nodes):
        if node.op == CONST:
            values[index] = f"const __int128 {prefix}v{index} = ({_int_literal(node.args[0])});"
        elif node.op == SCALAR:
            values[index] = f"const __int128 {prefix}v{index} = {prefix}scalar_{node.args[0]};"
        elif node.op == LANE:
            stored = program.node_shapes[index]
            target = len(program.shape) - len(stored) + node.args[0]
            if program.shape[target] == 1:
                atom = "0"
            elif target == len(program.shape) - 1:
                atom = f"({prefix}flat % {program.shape[target]})"
            else:
                atom = f"{prefix}ax{target}"
            values[index] = f"const __int128 {prefix}v{index} = {atom};"
        elif node.op in {ADD, SUB}:
            operator = "+" if node.op == ADD else "-"
            values[index] = (
                f"const __int128 {prefix}v{index} = "
                f"{prefix}v{node.args[0]} {operator} {prefix}v{node.args[1]};"
            )
        elif node.op == MULC:
            values[index] = (
                f"const __int128 {prefix}v{index} = "
                f"{prefix}v{node.args[0]} * (__int128)({_int_literal(node.args[1])});"
            )
        elif node.op == NEG:
            values[index] = f"const __int128 {prefix}v{index} = -{prefix}v{node.args[0]};"
        elif node.op in {DIVC, MODC}:
            function = "npu_mask_floordiv" if node.op == DIVC else "npu_mask_floormod"
            values[index] = (
                f"const __int128 {prefix}v{index} = {function}("
                f"{prefix}v{node.args[0]}, (__int128)({_int_literal(node.args[1])}));"
            )
        elif node.op == CMP:
            operator = {
                "eq": "==",
                "ne": "!=",
                "lt": "<",
                "le": "<=",
                "gt": ">",
                "ge": ">=",
            }[CMP_KINDS[node.args[2]]]
            booleans[index] = (
                f"const bool {prefix}b{index} = "
                f"{prefix}v{node.args[0]} {operator} {prefix}v{node.args[1]};"
            )
        elif node.op in {AND, OR}:
            operator = "&&" if node.op == AND else "||"
            booleans[index] = (
                f"const bool {prefix}b{index} = "
                f"{prefix}b{node.args[0]} {operator} {prefix}b{node.args[1]};"
            )
        elif node.op == BROADCAST:
            booleans[index] = f"const bool {prefix}b{index} = {prefix}b{node.args[0]};"
        else:
            assert node.op == NOT
            booleans[index] = f"const bool {prefix}b{index} = !{prefix}b{node.args[0]};"
    lines.extend("  " + line for line in values.values())
    lines.extend("  " + line for line in booleans.values())
    lines.append(f"  if ({prefix}b{len(program.nodes) - 1}) {{")
    lines.append("    npu_mask_word |= (UINT64_C(1) << npu_lane);")
    lines.append("  }")
    lines.append("}")
    return lines


def _byte_pointer(pointer, offset, dtype="float"):
    if not offset:
        return pointer
    return f"(__ubuf__ {dtype}*)((__ubuf__ uint8_t*)({pointer}) + ({offset}))"


def _scalar_value(value):
    if isinf(value):
        return ("-" if value < 0 else "") + "__builtin_inff()"
    return f"float({value!r})"


def _element_call(native, pointers, scalar, repeat, offset):
    refs = [_byte_pointer(pointer, offset) for pointer in pointers]
    if native == "mask_fill":
        return f"asc_duplicate({refs[0]}, {scalar}, uint8_t({repeat}), 1, 8);"
    if native == "mask_muls":
        return f"asc_mul_scalar({refs[1]}, {refs[0]}, {scalar}, uint8_t({repeat}), 1, 1, 8, 8);"
    if native == "mask_adds":
        return f"asc_add_scalar({refs[1]}, {refs[0]}, {scalar}, uint8_t({repeat}), 1, 1, 8, 8);"
    if native in {"mask_exp", "mask_sqrt", "mask_rsqrt"}:
        function = native.removeprefix("mask_")
        return f"asc_{function}({refs[1]}, {refs[0]}, uint8_t({repeat}), 1, 1, 8, 8);"
    function = {
        "mask_add": "add",
        "mask_mul": "mul",
        "mask_sub": "sub",
        "mask_div": "div",
        "mask_max": "max",
    }[native]
    return f"asc_{function}({refs[2]}, {refs[0]}, {refs[1]}, uint8_t({repeat}), 1, 1, 1, 8, 8, 8);"


def _flush_lines(native, pointers, scalar, start, stop, indent):
    """Emit bounded repeat calls for a run of all-ones blocks."""
    pad = " " * indent
    return [
        f"{pad}int64_t npu_left = ({stop}) - ({start});",
        f"{pad}int64_t npu_at = ({start});",
        f"{pad}while (npu_left > 0) {{",
        f"{pad}  int64_t npu_rep = npu_left > 255 ? 255 : npu_left;",
        f"{pad}  asc_set_vector_mask(0, UINT64_MAX);",
        f"{pad}  {_element_call(native, pointers, scalar, 'npu_rep', 'npu_at * 256')}",
        f"{pad}  npu_at += npu_rep;",
        f"{pad}  npu_left -= npu_rep;",
        f"{pad}}}",
    ]


def _tile_forms(program, prefix):
    """Exact per-node affine form over the whole tile.

    Each value node yields a constant text (compile-time integers plus the
    uniform scalar atoms) and one compile-time lane coefficient per tile axis,
    exactly the bounded expression the admitted program denotes. A periodic
    node has no affine form; the affine recipe is never selected with one.
    """
    require(not program.is_periodic, "affine tile bounds cannot represent a periodic lane")
    axes = affine_axes(program)
    source = {}
    for index, node in enumerate(program.nodes):
        op, args = node.op, node.args
        if op not in VALUE_OPS:
            continue
        if op == CONST:
            source[index] = f"(__int128)({_int_literal(args[0])})"
        elif op == SCALAR:
            source[index] = f"(__int128)({prefix}scalar_{args[0]})"
        elif op == LANE:
            source[index] = "(__int128)(0)"
        elif op == NEG:
            source[index] = f"(-{source[args[0]]})"
        elif op == MULC:
            source[index] = f"({source[args[0]]} * (__int128)({_int_literal(args[1])}))"
        elif op == ADD:
            source[index] = f"({source[args[0]]} + {source[args[1]]})"
        else:
            assert op == SUB
            source[index] = f"({source[args[0]]} - {source[args[1]]})"
    return source, axes


def _tile_bound_lines(program, prefix):
    """Runtime conservative (all, none) truth of the whole tile.

    Per value node the emitted bounds are exact for the current scalar atoms:
    a positive lane coefficient contributes at most `coefficient * (extent-1)`
    and a negative one at least that. Each node stays within the admitted
    per-node 2**125 magnitude, so every bound and every compared difference is
    exact in signed 128 bits.
    """
    source, axes = _tile_forms(program, prefix)
    lines = []
    for index in source:
        low = [source[index]]
        high = [source[index]]
        for coefficient, extent in zip(axes[index], program.shape, strict=True):
            if not coefficient:
                continue
            term = f"(__int128)({_int_literal(coefficient)}) * {extent - 1}"
            (high if coefficient > 0 else low).append(term)
        lines.append(f"const __int128 {prefix}lo{index} = {' + '.join(low)};")
        lines.append(f"const __int128 {prefix}hi{index} = {' + '.join(high)};")
    for index, node in enumerate(program.nodes):
        args = node.args
        if node.op == CMP:
            left, right, kind = args
            low = f"({prefix}lo{left} - {prefix}hi{right})"
            high = f"({prefix}hi{left} - {prefix}lo{right})"
            if kind == 0:
                all_true = f"({low} == 0 && {high} == 0)"
                all_false = f"({low} > 0 || {high} < 0)"
            elif kind == 1:
                all_true = f"({low} > 0 || {high} < 0)"
                all_false = f"({low} == 0 && {high} == 0)"
            elif kind == 2:
                all_true, all_false = f"({high} < 0)", f"({low} >= 0)"
            elif kind == 3:
                all_true, all_false = f"({high} <= 0)", f"({low} > 0)"
            elif kind == 4:
                all_true, all_false = f"({low} > 0)", f"({high} <= 0)"
            else:
                all_true, all_false = f"({low} >= 0)", f"({high} < 0)"
        elif node.op == AND:
            all_true = f"({prefix}all{args[0]} && {prefix}all{args[1]})"
            all_false = f"({prefix}none{args[0]} || {prefix}none{args[1]})"
        elif node.op == OR:
            all_true = f"({prefix}all{args[0]} || {prefix}all{args[1]})"
            all_false = f"({prefix}none{args[0]} && {prefix}none{args[1]})"
        elif node.op == NOT:
            all_true = f"{prefix}none{args[0]}"
            all_false = f"{prefix}all{args[0]}"
        elif node.op == BROADCAST:
            all_true = f"{prefix}all{args[0]}"
            all_false = f"{prefix}none{args[0]}"
        else:
            continue
        lines.append(f"const bool {prefix}all{index} = {all_true};")
        lines.append(f"const bool {prefix}none{index} = {all_false};")
    return lines, f"{prefix}all{len(program.nodes) - 1}", f"{prefix}none{len(program.nodes) - 1}"


def _chunk_lines(native, program, pointers, scalar, chunks, strategy, indent):
    """Per-64-lane replay: own bitmap for partial blocks, coalesced full runs."""
    pad = " " * indent
    lines = [
        f"{pad}int64_t npu_mask_full = -1;",
        f"{pad}for (int64_t npu_chunk = 0; npu_chunk < {chunks}; ++npu_chunk) {{",
    ]
    lines.extend(f"{pad}  {line}" for line in _word_lines(program, "npu_chunk", strategy))
    lines.extend(
        [
            f"{pad}  if (npu_mask_word == UINT64_MAX) {{",
            f"{pad}    if (npu_mask_full < 0) npu_mask_full = npu_chunk;",
            f"{pad}  }} else {{",
            f"{pad}    if (npu_mask_full >= 0) {{",
            *_flush_lines(native, pointers, scalar, "npu_mask_full", "npu_chunk", indent + 6),
            f"{pad}      npu_mask_full = -1;",
            f"{pad}    }}",
            f"{pad}    if (npu_mask_word) {{",
            f"{pad}      asc_set_vector_mask(0, npu_mask_word);",
            f"{pad}      {_element_call(native, pointers, scalar, '1', 'npu_chunk * 256')}",
            f"{pad}    }}",
            f"{pad}  }}",
            f"{pad}}}",
            f"{pad}if (npu_mask_full >= 0) {{",
            *_flush_lines(native, pointers, scalar, "npu_mask_full", str(chunks), indent + 2),
            f"{pad}}}",
        ]
    )
    return lines


def _row_span_lines(native, pointers, scalar, indent, offset, lo, hi, full_repeats):
    """Issue one half-open lane span `[lo, hi)` inside one logical row.

    A span is at most a leading partial bitmap, whole chunks coalesced into
    repeats of at most 255, and a trailing partial bitmap; a span shorter than
    one whole block is exactly one bitmap. No zero bitmap is issued and no lane
    is written twice. `full_repeats` is the static admitted row extent in whole
    blocks, so a run inside one row never needs more than that many repeats:
    when it fits one ISA call the whole run is a single repeat, otherwise the
    bounded 255 split stays.
    """
    pad = " " * indent
    inner = pad + "  "
    # Full runs start chunk-aligned; a partial bitmap is relative to its
    # chunk-aligned base address, exactly like the per-chunk replay.
    at = f"(({offset}) + npu_at) * 4"
    chunk_at = f"(({offset}) + npu_chunk * 64) * 4"
    if full_repeats <= 255:
        run = [
            f"{inner}    int64_t npu_rep = (npu_end - npu_at) / 64;",
            f"{inner}    asc_set_vector_mask(0, UINT64_MAX);",
            f"{inner}    {_element_call(native, pointers, scalar, 'npu_rep', at)}",
            f"{inner}    npu_at += npu_rep * 64;",
        ]
    else:
        run = [
            f"{inner}    int64_t npu_left = (npu_end - npu_at) / 64;",
            f"{inner}    while (npu_left > 0) {{",
            f"{inner}      int64_t npu_rep = npu_left > 255 ? 255 : npu_left;",
            f"{inner}      asc_set_vector_mask(0, UINT64_MAX);",
            f"{inner}      {_element_call(native, pointers, scalar, 'npu_rep', at)}",
            f"{inner}      npu_at += npu_rep * 64;",
            f"{inner}      npu_left -= npu_rep;",
            f"{inner}    }}",
        ]
    return [
        f"{pad}{{",
        f"{inner}int64_t npu_at = ({lo});",
        f"{inner}const int64_t npu_end = ({hi});",
        f"{inner}while (npu_at < npu_end) {{",
        f"{inner}  const int64_t npu_chunk = npu_at / 64;",
        f"{inner}  const int64_t npu_from = npu_at - npu_chunk * 64;",
        f"{inner}  if (npu_from == 0 && npu_end - npu_at >= 64) {{",
        *run,
        f"{inner}    // Recompute chunk/from/span for any remaining tail.",
        f"{inner}    continue;",
        f"{inner}  }}",
        f"{inner}  int64_t npu_span = 64 - npu_from;",
        f"{inner}  if (npu_span > npu_end - npu_at) npu_span = npu_end - npu_at;",
        f"{inner}  const uint64_t npu_word = npu_mask_interval(npu_from, npu_from + npu_span);",
        f"{inner}  if (npu_word) {{",
        f"{inner}    asc_set_vector_mask(0, npu_word);",
        f"{inner}    {_element_call(native, pointers, scalar, '1', chunk_at)}",
        f"{inner}  }}",
        f"{inner}  npu_at += npu_span;",
        f"{inner}}}",
        f"{pad}}}",
    ]


def _row_lines(native, program, pointers, scalar, indent):
    """Replay one unit-monotone comparison as clipped spans per logical row."""
    require(not program.is_periodic, "row replay cannot represent a periodic lane")
    node = comparison_root(program)
    require(node is not None, "row replay lost its admitted single comparison")
    source, axes = _tile_forms(program, "npu_m_")
    left, right, kind = node.args
    difference = axes[left][-1] - axes[right][-1]
    require(difference in {1, -1}, "row replay needs a unit last-axis coefficient")
    rank = len(program.shape)
    extent = program.shape[-1]
    rows = prod(program.shape[:-1])
    used = [axis for axis in range(rank - 1) if axes[left][axis] or axes[right][axis]]

    def base(index):
        parts = [source[index]]
        parts.extend(
            f"(__int128)({_int_literal(axes[index][axis])}) * npu_m_ax{axis}"
            for axis in used
            if axes[index][axis]
        )
        return "(" + " + ".join(parts) + ")"

    kind = kind if difference == 1 else _FLIP_KIND[kind]
    threshold = "(-(npu_m_b))" if difference == 1 else "(npu_m_b)"
    low = f"npu_mask_clamp({threshold}, {extent})"
    high = f"npu_mask_clamp({threshold} + 1, {extent})"
    # A full run stays inside one row, so the static extent bounds it.
    full_repeats = extent // 64
    pad = " " * indent
    offset = f"npu_row * {extent}"
    inner = indent + 2
    lines = [f"{pad}for (int64_t npu_row = 0; npu_row < {rows}; ++npu_row) {{"]
    for axis in used:
        # Row index space excludes the last axis, so each fixed axis strides
        # over the remaining non-last extents only.
        stride = prod(program.shape[axis + 1 : rank - 1])
        lines.append(
            f"{' ' * inner}const int64_t npu_m_ax{axis} = "
            f"((npu_row) / {stride}) % {program.shape[axis]};"
        )
    lines.append(f"{' ' * inner}const __int128 npu_m_b = {base(left)} - {base(right)};")
    if kind in {2, 3}:
        end = high if kind == 3 else low
        lines.extend(
            _row_span_lines(native, pointers, scalar, inner, offset, "0", end, full_repeats)
        )
    elif kind in {4, 5}:
        begin = high if kind == 4 else low
        lines.extend(
            _row_span_lines(
                native, pointers, scalar, inner, offset, begin, str(extent), full_repeats
            )
        )
    elif kind == 0:
        lines.extend(
            _row_span_lines(native, pointers, scalar, inner, offset, low, high, full_repeats)
        )
    else:
        lines.extend(
            _row_span_lines(native, pointers, scalar, inner, offset, "0", low, full_repeats)
        )
        lines.extend(
            _row_span_lines(
                native, pointers, scalar, inner, offset, high, str(extent), full_repeats
            )
        )
    lines.append(f"{pad}}}")
    return lines


def masked_element_lines(task, pointers):
    """Replay one coordinate-masked element native over its packed chunks.

    The admitted `affine64` word recipe also prescribes a runtime whole-tile
    shortcut: exact 128-bit bounds over the current scalar atoms decide
    (all, none) truth for the tile, so an all-true tile issues the ordinary
    full-run recipe and an all-false tile issues no element call at all (its
    declared RW footprint keeps the destination initialized-before-use). An
    unproven tile keeps the per-block replay, which is also the only replay for
    the `lane64` recipe. The emitter selects no policy of its own.
    """
    program = task.mask
    require(program is not None, "masked native lost its lane-predicate program")
    from_scalar = task.native in {"mask_fill", "mask_muls", "mask_adds"}
    scalar = _scalar_value(task.scalar) if from_scalar else ""
    chunks = program.lanes // 64
    require(
        task.mask_repeat == MASK_REPEAT_POLICY,
        "masked element lost the admitted full-run repeat policy",
    )
    require(
        task.mask_iteration in ITERATION_VALUES,
        "masked element lost its admitted iteration recipe",
    )
    if task.mask_iteration == ITERATION_ROWS:
        require(
            task.mask_word == WORD_AFFINE,
            "row iteration needs the admitted affine word recipe",
        )
    lines = ["{", "asc_set_mask_norm();", *_scalar_lines(task.operands)]
    if task.mask_word == WORD_AFFINE:
        bounds, all_true, all_false = _tile_bound_lines(program, "npu_m_")
        lines.extend(bounds)
        lines.append(f"if ({all_true}) {{")
        lines.extend(
            "  " + line for line in _flush_lines(task.native, pointers, scalar, "0", str(chunks), 2)
        )
        lines.append(f"}} else if (!({all_false})) {{")
        if task.mask_iteration == ITERATION_ROWS:
            lines.extend(_row_lines(task.native, program, pointers, scalar, 2))
        else:
            lines.extend(
                _chunk_lines(task.native, program, pointers, scalar, chunks, WORD_AFFINE, 2)
            )
        lines.append("}")
    else:
        require(task.mask_word == WORD_LANE, "masked element lost its selected word recipe")
        lines.extend(_chunk_lines(task.native, program, pointers, scalar, chunks, WORD_LANE, 0))
    lines.append("}")
    return lines


def mask_zero_lines(task, pointers):
    """Define one packed predicate block with a real count-mode Vector Dup."""
    words = task.params[0]
    require(
        words > 0 and words % 16 == 0,
        "packed predicate zeroing lost its whole-block uint16 count",
    )
    chunks = words // 16
    base = f"(__ubuf__ uint8_t*)({pointers[0]})"
    lines = [
        "{",
        f"  for (int64_t npu_chunk = 0; npu_chunk < {chunks}; ++npu_chunk) {{",
        "    asc_duplicate((__ubuf__ uint16_t*)"
        + f"({base} + npu_chunk * 32), uint16_t(0), uint32_t(16));",
        "  }",
        "}",
    ]
    return lines


def compare_lines(task, pointers):
    chunks = task.params[0]
    scalar = task.native.startswith("compare_scalar_")
    kind = task.native.removeprefix("compare_scalar_").removeprefix("compare_")
    out = _byte_pointer(pointers[-1], "npu_chunk * 32", "uint8_t")
    if scalar:
        source = _byte_pointer(pointers[0], "npu_chunk * 256")
        call = (
            f"asc_{kind}_scalar({out}, {source}, float({task.scalar!r}), uint8_t(1), 1, 1, 8, 8);"
        )
    else:
        source0 = _byte_pointer(pointers[0], "npu_chunk * 256")
        source1 = _byte_pointer(pointers[1], "npu_chunk * 256")
        call = f"asc_{kind}({out}, {source0}, {source1}, uint8_t(1), 1, 1, 1, 8, 8, 8);"
    return [
        "{",
        f"  for (int64_t npu_chunk = 0; npu_chunk < {chunks}; ++npu_chunk) {{",
        f"    {call}",
        "  }",
        "}",
    ]


def mask_bits_lines(task, pointers):
    program = task.mask
    require(program is not None, "predicate materialization lost its program")
    chunks = program.lanes // 64
    lines = ["{", *_scalar_lines(task.operands)]
    lines.append(f"  for (int64_t npu_chunk = 0; npu_chunk < {chunks}; ++npu_chunk) {{")
    lines.extend("  " + line for line in _word_lines(program, "npu_chunk", task.mask_word))
    base = f"((__ubuf__ uint8_t*)({pointers[0]}) + npu_chunk * 32)"
    lines += [
        f"    *(__ubuf__ uint64_t*)({base}) = npu_mask_word;",
        f"    *(__ubuf__ uint64_t*)({base} + 8) = UINT64_C(0);",
        f"    *(__ubuf__ uint64_t*)({base} + 16) = UINT64_C(0);",
        f"    *(__ubuf__ uint64_t*)({base} + 24) = UINT64_C(0);",
        "  }",
        "}",
    ]
    return lines


def combine_lines(task, pointers):
    words = task.params[0]
    kind = task.native.removeprefix("mask_")
    typed = [f"(__ubuf__ uint16_t*)({pointer})" for pointer in pointers]
    if kind == "not":
        # A real count-mode Vector NOT over the declared packed block.
        return [f"asc_not({typed[1]}, {typed[0]}, uint32_t({words}));"]
    function = "and" if kind == "and" else "or"
    return [f"asc_{function}({typed[2]}, {typed[0]}, {typed[1]}, uint32_t({words}));"]


def select_lines(task, pointers):
    chunks = task.params[0]
    update = task.native == "select_masked_update"
    lines = [
        "{",
        "  asc_set_mask_norm();",
        "  asc_set_vector_mask(0, UINT64_MAX);",
        f"  for (int64_t npu_chunk = 0; npu_chunk < {chunks}; ++npu_chunk) {{",
        f"    asc_set_cmp_mask((__ubuf__ void*)((__ubuf__ uint8_t*)({pointers[0]}) + npu_chunk * 32));",
        "    asc_sync_pipe(PIPE_V);",
    ]
    if update:
        destination = _byte_pointer(pointers[2], "npu_chunk * 256")
        value = _byte_pointer(pointers[1], "npu_chunk * 256")
        lines.append(
            f"    asc_select({destination}, {value}, {destination}, uint8_t(1), 1, 1, 1, 8, 8, 8);"
        )
    else:
        destination = _byte_pointer(pointers[3], "npu_chunk * 256")
        source0 = _byte_pointer(pointers[1], "npu_chunk * 256")
        source1 = _byte_pointer(pointers[2], "npu_chunk * 256")
        lines.append(
            f"    asc_select({destination}, {source0}, {source1}, uint8_t(1), 1, 1, 1, 8, 8, 8);"
        )
    lines += ["  }", "}"]
    return lines
