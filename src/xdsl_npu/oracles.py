"""CPU numerical contracts, independent of the sync planner and native simulator."""

import struct
from array import array
from math import cos, exp, sin, sqrt


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def operator_inputs(program, *, data_case="standard"):
    """Return parameter bytes, expected output, tolerance and logical shape."""
    if data_case not in {"standard", "numerical-edges"}:
        raise ValueError(f"unknown numerical input case {data_case}")
    if data_case != "standard" and program.name != "rms_norm":
        raise ValueError("numerical-edges currently applies to RMSNorm")
    if program.name == "gemm_nt":
        reduction = next(t for t in program.tasks if t.uid == "load_a").refs[0].row_stride // 2
        m = next(s.nbytes for s in program.storages if s.name == "a") // (reduction * 2)
        n = next(s.nbytes for s in program.storages if s.name == "bt") // (reduction * 2)
        a = [((3 * i + 5 * l) % 17 - 8) / 8 for i in range(m) for l in range(reduction)]
        bt = [((2 * j + 3 * l) % 13 - 6) / 8 for j in range(n) for l in range(reduction)]
        rhs = [bt[j * reduction : (j + 1) * reduction] for j in range(n)]
        expected = array("f")
        for i in range(m):
            lhs = a[i * reduction : (i + 1) * reduction]
            expected.extend(sum(x * y for x, y in zip(lhs, row)) for row in rhs)
        return (
            [
                struct.pack(f"<{len(a)}e", *a),
                struct.pack(f"<{len(bt)}e", *bt),
                b"\xcd" * (m * n * 4),
            ],
            expected,
            (0, 0),
            [m, n, reduction],
        )
    if program.name == "batched_matmul_nt":
        m, n, grain, _ = next(t.params for t in program.tasks if t.uid == "mad_init")
        reduction, batch = grain * program.count, program.blocks
        # Exact binary fractions keep this oracle independent of MAD grouping.
        a = [
            ((3 * i + 5 * l + g) % 17 - 8) / 8
            for g in range(batch)
            for i in range(m)
            for l in range(reduction)
        ]
        bt = [
            ((2 * j + 3 * l + 2 * g) % 13 - 6) / 8
            for g in range(batch)
            for j in range(n)
            for l in range(reduction)
        ]
        expected = array("f")
        for g in range(batch):
            for i in range(m):
                lhs = a[(g * m + i) * reduction : (g * m + i + 1) * reduction]
                for j in range(n):
                    rhs = bt[(g * n + j) * reduction : (g * n + j + 1) * reduction]
                    expected.append(sum(x * y for x, y in zip(lhs, rhs)))
        return (
            [
                struct.pack(f"<{len(a)}e", *a),
                struct.pack(f"<{len(bt)}e", *bt),
                b"\xcd" * (batch * m * n * 4),
            ],
            expected,
            (0, 0),
            [batch, m, n, reduction],
        )
    if program.name == "rms_norm":
        r, columns = next(t.params for t in program.tasks if t.uid == "partial_sum")
        rows = program.blocks * program.count * r
        epsilon = next(t.scalar for t in program.tasks if t.uid == "epsilon")
        x = array("f", ((i % 127 - 63) / 16 for i in range(rows * columns)))
        gamma = array("f", (1 + (j % 7 - 3) / 16 for j in range(columns)))
        if data_case == "numerical-edges":
            x = array(
                "f",
                (
                    (0, 1e-5, (j % 17 - 8) * 1000, 10000 if j == 0 else (j % 3 - 1))[row % 4]
                    for row in range(rows)
                    for j in range(columns)
                ),
            )
            gamma = array("f", ((-2, 0, 0.125, 1, 3)[j % 5] for j in range(columns)))
        expected = array("f")
        for row in range(rows):
            values = x[row * columns : (row + 1) * columns]
            # Each algebraic node has its declared FP32 boundary.
            partials = [
                f32(sum(f32(v * v) for v in values[i : i + 64])) for i in range(0, columns, 64)
            ]
            mean = f32(f32(sum(partials)) * f32(1 / columns))
            inverse = f32(1 / f32(sqrt(f32(mean + epsilon))))
            expected.extend(f32(f32(v * inverse) * g) for v, g in zip(values, gamma))
        return (
            [x.tobytes(), gamma.tobytes(), b"\xcd" * (rows * columns * 4)],
            expected,
            (5e-5, 5e-6),
            [rows, columns],
        )
    if program.name == "rope":
        r, half = next(t.params for t in program.tasks if t.uid == "cos_low")
        columns, rows = half * 2, program.blocks * program.count * r
        x = array("f", ((i % 127 - 63) / 16 for i in range(rows * columns)))
        c = array("f", (cos(row / 11 + j / 7) for row in range(rows) for j in range(half)))
        s = array("f", (sin(row / 11 + j / 7) for row in range(rows) for j in range(half)))
        expected = array("f")
        for row in range(rows):
            base, trig = row * columns, row * half
            low = [
                f32(f32(x[base + j] * c[trig + j]) - f32(x[base + half + j] * s[trig + j]))
                for j in range(half)
            ]
            high = [
                f32(f32(x[base + half + j] * c[trig + j]) + f32(x[base + j] * s[trig + j]))
                for j in range(half)
            ]
            expected.extend(low + high)
        return (
            [x.tobytes(), c.tobytes(), s.tobytes(), b"\xcd" * (rows * columns * 4)],
            expected,
            (0, 0),
            [rows, columns],
        )
    if program.name == "softmax":
        r, columns = next(t.params for t in program.tasks if t.uid == "partial_max")
        rows = program.blocks * program.count * r
        # Row shifts stress stable max subtraction; every 4th row is constant.
        x = array(
            "f",
            (
                1000 * (row % 3 - 1) + (0 if row % 4 == 0 else (j % 127 - 63) / 8)
                for row in range(rows)
                for j in range(columns)
            ),
        )
        expected = array("f")
        for row in range(rows):
            values = x[row * columns : (row + 1) * columns]
            maximum = max(values)
            numerators = [f32(exp(f32(v - maximum))) for v in values]
            partials = [f32(sum(numerators[i : i + 64])) for i in range(0, columns, 64)]
            inverse = f32(1 / f32(sum(partials)))
            expected.extend(f32(v * inverse) for v in numerators)
        return (
            [x.tobytes(), b"\xcd" * (rows * columns * 4)],
            expected,
            (5e-5, 2e-7),
            [rows, columns],
        )
    if program.name == "swiglu":
        elements = next(s.nbytes for s in program.storages if s.name == "gate") // 4
        rows, columns = next(t.params for t in program.tasks if t.uid == "multiply_up")
        rows *= program.count * program.blocks
        gate = array("f", ((i % 257 - 128) / 8 for i in range(elements)))
        up = array("f", (((3 * i) % 127 - 63) / 16 for i in range(elements)))
        expected = array("f", (f32(f32(g / f32(1 + f32(exp(-g)))) * u) for g, u in zip(gate, up)))
        return (
            [gate.tobytes(), up.tobytes(), b"\xcd" * (elements * 4)],
            expected,
            (5e-5, 5e-6),
            [rows, columns],
        )
    raise ValueError(f"no operator oracle for {program.name}")
