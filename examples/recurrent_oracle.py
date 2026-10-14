"""Independent CPU GDN reference, including every FP32 snapshot and carry.

The reduction here uses a double-precision sum of rounded FP32 products rather
than pretending to know the hardware vcadd tree. Device comparisons require
an explicit tolerance. The state update uses libm fmaf for its single rounding.
"""

import ctypes
from array import array
from math import exp

from oracles import f32


def gdn_reference(q, keys, values, g, beta, initial, *, sequences, steps, value_dim, scale):
    fma = ctypes.CDLL("libm.so.6").fmaf
    fma.argtypes = [ctypes.c_float] * 3
    fma.restype = ctypes.c_float
    attention, snapshots = array("f"), array("f")
    padded_steps = max(8, (steps + 7) // 8 * 8)
    for head in range(sequences):
        state = list(initial[head * value_dim * 64 : (head + 1) * value_dim * 64])
        for token in range(steps):
            decay = f32(exp(g[head * padded_steps + token]))
            weight = beta[head * padded_steps + token]
            key = keys[(head * steps + token) * 64 : (head * steps + token + 1) * 64]
            query = [
                f32(x * scale)
                for x in q[(head * steps + token) * 64 : (head * steps + token + 1) * 64]
            ]
            state = [f32(x * decay) for x in state]
            delta = [
                f32(
                    f32(
                        values[(head * steps + token) * value_dim + row]
                        - f32(sum(f32(state[row * 64 + col] * key[col]) for col in range(64)))
                    )
                    * weight
                )
                for row in range(value_dim)
            ]
            state = [
                fma(delta[row], key[col], state[row * 64 + col])
                for row in range(value_dim)
                for col in range(64)
            ]
            attention.extend(
                f32(sum(f32(state[row * 64 + col] * query[col]) for col in range(64)))
                for row in range(value_dim)
            )
            snapshots.extend(state)
    return {"y": attention, "snapshots": snapshots}


def recurrent_inputs(program):
    sequences, steps = program.blocks, program.count
    rows = next(t.params[0] for t in program.tasks if t.uid == "update_state")
    scale = next(t.scalar for t in program.tasks if t.uid == "scale_q")
    padded_steps = max(8, (steps + 7) // 8 * 8)
    q = array("f", (((3 * i + i // 64) % 37 - 18) / 64 for i in range(sequences * steps * 64)))
    keys = array(
        "f", (((5 * i + 7 * (i // 64)) % 31 - 15) / 128 for i in range(sequences * steps * 64))
    )
    values = array("f", (((7 * i) % 41 - 20) / 32 for i in range(sequences * steps * rows)))
    g = array(
        "f",
        (
            -0.01 * (1 + (head * 3 + token) % 19)
            for head in range(sequences)
            for token in range(padded_steps)
        ),
    )
    beta = array(
        "f",
        (
            (1 + (head + token * 7) % 23) / 32
            for head in range(sequences)
            for token in range(padded_steps)
        ),
    )
    initial = array(
        "f", (((11 * i + i // (64 * rows)) % 29 - 14) / 256 for i in range(sequences * rows * 64))
    )
    expected = gdn_reference(
        q,
        keys,
        values,
        g,
        beta,
        initial,
        sequences=sequences,
        steps=steps,
        value_dim=rows,
        scale=scale,
    )
    buffers = {
        "q": q.tobytes(),
        "k": keys.tobytes(),
        "v": values.tobytes(),
        "log_decay": g.tobytes(),
        "beta": beta.tobytes(),
        "initial_state": initial.tobytes(),
    }
    return (
        [buffers.get(s.name, b"\xcd" * s.nbytes) for s in program.storages if s.external],
        expected,
        (5e-5, 5e-6),
        [sequences, steps, rows, 64],
    )
