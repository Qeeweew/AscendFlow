"""Typed local and bidirectional CV software pipelines.

These small examples exercise stage displacement, exact ownership, and reuse.
They are not attention algorithms or performance benchmarks.
"""

from xdsl_npu import language as N


def _extent(count):
    return max(1, count) if type(count) is int else count


def vector_source(count=4):
    @N.prim_func(disjoint_external=True)
    def scale(
        X: N.Tensor((2, _extent(count), 64), N.f32),
        Y: N.Tensor((2, _extent(count), 64), N.f32, access="write"),
    ):
        with N.Kernel(2, kind="vector") as group:
            inputs = N.alloc_ub((64,), versions=2, name="inputs")
            outputs = N.alloc_ub((64,), versions=2, name="outputs")
            for task in N.Pipelined(count, stages=3):
                with N.Stage(0, name="load"):
                    value = N.copy(X[group, task, :], inputs)
                with N.Stage(1, name="compute"), N.Vector():
                    value = N.mul(value, 2.0, out=outputs)
                with N.Stage(2, name="store"):
                    N.copy(value, Y[group, task, :])

    return scale


def cv_source(count=4):
    shape = (1, _extent(count), 16, 16)

    @N.prim_func(disjoint_external=True)
    def cv(
        Q: N.Tensor(shape, N.f16),
        K: N.Tensor(shape, N.f16),
        V: N.Tensor(shape, N.f16),
        O: N.Tensor(shape, N.f32, access="write"),
    ):
        with N.Kernel(1, kind="mixed") as group:
            q = N.alloc_l1((16, 16))
            k = N.alloc_l1((16, 16), versions=2)
            p = N.alloc_l1((16, 16), versions=2)
            v = N.alloc_l1((16, 16), versions=2)
            score_acc = N.alloc_l0c((16, 16))
            pv_acc = N.alloc_l0c((16, 16))
            scores = N.alloc_workspace((16, 16), versions=2, name="scores")
            probabilities = N.alloc_workspace((16, 16), N.f16, versions=2, name="probabilities")
            pv = N.alloc_workspace((16, 16), versions=2, name="pv")
            score_ub = N.alloc_ub((8, 16), versions=2)
            p_ub = N.alloc_ub((8, 16), N.f16, versions=2)
            output_ub = N.alloc_ub((8, 16), versions=2)
            for task in N.Pipelined(count, stages=3):
                with N.Stage(0, name="qk"), N.Cube():
                    N.copy(Q[group, task, :, :], q)
                    N.copy(K[group, task, :, :], k)
                    score = N.gemm(q, k, score_acc, transpose_b=True)
                    N.copy(score, scores[task])
                with N.Stage(1, name="vector"), N.Vector() as owner:
                    rows = slice(owner * 8, owner * 8 + 8)
                    score = N.copy(scores[task][rows, :], score_ub)
                    scaled = N.mul(score, 2.0, out=score_ub)
                    rounded = N.cast(scaled, out=p_ub)
                    N.copy(rounded, probabilities[task][rows, :])
                with N.Stage(1, name="pv"), N.Cube():
                    N.copy(probabilities[task], p)
                    N.copy(V[group, task, :, :], v)
                    result = N.gemm(p, v, pv_acc)
                    N.copy(result, pv[task])
                with N.Stage(2, name="output"), N.Vector() as owner:
                    rows = slice(owner * 8, owner * 8 + 8)
                    result = N.copy(pv[task][rows, :], output_ub)
                    N.copy(result, O[group, task, rows, :])

    return cv


def segmented_cv_source(count=4, segments=3, groups=2):
    """QK -> RN_f16(2 * scores) -> PV with independent inner prefetch.

    Each CV task owns `segments` score/P contributions. PV initializes once
    and accumulates them in chunk order; it publishes one complete output.
    This is a synchronization example, not online softmax.
    """
    q_shape = (groups, _extent(count), 16, 16)
    kv_shape = (groups, _extent(count), segments, 16, 16)

    @N.prim_func(disjoint_external=True)
    def segmented_cv(
        Q: N.Tensor(q_shape, N.f16),
        K: N.Tensor(kv_shape, N.f16),
        V: N.Tensor(kv_shape, N.f16),
        Output: N.Tensor(q_shape, N.f32, access="write"),
    ):
        with N.Kernel(groups, kind="mixed") as group:
            q = N.alloc_l1((16, 16))
            k = N.alloc_l1((16, 16), versions=2)
            p = N.alloc_l1((16, 16), versions=2)
            v = N.alloc_l1((16, 16), versions=2)
            score_acc = N.alloc_l0c((16, 16))
            pv_acc = N.alloc_l0c((16, 16))
            scores = N.alloc_workspace((16, segments * 16), versions=2, name="scores")
            probabilities = N.alloc_workspace(
                (16, segments * 16), N.f16, versions=2, name="probabilities"
            )
            pv = N.alloc_workspace((16, 16), versions=2, name="pv")
            score_ub = N.alloc_ub((8, 16), versions=2)
            p_ub = N.alloc_ub((8, 16), N.f16, versions=2)
            output_ub = N.alloc_ub((8, 16), versions=2)
            for task in N.Pipelined(count, stages=3):
                with N.Stage(0, name="qk"), N.Cube():
                    q_value = N.copy(Q[group, task, :, :], q)
                    for part in N.Pipelined(segments, stages=2):
                        with N.Stage(0, name="prefetch_k"):
                            k_value = N.copy(K[group, task, part, :, :], k)
                        with N.Stage(1, name="qk_segment"):
                            score = N.gemm(q_value, k_value, score_acc, transpose_b=True)
                            N.copy(score, scores[task][:, part * 16 : part * 16 + 16])
                with N.Stage(1, name="vector"), N.Vector() as owner:
                    rows = slice(owner * 8, owner * 8 + 8)
                    for part in N.Pipelined(segments, stages=1):
                        columns = slice(part * 16, part * 16 + 16)
                        score = N.copy(scores[task][rows, columns], score_ub)
                        scaled = N.mul(score, 2.0, out=score_ub)
                        rounded = N.cast(scaled, out=p_ub)
                        N.copy(rounded, probabilities[task][rows, columns])
                with N.Stage(1, name="pv"), N.Cube():
                    for part in N.Pipelined(segments, stages=2):
                        with N.Stage(0, name="prefetch_pv"):
                            p_value = N.copy(probabilities[task][:, part * 16 : part * 16 + 16], p)
                            v_value = N.copy(V[group, task, part, :, :], v)
                        with N.Stage(1, name="pv_segment"):
                            result = N.gemm(p_value, v_value, pv_acc, initialize=(part == 0))
                    N.copy(result, pv[task])
                with N.Stage(2, name="output"), N.Vector() as owner:
                    rows = slice(owner * 8, owner * 8 + 8)
                    result = N.copy(pv[task][rows, :], output_ub)
                    N.copy(result, Output[group, task, rows, :])

    return segmented_cv


def main():
    import argparse
    import json
    from pathlib import Path

    from xdsl_npu.cli import build_module

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["vector", "cv", "segmented-cv"])
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--runtime", action="store_true", help="one by-value SSA tile-count parameter")
    parser.add_argument("--segments", type=int, default=3)
    parser.add_argument("--groups", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--shared", action="store_true")
    args = parser.parse_args()
    count = N.dynamic("tasks") if args.runtime else args.count
    source = (
        segmented_cv_source(count, args.segments, args.groups)
        if args.kind == "segmented-cv"
        else {"vector": vector_source, "cv": cv_source}[args.kind](count)
    )
    report = build_module(source, args.output, compile_asc=args.compile, shared=args.shared)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
