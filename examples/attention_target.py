"""P0 design fixture: complete FA target source, deliberately not yet executable.

No finite builder or legacy fallback is called. Both organizations use the same
mathematical body; only their retained device control differs. compile() currently
rejects missing frontend/lowering capabilities. See docs/attention-target.md.
"""

import textwrap
from pathlib import Path

from xdsl_npu import language as N

POLICY = '{"cube": ("qk", "pv"), "vector": ("update", "softmax")}'


def target_source(*, continuous=False):
    if type(continuous) is not bool:
        raise ValueError("continuous must be an explicit boolean policy")
    template = (Path(__file__).parent / "design" / "attention_target.py.in").read_text()
    before, body = template.split("        @TASK_LOOP@\n")
    body = textwrap.dedent(body)
    if continuous:
        header = (
            f"        for task in N.Pipelined(local_queries * kv_tiles, stages=3, issue_order={POLICY}):\n"
            "            query = task // kv_tiles\n"
            "            kv = task % kv_tiles\n"
        )
        return before + header + textwrap.indent(body, " " * 12)
    header = (
        "        for query in N.serial(local_queries):\n"
        f"            for kv in N.Pipelined(kv_tiles, stages=3, issue_order={POLICY}):\n"
        "                task = query * kv_tiles + kv\n"
    )
    return before + header + textwrap.indent(body, " " * 16)


def attention_target(*, continuous=False):
    return N.from_source(
        target_source(continuous=continuous),
        constants={"NEG_INF": float("-inf")},
        filename=f"<attention-target:{'continuous' if continuous else 'nested'}>",
        disjoint_external=True,
    )


def activation_trace(queries, kv_tiles, *, continuous=False):
    """Small P0 dispatch table, not the compiler or a hardware simulator."""
    if (
        type(queries) is not int
        or type(kv_tiles) is not int
        or min(queries, kv_tiles) < 0
        or queries * kv_tiles > 64
        or type(continuous) is not bool
    ):
        raise ValueError(
            "activation table requires nonnegative counts, <=64 tasks and a boolean policy"
        )
    result = {"cube": [], "vector": []}
    lengths = [queries * kv_tiles] if continuous else [kv_tiles] * queries
    for call, count in enumerate(lengths):
        for step in range(count + 2):
            for actor, stages in (
                ("cube", (("qk", 0), ("pv", 1))),
                ("vector", (("update", 2), ("softmax", 1))),
            ):
                for name, offset in stages:
                    g = step - offset
                    if 0 <= g < count:
                        query, kv = divmod(g, kv_tiles) if continuous else (call, g)
                        result[actor].append(
                            {"step": step, "call": call, "stage": name, "query": query, "kv": kv}
                        )
    return result


def main():
    import argparse
    import hashlib
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = {}
    for mode in (False, True):
        name = "continuous" if mode else "nested"
        source = target_source(continuous=mode)
        (args.output / f"{name}.dsl.py").write_text(source)
        records[name] = {
            "status": "design fixture; frontend/physical lowering incomplete",
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "two_queries_three_kv": activation_trace(2, 3, continuous=mode),
        }
    (args.output / "contract.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
