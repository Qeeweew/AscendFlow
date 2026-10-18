"""The supported nested example and its unsupported continuous design policy.

The sole mathematical body belongs to examples.flash_attention.attention.
Continuous control rebinds the real descriptors and moves each query's
initialization to its first stage. Runtime-cycle predicates remain unsupported.
"""

import ast
from pathlib import Path

from ascendflow import language as N
from examples.flash_attention.attention import NEG_INF
from examples.flash_attention.attention import attention_target as _attention_target


def target_source(*, continuous=False):
    if type(continuous) is not bool:
        raise ValueError("continuous must be an explicit boolean policy")
    if not continuous:
        return _attention_target.source
    module = ast.parse(_attention_target.source)
    function = module.body[0]
    kernel = function.body[0]
    for index, query in enumerate(kernel.body):
        if (
            isinstance(query, ast.For)
            and isinstance(query.iter, ast.Call)
            and isinstance(query.iter.func, ast.Attribute)
            and query.iter.func.attr == "serial"
        ):
            kv = next(node for node in query.body if isinstance(node, ast.For))
            prefix = query.body[: query.body.index(kv)]
            suffix = query.body[query.body.index(kv) + 1 :]
            initial = {"qk": [], "softmax": [], "update": []}
            descriptors = []
            for node in prefix:
                if isinstance(node, ast.With) and isinstance(node.items[0].context_expr, ast.Call):
                    call = node.items[0].context_expr
                    if isinstance(call.func, ast.Attribute) and call.func.attr == "Cube":
                        initial["qk"].extend(node.body)
                        continue
                    if isinstance(call.func, ast.Attribute) and call.func.attr == "Vector":
                        for fill in node.body:
                            name = fill.value.args[0].id
                            initial["update" if name == "o" else "softmax"].append(fill)
                        continue
                descriptors.append(node)
            for stage in kv.body:
                if not isinstance(stage, ast.With):
                    continue
                for item in stage.items:
                    call = item.context_expr
                    if (
                        not isinstance(call, ast.Call)
                        or not isinstance(call.func, ast.Attribute)
                        or call.func.attr != "Stage"
                    ):
                        continue
                    name = next(k.value.value for k in call.keywords if k.arg == "name")
                    if name in initial:
                        stage.body.insert(
                            0,
                            ast.If(
                                test=ast.parse("kv == 0", mode="eval").body,
                                body=initial[name],
                                orelse=[],
                            ),
                        )
            expected = ast.parse("task = query * kv_tiles + kv").body[0]
            if ast.dump(kv.body[0], include_attributes=False) != ast.dump(
                expected, include_attributes=False
            ):
                raise ValueError("continuous fixture needs the example's actual task ordinal")
            kv.target = ast.Name(id="task", ctx=ast.Store())
            kv.iter.args[0] = ast.parse("local_queries * kv_tiles", mode="eval").body
            orders = [
                n.value
                for n in ast.walk(ast.Module(body=prefix, type_ignores=[]))
                if isinstance(n, ast.Assign)
                and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id == "kv_order"
            ]
            next(k for k in kv.iter.keywords if k.arg == "issue_order").value = orders[-1]
            kv.body = (
                ast.parse("query = task // kv_tiles\nkv = task % kv_tiles").body
                + descriptors
                + kv.body[1:]
                + suffix
            )
            kernel.body[index] = kv
            kernel.body.insert(
                0,
                ast.Assert(
                    test=ast.parse("CAUSAL == False", mode="eval").body,
                    msg=ast.Constant("continuous fixture is dense only"),
                ),
            )
            return ast.unparse(ast.fix_missing_locations(module)) + "\n"
    raise ValueError("continuous fixture needs the example's actual serial query loop")


def attention_target(*, continuous=False):
    if type(continuous) is not bool:
        raise ValueError("continuous must be an explicit boolean policy")
    if not continuous:
        return _attention_target
    return N.from_source(
        target_source(continuous=True),
        constants={"NEG_INF": NEG_INF},
        filename="<attention-target:continuous>",
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
    # Use the example's policy; this finite dispatch model must not retain a
    # previous issue order after a source policy experiment.
    tree = ast.parse(target_source(continuous=continuous))
    policies = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "kv_order"
        and isinstance(node.value, ast.Dict)
    ]
    policy = (
        ast.literal_eval(policies[-1])
        if policies
        else next(
            value
            for node in ast.walk(tree)
            if isinstance(node, ast.keyword) and node.arg == "issue_order"
            if set(value := ast.literal_eval(node.value)) == {"cube", "vector"}
        )
    )
    offsets = {"qk": 0, "softmax": 1, "pv": 1, "update": 2}
    lengths = [queries * kv_tiles] if continuous else [kv_tiles] * queries
    for call, count in enumerate(lengths):
        for step in range(count + 2):
            for actor, names in policy.items():
                for name in names:
                    offset = offsets[name]
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
            "status": (
                "unsupported continuous runtime-cycle predicates"
                if mode
                else "supported nested example; device validation is limited to documented f16 cases"
            ),
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "two_queries_three_kv": activation_trace(2, 3, continuous=mode),
        }
    (args.output / "contract.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
