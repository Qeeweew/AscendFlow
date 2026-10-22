"""Generate a static region-count workload for the generic cold compile benchmark.

The runtime length n is always a device argument. Only the number of source
regions/declarations changes; unused declarations expose repeated binding work.
"""

import argparse
from pathlib import Path


def source(streams=32, unused=128):
    if not (1 <= streams <= 128 and 0 <= unused <= 512):
        raise ValueError("source-size experiment requires 1..128 streams and 0..512 declarations")
    result = """from __future__ import annotations
from ascendflow import language as N
@N.prim_func(disjoint_external=True)
def chain(A:N.Tensor((64,),N.f32),O:N.Tensor((64,),N.f32,access="write"),n:N.i64):
    with N.Kernel():
        x=N.alloc_ub((64,))
"""
    result += "".join(f"        unused{i}=N.alloc_ub((64,))\n" for i in range(unused))
    result += "        N.copy(A,x)\n"
    for i in range(streams):
        result += (
            f"        for t{i} in N.Pipelined(n):\n"
            f'            with N.Stage(0,name="scale{i}"):\n'
            "                N.mul(x,2.0,out=x)\n"
        )
    return result + "        N.copy(x,O)\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--streams", type=int, default=32)
    parser.add_argument("--unused", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = source(args.streams, args.unused)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result)


if __name__ == "__main__":
    main()
