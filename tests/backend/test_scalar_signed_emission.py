"""Inlined unsigned operations must retain their signed index result type."""

import subprocess

import pytest

from ascendflow import compile
from ascendflow import language as T
from tests.support.asc_trace import TRACE_SHIM, compile_cpp


@pytest.mark.parametrize("value", ("n // 16", "n % 16", "T.ceildiv(n, 16)"))
def test_unsigned_index_result_does_not_promote_its_signed_consumer(value, tmp_path):
    definition = T.from_source(f"""
def owners(Ends:T.Tensor((1,),T.i64)):
    with T.Kernel(5,kind="vector") as group:
        tile=T.alloc_ub((64,))
        n=T.load_index(Ends[0:1],minimum=0)
        count=T.max(0,({value})*32-group)
        for task in T.serial(T.ceildiv(count,5)):
            T.fill(tile,1.0)
""")
    result = compile(definition)
    asc = result.asc.split('extern "C" int kernel_config_')[0]
    asc = "\n".join(line for line in asc.splitlines() if not line.startswith("#include"))
    # Abort an erroneous giant loop immediately, without relying on a device timeout.
    shim = TRACE_SHIM.replace(
        ' std::cout<<name<<" "<<scalar;',
        ' if (++trace_calls > 1000) std::abort(); std::cout<<name<<" "<<scalar;',
    )
    shim = shim.replace(
        "int64_t trace_group=0,trace_actor=1;",
        "int trace_calls=0; int64_t trace_group=0,trace_actor=1;",
    )
    oracle = {
        "n // 16": "n / 16",
        "n % 16": "n % 16",
        "T.ceildiv(n, 16)": "n / 16 + (n % 16 != 0)",
    }[value]
    main = f"""
#include <cassert>
int main() {{
  for (int64_t n : {{0,1,15,16,17,32}}) {{
    for (trace_group=0; trace_group<5; ++trace_group) {{
      const int64_t left=({oracle})*32-trace_group;
      const int64_t count=left>0?left:0;
      const int64_t expected=count/5+(count%5!=0);
      trace_calls=0;
      owners(reinterpret_cast<uint8_t*>(&n));
      assert(trace_calls==expected);
    }}
  }}
}}
"""
    binary = compile_cpp(tmp_path, shim + asc + main, sanitize=True)
    subprocess.run([str(binary)], check=True, capture_output=True, timeout=5)
