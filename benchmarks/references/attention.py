"""Independent fixed-policy AscendC FA source, with no device side effects."""

from benchmarks.references import ascendc_source
from xdsl_npu.model import require

NAME = "attention_reference"


def reference_source(*, batch=1, heads=8, kv_heads=8, groups=20, dtype="f16"):
    require(
        all(type(v) is int and v > 0 for v in (batch, heads, kv_heads, groups))
        and heads % kv_heads == 0
        and groups <= 20
        and batch * heads < 2**20
        and dtype in {"f16", "bf16"},
        "reference requires positive B/H/G, HQ divisible by HKV, groups<=20 and b16 dtype",
    )
    source = ascendc_source("ascendc_attention.asc.in")
    for key, value in {
        "B": batch,
        "HQ": heads,
        "HKV": kv_heads,
        "GROUPS": groups,
        "TYPE": "half" if dtype == "f16" else "bfloat16_t",
    }.items():
        source = source.replace("@" + key + "@", str(value))
    require("@" not in source, "unbound attention reference policy")
    return source
