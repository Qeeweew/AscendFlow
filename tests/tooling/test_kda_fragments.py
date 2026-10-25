"""Fragment runners pass external IO in the source kernel's actual order."""

import runpy
from pathlib import Path

import pytest

from ascendflow.dialects.source import AllocationOp
from benchmarks.kda.fragments import CASES, PARAMETERS, sample, source

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("case", CASES)
def test_fragment_samples_match_external_tensor_contract(case):
    filename, entry = source(case)
    definition = runpy.run_path(str(ROOT / filename))[entry]
    module = definition.specialize(**(PARAMETERS | CASES[case][1])).to_ir()
    external = {
        op.sym_name.data: op.ref.type
        for op in module.walk()
        if isinstance(op, AllocationOp)
        and op.ref.type.memory.data == "GM"
        and op.access.data != "local"
    }
    pytest.importorskip("torch")
    host, expected = sample(case, names=tuple(external))
    assert tuple(host) == tuple(external)
    assert expected.keys() <= host.keys()
    for name, tensor in host.items():
        assert tuple(tensor.shape) == tuple(n.data for n in external[name].shape)
