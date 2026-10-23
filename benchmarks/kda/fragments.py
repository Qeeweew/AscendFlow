"""Build and numerically check the standalone KDA prefill fragments."""

import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path

from ascendflow.build import build_file
from ascendflow.device import health_snapshot
from ascendflow.diagnostics import require
from benchmarks.kda.run import digest, error_metrics

# Small complete cohorts exercise repeated ring reuse and both AIV participants.
PARAMETERS = {"CHUNKS": 2, "H": 2, "GROUPS": 1, "DTYPE": "bf16"}
CASES = {
    "gram": ("prefill_gram", {}),
    "gram-cross": ("prefill_gram_cross", {"PACK": 2}),
    "solve16": ("prefill_solve_merge", {"WIDTH": 16}),
    "solve32": ("prefill_solve_merge", {"WIDTH": 32}),
    "transform": ("prefill_transform", {}),
    "wu": ("prefill_wu", {}),
}
ORDERS = {
    "gram": ("Qp", "Kp", "Gp", "Bp", "Aqk", "Diagonal", "Ai32", "Lengths"),
    "gram-cross": (
        "Qp",
        "Kp",
        "Gp",
        "Bp",
        "Negative",
        "Aqk",
        "Lengths",
        "GramLeft",
        "GramRight",
        "GramSmall",
        "GramLarge",
    ),
    "solve16": ("Negative", "Blocks", "Merge", "Off", "Inverse"),
    "solve32": ("Negative", "Blocks", "Merge", "Off", "Inverse"),
    "transform": ("Qp", "Kp", "Vp", "Gp", "Bp", "Right", "KG", "QG", "Aqk", "A16", "Lengths"),
    "wu": ("Inverse", "Right", "U", "W16Ring", "W32"),
}


def source(case):
    entry = CASES[case][0]
    filename = "prefill_solve" if case.startswith("solve") else entry
    return f"examples/kda/{filename}.py", entry


def sample(case):
    """Independent CPU equations, including untouched output regions and rounding."""
    import torch

    rng = torch.Generator().manual_seed(42)

    def random(shape):
        return torch.randn(shape, generator=rng)

    shape = (2, 2, 64, 128)
    host = {name: (random(shape) / 128**0.5).bfloat16() for name in ("Qp", "Kp", "Vp")}
    host["Gp"] = (-torch.rand(shape, generator=rng) * 0.01).cumsum(-2)
    host["Bp"] = torch.rand((2, 2, 64), generator=rng) * 0.2
    host["Lengths"] = torch.tensor([37, 64], dtype=torch.int64)
    expected = {}

    def workspace(name, dims, dtype=torch.float32, value=0.0):
        host[name] = torch.full(dims, value, dtype=dtype)
        return host[name]

    matrix = (2, 2, 64, 64)
    if case == "gram":
        for name in ("Aqk", "Diagonal", "Ai32"):
            workspace(name, matrix, value=7.0)
            expected[name] = torch.zeros(matrix)
        eye = torch.eye(16)
        for chunk, valid in enumerate(host["Lengths"].tolist()):
            for start in range(0, 64, 16):
                rows = slice(start, start + 16)
                g = host["Gp"][chunk, :, rows]
                decay = (g.unsqueeze(-2) - g.unsqueeze(-3)).exp()
                k, q = (host[n][chunk, :, rows].float() for n in ("Kp", "Qp"))
                kk = (k.unsqueeze(-2) * k.unsqueeze(-3) * decay).sum(-1)
                kk *= host["Bp"][chunk, :, rows, None]
                lower = kk.tril(-1)
                lower[:, max(0, valid - start) :] = 0
                inverse = torch.linalg.solve_triangular(
                    eye + lower, eye.expand(2, 16, 16), upper=False
                )
                qk = (q.unsqueeze(-2) * k.unsqueeze(-3) * decay).sum(-1) / 128**0.5
                qk = qk.tril()
                qk[:, max(0, valid - start) :] = 0
                for name in ("Diagonal", "Ai32"):
                    expected[name][chunk, :, rows, rows] = inverse
                expected["Aqk"][chunk, :, rows, rows] = qk
    elif case == "gram-cross":
        for name in ("Negative", "Aqk"):
            expected[name] = workspace(name, matrix, value=7.0).clone()
        for name, dims in {
            "GramLeft": (1, 2, 2, 128, 128),
            "GramRight": (1, 2, 2, 64, 128),
            "GramSmall": (1, 2, 2, 2, 32, 16),
            "GramLarge": (1, 2, 2, 64, 32),
        }.items():
            workspace(name, dims)
        for start, col, extent in ((16, 0, 16), (48, 32, 16), (32, 0, 32)):
            rows, columns = slice(start, start + extent), slice(col, col + extent)
            g = host["Gp"]
            decay = (g[:, :, rows].unsqueeze(-2) - g[:, :, columns].unsqueeze(-3)).exp()
            k, q = host["Kp"].float(), host["Qp"].float()
            kk = (k[:, :, rows].unsqueeze(-2) * k[:, :, columns].unsqueeze(-3) * decay).sum(-1)
            qk = (q[:, :, rows].unsqueeze(-2) * k[:, :, columns].unsqueeze(-3) * decay).sum(-1)
            for chunk, valid in enumerate(host["Lengths"].tolist()):
                kk[chunk, :, max(0, valid - start) :] = 0
                qk[chunk, :, max(0, valid - start) :] = 0
            expected["Negative"][:, :, rows, columns] = -kk * host["Bp"][:, :, rows, None]
            expected["Aqk"][:, :, rows, columns] = qk / 128**0.5
    elif case.startswith("solve"):
        width = CASES[case][1]["WIDTH"]
        host["Negative"] = random(matrix) / 128
        blocks = workspace("Blocks", matrix)
        for start in range(0, 64, width):
            blocks[:, :, start : start + width, start : start + width] = torch.eye(width) + (
                random((2, 2, width, width)) / 128
            ).tril(-1)
        workspace("Merge", (1, 2, width, width))
        off = workspace("Off", matrix if width == 16 else (1, 2, width, width), value=7.0)
        inverse = workspace("Inverse", matrix, torch.bfloat16, value=7.0)
        if width == 16:
            expected["Off"] = off.clone()
            expected["Inverse"] = inverse.clone()  # This specialization must leave it untouched.
        else:
            expected["Inverse"] = blocks.clone()
        for left in range(0, 64, 2 * width):
            a, b = slice(left, left + width), slice(left + width, left + 2 * width)
            merged = blocks[:, :, b, b] @ host["Negative"][:, :, b, a] @ blocks[:, :, a, a]
            expected["Off" if width == 16 else "Inverse"][:, :, b, a] = merged
        if width == 32:
            expected["Inverse"] = expected["Inverse"].bfloat16()
    elif case == "transform":
        q, k, v = (host[n].float() for n in ("Qp", "Kp", "Vp"))
        g, beta = host["Gp"], host["Bp"]
        last = torch.stack([g[i, :, n - 1] for i, n in enumerate(host["Lengths"].tolist())])
        expected["Right"] = torch.stack(
            (k * beta[..., None] * g.exp(), v * beta[..., None]), dim=2
        ).bfloat16()
        expected["KG"] = (k * (last.unsqueeze(-2) - g).exp()).bfloat16()
        expected["QG"] = ((q / 128**0.5).bfloat16().float() * g.exp()).bfloat16()
        host["Aqk"] = random(matrix)
        expected["A16"] = host["Aqk"].bfloat16()
        for name, value in expected.items():
            workspace(name, value.shape, torch.bfloat16, value=7.0)
    else:
        host["Inverse"] = (torch.eye(64) + (random(matrix) / 128).tril(-1)).bfloat16()
        host["Right"] = (random((2, 2, 2, 64, 128)) / 8).bfloat16()
        products = (host["Inverse"].float().unsqueeze(2) @ host["Right"].float()).bfloat16()
        expected["U"] = products[:, :, 1]
        expected["W32"] = products[:, :, 0].float()
        workspace("U", shape, torch.bfloat16, value=7.0)
        workspace("W32", shape, value=7.0)
        workspace("W16Ring", (1, 2, 64, 128), torch.bfloat16)
    return {name: host[name].contiguous() for name in ORDERS[case]}, expected


def worker(args):
    health = health_snapshot(args.physical, args.output)
    import torch
    import torch_npu  # noqa: F401

    torch.set_num_threads(1)
    torch.npu.set_device(0)
    case = args.case
    filename, entry = source(case)
    config = {
        "case": case,
        "source": filename,
        "entry": entry,
        "parameters": PARAMETERS | CASES[case][1],
    }
    require(
        json.loads((args.artifacts / "fragment.json").read_text()) == config,
        "fragment specialization changed",
    )
    manifest = json.loads((args.artifacts / f"{entry}.compile.json").read_text())
    identity = {suffix: digest(args.artifacts / f"{entry}.{suffix}") for suffix in ("asc", "so")}
    require(
        identity["asc"] == manifest["source_sha256"]
        and identity["so"] == manifest["shared_sha256"],
        "fragment artifact changed",
    )
    host, expected = sample(case)
    inputs = {
        name: hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()
        for name, t in host.items()
    }
    device = {name: t.npu() for name, t in host.items()}
    library = ct.CDLL(str((args.artifacts / f"{entry}.so").resolve()))
    launch = getattr(library, "launch_" + entry)
    launch.argtypes, launch.restype = [ct.c_void_p] * (len(ORDERS[case]) + 1), ct.c_int
    results = []
    for _ in range(3):
        for name in expected:
            device[name].copy_(host[name])
        require(
            launch(
                torch.npu.current_stream().npu_stream, *(device[n].data_ptr() for n in ORDERS[case])
            )
            == 0,
            "fragment launch failed",
        )
        torch.npu.synchronize()
        measured = {name: device[name].cpu() for name in expected}
        metrics = {name: error_metrics(value, expected[name]) for name, value in measured.items()}
        for name, value in measured.items():
            rounded = value.dtype == torch.bfloat16 or (case == "wu" and name == "W32")
            torch.testing.assert_close(
                value,
                expected[name],
                rtol=0.008 if rounded else 1e-5,
                atol=1e-4 if rounded else 1e-6,
            )
        results.append(metrics)
    (args.output / "results.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "configuration": config,
                "health": health,
                "artifacts": identity,
                "commands": manifest["commands"],
                "input_sha256": inputs,
                "checks": results,
                "invocations": 3,
            },
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=CASES)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--physical", type=int)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    if args.build:
        filename, entry = source(args.case)
        parameters = PARAMETERS | CASES[args.case][1]
        build_file(
            Path(__file__).resolve().parents[2] / filename,
            args.output,
            entry=entry,
            parameters=parameters,
            shared=True,
            timeout_s=300,
        )
        (args.output / "fragment.json").write_text(
            json.dumps(
                {"case": args.case, "source": filename, "entry": entry, "parameters": parameters},
                indent=2,
            )
            + "\n"
        )
    else:
        require(
            args.artifacts is not None and args.physical is not None,
            "--artifacts and --physical are required for device checks",
        )
        worker(args)


if __name__ == "__main__":
    main()
