"""One collection per key case, real CSV summarization, and no PMU overrides."""

import csv
import json
import sys
from pathlib import Path

from benchmarks import profile_defaults


def test_two_default_collections_extract_device_time_without_warmup(monkeypatch, tmp_path):
    collected = []
    monkeypatch.setattr(profile_defaults, "health_snapshot", lambda *_: {"healthy": True})
    monkeypatch.setattr(profile_defaults, "bounded_process", lambda *_a, **_k: {"status": "passed"})

    def collect(command, directory):
        # Only the output location is an msprof option. All later options
        # belong to the actual application, which reuses the built artifact.
        assert command[0] == "msprof" and command[1].startswith("--output=")
        assert Path(command[2]).name == "profile.sh"
        assert "--build" not in command
        collected.append(command[3])
        warmup, repeats = (1, 3) if command[3] == "benchmarks.attention.run" else (5, 30)
        profile = Path(command[1].split("=", 1)[1])
        profile.mkdir(parents=True)
        (profile / "device.json").write_text(
            json.dumps({"status": "passed", "cases": [{"warmup": warmup, "repeats": repeats}]})
        )
        with (profile / "op_summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "Task Type",
                    "Op Name",
                    "Task Start Time(us)",
                    "Task Duration(us)",
                    "Block Num",
                ],
            )
            writer.writeheader()
            for i in range(warmup + repeats):
                writer.writerow(
                    {
                        "Task Type": "AI_CORE",
                        "Op Name": "test_kernel",
                        "Task Start Time(us)": i * 1000,
                        "Task Duration(us)": 999 if i < warmup else 7.5,
                        "Block Num": 20,
                    }
                )

    monkeypatch.setattr(profile_defaults, "run_device", collect)
    monkeypatch.setattr(
        sys, "argv", ["profile_defaults", "--output", str(tmp_path / "profiles"), "--physical", "2"]
    )
    assert profile_defaults.main() == 0
    assert collected == ["benchmarks.attention.run", "benchmarks.gemm.run_matmul"]
    report = json.loads((tmp_path / "profiles/results.json").read_text())
    assert report["status"] == "passed"
    assert [row["kernel_time_us"] for row in report["cases"]] == [7.5, 7.5]
