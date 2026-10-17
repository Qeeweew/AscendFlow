import pytest

from ascendflow.diagnostics import CompileError
from ascendflow.performance import profile_evidence, theoretical_metrics


def test_roofline_uses_explicit_array_and_bandwidth_assumptions():
    work = {
        "native_cube_flops": 68_719_476_736,
        "useful_cube_flops": 68_719_476_736,
        "gm_instruction_request_bytes": 1234567,
        "unique_gm_io_allocation_bytes": 33554432,
    }
    hardware = {"aic_cores": 20, "clocks": {"aic_max": {"mhz": 1800}}}
    metrics = theoretical_metrics(work, hardware, 2500)
    assert metrics["conditional_cube_peak_tflops"] == pytest.approx(294.912)
    assert metrics["conditional_cube_only_min_us"] == pytest.approx(233.016888889)
    assert metrics["bandwidth_scenarios"] == {} and metrics["hbm_gbps"] is None
    scenario = theoretical_metrics(work, hardware, 2500, hbm_gbps=1600)
    assert scenario["bandwidth_scenarios"]["cold_unique_io"]["ideal_us"] == pytest.approx(20.97152)


@pytest.mark.parametrize("task_type", ["MIX_AIC", "AI_VECTOR_CORE"])
def test_profile_counters_reject_stale_timing_and_wrong_card(tmp_path, task_type):
    import hashlib
    import sqlite3

    source = tmp_path / "op_summary.csv"
    source.write_text(
        "Device_id,Op Name,Task Type,Task Start Time(us),Task Duration(us),aic_mac_ratio\n"
        f"1,flash_attention,{task_type},1,50,0.9\n"
        f"1,flash_attention,{task_type},2,10,0.2\n"
        f"1,flash_attention,{task_type},3,12,0.4\n"
    )
    summary = {
        "csv": str(source),
        "csv_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "warmup": 1,
        "repeats": 2,
        "durations_us": [10.0, 12.0],
    }
    db = tmp_path / "PROF_example/device_1/sqlite/freq.db"
    db.parent.mkdir(parents=True)
    with sqlite3.connect(db) as database:
        database.execute("CREATE TABLE FreqParse(syscnt INTEGER, freq INTEGER)")
        database.execute("INSERT INTO FreqParse VALUES (100, 1800)")
    evidence = profile_evidence(
        tmp_path,
        summary,
        physical=1,
        warmup=1,
        repeats=2,
        kernel="flash_attention",
        task_type=task_type,
    )
    assert evidence["pipe_activity_medians"]["aic_mac_ratio"] == pytest.approx(0.3)
    assert evidence["frequency_records"][0]["samples_syscnt_mhz"] == [(100, 1800)]
    with pytest.raises(CompileError, match="one matching device task"):
        profile_evidence(
            tmp_path,
            summary,
            physical=0,
            warmup=1,
            repeats=2,
            kernel="flash_attention",
            task_type=task_type,
        )
    source.write_text(source.read_text().replace(",12,", ",120,"))
    with pytest.raises(CompileError, match="CSV changed"):
        profile_evidence(
            tmp_path,
            summary,
            physical=1,
            warmup=1,
            repeats=2,
            kernel="flash_attention",
            task_type=task_type,
        )
