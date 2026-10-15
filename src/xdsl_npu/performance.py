"""Exact physical work accounting and explicitly conditional hardware ceilings.

Effects are byte footprints, not DRAM counters. GM requests can hit the 192 MiB
L2; a warmed benchmark need not fetch each input allocation from HBM again.
This module never substitutes these estimates for measured performance.
"""

import configparser
import csv
import ctypes as ct
import hashlib
import re
import sqlite3
import subprocess
from collections import Counter
from pathlib import Path
from statistics import median

from .model import ITEMSIZE, require, storage_layout
from .toolchain import CANN


def runtime_work(module, arguments):
    """Count a verified loop invocation without enumerating runtime N.

    After final verification, counting visits refs and command residue sets.
    Verification retains its own graph/search cost; this is not a bound for it.
    These are instruction-requested bytes/elements, not HBM traffic or FLOPs.
    The caller records actual kernel timings and hardware rate assumptions.
    """
    from .sequence_model import is_sequence

    if is_sequence(module):
        from .sequence_sync import _task, verify_sequence

        _, checked = verify_sequence(module, _details=True)
        parts = [_runtime_leaf(analysis, arguments) for analysis in checked.analyses.values()]
        result = dict(parts[0])
        for key in (
            "native_invocations",
            "vector_requested_elements",
            "memory_requested_bytes_by_pipe_bank_mode",
            "transfer_requested_bytes",
            "sync_invocations",
        ):
            total = Counter()
            for part in parts:
                total.update(part[key])
            result[key] = dict(total)
        count = (
            0
            if checked.sequence.summary.bound is None
            else checked.sequence.summary.bound.evaluate(arguments)
        )
        invocations = (
            1
            if checked.sequence.invocation is None
            else checked.sequence.invocation.bound.evaluate(arguments)
        )
        for bridge in checked.bridges:
            if not any(d.contains(count) for d in bridge.domains):
                continue
            for action, point in (
                (("set", bridge.source), ("wait", bridge.destination))
                if bridge.event_id >= 0
                else (("barrier", bridge.destination),)
            ):
                key = f"{action}:{_task(checked.sequence.regions, point).pipe}"
                result["sync_invocations"][key] = result["sync_invocations"].get(key, 0) + result[
                    "groups"
                ] * (max(0, invocations - 1) if bridge.carried else invocations)
        result.pop("tasks_per_group")
        result["region_tasks_per_group"] = {
            name: part["tasks_per_group"] for name, part in zip(checked.analyses, parts)
        }
        if checked.sequence.invocation is not None:
            result.pop("tasks_per_invocation")
            result["region_tasks_per_invocation"] = {
                name: part["tasks_per_invocation"] for name, part in zip(checked.analyses, parts)
            }
        return result
    from .loop_sync import verify_loop

    _, analysis = verify_loop(module, _details=True)
    return _runtime_leaf(analysis, arguments)


def _runtime_leaf(analysis, arguments):
    from .ir import InvocationChannelOp
    from .scalar import from_ssa

    loop, channels, sequences, _, _ = analysis
    require(
        set(arguments) == {size.sym_name.data for size in loop.parameters}
        and all(type(value) is int and value >= 0 for value in arguments.values()),
        "runtime accounting needs every nonnegative integer scalar argument",
    )
    count = loop.constant_count if loop.bound is None else loop.bound.evaluate(arguments)
    invocations = 1 if loop.invocation is None else loop.invocation.bound.evaluate(arguments)
    calls, vectors, memory, transfers, syncs = Counter(), Counter(), Counter(), Counter(), Counter()
    peaks = {}
    for actor, data in loop.actors.items():
        _, peaks[actor] = storage_layout(data.program.storages)
        for task in data.program.tasks:
            iterations = min(count, 1) if task.last else count
            instances = iterations * loop.groups * invocations
            calls[task.native] += instances
            amounts = []
            for index, (ref, mode) in enumerate(zip(task.refs, task.modes)):
                amount = (
                    (
                        iterations * ref.total_nbytes
                        if ref.prefix_extent is None
                        else (
                            max(
                                0,
                                ref.prefix_extent.evaluate(arguments)
                                - max(0, count - 1) * ref.prefix_elements,
                            )
                            if task.last
                            else ref.prefix_extent.evaluate(arguments)
                        )
                        * ITEMSIZE[ref.dtype]
                    )
                    * loop.groups
                    * invocations
                )
                amounts.append(amount)
                for access in mode:
                    skipped = (
                        ref.total_nbytes * loop.groups * invocations
                        if (count and task.native == "mad_first" and index == 2 and access == "R")
                        else 0
                    )
                    memory[f"{task.pipe}:{ref.storage.bank}:{access}"] += amount - skipped
            if task.pipe == "V":
                vectors[task.native] += max(
                    amount // ITEMSIZE[ref.dtype] for amount, ref in zip(amounts, task.refs)
                )
            if task.pipe in {"MTE1", "MTE2", "MTE3", "FIX"}:
                transfers[f"{task.refs[0].storage.bank}->{task.refs[-1].storage.bank}"] += amounts[
                    0
                ]
        for command in sequences[actor]:
            if command.kind == "task":
                continue
            start, end = command.lower, count - command.upper
            if command.last_gap is not None:
                g = count - 1 - command.last_gap
                active = int(start <= g < end and g % command.period in command.residues)
            else:
                active = (
                    sum(
                        max(0, (end - 1 - r) // command.period - (start - 1 - r) // command.period)
                        for r in command.residues
                    )
                    if end > start
                    else 0
                )
            occurrences = active * invocations
            channel = channels.get(command.channel)
            if channel is not None and channel.carry:
                occurrences += min(count, channel.distance) * max(0, invocations - 1)
            syncs[f"{command.kind}:{command.task.pipe}"] += occurrences * loop.groups
    native = {
        (actor, task.uid): task
        for actor, data in loop.actors.items()
        for task in data.program.tasks
    }
    for protocol in loop.op.body.block.ops:
        if isinstance(protocol, InvocationChannelOp):
            occurrences = (
                max(0, invocations - 1)
                * int(count > max(protocol.source_gap.data, protocol.destination_index.data))
                * loop.groups
            )
            for action, endpoint in [
                ("set", protocol.source.data),
                ("wait", protocol.destination.data),
            ]:
                syncs[f"{action}:{native[tuple(endpoint.split('/'))].pipe}"] += occurrences
    footprint = sum(
        root.nbytes.data
        if root.extent_count is None
        else from_ssa(root.extent_count).evaluate(arguments) * root.count_scale.data
        for root in loop.roots.values()
    )
    return {
        "arguments": arguments,
        "tasks_per_group": count * invocations,
        **(
            {"tasks_per_invocation": count, "invocations_per_group": invocations}
            if loop.invocation is not None
            else {}
        ),
        "groups": loop.groups,
        "native_invocations": dict(calls),
        "vector_requested_elements": dict(vectors),
        "memory_requested_bytes_by_pipe_bank_mode": dict(memory),
        "transfer_requested_bytes": dict(transfers),
        "sync_invocations": dict(syncs),
        "terminal_drains": len(loop.actors) * loop.groups,
        "initial_fix_handshakes": sum(data.program.config for data in loop.actors.values())
        * loop.groups,
        "initial_load3d_handshakes": sum(data.program.load3d is not None for data in loop.actors.values()) * loop.groups,
        "local_peak_bytes_per_actor": peaks,
        "declared_gm_working_set_bytes": footprint,
        "scope": "instruction-requested footprints; not HBM transactions, measured bandwidth, or hardware FLOP peak",
    }


def profile_evidence(profile, summary, *, physical, warmup, repeats, kernel, task_type="MIX_AIC"):
    """Bind measured counters/clocks to the exact single-kernel timing CSV.

    The pipe ratios overlap; they are activity counters, not mutually
    exclusive time fractions or a direct attribution of wait latency.
    FreqParse records are samples, not a frequency trace for every cycle.
    """
    require(
        task_type in {"AI_CORE", "AI_VECTOR_CORE", "MIX_AIC", "MIX_AIV"},
        "counter accounting requires a supported NPU task type",
    )
    profile = Path(profile).resolve()
    source = Path(summary["csv"]).resolve()
    require(source.is_relative_to(profile), "timing CSV is outside its profiling run")
    require(
        hashlib.sha256(source.read_bytes()).hexdigest() == summary["csv_sha256"],
        "timing CSV changed since its summary was recorded",
    )
    require(
        summary["warmup"] == warmup and summary["repeats"] == repeats,
        "profiling and device launch counts disagree",
    )
    rows = sorted(csv.DictReader(source.open()), key=lambda row: float(row["Task Start Time(us)"]))
    require(
        len(rows) == warmup + repeats
        and all(
            row["Op Name"] == kernel
            and row["Task Type"] == task_type
            and int(row["Device_id"]) == physical
            for row in rows
        ),
        "counter accounting requires one matching device task per launch",
    )
    measured = rows[warmup:]
    durations = [float(row["Task Duration(us)"]) for row in measured]
    require(durations == summary["durations_us"], "CSV task durations disagree with summary")
    ratios = {
        column: median(float(row[column]) for row in measured)
        for column in measured[0]
        if column.endswith("_ratio") and all(row[column] not in {"", "N/A"} for row in measured)
    }
    frequencies = []
    for path in sorted(profile.glob(f"PROF_*/device_{physical}/sqlite/freq.db")):
        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as database:
            records = database.execute(
                "SELECT syscnt, freq FROM FreqParse ORDER BY syscnt"
            ).fetchall()
        frequencies.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "samples_syscnt_mhz": records,
            }
        )
    return {
        "csv_sha256": summary["csv_sha256"],
        "task_duration_median_us": median(durations),
        "pipe_activity_medians": ratios,
        "frequency_records": frequencies,
        "limits": "overlapping activity ratios do not identify critical-path stalls; recorded clocks are samples",
    }


def hardware_snapshot(physical):
    """Read installed platform metadata and current/max clocks; no device mutation."""
    board = subprocess.check_output(
        ["npu-smi", "info", "-t", "board", "-i", str(physical), "-c", "0"], text=True
    )
    match = re.search(r"Chip Name\s*:\s*(910B\w*)", board)
    require(match is not None, "hardware snapshot needs an actual 910B device")
    name = "Ascend" + match.group(1)
    path = CANN / "aarch64-linux/data/platform_config" / (name + ".ini")
    data = path.read_bytes()
    config = configparser.ConfigParser(interpolation=None)
    config.read_string(data.decode())
    require(config.getint("version", "NpuArch") == 2201, "hardware is not dav-2201")
    library = ct.CDLL("/usr/local/dcmi/libdcmi.so")
    require(library.dcmi_init() == 0, "DCMI read-only initialization failed")
    query = library.dcmi_get_device_frequency
    query.argtypes = [ct.c_int, ct.c_int, ct.c_int, ct.POINTER(ct.c_uint)]
    query.restype = ct.c_int
    clocks = {}
    for kind, label in ((6, "hbm"), (7, "aic_current"), (9, "aic_max"), (12, "aiv_current")):
        value = ct.c_uint()
        status = query(physical, 0, kind, ct.byref(value))
        clocks[label] = {"status": status, "mhz": value.value if status == 0 else None}
    return {
        "physical_device": physical,
        "soc": name,
        "aic_cores": config.getint("SoCInfo", "cube_core_cnt"),
        "aiv_cores": config.getint("SoCInfo", "vector_core_cnt"),
        "hbm_bytes": config.getint("SoCInfo", "memory_size"),
        "l2_bytes": config.getint("SoCInfo", "l2_size"),
        "local_capacity_bytes": {
            bank: config.getint("AICoreSpec", key)
            for bank, key in (
                ("UB", "ub_size"),
                ("L1", "l1_size"),
                ("L0A", "l0_a_size"),
                ("L0B", "l0_b_size"),
                ("L0C", "l0_c_size"),
            )
        },
        "configured_cube_mhz": config.getint("AICoreSpec", "cube_freq"),
        "clocks": clocks,
        "clock_scope": "single read-only snapshot; current clock is not the run's sustained frequency",
        "configured_memory_rates_raw": dict(config.items("AICoreMemoryRates")),
        "memory_rate_units": "not inferred; raw rates are not treated as measured GB/s",
        "platform_file": str(path.resolve()),
        "platform_sha256": hashlib.sha256(data).hexdigest(),
        "dcmi_header": "/usr/local/dcmi/dcmi_interface_api.h (frequency enums6,7,9,12)",
        "board": board,
    }


def physical_work(plan, *, useful_cube_flops=None):
    """Count EVERY finite native activation, weighted by its actual group grid."""
    from .dispatch import extract_dispatch
    from .ir import StaticDispatchOp
    from .mixed import extract_mixed, mixed_execution
    from .sync import verify_plan

    verify_plan(plan)
    if isinstance(plan.body.block.first_op, StaticDispatchOp):
        cases = extract_dispatch(plan).cases
        mixed_programs = [mixed for _, mixed in cases]
    else:
        mixed_programs = [extract_mixed(plan)]
    cube_flops = native_instances = 0
    dma, gm, vector_elements, vector_calls, syncs = (
        Counter(),
        Counter(),
        Counter(),
        Counter(),
        Counter(),
    )
    capacities = {}
    for mixed in mixed_programs:
        groups = mixed.op.blocks.data
        # Commands must include the actual optimized Set/Wait/drain placement.
        from xdsl.dialects.builtin import ModuleOp

        _, _, commands = mixed_execution(ModuleOp([mixed.op.clone()]))
        for sequence in commands.values():
            syncs.update(
                {
                    kind: n * groups
                    for kind, n in Counter(c.kind for c in sequence if c.kind != "task").items()
                }
            )
        for actor, program in mixed.programs.items():
            _, local = storage_layout(program.storages)
            capacities[f"case{len(capacities) // 3}/{actor}"] = local
            for inst in program.instances():
                task = program.tasks[inst.task]
                native_instances += groups
                lengths = [r.length(inst.generation) * r.rows for r in task.refs]
                if task.native in {"mad", "mad_cycle", "mad_first", "mad16"}:
                    m, n, k = task.params[:3] if task.native != "mad16" else (16, 16, 16)
                    cube_flops += groups * 2 * m * n * k
                if task.pipe in {"MTE1", "MTE2", "MTE3", "FIX"}:
                    route = f"{task.pipe}:{task.refs[0].storage.bank}->{task.refs[-1].storage.bank}"
                    dma[route] += groups * lengths[0]
                    for ref, mode, size in zip(task.refs, task.modes_for(inst.generation), lengths):
                        if ref.storage.bank == "GM":
                            for access in ("R", "W"):
                                if access in mode:
                                    gm[f"{ref.storage.name}:{access}"] += groups * size
                if task.pipe == "V":
                    vector_elements[task.native] += (
                        groups * lengths[0] // ITEMSIZE[task.refs[0].dtype]
                    )
                    calls = 1
                    if (
                        task.native in {"reduce_sum64", "reduce_max64"}
                        and not task.refs[1].contiguous
                    ):
                        calls = task.params[0]
                    elif task.native.endswith("_row_broadcast"):
                        calls = min(task.params[0], task.params[1] // 64)
                    elif task.native in {
                        "reduce_tile_sum64",
                        "reduce_tile_max64",
                    }:
                        calls = task.params[1] // 64
                    vector_calls[task.native] += groups * calls
    roots = mixed_programs[0].roots
    unique_io = sum(
        root.nbytes.data
        for name, root in roots.items()
        if root.initialized.data or gm[f"{name}:W"] and not gm[f"{name}:R"]
    )
    return {
        "native_cube_flops": cube_flops,
        "useful_cube_flops": useful_cube_flops,
        "native_instances": native_instances,
        "dma_source_bytes_by_route": dict(dma),
        "gm_instruction_requests_by_root": dict(gm),
        "gm_instruction_request_bytes": sum(gm.values()),
        "gm_allocation_bytes": sum(root.nbytes.data for root in roots.values()),
        "unique_gm_io_allocation_bytes": unique_io,
        "vector_elements_by_native": dict(vector_elements),
        "vector_intrinsic_call_upper_bounds": dict(vector_calls),
        "sync_instructions": dict(syncs),
        "local_bank_bytes_by_actor": capacities,
        "terminal_pipe_all_calls": sum(m.op.blocks.data * 3 for m in mixed_programs),
        "initial_load3d_handshakes": sum(m.op.blocks.data * sum(p.load3d is not None for p in m.programs.values()) for m in mixed_programs),
        "scope": "exact finite Cube/DMA work; masked Vector call counts are upper bounds; no inferred DRAM misses or CANN reference traffic",
    }


def theoretical_metrics(
    work, hardware, duration_us, *, cube_ops_per_core_cycle=8192, hbm_gbps=None, l2_gbps=None
):
    """A conditional roofline at the observed device's maximum clock.

    8192 assumes one 16x16x16 b16 FMA fractal/core/cycle, counting FMA as
    two operations. It is an analytic array model, not a measured throughput.
    Bandwidth scenarios are optional explicit inputs; no peak is invented
    from HBM MHz or undocumented units in a platform configuration.
    """
    require(
        duration_us > 0 and cube_ops_per_core_cycle > 0, "performance counts/rates must be positive"
    )
    mhz = hardware["clocks"]["aic_max"]["mhz"]
    require(mhz is not None and mhz > 0, "maximum AIC clock is unknown")
    peak = hardware["aic_cores"] * mhz * 1e6 * cube_ops_per_core_cycle
    flop = work["native_cube_flops"]
    requested = work["gm_instruction_request_bytes"]
    useful = work["useful_cube_flops"]
    report = {
        "measured_us": duration_us,
        "native_cube_tflops": flop / duration_us / 1e6,
        "useful_cube_tflops": useful / duration_us / 1e6 if useful is not None else None,
        "conditional_cube_peak_tflops": peak / 1e12,
        "conditional_cube_only_min_us": flop / peak * 1e6,
        "fraction_of_conditional_cube_peak": flop / (duration_us * 1e-6) / peak,
        "cube_model": {
            "ops_per_core_cycle": cube_ops_per_core_cycle,
            "clock_mhz": mhz,
            "assumption": "one complete b16 FMA fractal/core/cycle; ignores instruction startup, Vector, DMA and stalls",
        },
        "gm_requested_gbps": requested / duration_us / 1e3,
        "cube_ops_per_gm_request_byte": flop / requested if requested else None,
        "useful_to_native_cube_work": useful / flop if useful is not None and flop else None,
        "gm_allocation_to_l2_capacity": work["gm_allocation_bytes"] / hardware["l2_bytes"]
        if "gm_allocation_bytes" in work and hardware.get("l2_bytes")
        else None,
        "hbm_gbps": hbm_gbps,
        "l2_gbps": l2_gbps,
        "bandwidth_scenarios": {},
    }
    for label, bandwidth, size in (
        ("cold_unique_io", hbm_gbps, work["unique_gm_io_allocation_bytes"]),
        ("all_gm_requests_via_l2", l2_gbps, work["gm_instruction_request_bytes"]),
    ):
        if bandwidth is not None:
            require(bandwidth > 0, "bandwidth scenario must be positive GB/s")
            report["bandwidth_scenarios"][label] = {
                "assumed_gbps": bandwidth,
                "bytes": size,
                "ideal_us": size / bandwidth / 1e3,
            }
    report["limits"] = (
        "conditional ceilings, not a latency prediction; warm L2 can remove HBM misses, and shared ports/cache effects need device evidence"
    )
    return report
