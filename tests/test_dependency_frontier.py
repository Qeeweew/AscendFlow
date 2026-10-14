"""Finite byte-frontier differential; not an arbitrary-size formal proof."""

import random

import pytest

from xdsl_npu.model import CompileError, Instance, Program, Ref, Storage, Task, dependencies


def byte_frontier(program):
    writers, readers, hazards = {}, {}, set()
    for instance in program.instances():
        task = program.tasks[instance.task]
        accesses = [
            ((ref.storage.name, address), mode, ref.storage.initialized)
            for ref, mode in zip(task.refs, task.modes_for(instance.generation))
            for start, stop in ref.segments(instance.generation)
            for address in range(start, stop)
        ]
        for cell, mode, initialized in accesses:
            if "R" in mode:
                if cell not in writers and not initialized:
                    raise CompileError("undefined")
                if cell in writers and writers[cell] != instance:
                    hazards.add((writers[cell], instance, "RAW", *cell))
                readers.setdefault(cell, set()).add(instance)
        for cell, mode, _ in accesses:
            if "W" in mode:
                if cell in writers and writers[cell] != instance:
                    hazards.add((writers[cell], instance, "WAW", *cell))
                hazards.update(
                    (reader, instance, "WAR", *cell)
                    for reader in readers.get(cell, ())
                    if reader != instance
                )
                writers[cell] = instance
                readers[cell] = set()
    return hazards


@pytest.mark.parametrize("seed", range(32))
def test_exact_cells_and_coalesced_hazards_match_finite_byte_oracle(seed):
    rng = random.Random(seed)
    roots = tuple(Storage(name, "GM", 256, "f32", "linear", True, True) for name in ["x", "y"])
    tasks = []
    for number in range(12):
        refs, modes = [], []
        for _ in range(rng.randint(1, 3)):
            refs.append(
                Ref(
                    rng.choice(roots),
                    ((rng.randrange(8) * 4, rng.randrange(4) * 4, 2, 1),),
                    rng.choice([4, 8, 16]),
                    "f32",
                    "linear",
                    rows=rng.choice([1, 2, 3]),
                    row_stride=32,
                )
            )
            modes.append(rng.choice(["R", "W", "RW"]))
        tasks.append(
            Task(
                f"t{number}",
                "materialized_effects",
                "V",
                rng.randrange(3),
                tuple(refs),
                tuple(modes),
                0,
                (),
                upper=6,
                period=3,
                phase=rng.randrange(3),
                logical_phase=rng.randrange(3),
            )
        )
    program = Program("finite_frontier", "mixed", 6, roots, tuple(tasks), False)
    actual = {
        (o.source, o.destination, o.hazard, o.storage, address)
        for o in dependencies(program)
        for address in range(o.start, o.stop)
    }
    assert actual == byte_frontier(program)


def test_coalescing_does_not_fill_a_hole_or_drop_a_last_reader():
    root = Storage("ring", "GM", 128, "f32", "linear", True, True)
    view = Ref(root, ((0, 0, 0, 1),), 8, "f32", "linear", rows=2, row_stride=32)
    tasks = tuple(
        Task(uid, "materialized_effects", "V", 0, (view,), (mode,), 0, (), upper=1)
        for uid, mode in [("r0", "R"), ("r1", "R"), ("write", "W")]
    )
    program = Program("holes", "mixed", 1, (root,), tasks, False)
    war = [o for o in dependencies(program) if o.hazard == "WAR"]
    assert {(o.start, o.stop) for o in war} == {(0, 8), (32, 40)}
    assert {o.source for o in war} == {Instance(0, 0), Instance(0, 1)}


def test_undefined_cells_are_rejected_without_an_import_assumption():
    root = Storage("local", "UB", 32, "f32", "linear", False, False)
    view = Ref(root, ((0, 0, 0, 1),), 32, "f32", "linear")
    task = Task("bad", "materialized_effects", "V", 0, (view,), ("R",), 0, (), upper=1)
    program = Program("bad", "aiv", 1, (root,), (task,), False)
    with pytest.raises(CompileError, match="reads undefined local"):
        dependencies(program)
