"""Automatic cache lifetime, owner isolation and complete snapshot keys."""

import gc
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Barrier

import pytest

from ascendflow.memo import memo_scope, memo_table, memoized, memoized_method
from ascendflow.scalar import Fact


@pytest.mark.parametrize("frozen", (False, True))
def test_method_cache_belongs_to_each_owner_and_does_not_retain_it(frozen):
    calls = []

    @dataclass(frozen=frozen)
    class Owner:
        name: str

        @memoized_method
        def query(self, value):
            calls.append((self.name, type(value), value))
            return None if value is None else False if value is False else (type(value), value)

    first, second = Owner("first"), Owner("second")
    for value in (None, False, 1, True):
        assert first.query(value) is first.query(value)
        second.query(value)
    assert len(calls) == 8
    assert first.query(1) != first.query(True)
    reference = weakref.ref(first)
    del first
    gc.collect()
    assert reference() is None


def test_build_cache_snapshots_facts_retries_exceptions_and_preserves_argument_errors():
    calls = []

    @memoized(key=lambda facts: frozenset(facts.items()))
    def query(facts):
        calls.append(dict(facts))
        if facts["x"].minimum == 0:
            raise RuntimeError("failed calculation")
        return object()

    facts = {"x": Fact()}
    with memo_scope():
        for _ in range(2):
            with pytest.raises(RuntimeError, match="failed calculation"):
                query(facts)
        facts["x"] = Fact(minimum=1)
        first = query(facts)
        assert query(facts=facts) is first
        facts["x"] = Fact(minimum=2)
        assert query(facts) is not first
        with pytest.raises(TypeError, match="unexpected"):
            query(facts, unexpected=True)
    assert len(calls) == 4


def test_nested_build_restores_outer_cache_and_clears_aliased_tables_on_failure():
    @memoized(key=lambda value: value)
    def query(value):
        return object()

    with memo_scope():
        outer = query(1)
        with memo_scope():
            assert query(1) is outer
        with pytest.raises(RuntimeError), memo_scope(fresh=True):
            assert query(1) is not outer
            inner_table = memo_table(query)
            raise RuntimeError("cancelled build")
        assert inner_table == {}
        assert query(1) is outer
        outer_table = memo_table(query)
    assert outer_table == {} and memo_table(query) is None
    assert query(1) is not query(1)


def test_concurrent_builds_never_share_tables():
    barrier = Barrier(2)

    @memoized(key=lambda: ())
    def query():
        return object()

    def build(_):
        with memo_scope(fresh=True):
            value = query()
            barrier.wait(timeout=5)
            assert query() is value
            return value

    with ThreadPoolExecutor(max_workers=2) as pool:
        left, right = pool.map(build, range(2))
    assert left is not right
