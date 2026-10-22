"""Memoization owned by one compilation or one immutable analysis instance.

Keys must include every semantic input. Mutable IR/range analyses keep their
explicit invalidation; these helpers never infer that an input is unchanged.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache, update_wrapper, wraps

_active = ContextVar("ascendflow_memo", default=None)
_MISSING = object()


@contextmanager
def memo_scope(*, fresh=False):
    """Release all build tables on exit; nested builds can request isolation."""
    if not fresh and _active.get() is not None:
        yield
        return
    tables = {}
    token = _active.set(tables)
    try:
        yield
    finally:
        _active.reset(token)
        for table in tables.values():
            table.clear()
        tables.clear()


def memo_table(namespace):
    """A table in the current build, or None outside a build scope."""
    tables = _active.get()
    return None if tables is None else tables.setdefault(namespace, {})


def memoized(*, key):
    """Automatically cache a pure build query with an explicit semantic key."""

    def decorate(function):
        @wraps(function)
        def query(*args, **kwargs):
            table = memo_table(query)
            if table is None:
                return function(*args, **kwargs)
            try:
                query_key = key(*args, **kwargs)
                result = table.get(query_key, _MISSING)
            except TypeError:
                return function(*args, **kwargs)
            if result is _MISSING:
                result = function(*args, **kwargs)
                table[query_key] = result
            return result

        return query

    return decorate


class memoized_method:
    """Install an unbounded functools cache on the instance, never the class.

    The owner need not be hashable. Its dictionary owns the bound cache and is
    collected with it; no global method cache retains old analysis objects.
    """

    def __init__(self, function):
        update_wrapper(self, function)

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        method = lru_cache(maxsize=None, typed=True)(self.__wrapped__.__get__(instance, owner))
        instance.__dict__[self.__name__] = method
        return method

    def __call__(self, *args, **kwargs):
        return self.__wrapped__(*args, **kwargs)
