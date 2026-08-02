"""General-purpose helpers used across the course notebooks."""

from __future__ import annotations

import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

# Binary unit ladder. We use 1024 (not 1000) because that is what `sys.getsizeof`
# and pandas' `memory_usage` report against.
_UNITS = ("B", "KiB", "MiB", "GiB", "TiB")


def format_bytes(n_bytes: float, precision: int = 1) -> str:
    """Render a byte count in the largest unit that keeps it readable.

    Parameters
    ----------
    n_bytes : float
        Size in bytes.
    precision : int, default 1
        Digits after the decimal point.

    Returns
    -------
    str
        Human-readable size, e.g. ``"1.5 MiB"``.

    Examples
    --------
    >>> format_bytes(512)
    '512.0 B'
    >>> format_bytes(1536)
    '1.5 KiB'
    >>> format_bytes(1_073_741_824)
    '1.0 GiB'
    """
    size = float(n_bytes)
    for unit in _UNITS:
        # Stop at the first unit where the value is below 1024, so we never
        # print something like "2048.0 KiB".
        if abs(size) < 1024.0 or unit == _UNITS[-1]:
            return f"{size:.{precision}f} {unit}"
        size /= 1024.0
    # Unreachable, but keeps type checkers happy.
    return f"{size:.{precision}f} {_UNITS[-1]}"


def memory_footprint(obj: Any) -> str:
    """Report the shallow memory footprint of a Python object.

    Returns
    -------
    str
        Formatted size, e.g. ``"56.0 B"``.

    Notes
    -----
    This is :func:`sys.getsizeof`, which is **shallow**: for a container it
    counts the container's own overhead plus its pointers, *not* the objects
    those pointers reference. A list of one million integers therefore reports
    only the list's array of pointers. Module 02 shows why NumPy arrays do not
    have this problem.

    Examples
    --------
    >>> isinstance(memory_footprint([1, 2, 3]), str)
    True
    """
    return format_bytes(sys.getsizeof(obj))


@contextmanager
def timer(label: str = "elapsed", precision: int = 4) -> Iterator[None]:
    """Time a block of code and print how long it took.

    Uses :func:`time.perf_counter`, the highest-resolution clock available, and
    reports even if the block raises — so a failed run still tells you where the
    time went.

    Parameters
    ----------
    label : str, default "elapsed"
        Text printed alongside the duration.
    precision : int, default 4
        Digits after the decimal point, in seconds.

    Examples
    --------
    >>> with timer("summing"):        # doctest: +SKIP
    ...     total = sum(range(1_000_000))
    summing: 0.0231 s
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        # `finally` guarantees the timing prints even when the block raises,
        # which is exactly when you most want to know how far it got.
        duration = time.perf_counter() - start
        print(f"{label}: {duration:.{precision}f} s")
