# Copyright (c) 2026 Morehead State University
# Author: Andrea Togni
# SPDX-License-Identifier: MIT

"""Shared helpers for the RDEF reader/writer.

Time coercion (CCSDS ASCII Time Code B and ISO-8601), the colon-aligned
row formatter and number formatter used by every ``__str__``, the
progress display, and the thread pool that splits large decodes.
"""

from __future__ import annotations

import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import numpy as np

_CCSDS_B = re.compile(r"(\d{4})-?(\d{3}T\d{2}:\d{2}:\d{2})(?:\.(\d+))?Z?")  # CCSDS 301.0-B-4 §3.5.1.2
_ISO_FRACTION = re.compile(r"\d{2}:\d{2}:\d{2}[.,](\d{7,})")  # digits beyond microseconds


def _parse_ccsds_string(text: str) -> datetime:
    """Parse a CCSDS ASCII Time Code B or ISO-8601 string to datetime.

    Accepts ``YYYYDDDThh:mm:ss``, ``YYYY-DDDThh:mm:ss``, either with a
    fraction of a second (``.5``, any number of digits; microseconds
    here, see ``_to_datetime64`` for the rest), and standard ISO-8601
    strings (e.g. ``2026-02-08T12:00:00``). Out-of-range fields raise
    ValueError instead of rolling over.
    """
    m = _CCSDS_B.fullmatch(text)
    if m:
        # Same checks as the observation-file reader; leap second :60 is rejected.
        try:
            dt = datetime.strptime(m[1] + m[2], "%Y%jT%H:%M:%S")
        except ValueError:
            raise ValueError(f"time {text!r}: field out of range") from None
        if dt.year != int(m[1]):  # strptime rolls day 366 of a common year over
            raise ValueError(f"time {text!r}: day of year out of range")
        if m[3]:
            dt = dt.replace(microsecond=int(m[3][:6].ljust(6, "0")))
        return dt.replace(tzinfo=UTC)
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def _to_datetime(value: datetime | str) -> datetime:
    """Coerce *value* to a timezone-aware ``datetime``.

    Accepts ``datetime`` or an ISO-8601 / CCSDS
    string (e.g. ``"2026-02-08T12:00:00"``, ``"2026-039T12:00:00"``).
    """
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value
    return _parse_ccsds_string(value)  # the caller has checked the type


def _to_datetime64(
    value: datetime | np.datetime64 | str | None,
) -> np.datetime64 | None:
    """Coerce *value* to ``numpy.datetime64[ns]`` for time-window filtering.

    Returns ``None`` if the input is ``None``.
    """
    if value is None:
        return None
    if isinstance(value, np.datetime64):
        return np.datetime64(value, "ns")
    if isinstance(value, str | datetime):
        # Same rules as _to_datetime (naive = UTC), then an exact conversion:
        # datetime.timestamp() would read a naive value as LOCAL time and
        # round through a float.
        dt = _to_datetime(value).astimezone(UTC).replace(tzinfo=None)
        out = np.datetime64(dt, "ns")
        # datetime stops at microseconds: add the string's 7th to 9th
        # fractional digits back, so "...00.000000250" keeps its 250 ns.
        frac = _ISO_FRACTION.search(value) if isinstance(value, str) else None
        if frac:
            out += np.timedelta64(int(frac[1][6:9].ljust(3, "0")), "ns")
        return out
    raise TypeError(
        f"Expected datetime, np.datetime64, str, or None, got {type(value).__name__}"
    )


def _fmt_num(x: float, decimals: int = 6) -> str:
    """Thousands separators, at most *decimals* places, trailing zeros trimmed.

    A nonzero value too small for *decimals* places keeps its significant
    digits (``1e-07``) rather than printing as ``0``.
    """
    if x == 0:
        return "0"  # also covers -0.0
    if abs(x) < 10.0**-decimals or abs(x) >= 1e18:  # also no 300-digit row
        return f"{x:.6g}"
    return f"{x:,.{decimals}f}".rstrip("0").rstrip(".")


def _align_rows(rows: list[tuple[str, str]], indent: str = "") -> str:
    """Format ``(label, value)`` pairs with the colons lined up.

    Labels are right-justified to the width of the longest one.
    """
    width = max(len(label) for label, _ in rows)
    return "\n".join(
        f"{indent}{label:>{width}}: {value}".rstrip() for label, value in rows
    )


class _Progress:
    """Progress bar over a byte or record count, or a no-op when disabled.

    A row of ``*`` on stderr under a ``0% ... 100%`` scale, standard library
    only. ``_Progress(total, enabled=False)`` costs two attribute lookups
    per record and draws nothing.
    """

    __slots__ = ("_on", "_n", "_seen", "_drawn")

    _WIDTH = 30

    def __init__(self, total: float, enabled: bool):
        # total <= 0 happens on an empty file; the bar would divide by zero,
        # so disable rather than special-case every call site.
        self._on = enabled and total > 0
        self._n = 0.0
        self._seen = total
        self._drawn = 0
        if self._on:
            w = self._WIDTH
            sys.stderr.write(" " * 8 + "0%" + " " * (w - 6) + "100%\n" + " " * 8)

    def update(self, amount: float) -> None:
        """Advance the bar by *amount* (bytes or records)."""
        self._n += amount
        if self._on:
            self._draw(min(self._n, self._seen))

    def _draw(self, n: float) -> None:
        stars = int(n / self._seen * self._WIDTH)
        if stars > self._drawn:
            sys.stderr.write("*" * (stars - self._drawn))
            sys.stderr.flush()
            self._drawn = stars

    def finish(self) -> None:
        """Jump the bar to 100%.

        For an early break -- e.g. ``stop_epoch`` ending a read before EOF --
        so the bar does not stop half-drawn.
        """
        self._n = self._seen
        if self._on:
            self._draw(self._seen)

    def close(self) -> None:
        """Finish the bar, emitting the trailing newline."""
        if self._on:
            sys.stderr.write("\n")
            sys.stderr.flush()
            self._on = False

    def __enter__(self) -> _Progress:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# Decodes and complex builds of at least this many samples (8 MB of int8
# I/Q, 64 MB at complex128) run in _THREADS disjoint chunks: a fresh array
# that size is bound by first-touch page faults, which threads overlap.
# Measured on one 100 M-sample record, complex128 build: 490 ms on one
# thread, 112 ms on 8 (Cython kernel; 158 ms with NumPy stores).  On 24
# cores, 16 threads cut a 20 Msps complex build 10-18% further; 24 lost.
# Smaller machines get 8, or one per core below that.
_THREADED_MIN_SAMPLES = 4_000_000
_CPUS = os.cpu_count() or 1
_DEFAULT_THREADS = 16 if _CPUS >= 24 else min(8, _CPUS)
_THREADS = _DEFAULT_THREADS
# Lazy module-level pools, never shut down. "cpu" decodes, "io"
# reads, so a read-ahead never queues behind the decode it should overlap.
_pools: dict[str, ThreadPoolExecutor] = {}
_pool_lock = threading.Lock()  # the read-ahead thread and the caller may race to create one


def _drop_pool() -> None:
    # A forked child inherits the pools but none of their threads: they
    # would queue chunks that nothing runs.  Start fresh pools there instead,
    # and a fresh lock: one held by another thread at the fork stays held.
    global _pool_lock
    _pool_lock = threading.Lock()
    _pools.clear()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_drop_pool)


def set_threads(n: int | None = None) -> None:
    """
    Set how many threads split each large read, decode and complex build.

    The default is 16 on machines with at least 24 cores, otherwise 8, or
    one per core below 8.  It was tuned on a 24-core machine: 16 threads
    built 20 Msps complex samples 10-18% faster than 8, and 24 were slower.
    Machines with more cores may gain from more; 1 gives each pool a single
    worker.  Takes effect for calls made after it returns.

    Parameters
    ----------
    n : int or None
        Threads per pool (positive), or None to restore the default.

    Raises
    ------
    ValueError
        If *n* is not a positive int.
    """
    global _THREADS
    if n is None:
        n = _DEFAULT_THREADS
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"threads must be a positive int, got {n!r}")
    with _pool_lock:
        _THREADS = n
        # Next call starts pools of the new size.  No shutdown: a caller may
        # hold an old pool between taking it and mapping on it; dropped pools'
        # threads exit once nothing references them.
        _pools.clear()


def _in_chunks(n: int, per: int, fn, *, pool: str = "cpu") -> None:
    """Call ``fn(a, b)`` over ``[0, n)``: once, or in parallel chunks when large.

    *n* counts input units of *per* samples each; chunk edges fall on whole
    units.  *fn* must release the GIL to gain anything, and write only its
    own ``[a, b)``.
    """
    if n * per < _THREADED_MIN_SAMPLES:
        fn(0, n)
        return
    with _pool_lock:
        ex = _pools.get(pool)
        if ex is None:
            ex = _pools[pool] = ThreadPoolExecutor(_THREADS)
        k = ex._max_workers
    edges = np.linspace(0, n, k + 1).astype(np.int64).tolist()
    list(ex.map(fn, edges[:-1], edges[1:]))
