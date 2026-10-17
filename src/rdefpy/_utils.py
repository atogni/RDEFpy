# Copyright (c) 2026 Morehead State University
# Author: Andrea Togni
# SPDX-License-Identifier: MIT

"""Shared helpers for the RDEF reader/writer.

Time coercion (CCSDS ASCII Time Code B and ISO-8601), the colon-aligned
row formatter and number formatter used by every ``__str__``, and the
progress display.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta

import numpy as np


def _parse_ccsds_string(text: str) -> datetime:
    """Parse a CCSDS ASCII Time Code B or ISO-8601 string to datetime.

    Accepts ``YYYYDDDThh:mm:ss``, ``YYYY-DDDThh:mm:ss``, and standard
    ISO-8601 strings (e.g. ``2026-02-07T12:00:00``).
    """
    if len(text) == 16 and text[7] == "T" and text[:4].isdigit():
        try:
            year = int(text[0:4])
            doy = int(text[4:7])
            hour = int(text[8:10])
            minute = int(text[11:13])
            second = int(text[14:16])
            base = datetime(year, 1, 1, tzinfo=UTC)
            return base + timedelta(
                days=doy - 1, hours=hour, minutes=minute, seconds=second
            )
        except ValueError:
            pass  # malformed numeric field; fall through to ISO-8601
    if len(text) == 17 and text[4] == "-" and text[8] == "T":
        try:
            year = int(text[0:4])
            doy = int(text[5:8])
            hour = int(text[9:11])
            minute = int(text[12:14])
            second = int(text[15:17])
            base = datetime(year, 1, 1, tzinfo=UTC)
            return base + timedelta(
                days=doy - 1, hours=hour, minutes=minute, seconds=second
            )
        except ValueError:
            pass  # malformed numeric field; fall through to ISO-8601
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def _to_datetime(value: datetime | np.datetime64 | str) -> datetime:
    """Coerce *value* to a timezone-aware ``datetime``.

    Accepts ``datetime``, ``numpy.datetime64``, or an ISO-8601 / CCSDS
    string (e.g. ``"2026-01-15T12:00:00"``, ``"2026-039T12:00:00"``).
    """
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value
    if isinstance(value, np.datetime64):
        ts = (value - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(1, "s")
        return datetime.fromtimestamp(float(ts), tz=UTC)
    if isinstance(value, str):
        return _parse_ccsds_string(value)
    raise TypeError(
        f"Expected datetime, np.datetime64, or str, got {type(value).__name__}"
    )


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
        return np.datetime64(dt, "ns")
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
    if abs(x) < 10.0**-decimals:
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
