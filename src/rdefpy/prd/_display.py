# Copyright (c) 2026 Morehead State University
# Author: Andrea Togni
# SPDX-License-Identifier: MIT

"""
Human-readable rendering of RDEF product-file objects.

Everything that formats a ``RecordHeader``, ``RecordData``, ``Record`` or
``ProductFile`` lives here; the classes in ``classes.py`` keep only thin
``__repr__`` / ``__str__`` / ``info`` / ``dump*`` delegates.

Conventions:

* ``repr`` is one line, ``<ClassName ...>``; ``str`` is a title line --
  the class name plus an identifier, nothing else -- then right-aligned
  ``label: value`` rows.
* Labels are the attribute names (``station_id``, ``timetag``), so what you
  read is what you type.  Derived values (``band``, ``fixed_downconv_hz``,
  ``frequency_steering``) are named for what they are.
* Units: a label that carries its unit (``duration_s``,
  ``fixed_downconv_hz``) is not repeated in the value;
  otherwise every physical value shows its unit (Table 5-1 units, in
  plain ASCII: ``turns/s^2``).
* Numbers: thousands separators, at most six decimals, trailing zeros
  trimmed.  Each block shows only its own level; nothing repeats.
* Time tags are full ISO 8601 with nanoseconds and a ``Z`` suffix --
  never truncated.  They are station time, an approximation to UTC
  (§5.2.5), not converted.
* Plain text only: no terminal colour codes.  Nothing here prints except
  the dumps, which write where they are told.  The dumps write raw
  values: no separators, no units; reserved byte fields as length
  summaries.
* Rendering never raises on a half-built object; unusable values show as
  ``?``.

The module has no runtime import from ``classes.py`` -- field order comes
from ``_binary._FIELDS`` (a header's ``__slots__`` also hold the per-file
settings ``_t0_offset`` and ``_agency_layout``) -- and imports ``reader``
lazily, so there is no import cycle.
"""

from __future__ import annotations

import math
import numbers
import os
import sys
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

import numpy as np

from rdefpy._utils import _align_rows, _fmt_num
from rdefpy.prd._binary import _FIELDS

if TYPE_CHECKING:
    from rdefpy.prd.classes import ProductFile, Record, RecordData, RecordHeader

# Reserved byte fields of Table 5-1, shown as length summaries.
_RESERVED = frozenset(
    {"empty_fields_future_extension", "empty_fields_internal_agency_use"}
)

# Identifiers, flags and calendar fields read wrong with thousands separators.
_NO_GROUPING = {
    "record_version_id", "station_id", "spacecraft_id", "validity_flag",
    "agency_flag", "time_tag_year", "time_tag_doy",
}


# Units of the Table 5-1 fields that have one, shown after the value.
_UNITS = {
    "record_length": "bytes",
    "sample_size": "bit",
    "sample_rate": "Hz",
    "rf_to_if_downconv": "Hz",
    "if_to_channel_downconv": "Hz",
    "time_tag_second_of_day": "s",
    "time_tag_picoseconds_of_the_second": "ps",
    "channel_accumulated_phase": "turns",
    "channel_phase_polynomial_coefficient_0": "turns",
    "channel_phase_polynomial_coefficient_1": "turns/s",
    "channel_phase_polynomial_coefficient_2": "turns/s^2",
    "channel_phase_polynomial_coefficient_3": "turns/s^3",
}


# =============================================================================
# Value formatting
# =============================================================================


def _is_num(val) -> bool:
    """A real number, NumPy scalars included; bool is a flag, not a number."""
    return isinstance(val, numbers.Real) and not isinstance(val, bool)


def _fmt_header_value(name: str, val) -> str:
    """Header value for display: magnitudes get thousands separators."""
    if val is None:
        return "?"
    if name in _NO_GROUPING:
        return str(val)
    if isinstance(val, numbers.Integral) and not isinstance(val, bool):
        return f"{int(val):,}"
    if _is_num(val):
        return _fmt_num(float(val))
    return str(val)


def _fmt_id(val) -> str:
    return "?" if val is None else str(val)


def _fmt_timetag(hdr: RecordHeader) -> str:
    """Full time tag, ISO 8601 UTC with nanoseconds: ``...T12:00:00.000000123Z``."""
    try:
        return np.datetime_as_string(hdr.timetag, unit="ns") + "Z"
    except (TypeError, ValueError, OverflowError):
        return "?"


def _fmt_rate(rate) -> str:
    """Whole-number rate in Hz (a float such as 1000.0 included), else ``?``."""
    if _is_num(rate) and math.isfinite(rate) and rate == int(rate):
        return f"{int(rate):,} Hz"
    return "?"


def _fmt_size(size) -> str:
    return f"{size} bit" if size is not None else "?"


def _fixed_downconv_hz(hdr: RecordHeader) -> float | None:
    """Fixed part ``rf_to_if + if_to_channel`` (§5.2.6), or None if unset."""
    rf, ifc = hdr.rf_to_if_downconv, hdr.if_to_channel_downconv
    if rf is None or ifc is None:
        return None
    return float(rf) + float(ifc)


def _downlink_band(hdr: RecordHeader) -> str:
    """
    Classify the fixed downconversion frequency into a DSN band.

    Returns ``'S'``, ``'X'`` or ``'Ka'`` when ``rf_to_if + if_to_channel``
    falls in that band's nominal allocation (+/- guard), else ``'unknown'``.
    """
    # Lazy import: reader -> classes -> _display -> reader is a cycle.
    from rdefpy.prd.reader import _BAND_GUARD_HZ, _DOWNLINK_BANDS_HZ

    total = _fixed_downconv_hz(hdr)
    if total is None:
        return "unknown"
    for band, (lo, hi) in _DOWNLINK_BANDS_HZ.items():
        if lo - _BAND_GUARD_HZ <= total <= hi + _BAND_GUARD_HZ:
            return "Ka" if band == "KA" else band
    return "unknown"


def _steering(hdr: RecordHeader) -> bool:
    """Frequency steering within the record: any nonzero c2 or c3 (§5.2.6)."""
    return any(
        float(getattr(hdr, f"channel_phase_polynomial_coefficient_{k}") or 0.0) != 0.0
        for k in (2, 3)
    )


def _field_text(name: str, val) -> str:
    """One header field for display; reserved bytes as a length summary."""
    if name in _RESERVED and isinstance(val, bytes | bytearray):
        nonzero = sum(1 for b in val if b)
        return f"{len(val)} bytes, " + (f"{nonzero} non-zero" if nonzero else "all zero")
    text = _fmt_header_value(name, val)
    if val is not None and name in _UNITS:
        text += " " + _UNITS[name]
    return text


# =============================================================================
# repr: one line
# =============================================================================


def _short(hdr: RecordHeader) -> str:
    return (
        f"{_fmt_timetag(hdr)} st={_fmt_id(hdr.station_id)} sc={_fmt_id(hdr.spacecraft_id)} "
        f"{_fmt_size(hdr.sample_size)} {_fmt_rate(hdr.sample_rate)}"
    )


def header_repr(hdr: RecordHeader) -> str:
    return f"<RecordHeader {_short(hdr)}>"


def _fmt_samples(data: RecordData) -> str:
    """Sample count; ``I n / Q m`` when only one of them is set (``len`` raises then)."""
    if data._i.shape != data._q.shape:
        return f"I {data._i.size:,} / Q {data._q.size:,}"
    return f"{len(data):,}"


def data_repr(data: RecordData) -> str:
    return (
        f"<RecordData {_fmt_samples(data)} samples {_fmt_size(data.sample_size)} "
        f"{_fmt_rate(data.sample_rate)}>"
    )


def record_repr(rec: Record) -> str:
    # Header only: len(rec.data) would read an unloaded record from disk.
    return f"<Record {_short(rec.header)}>"


def product_repr(pf: ProductFile) -> str:
    name = f" {pf.name!r}" if pf.name else ""
    return f"<ProductFile{name} {len(pf.records):,} records>"


# =============================================================================
# str: aligned blocks
# =============================================================================


def header_str(hdr: RecordHeader) -> str:
    """Every Table 5-1 field by attribute name, then the derived ``timetag``."""
    rows = [(name, _field_text(name, getattr(hdr, name))) for name in _FIELDS]
    rows.append(("timetag", _fmt_timetag(hdr)))
    return "RecordHeader\n" + _align_rows(rows, indent="  ")


def data_str(data: RecordData) -> str:
    """Size, rate and storage of one record's I/Q (no pass over the samples)."""
    rows: list[tuple[str, str]] = [
        ("samples", _fmt_samples(data)),
        ("sample_size", _fmt_size(data.sample_size)),
        ("sample_rate", _fmt_rate(data.sample_rate)),
    ]
    try:
        rows.append(("duration_s", _fmt_num(data.duration_s)))
    except (TypeError, ValueError, ZeroDivisionError):  # no usable rate
        pass
    rows += [
        ("dtype", np.dtype(data.dtype).name),
        ("complex_dtype", np.dtype(data.complex_dtype).name),
    ]
    return "RecordData\n" + _align_rows(rows, indent="  ")


def _fmt_f_var(hdr: RecordHeader) -> str:
    """
    Variable downconversion frequency ``c1 + 2*c2*(t-t0) + 3*c3*(t-t0)^2``
    (§5.2.6.3) in Hz, t0 the polynomial origin, factors
    multiplied in.
    Display only: zero terms are left out, decimals cut to 6 significant
    digits (integer digits are all kept).
    """
    try:
        a, b, c = (k * float(getattr(hdr, f"channel_phase_polynomial_coefficient_{k}"))
                   for k in (1, 2, 3))
    except (TypeError, ValueError):  # half-built header
        return "?"

    def sig(x: float) -> str:
        """*x* >= 0 to 6 significant digits; exponent form below 1e-9 or from 1e18."""
        if not math.isfinite(x) or x < 1e-9 or x >= 1e18:  # garbage: no 300-digit row
            return f"{x:.6g}"
        d = max(0, 5 - math.floor(math.log10(x)))
        s = f"{x:,.{d}f}"
        return s.rstrip("0").rstrip(".") if d else s

    terms = [(x, u) for x, u in ((a, ""), (b, "*(t-t0)"), (c, "*(t-t0)^2")) if x != 0]
    if not terms:
        return "0"
    text = " ".join(f"{'-' if x < 0 else '+'} {sig(abs(x))}{u}" for x, u in terms)
    return text[2:] if text[0] == "+" else "-" + text[2:]


def _fmt_count(rec: Record) -> str:
    """Sample count from memory or the data-section size; never a disk read."""
    try:
        return f"{len(rec):,}"
    except (TypeError, ValueError, ZeroDivisionError):  # half-built header
        return "?"


def record_str(rec: Record) -> str:
    """One-record summary from the header only (never reads samples from disk)."""
    hdr = rec.header
    title = "Record" if rec._index is None else f"Record {rec._index:,}"
    f_fixed = _fixed_downconv_hz(hdr)
    rows = [
        ("timetag", _fmt_timetag(hdr)),
        ("sample_rate", _fmt_rate(hdr.sample_rate)),
        ("sample_size", _fmt_size(hdr.sample_size)),
        ("samples", _fmt_count(rec)),
        ("loaded", str(rec._data is not None)),
        ("spacecraft_id", _fmt_id(hdr.spacecraft_id)),
        ("station_id", _fmt_id(hdr.station_id)),
        ("band", _downlink_band(hdr)),
        ("fixed_downconv_hz", "?" if f_fixed is None else _fmt_num(f_fixed)),
        ("variable_downconv_hz", _fmt_f_var(hdr)),
    ]
    for k in range(4):
        name = f"channel_phase_polynomial_coefficient_{k}"
        rows.append((f"phase_coef_{k}", _field_text(name, getattr(hdr, name))))
    return title + "\n" + _align_rows(rows, indent="  ")


def product_str(pf: ProductFile) -> str:
    """
    Product-file summary (see ``ProductFile.info``).

    Fields that vary across records list each distinct value with its
    record count, in first-seen order.
    """
    title = f"ProductFile {pf.name}" if pf.name else "ProductFile"
    if not pf.records:
        return f"{title} (empty, 0 records)"

    headers = [r.header for r in pf.records]
    try:
        tt = np.array([h.timetag for h in headers], dtype="datetime64[ns]")
    except (TypeError, ValueError, OverflowError):
        tt = None

    def _one(values, fmt=str) -> str:
        # One value: as is. Several: each with its record count, so a stray
        # station or spacecraft ID shows how many records it owns.
        counts = Counter(values)  # insertion-ordered: first-seen order
        if len(counts) == 1:
            return fmt(next(iter(counts)))
        return ", ".join(f"{fmt(v)} ({c:,} rec)" for v, c in counts.items())

    if tt is None:
        first = last = consecutive = "?"
    else:
        first = _fmt_timetag(headers[int(np.argmin(tt))])
        last = _fmt_timetag(headers[int(np.argmax(tt))])
        # 1 s record spacing (§5.1) on full time tags: crosses year
        # boundaries and includes picoseconds.
        dt_sec = np.diff(tt) / np.timedelta64(1, "s")
        if dt_sec.size == 0 or np.allclose(dt_sec, 1.0, rtol=0, atol=1e-6):
            consecutive = "True"
        else:
            consecutive = f"False (min dt {dt_sec.min():.6f} s, max dt {dt_sec.max():.6f} s)"

    # Steering across the file: curvature in any record, or c1 changing.
    try:
        c1 = {float(h.channel_phase_polynomial_coefficient_1 or 0.0) for h in headers}
        c1 = {"nan" if v != v else v for v in c1}  # NaN != NaN: one NaN is no change
        steering = str(any(_steering(h) for h in headers) or len(c1) > 1)
    except (TypeError, ValueError):  # a coefficient that is not a number
        steering = "?"

    rows = [
        ("path", _fmt_id(pf.path)),
        ("records", f"{len(headers):,}"),
        ("first_timetag", first),
        ("last_timetag", last),
        ("consecutive", consecutive),
        ("sample_rate", _one([h.sample_rate for h in headers], _fmt_rate)),
        ("sample_size", _one([h.sample_size for h in headers], _fmt_size)),
        ("spacecraft_id", _one([h.spacecraft_id for h in headers], _fmt_id)),
        ("station_id", _one([h.station_id for h in headers], _fmt_id)),
        ("band", _one([_downlink_band(h) for h in headers])),
        ("frequency_steering", steering),
    ]
    return title + "\n" + _align_rows(rows, indent="  ")


# =============================================================================
# Printing and text dumps
# =============================================================================


def dump(
    pf: ProductFile, dest: str | os.PathLike | TextIO | None = None, *, data: bool = True
) -> None:
    """Write headers, and I/Q pairs if *data*, as text (see
    ``ProductFile.dump`` / ``ProductFile.dump_headers``).

    *dest*: None prints to stdout; a path writes a new UTF-8 file, removed
    again if the dump fails; a text stream is written to and left open.
    A path naming the product file (or a record's source file) is refused:
    opening it for writing would empty it before its samples are read.
    """
    if dest is None:
        return _write_dump(pf, sys.stdout, data)
    if not isinstance(dest, str | os.PathLike):
        return _write_dump(pf, dest, data)

    out = Path(dest)
    if out.exists():
        sources = {pf.path, *(r._source.path for r in pf.records if r._source is not None)}
        for src in sources:
            if src is not None and Path(src).exists() and os.path.samefile(out, src):
                raise ValueError(f"dump would overwrite the product file it reads: {out}")
    with out.open("w", encoding="utf-8") as f:  # failing here deletes nothing
        try:
            _write_dump(pf, f, data)
        except BaseException:
            f.close()
            out.unlink(missing_ok=True)  # no truncated dump left behind
            raise


def _write_dump(pf: ProductFile, f: TextIO, data: bool) -> None:
    f.write(f"RDEF {'Full' if data else 'Header'} Dump -- {len(pf.records)} records\n")
    f.write("=" * 70 + "\n\n")
    for idx, rec in enumerate(pf.records):
        f.write(f"Record {idx}" + (" -- Header\n" if data else "\n"))
        f.write("-" * 40 + "\n")
        for field in _FIELDS:
            val = getattr(rec.header, field)
            text = _field_text(field, val) if field in _RESERVED else val
            f.write(f"  {field}: {text}\n")
        f.write("\n")
        if not data:
            continue

        d = rec.data  # one read per record when it is not loaded
        f.write(f"Record {idx} -- Data ({len(d)} samples)\n")
        f.write("-" * 40 + "\n")
        f.writelines(
            f"  {i}, {q}\n" for i, q in zip(d.i_samples.tolist(), d.q_samples.tolist())
        )
        f.write("\n")
