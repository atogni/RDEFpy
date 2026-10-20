# Copyright (c) 2026 Morehead State University
# Author: Andrea Togni
# SPDX-License-Identifier: MIT

"""
Binary I/O utilities for RDEF product files (.prd).

Implements the low-level binary packing defined in CCSDS 506.1-B-1
(June 2013), "Delta-DOR Raw Data Exchange Format", §5.

Components
----------
Header Unpacking
    unpack_header : Unpack full 176-byte RDEF record header (§5.2, Table 5-1)
    read_header_length : Read minimal 8-byte prefix (label + record_length)

I/Q Data Unpacking
    unpack_iq_16 : 16-bit I/Q samples (bias-corrected, int32 output)
    unpack_iq_8 : 8-bit I/Q samples (int16 output)
    unpack_iq_4 : 4-bit I/Q samples (int8 output)
    unpack_iq_2 : 2-bit I/Q samples (int8 output)
    unpack_iq_1 : 1-bit I/Q samples (int8 output)

Notes
-----
All I/Q unpacking functions apply the ``2k+1`` transform of §5.3.6,
which compensates the offset of the §5.3.5 truncation.  Header layout
follows Table 5-1 / Figure 5-3.  I/Q packing per sample size follows §5.3
and Table 5-2.  Byte order is little-endian throughout (§5.1.5).

The ``2k+1`` transform, the sign extension and the bit offsets ARE the
format: a plausible-looking simplification changes recorded sample values.
Any edit here must keep two test files green:

- ``tests/test_spec_prd_samples.py`` checks every decoder against an
  independent transcription of Table 5-2 (``tests/rdef_spec.py``),
  including hand-worked words, one-slot sweeps and all 256 byte values.
- ``tests/test_rdefpy.py`` pins the decoded I/Q of the test files to fixed
  SHA-256 digests.

These NumPy paths are the reference and the fallback.  The Cython port
(``unpack_cyth.pyx``, built by ``setup.py``) is what ``reader._decode_iq``
uses when the extension is present; it must return identical arrays, and
it copies its 4/2/1-bit tables from ``_BYTE_TABLES`` here.
"""

from __future__ import annotations

import math
import struct
import sys
import warnings
from typing import BinaryIO

import numpy as np

# =============================================================================
# Constants and Structure Definitions
# =============================================================================

# Header struct per CCSDS 506.1-B-1 (Table 5-1 / Figure 5-3).
_RDEF_HEADER_STRUCT = struct.Struct(
    "<4s I H H H H I H H d d H H I d d d d d d 36s 40s i"
)
# Names of the fields between RECORD LABEL and END LABEL, in file order.
_FIELDS = (
    "record_length",
    "record_version_id",
    "station_id",
    "spacecraft_id",
    "sample_size",
    "sample_rate",
    "validity_flag",
    "agency_flag",
    "rf_to_if_downconv",
    "if_to_channel_downconv",
    "time_tag_year",
    "time_tag_doy",
    "time_tag_second_of_day",
    "time_tag_picoseconds_of_the_second",
    "channel_accumulated_phase",
    "channel_phase_polynomial_coefficient_0",
    "channel_phase_polynomial_coefficient_1",
    "channel_phase_polynomial_coefficient_2",
    "channel_phase_polynomial_coefficient_3",
    "empty_fields_future_extension",
    "empty_fields_internal_agency_use",
)
_RDEF_HEADER_PREFIX_STRUCT = struct.Struct("<4s I")
# Derived sizes
_HEADER_SIZE = _RDEF_HEADER_STRUCT.size
_HEADER_PREFIX_SIZE = _RDEF_HEADER_PREFIX_STRUCT.size

# Valid sample sizes per CCSDS 506.1-B-1
_VALID_SAMPLE_SIZES = frozenset([1, 2, 4, 8, 16])

# float64 header fields (Table 5-1); §5.2.4 rules out NaN, -Inf, +Inf and -0.
_FLOAT_FIELDS = (
    "rf_to_if_downconv",
    "if_to_channel_downconv",
    "time_tag_picoseconds_of_the_second",
    "channel_accumulated_phase",
    "channel_phase_polynomial_coefficient_0",
    "channel_phase_polynomial_coefficient_1",
    "channel_phase_polynomial_coefficient_2",
    "channel_phase_polynomial_coefficient_3",
)


def _byte_tables(bits: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Decode tables for the sub-byte sample sizes (4, 2, 1 bits).

    One packed byte holds ``n = 8 // (2*bits)`` complex samples (Table 5-2).
    Row *b* of the I table is the I values of those *n* samples, in time
    order; likewise for Q.  Built from the format rule, not typed in:

    - sample *s* of the byte has I at bit ``2*bits*s`` and Q at
      ``2*bits*s + bits`` (Table 5-2; time runs LSB to MSB, §5.3.4);
    - a *bits*-wide field is the two's-complement code *k*, decoded as
      ``2k+1`` (§5.3.6).

    Each row of *n* int8 values is viewed as one ``n``-byte integer, so a
    decode is a single 1-D gather per lane, ``table[byte_array]``, viewed back
    as int8.  The views only regroup bytes in memory and are undone by the
    final ``.view(np.int8)``, so the result does not depend on endianness.
    ``unpack_cyth.pyx`` copies these same tables, so the two decoders cannot
    drift apart.
    """
    n = 8 // (2 * bits)
    mask = (1 << bits) - 1
    sign = 1 << (bits - 1)
    fields = np.arange(256)[:, None] >> (2 * bits * np.arange(n))[None, :]

    def values(codes: np.ndarray) -> np.ndarray:
        k = (codes ^ sign) - sign  # sign-extend
        return np.ascontiguousarray(2 * k + 1, dtype=np.int8)  # §5.3.6

    lane = {1: np.int8, 2: np.int16, 4: np.int32}[n]
    i = values(fields & mask).view(lane).ravel()
    q = values((fields >> bits) & mask).view(lane).ravel()
    return i, q


# byte value -> (I table, Q table), for 4-, 2- and 1-bit samples.
_BYTE_TABLES = {bits: _byte_tables(bits) for bits in (4, 2, 1)}


# =============================================================================
# Validation Helpers
# =============================================================================


def _validate_label(label: bytes, record_index: int) -> None:
    """Validate RDEF record header label."""
    if label != b"RDEF":
        raise ValueError(
            f"Record {record_index}: RECORD LABEL {label!r}, "
            "expected b'RDEF' (Table 5-1)"
        )


def _user_stacklevel() -> int:
    """``stacklevel`` that points a warning at the first frame outside rdefpy."""
    frame, level = sys._getframe(1), 1
    while frame is not None:
        module = frame.f_globals.get("__name__") or ""
        if module.partition(".")[0] != "rdefpy":
            break
        frame, level = frame.f_back, level + 1
    return level


def _out_of_range(h: dict) -> list[str]:
    """
    Field values outside Table 5-1's allowed values or §5.2.4, one line each.

    Shared by the reader (warns, so a bad file can still be inspected and
    corrected) and the writer (refuses, so RDEFpy never writes one):

    - RECORD VERSION ID != 1
    - AGENCY FLAG outside 0-3
    - TIME TAG DOY outside 1-366
    - TIME TAG SECOND OF DAY outside 0-86400
    - TIME TAG PICOSECONDS negative ("positive non-zero ... or 0")
    - any float64 field NaN, -Inf, +Inf or -0 (§5.2.4)
    """
    bad = []
    if h["record_version_id"] != 1:
        bad.append(f"  - RECORD VERSION ID: {h['record_version_id']} (expected: 1)")
    if not 0 <= h["agency_flag"] <= 3:
        bad.append(f"  - AGENCY FLAG: {h['agency_flag']} (valid range: 0-3)")
    if not 1 <= h["time_tag_doy"] <= 366:
        bad.append(f"  - TIME TAG DOY: {h['time_tag_doy']} (valid range: 1-366)")
    if not 0 <= h["time_tag_second_of_day"] <= 86400:
        bad.append(f"  - TIME TAG SECOND OF DAY: {h['time_tag_second_of_day']} "
                   "(valid range: 0-86400)")
    if h["time_tag_picoseconds_of_the_second"] < 0:
        bad.append("  - TIME TAG PICOSECONDS: "
                   f"{h['time_tag_picoseconds_of_the_second']} (must be >= 0)")
    for f in _FLOAT_FIELDS:
        v = h[f]
        if not math.isfinite(v) or (v == 0 and math.copysign(1.0, v) < 0):
            bad.append(f"  - {f}: {v!r} "
                       "(§5.2.4: NaN, -Inf, +Inf and -0 not supported)")
    return bad


def _validate_header(
    *,
    label: bytes,
    end_label: int,
    record_length: int,
    sample_rate: int,
    sample_size: int,
    record_index: int,
) -> None:
    """
    Structural checks of an RDEF header (CCSDS 506.1-B-1 §5.2), raising
    ``ValueError``: without them the record cannot be parsed, nor the next
    one found.  Out-of-range field values are :func:`_out_of_range`'s job.

    - RECORD LABEL is ``b'RDEF'`` (Table 5-1)
    - END LABEL is ``-99999`` (Table 5-1, last 4 bytes)
    - SAMPLE SIZE is valid: 1, 2, 4, 8, or 16 bits (Table 5-1)
    - ``2 * sample_rate * sample_size`` is a multiple of 32 (Table 5-1
      SAMPLE RATE, §5.3.2: 32-bit sample words)
    - RECORD LENGTH matches ``176 + data_bytes`` (Table 5-1)
    """
    _validate_label(label, record_index)

    i = record_index
    if end_label != -99999:
        raise ValueError(
            f"Record {i}: END LABEL {end_label}, expected -99999 (Table 5-1)"
        )

    if sample_size not in _VALID_SAMPLE_SIZES:
        raise ValueError(
            f"Record {i}: SAMPLE SIZE {sample_size}, "
            "expected 1, 2, 4, 8 or 16 (Table 5-1)"
        )

    bits = 2 * sample_rate * sample_size
    if bits % 32 != 0:
        raise ValueError(
            f"Record {i}: 2 * SAMPLE RATE * SAMPLE SIZE = {bits}, "
            "not a multiple of 32 (Table 5-1, §5.3.2)"
        )

    expected_length = _HEADER_SIZE + bits // 8
    if record_length != expected_length:
        raise ValueError(
            f"Record {i}: RECORD LENGTH {record_length}, expected {expected_length} "
            f"= {_HEADER_SIZE} + 2 * SAMPLE RATE * SAMPLE SIZE / 8 (Table 5-1)"
        )


# =============================================================================
# Header Unpacking Functions
# =============================================================================


def unpack_header(f: BinaryIO, record_index: int) -> dict | None:
    """
    Unpack and validate a full RDEF record header.

    Parameters
    ----------
    f : BinaryIO
        File object positioned at start of header
    record_index : int
        Current record index for error reporting

    Returns
    -------
    dict or None
        Dictionary of header fields, or None if EOF reached

    Raises
    ------
    EOFError
        If header is truncated
    ValueError
        If header validation fails

    Warns
    -----
    UserWarning
        If a field value is outside Table 5-1's allowed values or §5.2.4
        (see :func:`_out_of_range`); the header is still returned.
    """
    header_bytes = f.read(_HEADER_SIZE)
    if not header_bytes:
        return None
    if len(header_bytes) < _HEADER_SIZE:
        raise EOFError(
            f"Record {record_index}: header truncated, "
            f"{len(header_bytes)} of {_HEADER_SIZE} bytes"
        )

    label, *values, end_label = _RDEF_HEADER_STRUCT.unpack(header_bytes)
    header = dict(zip(_FIELDS, values, strict=True))
    _validate_header(
        label=label,
        end_label=end_label,
        record_length=header["record_length"],
        sample_rate=header["sample_rate"],
        sample_size=header["sample_size"],
        record_index=record_index,
    )
    bad = _out_of_range(header)
    if bad:
        warnings.warn(
            f"Record {record_index}: field values outside the allowed range "
            "(Table 5-1, §5.2.4); record read as is:\n" + "\n".join(bad),
            stacklevel=_user_stacklevel(),
        )
    return header


def read_header_length(f: BinaryIO, record_index: int) -> int | None:
    """
    Read the 8-byte RDEF header prefix (label + record_length).

    This minimal read is used for fast file scanning.

    Parameters
    ----------
    f : BinaryIO
        File object positioned at start of header
    record_index : int
        Current record index for error reporting

    Returns
    -------
    int or None
        Record length in bytes, or None if EOF reached

    Raises
    ------
    EOFError
        If prefix is truncated
    ValueError
        If label validation fails, or RECORD LENGTH is smaller than the
        fixed 176-byte header (a value that would make a scanner seek
        backward into an infinite loop).
    """
    buf = f.read(_HEADER_PREFIX_SIZE)
    if not buf:
        return None
    if len(buf) < _HEADER_PREFIX_SIZE:
        raise EOFError(
            f"Record {record_index}: header truncated, "
            f"{len(buf)} of {_HEADER_PREFIX_SIZE} bytes"
        )

    label, record_length = _RDEF_HEADER_PREFIX_STRUCT.unpack(buf)
    _validate_label(label, record_index)
    if record_length < _HEADER_SIZE:
        raise ValueError(
            f"Record {record_index}: RECORD LENGTH {record_length}, "
            f"less than the {_HEADER_SIZE}-byte header (Table 5-1)"
        )

    return record_length


# =============================================================================
# I/Q Data Unpacking Functions
# =============================================================================


def unpack_iq_16(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    """
    Unpack 16-bit I/Q samples per CCSDS 506.1-B-1 Table 5-2.

    Format: MSB -> Q1, I1 -> LSB

    Parameters
    ----------
    data : bytes
        Raw binary data containing I/Q samples

    Returns
    -------
    i : np.ndarray (int32)
        In-phase samples, bias-corrected
    q : np.ndarray (int32)
        Quadrature samples, bias-corrected

    Notes
    -----
    Applies the §5.3.6 transform (compensates the §5.3.5 truncation):
    ``out = 2*k + 1``.
    Input range:  k in [-32768, 32767]
    Output range: [-65535, 65535] (odd integers only)
    """
    arr = np.frombuffer(data, dtype="<i2")
    n_pairs = arr.size // 2

    # 2k+1 computed in int32 directly from the strided int16 view into the
    # contiguous outputs: no interleaved temp, no deinterleave copy.
    i = np.empty(n_pairs, dtype=np.int32)
    q = np.empty(n_pairs, dtype=np.int32)
    np.multiply(arr[0 : 2 * n_pairs : 2], 2, out=i, dtype=np.int32)
    np.multiply(arr[1 : 2 * n_pairs : 2], 2, out=q, dtype=np.int32)
    i += 1
    q += 1
    return i, q


def unpack_iq_8(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    """
    Unpack 8-bit I/Q samples per CCSDS 506.1-B-1 Table 5-2.

    Format: MSB -> Q2, I2, Q1, I1 -> LSB

    Parameters
    ----------
    data : bytes
        Raw binary data containing I/Q samples

    Returns
    -------
    i : np.ndarray (int16)
        In-phase samples, bias-corrected
    q : np.ndarray (int16)
        Quadrature samples, bias-corrected

    Notes
    -----
    Applies the §5.3.6 transform (compensates the §5.3.5 truncation):
    ``out = 2*k + 1``.
    Input range:  k in [-128, 127]
    Output range: [-255, 255] (odd integers only)
    """
    arr = np.frombuffer(data, dtype="<i1")
    n_pairs = arr.size // 2

    i =np.empty(n_pairs, dtype=np.int16)
    q = np.empty(n_pairs, dtype=np.int16)
    np.multiply(arr[0 : 2 * n_pairs : 2], 2, out=i, dtype=np.int16)
    np.multiply(arr[1 : 2 * n_pairs : 2], 2, out=q, dtype=np.int16)
    i += 1
    q += 1
    return i, q


def unpack_iq_4(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    """
    Unpack 4-bit I/Q samples per CCSDS 506.1-B-1 Table 5-2.

    Format: MSB -> Q4, I4, Q3, I3, Q2, I2, Q1, I1 -> LSB

    Parameters
    ----------
    data : bytes
        Raw binary data containing I/Q samples

    Returns
    -------
    i : np.ndarray (int8)
        In-phase samples, bias-corrected
    q : np.ndarray (int8)
        Quadrature samples, bias-corrected

    Notes
    -----
    Applies the §5.3.6 transform (compensates the §5.3.5 truncation):
    ``out = 2*k + 1``.
    Input range:  k in [-8, 7] (4-bit two's complement)
    Output range: [-15, 15] (odd integers only)
    """
    return _unpack_sub_byte(data, 4)


def unpack_iq_2(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    """
    Unpack 2-bit I/Q samples per CCSDS 506.1-B-1 Table 5-2.

    Format: MSB -> Q8, I8, Q7, I7, ... Q2, I2, Q1, I1 -> LSB

    Parameters
    ----------
    data : bytes
        Raw binary data containing I/Q samples

    Returns
    -------
    i : np.ndarray (int8)
        In-phase samples, bias-corrected
    q : np.ndarray (int8)
        Quadrature samples, bias-corrected

    Notes
    -----
    Applies the §5.3.6 transform (compensates the §5.3.5 truncation):
    ``out = 2*k + 1``.
    Input range:  k in [-2, 1] (2-bit two's complement)
    Output range: [-3, 3] (odd integers only)
    """
    return _unpack_sub_byte(data, 2)


def unpack_iq_1(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    """
    Unpack 1-bit I/Q samples per CCSDS 506.1-B-1 Table 5-2.

    Format: MSB -> Q16, I16, Q15, I15, ... Q2, I2, Q1, I1 -> LSB

    Parameters
    ----------
    data : bytes
        Raw binary data containing I/Q samples

    Returns
    -------
    i : np.ndarray (int8)
        In-phase samples, bias-corrected
    q : np.ndarray (int8)
        Quadrature samples, bias-corrected

    Notes
    -----
    Applies the §5.3.6 transform (compensates the §5.3.5 truncation):
    ``out = 2*k + 1``.
    Input range:  k in [-1, 0] (1-bit two's complement: bit 0 -> +1, 1 -> -1)
    Output range: [-1, 1] (odd integers only)
    """
    return _unpack_sub_byte(data, 1)


def _unpack_sub_byte(data: bytes, bits: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Decode 4-, 2- or 1-bit samples: one table gather per lane.

    A byte holds whole complex samples at these sizes, so the decode is
    ``table[byte]`` for I and for Q (see :func:`_byte_tables`), with no
    temporaries beyond the outputs.
    """
    byte_arr = np.frombuffer(data, dtype=np.uint8)
    table_i, table_q = _BYTE_TABLES[bits]
    return table_i[byte_arr].view(np.int8), table_q[byte_arr].view(np.int8)
