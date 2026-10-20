# Copyright (c) 2026 Morehead State University
# Author: Andrea Togni
# SPDX-License-Identifier: MIT

"""
Writer functions for RDEF product files (.prd).

Implements the inverse of the reader module, packing I/Q samples and
headers into the binary format defined in CCSDS 506.1-B-1 (June 2013),
"Delta-DOR Raw Data Exchange Format", §5.

Writer Functions
----------------
write_product
    Write complete ProductFile to disk.
_write_record
    Write single record to file path.

Packing Functions
-----------------
pack_record_header
    Pack RecordHeader into 176-byte binary format (§5.2, Table 5-1).
pack_record_data
    Pack RecordData (I/Q samples) into bit-packed binary format (§5.3).

Notes
-----
- All packing functions invert the ``2k+1`` bias correction (§5.3.6)
  applied during reading.
- Sample sizes 1, 2, 4, 8, 16 bits are supported per §5.3.
- Headers and data must satisfy CCSDS 506.1-B-1 constraints (32-bit
  word alignment, little-endian byte order per §1.3.3 / §5.1.5).
- Writing preserves exact binary format for round-trip compatibility.

Examples
--------
Write a complete product file:

    >>> from rdefpy.prd.reader import read_product
    >>> from rdefpy.prd.writer import write_product
    >>> pf = read_product("input.prd")
    >>> write_product("output.prd", pf)

Write a single record:

    >>> from rdefpy.prd.classes import Record, RecordHeader, RecordData
    >>> from rdefpy.prd.writer import _write_record
    >>> import numpy as np
    >>>
    >>> hdr = RecordHeader({
    ...     "record_version_id": 1,
    ...     "station_id": 24,
    ...     "spacecraft_id": 0,
    ...     "sample_size": 16,
    ...     "sample_rate": 200000,
    ...     # ... other required fields
    ... })
    >>> i = np.random.randint(-65535, 65535, 200000, dtype=np.int32)
    >>> q = np.random.randint(-65535, 65535, 200000, dtype=np.int32)
    >>> data = RecordData(sample_size=16, i_samples=i, q_samples=q)
    >>> rec = Record(header=hdr, data=data)
    >>> _write_record("output.prd", rec)

"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import BinaryIO

import numpy as np

from rdefpy._utils import _Progress
from rdefpy.prd._binary import (
    _FLOAT_FIELDS,
    _RDEF_HEADER_STRUCT,
    _VALID_SAMPLE_SIZES,
    _out_of_range,
)
from rdefpy.prd.classes import ProductFile, Record, RecordHeader

# =============================================================================
# Constants
# =============================================================================

# Header size in bytes per CCSDS 506.1-B-1
_HEADER_SIZE_BYTES = _RDEF_HEADER_STRUCT.size  # 176 bytes

# Required header fields that must not be None
_REQUIRED_HEADER_FIELDS = (
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
)


# =============================================================================
# Validation Helpers
# =============================================================================


def _validate_header_fields(header: RecordHeader) -> None:
    """
    Validate that all required header fields are present and not None.

    Parameters
    ----------
    header : RecordHeader
        Header to validate

    Raises
    ------
    ValueError
        If any required field is None
    """
    for field_name in _REQUIRED_HEADER_FIELDS:
        if getattr(header, field_name) is None:
            raise ValueError(f"Header field '{field_name}' must not be None")


def _validate_sample_parameters(sample_rate: int, sample_size: int) -> None:
    """
    Validate sample rate and size meet CCSDS 506.1-B-1 §5.3 constraints.

    Parameters
    ----------
    sample_rate : int
        Sample rate in Hz
    sample_size : int
        Bits per sample (1, 2, 4, 8, or 16)

    Raises
    ------
    ValueError
        If sample_rate <= 0, sample_size invalid, or CCSDS constraint violated
    """
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be > 0, got {sample_rate}")

    if sample_size not in _VALID_SAMPLE_SIZES:
        raise ValueError(
            f"sample_size must be one of "
            f"{sorted(_VALID_SAMPLE_SIZES)}, got {sample_size}"
        )

    # CCSDS constraint: SAMPLE_RATE * 2 * SAMPLE_SIZE must be multiple of 32
    # This ensures sample word length aligns to 32-bit boundaries
    if (sample_rate * 2 * sample_size) % 32 != 0:
        raise ValueError(
            f"sample_rate * 2 * sample_size must be a multiple of 32 "
            f"to keep the sample word length to 32 bits. "
            f"Got: {sample_rate} * 2 * {sample_size} = {sample_rate * 2 * sample_size}"
        )


def _prepare_empty_fields(header: RecordHeader) -> tuple[bytes, bytes]:
    """
    Prepare empty/reserved header fields with proper validation.

    Parameters
    ----------
    header : RecordHeader
        Header containing empty field data

    Returns
    -------
    empty_future : bytes
        36-byte future extension field
    empty_agency : bytes
        40-byte agency use field

    Raises
    ------
    ValueError
        If empty fields are wrong size
    """
    empty_future = header.empty_fields_future_extension
    if empty_future is None:
        empty_future = b"\x00" * 36
    else:
        empty_future = bytes(empty_future)
        if len(empty_future) != 36:
            raise ValueError(
                f"empty_fields_future_extension must be exactly "
                f"36 bytes, got {len(empty_future)}"
            )

    empty_agency = header.empty_fields_internal_agency_use
    if empty_agency is None:
        empty_agency = b"\x00" * 40
    else:
        empty_agency = bytes(empty_agency)
        if len(empty_agency) != 40:
            raise ValueError(
                f"empty_fields_internal_agency_use must be exactly "
                f"40 bytes, got {len(empty_agency)}"
            )

    return empty_future, empty_agency


# =============================================================================
# Bias Inversion Helper
# =============================================================================


def _invert_bias(
    y: np.ndarray, sample_size: int, min_val: int, max_val: int
) -> np.ndarray:
    """
    Invert bias correction per §5.3.6: y = 2*k + 1 → k, clipped to valid range.

    This is the inverse of the truncation encoding applied during reading.
    Converts from the expanded ``2k+1`` representation back to on-disk format.

    Parameters
    ----------
    y : np.ndarray
        Bias-corrected values (y = 2*k + 1)
    sample_size : int
        Bits per sample (determines output dtype)
    min_val : int
        Minimum valid value for k
    max_val : int
        Maximum valid value for k

    Returns
    -------
    k : np.ndarray
        Original values, clipped

    Notes
    -----
    Computed in the dtype of *y*, which holds every value involved.  The
    result is k, still in that dtype; the caller narrows it on store.
    """
    # Clipping y to [2*min+1, 2*max+1] first gives the same k as clipping
    # floor((y-1)/2) to [min, max] (both maps are monotone), and keeps y-1
    # from overflowing.  >> 1 on a signed integer is floor division by 2.
    k = np.clip(y, 2 * min_val + 1, 2 * max_val + 1)
    k -= 1
    k >>= 1
    return k


# =============================================================================
# Header Packing
# =============================================================================


def pack_record_header(header: RecordHeader) -> bytes:
    """
    Pack RecordHeader into 176-byte binary format per CCSDS 506.1-B-1.

    Automatically calculates record_length based on sample_rate and sample_size:
        record_length = (2 * sample_rate * sample_size) / 8 + 176

    This assumes one-second records with I/Q interleaved samples.

    Parameters
    ----------
    header : RecordHeader
        Header with all required fields populated

    Returns
    -------
    bytes
        176-byte packed header

    Raises
    ------
    ValueError
        If required fields are None or constraints violated
    TypeError
        If field types are incorrect

    Notes
    -----
    Header structure follows CCSDS 506.1-B-1 Table 5-1 / Figure 5-3:
        - 4 bytes: "RDEF" label
        - 4 bytes: Record length (uint32)
        - ... (see CCSDS spec for complete layout)
        - 176 bytes total

    Examples
    --------
    >>> hdr = RecordHeader({
    ...     "record_version_id": 1,
    ...     "station_id": 24,
    ...     "sample_size": 16,
    ...     "sample_rate": 200000,
    ...     # ... other fields
    ... })
    >>> header_bytes = pack_record_header(hdr)
    >>> len(header_bytes)
    176
    """
    # Validate all required fields are present
    _validate_header_fields(header)

    # Extract and validate sample parameters
    sample_rate = int(header.sample_rate)
    sample_size = int(header.sample_size)
    _validate_sample_parameters(sample_rate, sample_size)

    # Never write a value Table 5-1 or §5.2.4 rules out.  -0.0 is written as
    # +0.0 (same value; §5.2.4 forbids the sign bit, arithmetic easily sets it).
    h = {f: getattr(header, f) for f in _REQUIRED_HEADER_FIELDS}
    h.update({f: float(h[f]) + 0.0 for f in _FLOAT_FIELDS})  # -0.0 + 0.0 == +0.0
    bad = _out_of_range(h)
    if bad:
        raise ValueError(
            "Header breaks CCSDS 506.1-B-1 (Table 5-1, §5.2.4):\n" + "\n".join(bad)
        )

    # Calculate record length for one-second record
    # Data bytes = 2 (I+Q) * sample_rate * sample_size / 8
    data_bytes = (2 * sample_rate * sample_size) // 8
    record_length = _HEADER_SIZE_BYTES + data_bytes

    # Prepare empty/reserved fields
    empty_future, empty_agency = _prepare_empty_fields(header)

    # Pack header using CCSDS 506.1-B-1 structure
    header_bytes = _RDEF_HEADER_STRUCT.pack(
        b"RDEF",  # Record label (4 bytes)
        record_length,  # Record length (4 bytes)
        int(header.record_version_id),  # Version ID (2 bytes)
        int(header.station_id),  # Station ID (2 bytes)
        int(header.spacecraft_id),  # Spacecraft ID (2 bytes)
        sample_size,  # Sample size in bits (2 bytes)
        sample_rate,  # Sample rate in Hz (4 bytes)
        int(header.validity_flag),  # Validity flag (2 bytes)
        int(header.agency_flag),  # Agency flag (2 bytes)
        h["rf_to_if_downconv"],  # RF to IF downconversion (8 bytes)
        h["if_to_channel_downconv"],  # IF to channel downconversion (8 bytes)
        int(header.time_tag_year),  # Time tag year (2 bytes)
        int(header.time_tag_doy),  # Time tag day of year (2 bytes)
        int(header.time_tag_second_of_day),  # Time tag second of day (4 bytes)
        h["time_tag_picoseconds_of_the_second"],  # Picoseconds (8 bytes)
        h["channel_accumulated_phase"],  # Accumulated phase (8 bytes)
        h["channel_phase_polynomial_coefficient_0"],  # Phase coeff 0 (8 bytes)
        h["channel_phase_polynomial_coefficient_1"],  # Phase coeff 1 (8 bytes)
        h["channel_phase_polynomial_coefficient_2"],  # Phase coeff 2 (8 bytes)
        h["channel_phase_polynomial_coefficient_3"],  # Phase coeff 3 (8 bytes)
        empty_future,  # Future extension (36 bytes)
        empty_agency,  # Internal agency use (40 bytes)
        -99999,  # End label (4 bytes, sentinel value)
    )

    return header_bytes


def write_record_header(file_obj: BinaryIO, header: RecordHeader) -> int:
    """
    Write packed header to open binary file.

    Parameters
    ----------
    file_obj : BinaryIO
        Open binary file object (mode 'wb' or 'ab')
    header : RecordHeader
        Header to write

    Returns
    -------
    int
        Number of bytes written (always 176)

    Raises
    ------
    IOError
        If write operation fails or writes wrong number of bytes
    """
    header_bytes = pack_record_header(header)
    written = file_obj.write(header_bytes)

    if written != _HEADER_SIZE_BYTES:
        raise OSError(f"Expected to write {_HEADER_SIZE_BYTES} bytes, wrote {written}")

    return written


# =============================================================================
# Data Packing (Inverse of Reader Unpacking)
# =============================================================================


def pack_record_data(rec: Record) -> bytes:
    """
    Pack RecordData into compressed binary format per CCSDS 506.1-B-1.

    Inverts the bias correction and bit-packing applied during reading.
    Each sample size has a specific packing strategy:

    - 16-bit: Interleaved int16 (I, Q, I, Q, ...)
    - 8-bit: Interleaved int8 (I, Q, I, Q, ...)
    - 4-bit: Contiguous nibble packing (I low, Q high)
    - 2-bit: Four samples per 16-bit word
    - 1-bit: BPSK bit packing

    Parameters
    ----------
    rec : Record
        Record with header and data to pack

    Returns
    -------
    bytes
        Packed binary data section

    Raises
    ------
    ValueError
        If data is missing, length mismatches, or constraints violated
    TypeError
        If data dtype doesn't match expected type for sample_size

    Notes
    -----
    This function performs the exact inverse of the reader unpacking:
        - unpack_iq_16 ↔ pack_record_data (sample_size=16)
        - unpack_iq_8 ↔ pack_record_data (sample_size=8)
        - unpack_iq_4 ↔ pack_record_data (sample_size=4)
        - unpack_iq_2 ↔ pack_record_data (sample_size=2)
        - unpack_iq_1 ↔ pack_record_data (sample_size=1)

    Expected dtypes by sample size:
        - 16-bit: int32 (bias-corrected from int16)
        - 8-bit: int16 (bias-corrected from int8)
        - 4-bit: int8 (bias-corrected from 4-bit)
        - 2-bit: int8 (bias-corrected from 2-bit)
        - 1-bit: int8 (bipolar: -1 or +1)
    """
    hdr = rec.header
    data = rec.data

    if data is None:
        raise ValueError("Record has no data to pack")

    sample_size = int(hdr.sample_size)
    sample_rate = int(hdr.sample_rate)

    n_complex = len(data)
    if n_complex == 0:
        return b""

    if sample_rate != n_complex:
        raise ValueError(
            f"Record data length ({n_complex}) does not match "
            f"header.sample_rate ({sample_rate})"
        )

    i_vals = data.i_samples
    q_vals = data.q_samples

    # =========================================================================
    # 16-bit Sample Packing
    # =========================================================================
    if sample_size == 16:
        # File stores signed int16 k
        # RecordData stores y = 2*k + 1 as int32
        if i_vals.dtype != np.int32 or q_vals.dtype != np.int32:
            raise TypeError(
                f"For sample_size=16, RecordData must store int32 (2*k+1). "
                f"Got i.dtype={i_vals.dtype}, q.dtype={q_vals.dtype}"
            )

        # Invert bias: y = 2*k + 1 → k
        i_raw = _invert_bias(i_vals, 16, -32768, 32767)  # int16
        q_raw = _invert_bias(q_vals, 16, -32768, 32767)  # int16

        # Interleave: I1, Q1, I2, Q2, ...
        interleaved = np.empty(2 * n_complex, dtype="<i2")
        interleaved[0::2] = i_raw
        interleaved[1::2] = q_raw

        return interleaved.tobytes()

    # =========================================================================
    # 8-bit Sample Packing
    # =========================================================================
    if sample_size == 8:
        # File stores signed int8 k
        # RecordData stores y = 2*k + 1 as int16
        if i_vals.dtype != np.int16 or q_vals.dtype != np.int16:
            raise TypeError(
                f"For sample_size=8, RecordData must store int16 (2*k+1). "
                f"Got i.dtype={i_vals.dtype}, q.dtype={q_vals.dtype}"
            )

        # Invert bias: y = 2*k + 1 → k
        i_raw = _invert_bias(i_vals, 8, -128, 127)  # int8
        q_raw = _invert_bias(q_vals, 8, -128, 127)  # int8

        # Interleave: I1, Q1, I2, Q2, ...
        interleaved = np.empty(2 * n_complex, dtype="<i1")
        interleaved[0::2] = i_raw
        interleaved[1::2] = q_raw

        return interleaved.tobytes()

    # =========================================================================
    # 4-bit Sample Packing (Nibble-Interleaved)
    # =========================================================================
    if sample_size == 4:
        # File stores 4-bit nibbles (one I/Q pair per byte)
        # RecordData stores y = 2*k + 1 as int8
        if i_vals.dtype != np.int8 or q_vals.dtype != np.int8:
            raise TypeError(
                f"For sample_size=4, RecordData must store int8 (2*k+1). "
                f"Got i.dtype={i_vals.dtype}, q.dtype={q_vals.dtype}"
            )

        # Invert bias: y = 2*k + 1 → k in [-8, 7]
        i_k = _invert_bias(i_vals, 4, -8, 7)
        q_k = _invert_bias(q_vals, 4, -8, 7)

        # Pack nibbles into bytes with contiguous packing per §5.3:
        # Byte layout: [Q3 Q2 Q1 Q0] [I3 I2 I1 I0] (MSB -> LSB)
        # I in low nibble bits [0:3], Q in high nibble bits [4:7].  The low
        # 4 bits of int8 k are its 4-bit two's-complement code.
        i_k &= 0xF
        q_k <<= 4
        q_k |= i_k

        return q_k.tobytes()

    # =========================================================================
    # 2-bit Sample Packing (Word-Packed)
    # =========================================================================
    if sample_size == 2:
        # File stores 2-bit two's complement: {00, 01, 10, 11} → {0, 1, -2, -1}
        # RecordData stores y = 2*k + 1 as int8
        if i_vals.dtype != np.int8 or q_vals.dtype != np.int8:
            raise TypeError(
                f"For sample_size=2, RecordData must store int8 (2*k+1). "
                f"Got i.dtype={i_vals.dtype}, q.dtype={q_vals.dtype}"
            )

        # Invert bias: y = 2*k + 1 → k in [-2, 1]
        i_k = _invert_bias(i_vals, 2, -2, 1)
        q_k = _invert_bias(q_vals, 2, -2, 1)

        # The low 2 bits of int8 k are its 2-bit two's-complement code:
        # 0 → 00, 1 → 01, -2 → 10, -1 → 11.  One nibble per pair, Q above I.
        n_words = (n_complex + 3) // 4  # Ceiling division
        nib = np.zeros(4 * n_words, dtype=np.int8)  # zero-padded to whole words
        np.bitwise_and(i_k, 0x3, out=nib[:n_complex])
        q_k <<= 2
        q_k &= 0xC
        nib[:n_complex] |= q_k

        # Pack into 16-bit words: 4 I/Q pairs per word, two per byte
        # Word layout:
        #   bits  1–0 : I0
        #   bits  3–2 : Q0
        #   bits  5–4 : I1
        #   bits  7–6 : Q1
        #   bits  9–8 : I2
        #   bits 11–10: Q2
        #   bits 13–12: I3
        #   bits 15–14: Q3
        # i.e. little-endian bytes holding pairs (0, 1), (2, 3).
        out = nib[1::2] << 4
        out |= nib[0::2]

        return out.tobytes()

    # =========================================================================
    # 1-bit Sample Packing (BPSK)
    # =========================================================================
    if sample_size == 1:
        # File stores bits: 0 or 1
        # RecordData stores bipolar: -1 or +1 as int8
        if i_vals.dtype != np.int8 or q_vals.dtype != np.int8:
            raise TypeError(
                f"For sample_size=1, RecordData must store int8 in {{-1, +1}}. "
                f"Got i.dtype={i_vals.dtype}, q.dtype={q_vals.dtype}"
            )

        # Map amplitudes {-1, +1} → bits {1, 0} per §5.3.6:
        # Reader: bit 0 -> k=0 -> +1, bit 1 -> k=-1 -> -1
        # Inverse: +1 -> bit 0, -1 -> bit 1
        i_bits = (i_vals < 0).astype(np.uint8)
        q_bits = (q_vals < 0).astype(np.uint8)

        # Build bit stream: even positions = I, odd positions = Q
        bits = np.empty(2 * n_complex, dtype=np.uint8)
        bits[0::2] = i_bits
        bits[1::2] = q_bits

        # Pack bits into bytes (LSB-first within each byte)
        byte_arr = np.packbits(bits, bitorder="little")

        return byte_arr.tobytes()

    raise ValueError(f"Unsupported sample_size: {sample_size}")


# =============================================================================
# Record Writing
# =============================================================================


def _write_record_to_file(
    file_obj: BinaryIO, rec: Record, handles: dict | None = None
) -> None:
    """
    Write complete record (header + data) to open binary file.

    Parameters
    ----------
    file_obj : BinaryIO
        Open binary file object (mode 'wb' or 'ab')
    rec : Record
        Record with header and data to write
    handles : dict, optional
        Open source files by path, reused across calls; the caller closes them.

    Raises
    ------
    ValueError
        If record has no data, data length mismatches, or record_length inconsistent
    IOError
        If write operation fails

    Notes
    -----
    Automatically updates rec.header.record_length to match actual data size.

    Examples
    --------
    >>> with open("output.rdef", "wb") as f:
    ...     for rec in product_file.records:
    ...         _write_record_to_file(f, rec)
    """
    # An unloaded record whose samples nobody holds is copied from its file
    # verbatim: decoding and re-packing whole samples is the identity (the
    # 2k+1 map and the bit packing are one-to-one), so the bytes are the
    # same, without the decode, the I/Q arrays and the packing.
    src = rec._source
    hdr = rec.header
    verbatim = (
        rec._data is None
        and src is not None
        and (rec._weak is None or rec._weak() is None)
        and hdr.sample_size in _VALID_SAMPLE_SIZES
        and src.size * 8 % (2 * hdr.sample_size) == 0
    )
    if verbatim:
        n_complex = len(rec)
    else:
        # Hold the samples for the whole write: on an unloaded record this
        # is the one disk read, and pack_record_data's access reuses it.
        data = rec.data
        n_complex = 0 if data is None else len(data)
    if n_complex == 0:
        raise ValueError(
            "Record contains no I/Q data. "
            "Use a reader that loads samples (not read_headers), "
            "or construct RecordData with I/Q samples before writing."
        )

    # Pack header and data
    header_bytes = pack_record_header(hdr)
    if verbatim:
        if n_complex != int(hdr.sample_rate):
            raise ValueError(
                f"Record data length ({n_complex}) does not match "
                f"header.sample_rate ({int(hdr.sample_rate)})"
            )
        f = None if handles is None else handles.get(src.path)
        if f is None and handles is not None:
            f = handles[src.path] = src.open()
        data_bytes = src.read_bytes(f)
    else:
        data_bytes = pack_record_data(rec)

    # Verify record_length consistency
    expected_length = _HEADER_SIZE_BYTES + len(data_bytes)
    if (
        rec.header.record_length is not None
        and rec.header.record_length != expected_length
    ):
        raise ValueError(
            f"header.record_length={rec.header.record_length} does not match "
            f"computed length {expected_length}"
        )

    # Write to file
    file_obj.write(header_bytes)
    file_obj.write(data_bytes)


def _write_record(path: str | Path, rec: Record) -> None:
    """
    Write single record to file path (convenience wrapper).

    Opens file, writes record, and closes file.

    Parameters
    ----------
    path : str or Path
        Output file path
    rec : Record
        Record to write

    Raises
    ------
    ValueError
        If record is invalid
    IOError
        If file cannot be opened or written

    Examples
    --------
    >>> _write_record("single_record.rdef", my_record)
    """
    with open(path, "wb") as f:
        _write_record_to_file(f, rec)


def write_product(
    path: str | Path,
    product: ProductFile,
    *,
    progress: bool = False,
) -> None:
    """
    Write complete ProductFile to disk.

    Writes all records sequentially with proper alignment and formatting
    per CCSDS 506.1-B-1. Output file can be read back using read_product().

    Parameters
    ----------
    path : str or Path
        Output file path.
    product : ProductFile
        Product file containing records to write.
    progress : bool
        Draw a progress bar on stderr (default False).

    Raises
    ------
    ValueError
        If any record is invalid.
    IOError
        If file cannot be opened or written.

    Notes
    -----
    Records are written sequentially without padding or alignment bytes.
    Each record begins immediately after the previous record ends.

    Examples
    --------
    Round-trip test (read and write):

        >>> from rdefpy.prd.reader import read_product
        >>> from rdefpy.prd.writer import write_product
        >>> pf = read_product("input.prd")
        >>> write_product("output.prd", pf)

    """
    output_path = Path(path)

    # Write to a temporary file beside the target, then replace it in one
    # step: the target is never left half-written, and writing a product
    # file back over the file its records are read from is safe (the
    # records read the original until the replace).
    fd, tmp = tempfile.mkstemp(
        dir=output_path.parent, prefix=f".{output_path.name}.", suffix=".tmp"
    )
    handles: dict = {}  # source files of verbatim-copied records
    try:
        with (
            os.fdopen(fd, "wb") as fp,
            _Progress(len(product.records), progress) as pbar,
        ):
            try:
                for rec in product.records:
                    _write_record_to_file(fp, rec, handles)
                    pbar.update(1)
            finally:
                # Closed before the replace: Windows cannot replace an open file.
                for f in handles.values():
                    f.close()
            fp.flush()
            os.fsync(fp.fileno())
        # mkstemp creates the file owner-only (0600); give it the target's
        # existing mode, or the usual new-file mode.
        if output_path.exists():
            shutil.copymode(output_path, tmp)
        else:
            umask = os.umask(0)
            os.umask(umask)
            os.chmod(tmp, 0o666 & ~umask)
        os.replace(tmp, output_path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
