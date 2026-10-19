# Copyright (c) 2026 Morehead State University
# Author: Andrea Togni
# SPDX-License-Identifier: MIT

"""
Reader functions for RDEF product files (.prd).

Implements record-by-record reading of the binary format defined in
CCSDS 506.1-B-1 (June 2013), §5.  Product files can be multi-gigabyte, so
``read_product`` reads only the headers: each returned record knows where
its samples are on disk and reads them on access (or keeps them after
``load()``).

Functions
---------
scan_product
    Fast scan to count records and check length consistency.
read_headers
    Read only headers (§5.2); records carry empty sample arrays.
read_product
    Read headers, with optional time, header-field, and frequency/band
    filtering; samples on access or ``load()``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import BinaryIO
from datetime import datetime
from pathlib import Path

import numpy as np

from rdefpy._utils import _Progress, _to_datetime64
from rdefpy.prd.binary import (
    _HEADER_PREFIX_SIZE,
    _HEADER_SIZE,
    read_header_length,
    unpack_header,
    unpack_iq_1,
    unpack_iq_2,
    unpack_iq_4,
    unpack_iq_8,
    unpack_iq_16,
)
from rdefpy.prd.classes import (
    ProductFile,
    Record,
    RecordData,
    RecordHeader,
    _downconv_hz,
    _load_records,
)

# =============================================================================
# Constants
# =============================================================================

# Python unpacker dispatch table
_UNPACKERS = {
    16: unpack_iq_16,
    8: unpack_iq_8,
    4: unpack_iq_4,
    2: unpack_iq_2,
    1: unpack_iq_1,
}

# Cython fast path (setup.py builds it on install). Same 2k+1 output for
# every sample size; tests/test_rdefpy.py checks the two agree. Missing
# extension -> the NumPy table above, silently.
try:
    from rdefpy.prd.unpack_cyth import unpack_iq_cython as _unpack_fast
except ImportError:  # not built (uv_build, no compiler, ABI mismatch)
    _unpack_fast = None

# Nominal downlink RF bands (DSN/CCSDS), in Hz. A ``band`` filter works
# over the total-downconversion-frequency window, which tracks the received
# downlink carrier. Edges are the nominal allocation; the guard below is
# added on read.
_DOWNLINK_BANDS_HZ = {
    "S": (2.290e9, 2.300e9),
    "X": (8.400e9, 8.450e9),
    "KA": (31.800e9, 32.300e9),
}
# Shielding guard added to each band edge (min - guard, max + guard).
_BAND_GUARD_HZ = 1.0e9

# Records at or below this sample rate are loaded by read_product: their
# samples are small, and per-access disk reads would cost more than the
# data. Above it, samples stay on disk until accessed or load()ed.
_AUTO_LOAD_MAX_RATE_HZ = 1000


# =============================================================================
# Private Helpers
# =============================================================================


def _validate_record_length(record_length: int, record_index: int) -> None:
    """
    Validate that record length is at least as large as the header.

    Raises
    ------
    ValueError
        If ``record_length < _HEADER_SIZE``.
    """
    if record_length < _HEADER_SIZE:
        raise ValueError(
            f"record_length {record_length} smaller than header size "
            f"{_HEADER_SIZE} at record {record_index} (§5.2)"
        )


def _decode_iq(
    data_bytes: bytes,
    sample_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Decode I/Q samples from raw bytes.

    Parameters
    ----------
    data_bytes : bytes
        Raw binary I/Q data.
    sample_size : int
        Bits per sample (1, 2, 4, 8, or 16).

    Returns
    -------
    i, q : np.ndarray
        In-phase and quadrature samples.

    Raises
    ------
    ValueError
        If *sample_size* is not supported.
    """
    if _unpack_fast is not None and sample_size in _UNPACKERS:
        return _unpack_fast(data_bytes, sample_size)
    unpacker = _UNPACKERS.get(sample_size)
    if unpacker is None:
        raise ValueError(f"Unsupported sample_size={sample_size} (§5.3)")
    return unpacker(data_bytes)


class _DiskSource:
    """
    Where one record's samples sit on disk, and how to read them back.

    Held by a ``Record`` returned from ``read_product``.  Carries the
    absolute file path (not an open handle), so records stay picklable and
    usable after the read call returns, plus the file's size and mtime at
    read time: if the file has since changed, reading raises instead of
    silently decoding different bytes.
    """

    __slots__ = ("path", "offset", "size", "complex_dtype", "stamp")

    def __init__(
        self,
        path: Path,
        offset: int,
        size: int,
        complex_dtype: np.dtype,
        stamp: tuple[int, int],
    ):
        self.path = path
        self.offset = offset
        self.size = size
        self.complex_dtype = complex_dtype
        self.stamp = stamp

    def n_samples(self, header: RecordHeader) -> int:
        """Complex samples in the data section: 2 values of sample_size bits each."""
        return self.size * 8 // (2 * int(header.sample_size))

    def open(self) -> BinaryIO:
        """Open the file for reading, after checking it is the one indexed."""
        st = self.path.stat()
        if (st.st_size, st.st_mtime_ns) != self.stamp:
            raise RuntimeError(
                f"{self.path} has changed since it was read; "
                "call read_product again before accessing samples."
            )
        return self.path.open("rb")

    def read_bytes(self, f: BinaryIO | None = None) -> bytes:
        """The data section exactly as on disk, through *f* if given."""
        if f is None:
            with self.open() as fh:
                fh.seek(self.offset)
                data_bytes = fh.read(self.size)
        else:
            f.seek(self.offset)
            data_bytes = f.read(self.size)
        if len(data_bytes) != self.size:
            raise EOFError(
                f"Truncated data section: expected {self.size} bytes, "
                f"got {len(data_bytes)} (§5.1)"
            )
        return data_bytes

    def read(
        self,
        header: RecordHeader,
        *,
        writeable: bool,
        f: BinaryIO | None = None,
    ) -> RecordData:
        """Read and decode the samples (§5.3), through *f* if given."""
        i, q = _decode_iq(self.read_bytes(f), header.sample_size)
        if not writeable:
            # Transient read: an in-place edit would change a copy nobody
            # keeps, so make it an error instead of a silent no-op.
            i.flags.writeable = False
            q.flags.writeable = False
        return RecordData(
            sample_size=header.sample_size,
            sample_rate=header.sample_rate,
            i_samples=i,
            q_samples=q,
            complex_dtype=self.complex_dtype,
        )


def _skip(f: BinaryIO, n: int, file_size: int, idx: int, record_length: int) -> None:
    """Seek past *n* data bytes; raise if the file ends before they do (§5.1)."""
    if f.seek(n, 1) > file_size:
        raise EOFError(
            f"Truncated product file: record {idx} declares "
            f"{record_length} bytes but the file ends early (§5.1)."
        )


# =============================================================================
# Reader Functions
# =============================================================================


def scan_product(path: str | Path, *, progress: bool = False) -> tuple[int, bool]:
    """
    Fast scan of an RDEF product file to count records and check
    length consistency.

    Reads only the 8-byte header prefix per record.

    Parameters
    ----------
    path : str or Path
        Path to RDEF product file.
    progress : bool
        Draw a progress bar on stderr (default False).

    Returns
    -------
    count : int
        Total number of records.
    consistent_length : bool
        True if all records share the same ``record_length``.
    """
    path = Path(path)
    file_size = path.stat().st_size

    count = 0
    consistent_length = True
    prev_length: int | None = None

    with (
        path.open("rb") as f,
        _Progress(file_size, progress) as pbar,
    ):
        while True:
            record_length = read_header_length(f, count)
            if record_length is None:
                break

            if prev_length is not None and record_length != prev_length:
                consistent_length = False
            prev_length = record_length

            f.seek(record_length - _HEADER_PREFIX_SIZE, 1)
            count += 1
            pbar.update(record_length)

    return count, consistent_length


def read_headers(path: str | Path, *, progress: bool = False) -> ProductFile:
    """
    Read only RDEF record headers, skipping all data sections.

    Records will contain headers but empty ``RecordData`` objects.
    Significantly faster than ``read_product`` when only header fields
    is needed.

    Parameters
    ----------
    path : str or Path
        Path to RDEF product file.
    progress : bool
        Draw a progress bar on stderr (default False).

    Returns
    -------
    ProductFile
        Product file with headers only (no I/Q data).

    Raises
    ------
    ValueError
        If a record length is smaller than the header (§5.2).
    """
    path = Path(path)
    file_size = path.stat().st_size
    records: list[Record] = []

    with (
        path.open("rb") as f,
        _Progress(file_size, progress) as pbar,
    ):
        idx = 0
        while True:
            header_dict = unpack_header(f, idx)
            if header_dict is None:
                break

            hdr = RecordHeader(header_dict)
            _validate_record_length(hdr.record_length, idx)

            records.append(
                Record(
                    header=hdr,
                    data=RecordData(sample_size=hdr.sample_size),
                )
            )

            new_pos = f.seek(hdr.record_length - _HEADER_SIZE, 1)
            if new_pos > file_size:
                raise EOFError(
                    f"Truncated product file: record {idx} declares "
                    f"{hdr.record_length} bytes but the file ends early (§5.2)."
                )
            pbar.update(hdr.record_length)
            idx += 1

    pf = ProductFile(name=path.stem)
    pf.path = path.resolve()
    pf.records = records

    pf._check_constant_fields()

    return pf


def read_product(
    path: str | Path,
    *,
    num_records: int | None = None,
    progress: bool = False,
    start_epoch: datetime | np.datetime64 | str | None = None,
    stop_epoch: datetime | np.datetime64 | str | None = None,
    filters: dict | None = None,
    min_downconv_hz: float | None = None,
    max_downconv_hz: float | None = None,
    band: str | None = None,
    dtype: type | np.dtype = np.complex128,
) -> ProductFile:
    """
    Read an RDEF product file (CCSDS 506.1-B-1 §5) into a :class:`ProductFile`.

    Records at or below 1 kHz sample rate -- the common case -- are loaded:
    their samples are small, so they are read now and kept in memory.
    Faster records are left on disk; each knows where its samples are:

    * ``record.data`` reads and decodes them on access, and nothing keeps
      them -- they are freed as soon as you drop the result, so looping
      over a multi-gigabyte file holds one record at a time;
    * ``record.load()`` / ``pf.load()`` read them and keep them in memory
      until ``release()``.

    ``pf.release()`` drops loaded samples of any rate; they are then read
    on access like the rest.  The file must stay in place, unchanged,
    while its records are used.

    Parameters
    ----------
    path : str or Path
        Path to RDEF product file.
    num_records : int, optional
        Read only the first *N* records.  When filters are active this caps
        the records *examined* (including skipped ones), not the number
        returned.
    progress : bool
        Draw progress bars on stderr (default False): one for the header
        pass, and one for loading when records are at or below 1 kHz.
    start_epoch, stop_epoch : datetime, np.datetime64, str, or None, optional
        Keep records with ``start_epoch <= timetag < stop_epoch``
        (half-open, so consecutive windows tile without overlap or gap),
        compared on the full time tag, picoseconds included.  Accepts
        datetime (naive = UTC), numpy.datetime64, or an ISO-8601 or CCSDS
        string (e.g. "2026-01-15T12:00:00", "2026-015T12:00:00").
    filters : dict, optional
        Header-field filters.  Keys are RecordHeader field names (e.g.
        ``"station_id"``, ``"spacecraft_id"``, ``"validity_flag"``,
        ``"sample_rate"``); values are the expected values.  Records whose
        header fields do not match **all** filters are skipped without
        reading their data.
    min_downconv_hz, max_downconv_hz : float, optional
        Total downconversion-frequency window in Hz, evaluated at the first
        sample (t = 0) as
        ``rf_to_if_downconv + if_to_channel_downconv + c1``.
    band : {"S", "X", "Ka"}, optional
        Select records by downlink RF band (case-insensitive).  Convenience
        over *min_downconv_hz* / *max_downconv_hz*: the nominal downlink
        allocation is widened by a 1 GHz shielding guard on each edge
        (S 2.290-2.300, X 8.400-8.450, Ka 31.800-32.300 GHz nominal).
        Mutually exclusive with *min_downconv_hz* / *max_downconv_hz*.
    dtype : np.complex64 or np.complex128, default np.complex128
        Precision of the decoded complex samples (``record.data.samples``).
        ``np.complex64`` halves the memory, worthwhile at high sample rates
        (e.g. 20 MHz), and is lossless: the 2k+1 sample values (§5.3.6) are
        small integers exact in float32.

    Returns
    -------
    ProductFile
        All kept records; samples loaded at <= 1 kHz, on disk above.

    Raises
    ------
    ValueError
        If a filter key is not a valid RecordHeader field name; if *band*
        is unknown or combined with *min_downconv_hz* / *max_downconv_hz*;
        if a record length is smaller than the header (§5.2); or if
        *start_epoch* / *stop_epoch* is an unparseable string.
    EOFError
        If a record's data section is truncated (§5.1).

    Examples
    --------
    Keep station-24 records in memory:

    >>> pf = read_product("file.prd", filters={"station_id": 24})
    >>> pf.load()

    Walk a long X-band file one record at a time, holding none:

    >>> for rec in read_product("file.prd", band="X"):
    ...     power = rec.data.power  # reads and decodes this record only
    """
    path = Path(path)

    # Resolve band -> downconversion window (mutually exclusive with min/max).
    if band is not None:
        if min_downconv_hz is not None or max_downconv_hz is not None:
            raise ValueError("Specify either `band` or min/max_downconv_hz, not both.")
        try:
            lo, hi = _DOWNLINK_BANDS_HZ[band.upper()]
        except (KeyError, AttributeError):
            raise ValueError(
                f"Unknown band {band!r}. Valid: {sorted(_DOWNLINK_BANDS_HZ)}"
            ) from None
        min_downconv_hz = lo - _BAND_GUARD_HZ
        max_downconv_hz = hi + _BAND_GUARD_HZ

    # Convert time parameters to datetime64
    start_epoch = _to_datetime64(start_epoch)
    stop_epoch = _to_datetime64(stop_epoch)

    # Validate filter field names upfront
    if filters:
        valid_fields = set(RecordHeader.__slots__)
        for field in filters:
            if field not in valid_fields:
                raise ValueError(
                    f"Unknown header field in filters: {field!r}. "
                    f"Valid fields: {sorted(valid_fields)}"
                )

    # Validate sample precision now, not on first .data access.
    if np.dtype(dtype) not in (np.dtype(np.complex64), np.dtype(np.complex128)):
        raise ValueError(
            f"dtype must be np.complex64 or np.complex128, got {np.dtype(dtype)}"
        )

    pf = ProductFile(name=path.stem)
    pf.path = path.resolve()
    pf.records = list(
        _walk_records(
            path,
            progress=progress,
            num_records=num_records,
            start_epoch=start_epoch,
            stop_epoch=stop_epoch,
            filters=filters,
            min_downconv_hz=min_downconv_hz,
            max_downconv_hz=max_downconv_hz,
            complex_dtype=np.dtype(dtype),
        )
    )

    _load_records(
        [r for r in pf.records if r.header.sample_rate <= _AUTO_LOAD_MAX_RATE_HZ],
        progress=progress,
    )

    pf._check_constant_fields()

    return pf


def _walk_records(
    path: str | Path,
    *,
    progress: bool,
    num_records: int | None,
    start_epoch: np.datetime64 | None,
    stop_epoch: np.datetime64 | None,
    filters: dict | None,
    min_downconv_hz: float | None,
    max_downconv_hz: float | None,
    complex_dtype: np.dtype,
) -> Iterator[Record]:
    """
    Walk and filter a product file's headers, yielding one record per kept record.

    Shared core of :func:`read_product`.  Inputs are assumed already
    validated and resolved by the caller: ``band`` -> min/max window, epochs
    -> ``datetime64``, filter keys checked.  Data sections are seeked over,
    never read; each kept record carries a :class:`_DiskSource` instead.
    """
    path = Path(path).resolve()
    st = path.stat()
    file_size = st.st_size
    stamp = (st.st_size, st.st_mtime_ns)

    use_record_limit = num_records is not None
    time_filtered = start_epoch is not None or stop_epoch is not None

    with (
        path.open("rb") as f,
        _Progress(num_records if use_record_limit else file_size, progress) as pbar,
    ):
        idx = 0
        while True:
            if use_record_limit and idx >= num_records:
                break

            header_dict = unpack_header(f, idx)
            if header_dict is None:
                break

            hdr = RecordHeader(header_dict)
            _validate_record_length(hdr.record_length, idx)
            data_size = hdr.record_length - _HEADER_SIZE
            offset = f.tell()
            _skip(f, data_size, file_size, idx, hdr.record_length)
            pbar.update(1 if use_record_limit else hdr.record_length)
            idx += 1

            # Time-window filtering (evaluated first so stop_epoch can
            # short-circuit before the other filters).
            if time_filtered:
                try:
                    rec_epoch = hdr.timetag
                except (ValueError, OverflowError):
                    rec_epoch = None

                if rec_epoch is not None:
                    if start_epoch is not None and rec_epoch < start_epoch:
                        continue
                    if stop_epoch is not None and rec_epoch >= stop_epoch:
                        pbar.finish()
                        break

            # Header field filtering
            if filters and any(getattr(hdr, k) != v for k, v in filters.items()):
                continue

            # Downconversion frequency filtering (evaluated at t=0)
            if min_downconv_hz is not None or max_downconv_hz is not None:
                f_dc_hz = _downconv_hz(hdr, 0.0)
                if (min_downconv_hz is not None and f_dc_hz < min_downconv_hz) or (
                    max_downconv_hz is not None and f_dc_hz > max_downconv_hz
                ):
                    continue

            yield Record._on_disk(
                hdr, _DiskSource(path, offset, data_size, complex_dtype, stamp)
            )
