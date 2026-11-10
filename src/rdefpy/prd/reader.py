# Copyright (c) 2026 Morehead State University
# Author: Andrea Togni
# SPDX-License-Identifier: MIT

"""
Reader functions for RDEF product files (.prd).

Implements record-by-record reading of the binary format defined in
CCSDS 506.1-B-1 (June 2013), §5.  Product files can be multi-gigabyte, so
``read_product`` reads every header but leaves the samples of records above
1 kHz on disk: each such record knows where its samples are and reads them
on access (or keeps them after ``load()``).

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

import os
import threading
import warnings
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import BinaryIO
from datetime import datetime
from pathlib import Path

import numpy as np

from rdefpy._utils import _in_chunks, _Progress, _to_datetime64
from rdefpy.prd._binary import (
    _HEADER_PREFIX_SIZE,
    _HEADER_SIZE,
    _user_stacklevel,
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
    _HEADER_FIELD_SET,
    _PS_0222_MAX,
    _check_agency_layout,
    _check_t0_offset,
    _filter_downconv_hz,
    _in_window,
    _record_epoch,
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
    from rdefpy.prd import unpack_cyth as _cyth
except ImportError:  # extension not built (no compiler, ABI mismatch)
    _cyth = None
_unpack_fast = getattr(_cyth, "unpack_iq_cython", None)
# An older build may lack the fused complex kernel: then the int path is used.
_complex_fast = getattr(_cyth, "unpack_complex_cython", None)

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

# Data sections at least this large are read in parallel slices, one handle
# per thread: one thread copying out of the OS cache, or one disk request in
# flight, bounds a 20 Msps record's read.
_PARALLEL_READ_MIN_BYTES = 8 << 20
# Data sections at least this large, read in file order, have the next
# record's section read on a background thread while this one decodes.
# 20-40% faster warm decode from 512 KiB, 2- and 16-bit.
_READ_AHEAD_MIN_BYTES = 512 << 10


# =============================================================================
# Private Helpers
# =============================================================================


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
        If *data_bytes* is not whole 32-bit words (§5.3.2).  Both decoders
        would drop a partial word.
    """
    if len(data_bytes) % 4:
        raise ValueError(
            f"I/Q data of {len(data_bytes)} bytes is not whole 32-bit words (§5.3.2)"
        )
    if _unpack_fast is not None and sample_size in _UNPACKERS:
        return _unpack_fast(data_bytes, sample_size)
    # sample_size was checked when the header was read (_binary._validate_header) and
    # equals the header's on every decode (_DiskSource.check).
    return _UNPACKERS[sample_size](data_bytes)


def _decode_complex(data_bytes: bytes, sample_size: int, out: np.ndarray) -> None:
    """Decode I/Q samples from raw bytes straight into *out* as ``I + jQ``.

    *out* is checked by the caller (``Record.read_samples``).  Same values as
    ``_decode_iq`` then ``RecordData.samples``; the Cython path skips the int
    I/Q arrays.
    """
    if _complex_fast is None or sample_size not in _UNPACKERS:
        i, q = _decode_iq(data_bytes, sample_size)
        out.real = i
        out.imag = q
        return
    if len(data_bytes) % 4:
        raise ValueError(
            f"I/Q data of {len(data_bytes)} bytes is not whole 32-bit words (§5.3.2)"
        )
    _complex_fast(data_bytes, sample_size, out)


class _DiskSource:
    """
    Where one record's samples sit on disk, and how to read them back.

    Held by a ``Record`` returned from ``read_product``.  Carries the
    absolute file path (not an open handle), so records stay picklable and
    usable after the read call returns, plus the file's size and mtime at
    read time: if the file has since changed, reading raises instead of
    silently decoding different bytes.  Also the sample size the bytes were
    written with: a header changed to another one cannot decode them.
    """

    __slots__ = ("path", "offset", "size", "complex_dtype", "stamp", "sample_size")

    def __init__(
        self,
        path: Path,
        offset: int,
        size: int,
        complex_dtype: np.dtype,
        stamp: tuple[int, int],
        sample_size: int,
    ):
        self.path = path
        self.offset = offset
        self.size = size
        self.complex_dtype = complex_dtype
        self.stamp = stamp
        self.sample_size = sample_size

    def check(self, header: RecordHeader) -> None:
        """Raise if *header* no longer gives the sample size of the bytes on disk."""
        if header.sample_size != self.sample_size:
            raise ValueError(
                f"header.sample_size is {header.sample_size} but the samples on disk "
                f"are {self.sample_size}-bit: load() the samples before changing it, "
                "or build a new Record with new data if the change is intended"
            )

    def n_samples(self, header: RecordHeader) -> int:
        """Complex samples in the data section: 2 values of sample_size bits each."""
        self.check(header)
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
        self.check(header)
        raw = _readahead.get(self, f)
        data = self.decode(raw, header, writeable=writeable)
        if isinstance(raw, np.ndarray):
            _buffers.give(raw)  # the decoders copy: nothing views raw any more
        return data

    def read_complex(self, header: RecordHeader, out: np.ndarray) -> np.ndarray:
        """Read the samples and decode them straight into *out* (checked by the caller)."""
        self.check(header)
        raw = _readahead.get(self, None)
        _decode_complex(raw, header.sample_size, out)
        if isinstance(raw, np.ndarray):
            _buffers.give(raw)  # only after success: a failed decode may still be reading it
        return out

    def _read_now(self, f: BinaryIO | None = None) -> bytes | np.ndarray:
        """The data section, in parallel slices when large."""
        if self.size >= _PARALLEL_READ_MIN_BYTES:
            return self._read_parallel()
        return self.read_bytes(f)

    def _read_parallel(self) -> np.ndarray:
        """The data section, read in parallel slices into a fresh uint8 array."""
        buf = _buffers.take(self.size)

        def part(a: int, b: int) -> None:
            with self.open() as fh:
                fh.seek(self.offset + a)
                if fh.readinto(memoryview(buf)[a:b]) != b - a:
                    raise EOFError(
                        f"Truncated data section: expected {self.size} bytes (§5.1)"
                    )

        _in_chunks(self.size, 1, part, pool="io")
        return buf

    def decode(
        self, data_bytes: bytes, header: RecordHeader, *, writeable: bool
    ) -> RecordData:
        """Decode this record's data section, already read, into samples."""
        i, q = _decode_iq(data_bytes, header.sample_size)
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


class _Buffers:
    """
    Raw data-section buffers kept for reuse: a fresh 80 MB array pays page
    faults when filled and again when freed.

    Keeps at most two (the section decoding, the one read ahead)
    for the life of the process -- 160 MB at 16-bit 20 Msps.  A read-ahead
    left pending (a loop abandoned, a stale guess) holds one more until the
    next read replaces it.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._free: list[np.ndarray] = []

    def take(self, n: int) -> np.ndarray:
        with self._lock:
            for k, b in enumerate(self._free):
                if b.size == n:
                    return self._free.pop(k)
        return np.empty(n, np.uint8)

    def give(self, b: np.ndarray) -> None:
        with self._lock:
            if len(self._free) < 2:
                self._free.append(b)


_buffers = _Buffers()


class _ReadAhead:
    """
    Read the next large data section while the caller decodes this one.

    Starts after two reads in file order, so header-only walks and random
    access never trigger it, and stays one section ahead: at most one extra
    record's bytes in memory.  The guess assumes the next record has the
    same size (one channel, so one rate, per file: §5.1.3); a wrong guess is a wasted read.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: ThreadPoolExecutor | None = None
        self._last: tuple | None = None  # (path, stamp, end) of the last read
        self._next: tuple | None = None  # (key, future) of the section ahead

    def get(self, src: _DiskSource, f: BinaryIO | None) -> bytes | np.ndarray:
        key = (src.path, src.stamp, src.offset, src.size)
        end = src.offset + src.size
        with self._lock:
            ahead, self._next = self._next, None
            in_order = self._last == (src.path, src.stamp, src.offset - _HEADER_SIZE)
            self._last = (src.path, src.stamp, end)
            if (
                in_order
                and src.size >= _READ_AHEAD_MIN_BYTES
                and end + _HEADER_SIZE + src.size <= src.stamp[0]
            ):
                nxt = _DiskSource(
                    src.path, end + _HEADER_SIZE, src.size, src.complex_dtype, src.stamp,
                    src.sample_size,
                )
                if self._thread is None:
                    self._thread = ThreadPoolExecutor(1)
                self._next = (
                    (nxt.path, nxt.stamp, nxt.offset, nxt.size),
                    self._thread.submit(nxt._read_now),
                )
        if ahead is not None:
            if ahead[0] == key:
                try:
                    data = ahead[1].result()
                    st = src.path.stat()
                    if (st.st_size, st.st_mtime_ns) == src.stamp:
                        return data
                except Exception:
                    pass  # read again below: errors surface as on a direct read
            else:
                ahead[1].cancel()
        return src._read_now(f)


_readahead = _ReadAhead()

if hasattr(os, "register_at_fork"):
    # A forked child inherits the executor but not its thread, and any lock
    # held at the fork stays held forever: start both afresh.
    os.register_at_fork(after_in_child=_readahead.__init__)
    os.register_at_fork(after_in_child=_buffers.__init__)


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

    Reads only the 8-byte header prefix per record: the label and RECORD
    LENGTH are checked, nothing else, so a file that ``read_product``
    rejects (e.g. RECORD LENGTH not matching the sample size and rate) can
    still scan clean.

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

    Raises
    ------
    ValueError
        If a record label is not ``RDEF`` or RECORD LENGTH is below the
        176-byte header (Table 5-1).
    EOFError
        If the file ends inside a record (§5.1).
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

            _skip(f, record_length - _HEADER_PREFIX_SIZE, file_size, count, record_length)
            count += 1
            pbar.update(record_length)

    return count, consistent_length


def _warn_picoseconds(pf: ProductFile) -> None:
    """One RuntimeWarning if any record's picoseconds exceed DSN 0222's range."""
    n = sum(1 for r in pf.records if (r.header.time_tag_picoseconds_of_the_second or 0.0) > _PS_0222_MAX)
    if n:
        warnings.warn(
            f"time_tag_picoseconds_of_the_second exceeds DSN 0222's range (0 to "
            f"{_PS_0222_MAX:,.0f} ps, Table 3-1) in {n} record(s): the phase-polynomial "
            "coefficients may refer to another origin than TIME TAG SECOND OF DAY. Pass "
            "t0_offset (e.g. 1.0 for the next second, or picoseconds*1e-12 for the first "
            "sample); passing any value, including 0.0, chooses explicitly and silences this",
            RuntimeWarning,
            stacklevel=_user_stacklevel(),
        )


def read_headers(
    path: str | Path,
    *,
    progress: bool = False,
    t0_offset: float | None = None,
    agency_layout: str | np.dtype = "auto",
) -> ProductFile:
    """
    Read only RDEF record headers, skipping all data sections.

    Records will contain headers but empty ``RecordData`` objects.
    Significantly faster than ``read_product`` when only header fields
    are needed.

    Parameters
    ----------
    path : str or Path
        Path to RDEF product file.
    progress : bool
        Draw a progress bar on stderr (default False).
    t0_offset : float or None, default None
        Rarely needed.  The downconversion polynomial origin is TIME TAG
        SECOND OF DAY (§5.2.6.2; DSN 0222 §3.4); for a producer known to use
        another origin, this moves it by *t0_offset* seconds for every
        record.  When it is not passed and a record has picoseconds above
        DSN 0222's 100,000, one ``RuntimeWarning``.
    agency_layout : "auto", "dsn" or np.dtype, default "auto"
        Layout of the internal-agency block for ``agency_fields``; sets the
        read-only ``ProductFile.agency_layout``.  Nothing is decoded at read
        time.  Any other string, a dtype that is not 40 bytes or one with
        object fields raises ``ValueError``.

    Returns
    -------
    ProductFile
        Product file with headers only (no I/Q data).

    Raises
    ------
    ValueError
        If a header fails the structural checks of §5.2 (labels, sample
        size, record length); if *t0_offset* or *agency_layout* is invalid.
    EOFError
        If the file ends inside a record (§5.1).

    Warns
    -----
    UserWarning
        If a header field is out of range (Table 5-1, §5.2.4); the record
        is read as is.
    RuntimeWarning
        If a file-constant field (station, spacecraft, agency, version,
        sample size, sample rate, fixed downconversion) varies between
        records (§5).
    RuntimeWarning
        Once per read, if *t0_offset* is not passed and a record's
        picoseconds exceed DSN 0222's 100,000 (Table 3-1): the polynomial
        coefficients may refer to another origin.
    """
    path = Path(path)
    offset = 0.0 if t0_offset is None else _check_t0_offset(t0_offset)
    agency_layout = _check_agency_layout(agency_layout)
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

            records.append(
                Record(
                    header=hdr,
                    data=RecordData(sample_size=hdr.sample_size),
                )
            )

            _skip(f, hdr.record_length - _HEADER_SIZE, file_size, idx, hdr.record_length)
            pbar.update(hdr.record_length)
            idx += 1

    pf = ProductFile(name=path.stem)
    pf.path = path.resolve()
    pf.records = records
    pf._set_read_settings(offset, agency_layout)
    if t0_offset is None:
        _warn_picoseconds(pf)

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
    t0_offset: float | None = None,
    agency_layout: str | np.dtype = "auto",
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
        Draw a progress bar on stderr (default False) for the header pass,
        which also loads the records at or below 1 kHz.
    start_epoch, stop_epoch : datetime, np.datetime64, str, or None, optional
        Keep records with ``start_epoch <= timetag < stop_epoch``
        (half-open, so consecutive windows tile without overlap or gap),
        compared on the full time tag, picoseconds included.  Accepts
        datetime (naive = UTC), numpy.datetime64, or an ISO-8601 or CCSDS
        string (e.g. "2026-02-08T12:00:00", "2026-039T12:00:00").
    filters : dict, optional
        Header-field filters.  Keys are RecordHeader field names (e.g.
        ``"station_id"``, ``"spacecraft_id"``, ``"validity_flag"``,
        ``"sample_rate"``); values are the expected values.  Records whose
        header fields do not match **all** filters are skipped without
        reading their data.
    min_downconv_hz, max_downconv_hz : float, optional
        Total downconversion-frequency window in Hz, evaluated at each
        record's first sample (``record_downconversion_frequency``).  A
        record whose c1, c2 and c3 are all NaN (DSN 0222 Millisecond
        Predict) is compared on its fixed part,
        ``rf_to_if_downconv + if_to_channel_downconv``.
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
    t0_offset : float or None, default None
        Rarely needed.  The downconversion polynomial origin is TIME TAG
        SECOND OF DAY (§5.2.6.2; DSN 0222 §3.4); for a producer known to use
        another origin, this moves it by *t0_offset* seconds for every
        record.  When it is not passed and a record has picoseconds above
        DSN 0222's 100,000, one ``RuntimeWarning``.
    agency_layout : "auto", "dsn" or np.dtype, default "auto"
        Layout of the internal-agency block for ``agency_fields``; sets the
        read-only ``ProductFile.agency_layout``.  Nothing is decoded at read
        time.  Any other string, a dtype that is not 40 bytes or one with
        object fields raises ``ValueError``.

    Returns
    -------
    ProductFile
        All kept records; samples loaded at <= 1 kHz, on disk above.

    Raises
    ------
    ValueError
        If a filter key is not a valid RecordHeader field name; if *band*
        is unknown or combined with *min_downconv_hz* / *max_downconv_hz*;
        if a header fails the structural checks of §5.2 (labels, sample
        size, record length); if *start_epoch* / *stop_epoch* is an
        unparseable string; or if *t0_offset* or *agency_layout* is invalid.
    TypeError
        If *start_epoch* / *stop_epoch* is not a datetime, np.datetime64,
        str or None.
    EOFError
        If the file ends inside a record (§5.1).

    Warns
    -----
    UserWarning
        If a header field is out of range (Table 5-1, §5.2.4); the record
        is read as is.
    RuntimeWarning
        If a file-constant field (station, spacecraft, agency, version,
        sample size, sample rate, fixed downconversion) varies between
        records (§5).
    RuntimeWarning
        Once per read, if *t0_offset* is not passed and a record's
        picoseconds exceed DSN 0222's 100,000 (Table 3-1): the polynomial
        coefficients may refer to another origin.

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
    offset = 0.0 if t0_offset is None else _check_t0_offset(t0_offset)
    agency_layout = _check_agency_layout(agency_layout)

    if filters:
        valid_fields = _HEADER_FIELD_SET
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
            t0_offset=offset,
        )
    )
    pf._set_read_settings(offset, agency_layout)
    if t0_offset is None:
        _warn_picoseconds(pf)

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
    t0_offset: float = 0.0,
) -> Iterator[Record]:
    """
    Walk and filter a product file's headers, yielding one record per kept record.

    Shared core of :func:`read_product`.  Inputs are assumed already
    validated and resolved by the caller: ``band`` -> min/max window, epochs
    -> ``datetime64``, filter keys checked.  Filters see the header only:
    data sections of skipped records, and of kept records above 1 kHz, are
    seeked over; kept records at or below 1 kHz are read and decoded now.
    Every kept record carries a :class:`_DiskSource`.
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
                pbar.finish()  # EOF before num_records: full bar
                break

            hdr = RecordHeader(header_dict)
            data_size = hdr.record_length - _HEADER_SIZE
            offset = f.tell()
            pbar.update(1 if use_record_limit else hdr.record_length)
            idx += 1

            # Filters run on the header alone, so a skipped record's data is
            # never read.  No early stop at stop_epoch: the records of a file
            # need not be in time order.  A time tag that cannot be evaluated
            # fails any time window.
            keep = True
            if time_filtered:
                rec_epoch = _record_epoch(hdr)
                keep = rec_epoch is not None and (
                    (start_epoch is None or rec_epoch >= start_epoch)
                    and (stop_epoch is None or rec_epoch < stop_epoch)
                )
            if keep and filters:
                keep = all(getattr(hdr, k) == v for k, v in filters.items())
            if keep and (min_downconv_hz is not None or max_downconv_hz is not None):
                keep = _in_window(_filter_downconv_hz(hdr, t0_offset), min_downconv_hz, max_downconv_hz)

            # Samples read_product loads anyway are read now, in this pass:
            # one sequential read instead of a second seek-and-read per record.
            if keep and hdr.sample_rate <= _AUTO_LOAD_MAX_RATE_HZ:
                data_bytes = f.read(data_size)
                if len(data_bytes) != data_size:
                    raise EOFError(
                        f"Truncated product file: record {idx - 1} declares "
                        f"{hdr.record_length} bytes but the file ends early (§5.1)."
                    )
            else:
                data_bytes = None
                _skip(f, data_size, file_size, idx - 1, hdr.record_length)
            if not keep:
                continue

            src = _DiskSource(path, offset, data_size, complex_dtype, stamp, hdr.sample_size)
            rec = Record._on_disk(hdr, src)
            if data_bytes is not None:
                rec._data = src.decode(data_bytes, hdr, writeable=True)
            yield rec
