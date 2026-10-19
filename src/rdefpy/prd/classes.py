# Copyright (c) 2026 Morehead State University
# Author: Andrea Togni
# SPDX-License-Identifier: MIT

"""
Data model classes for RDEF product files (.prd).

Implements the binary record structure defined in CCSDS 506.1-B-1
(June 2013), "Delta-DOR Raw Data Exchange Format", §5.

Classes
-------
RecordHeader
    Parsed metadata for a single RDEF record (§5.2, Table 5-1).
RecordData
    I/Q sample container (§5.3).
Record
    Single RDEF record combining header and data (§5.1).
ProductFile
    Top-level container for complete RDEF product files.

Notes
-----
These are containers, not estimators.  Spectra, carrier tracking, and
simulation belong to the caller; this module's job is to hand them
correct samples with correct time tags.
"""

from __future__ import annotations

import copy
import operator
import warnings
import weakref
from datetime import datetime
from pathlib import Path
from collections.abc import Iterator
from typing import Any, TextIO, overload

import numpy as np

from rdefpy._utils import _Progress, _to_datetime64
from rdefpy.prd import _display

# =============================================================================
# Constants
# =============================================================================

# Standard RDEF header fields per CCSDS 506.1-B-1 Table 5-1
_HEADER_FIELDS = (
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

_HEADER_SIZE_BYTES = 176

# Maximum absolute sample value per bit-depth after the 2k+1 transform (§5.3.6).
_SATURATION_THRESHOLD: dict[int, int] = {
    1: 1,
    2: 3,
    4: 15,
    8: 255,
    16: 65535,
}


# =============================================================================
# Downconversion reconstruction (§5.2.6) -- the one implementation
# =============================================================================

_PHASE_FIELDS = (
    "channel_accumulated_phase",
    "channel_phase_polynomial_coefficient_0",
    "channel_phase_polynomial_coefficient_1",
    "channel_phase_polynomial_coefficient_2",
    "channel_phase_polynomial_coefficient_3",
)


def _fixed_downconv_hz(header: RecordHeader) -> float:
    """
    Fixed downconversion ``rf_to_if + if_to_channel`` in Hz (§5.2.6).

    Raises ``ValueError`` when either field is unset: a 0 Hz local
    oscillator is never a valid default, so a half-built header must not
    silently yield a sky frequency off by the whole LO.
    """
    rf, ifc = header.rf_to_if_downconv, header.if_to_channel_downconv
    if rf is None or ifc is None:
        raise ValueError(
            "rf_to_if_downconv and if_to_channel_downconv must both be set to "
            f"reconstruct the downconversion (§5.2.6); got {rf!r} and {ifc!r}"
        )
    return float(rf) + float(ifc)


def _phase_terms(header: RecordHeader) -> tuple[float, float, float, float, float]:
    """
    ``(Phi, c0, c1, c2, c3)`` from the header (§5.2.6).

    Unset values count as 0: a header built by hand without a variable
    downconversion part has none, which is what 0 means.
    """
    return tuple(float(getattr(header, f) or 0.0) for f in _PHASE_FIELDS)


def _downconv_hz(header: RecordHeader, t):
    """
    Total downconversion frequency in Hz at *t* seconds from the record start.

    ``f_DC(t) = (rf_to_if + if_to_channel) + (c1 + 2*c2*t + 3*c3*t**2)``
    (§5.2.6).  *t* is a float or an array.  Every downconversion frequency
    in the package -- per sample, per record, the
    ``select`` / ``read_product`` band filters -- goes through here.
    """
    _, _, c1, c2, c3 = _phase_terms(header)
    return _fixed_downconv_hz(header) + (c1 + 2.0 * c2 * t + 3.0 * c3 * t**2)


def _downconv_phase(header: RecordHeader, t):
    """
    Downconversion phase in turns at *t* seconds from the record start.

    ``phi(t) = Phi + c0 + c1*t + c2*t**2 + c3*t**3`` (§5.2.6), where ``Phi``
    (CHANNEL ACCUMULATED PHASE) already includes the fixed downconverter.
    *t* is a float or an array.
    """
    phi, c0, c1, c2, c3 = _phase_terms(header)
    return phi + c0 + c1 * t + c2 * t**2 + c3 * t**3


def _header_sample_rate(header: RecordHeader) -> float:
    """Header ``sample_rate`` as float; raises if unset or non-positive."""
    if header.sample_rate is None:
        raise ValueError("sample_rate is required for downconversion computation")
    fs = float(header.sample_rate)
    if fs <= 0:
        raise ValueError(f"Invalid sample rate: {fs} Hz")
    return fs


def _record_epoch(header: RecordHeader) -> np.datetime64 | None:
    """``header.timetag``, or ``None`` when the time-tag fields are unusable."""
    try:
        return header.timetag
    except (TypeError, ValueError, OverflowError):
        return None


# Header fields that must be the same in every record of a product file:
# one scan, one channel, one station (§5).  Checked when a file is read and
# when two files are concatenated.
_FILE_CONSTANT_FIELDS = (
    "station_id",
    "spacecraft_id",
    "agency_flag",
    "record_version_id",
    "sample_size",
    "sample_rate",
    "rf_to_if_downconv",
    "if_to_channel_downconv",
)


# =============================================================================
# RecordHeader
# =============================================================================


class RecordHeader:
    """
    Parsed RDEF record header (CCSDS 506.1-B-1 Table 5-1).

    Attributes
    ----------
    record_length : int
        Total record length in bytes (header + data).
    record_version_id : int
        RDEF format version identifier.
    station_id : int
        Ground station identifier.
    spacecraft_id : int
        Spacecraft identifier.
    sample_size : int
        Bits per I/Q component (1, 2, 4, 8, or 16).
    sample_rate : int
        Complex samples per second.
    validity_flag : int
        0 = no error.
    agency_flag : int
        0=unused, 1=ESA, 2=JAXA, 3=NASA.
    rf_to_if_downconv : float
        RF-to-IF downconversion frequency (Hz).
    if_to_channel_downconv : float
        IF-to-channel downconversion frequency (Hz).
    time_tag_year : int
        Time tag year.
    time_tag_doy : int
        Time tag day of year (1-366).
    time_tag_second_of_day : int
        Time tag second of day (0-86400).
    time_tag_picoseconds_of_the_second : float
        Sub-second precision in picoseconds.
    channel_accumulated_phase : float
        Accumulated phase in turns at record start (§5.2.6).
    channel_phase_polynomial_coefficient_0 : float
        Phase polynomial coefficient c0 in turns (§5.2.6).
    channel_phase_polynomial_coefficient_1 : float
        Phase polynomial coefficient c1 in turns/s (§5.2.6).
    channel_phase_polynomial_coefficient_2 : float
        Phase polynomial coefficient c2 in turns/s**2 (§5.2.6).
    channel_phase_polynomial_coefficient_3 : float
        Phase polynomial coefficient c3 in turns/s**3 (§5.2.6).
    empty_fields_future_extension : bytes
        Reserved 36 bytes for future extensions.
    empty_fields_internal_agency_use : bytes
        Reserved 40 bytes for internal agency use.
    """

    __slots__ = (*_HEADER_FIELDS,)

    # Declared for type checkers and IDEs (the slots above come from
    # _HEADER_FIELDS at runtime).  Every field is None until set.
    record_length: int | None
    record_version_id: int | None
    station_id: int | None
    spacecraft_id: int | None
    sample_size: int | None
    sample_rate: int | None
    validity_flag: int | None
    agency_flag: int | None
    rf_to_if_downconv: float | None
    if_to_channel_downconv: float | None
    time_tag_year: int | None
    time_tag_doy: int | None
    time_tag_second_of_day: int | None
    time_tag_picoseconds_of_the_second: float | None
    channel_accumulated_phase: float | None
    channel_phase_polynomial_coefficient_0: float | None
    channel_phase_polynomial_coefficient_1: float | None
    channel_phase_polynomial_coefficient_2: float | None
    channel_phase_polynomial_coefficient_3: float | None
    empty_fields_future_extension: bytes | None
    empty_fields_internal_agency_use: bytes | None

    def __init__(self, info: dict[str, Any] | None = None):
        """
        Initialise header fields.

        Parameters
        ----------
        info : dict or None
            Mapping of field names to values.  Unknown keys raise
            ``KeyError``.  All fields default to ``None``.

        Notes
        -----
        Headers are mutable: fields can be modified by direct assignment or
        via ``Record.correct_header()``.
        """
        for field in _HEADER_FIELDS:
            setattr(self, field, None)
        if info is not None:
            for key, value in info.items():
                if key not in _HEADER_FIELDS:
                    raise KeyError(f"Unknown header field: {key}")
                setattr(self, key, value)

    def __deepcopy__(self, memo: dict) -> RecordHeader:
        """Deep-copy every header field into a new ``RecordHeader``."""
        info = {f: copy.deepcopy(getattr(self, f), memo) for f in _HEADER_FIELDS}
        return RecordHeader(info)

    def __repr__(self) -> str:
        return _display.header_repr(self)

    def __str__(self) -> str:
        return _display.header_str(self)

    @property
    def timetag(self) -> np.datetime64:
        """
        Record time tag as a ``numpy.datetime64[ns]``.

        Combines the header time-tag fields (year, day-of-year,
        second-of-day, and picoseconds-of-the-second) into a single
        ``numpy.datetime64`` value with nanosecond resolution.
        Picoseconds are rounded to the nearest nanosecond: the field is a
        float64 (Table 5-1), and a value one ulp below a whole nanosecond
        must not lose that nanosecond.

        Returns
        -------
        numpy.datetime64
            UTC timestamp with ``datetime64[ns]`` precision.
        """
        base = np.datetime64(f"{int(self.time_tag_year):04d}-01-01", "ns")
        base += np.timedelta64(int(self.time_tag_doy) - 1, "D")
        base += np.timedelta64(int(self.time_tag_second_of_day), "s")
        pico = self.time_tag_picoseconds_of_the_second
        if pico:
            base += np.timedelta64(round(float(pico) / 1000), "ns")
        return base


# =============================================================================
# RecordData
# =============================================================================


class RecordData:
    """
    I/Q sample container for one RDEF record.

    Each RDEF record holds exactly one second of data at a fixed sample
    rate (CCSDS 506.1-B-1 §5.1).  Sample values use the ``2k+1``
    bias-corrected representation defined in §5.3.6 and are stored as
    integers -- exactly the values on disk, never scaled or centred.

    The scalar properties below are data-quality checks (level, dynamic
    range, clipping), not signal processing.  For spectra and carrier
    tracking, hand ``samples`` to your estimator.

    Parameters
    ----------
    sample_size : int
        Bits per I/Q component (1, 2, 4, 8, or 16) per §5.3.
    sample_rate : int, optional
        Complex samples per second.  Required by ``duration_s`` and
        ``time_array``.
    i_samples, q_samples : np.ndarray, optional
        In-phase / quadrature arrays.  Must be both provided or both
        omitted.
    complex_dtype : type or np.dtype
        Precision of ``samples`` (complex64 or complex128).

    Attributes
    ----------
    dtype : numpy dtype
        Storage type: int8 for 1/2/4-bit, int16 for 8-bit, int32 for
        16-bit (the smallest type holding the 2k+1 values exactly).
    """

    __slots__ = (
        "sample_size", "sample_rate", "dtype", "_i", "_q", "complex_dtype", "_x",
        "__weakref__",
    )

    def __init__(
        self,
        sample_size: int,
        sample_rate: int | None = None,
        i_samples: np.ndarray | None = None,
        q_samples: np.ndarray | None = None,
        *,
        complex_dtype: type | np.dtype = np.complex128,
    ):
        self.sample_size = int(sample_size)
        if self.sample_size not in (1, 2, 4, 8, 16):
            raise ValueError(
                f"sample_size must be one of (1, 2, 4, 8, 16) "
                f"(§5.3, CCSDS 506.1-B-1), got {self.sample_size}"
            )
        self.sample_rate = int(sample_rate) if sample_rate is not None else None

        cdt = np.dtype(complex_dtype)
        if cdt not in (np.dtype(np.complex64), np.dtype(np.complex128)):
            raise ValueError(
                f"complex_dtype must be np.complex64 or np.complex128, got {cdt}"
            )
        # Precision of the materialised complex samples. complex64 halves the
        # memory bandwidth for high-rate (e.g. 20 MHz) processing; complex128
        # (default) and complex64 are both exact, since the 2k+1 values (§5.3.6)
        # are small integers that fit losslessly in float32.
        self.complex_dtype = cdt

        # Smallest integer type that holds 2k+1 exactly (§5.3.6 doubles the
        # raw range): 16-bit -> +-65535 needs int32, 8-bit -> +-255 needs
        # int16, 4/2/1-bit -> +-15/+-3/+-1 fit int8.
        if self.sample_size == 16:
            self.dtype = np.int32
        elif self.sample_size == 8:
            self.dtype = np.int16
        else:
            self.dtype = np.int8

        if (i_samples is None) != (q_samples is None):
            raise ValueError(
                "i_samples and q_samples must be both provided or both None"
            )
        self._x: np.ndarray | None = None  # cached complex view, see samples
        if i_samples is None:
            self._i = np.empty(0, dtype=self.dtype)
            self._q = np.empty(0, dtype=self.dtype)
            return

        i_arr = np.asarray(i_samples, dtype=self.dtype)
        q_arr = np.asarray(q_samples, dtype=self.dtype)
        if i_arr.shape != q_arr.shape:
            raise ValueError("i_samples and q_samples must have the same shape")
        self._i = i_arr
        self._q = q_arr

    # ---- basic --------------------------------------------------------

    def __len__(self) -> int:
        """Return number of complex sample pairs."""
        return self._i.shape[0]

    def __repr__(self) -> str:
        return _display.data_repr(self)

    def __str__(self) -> str:
        return _display.data_str(self)

    def _require_sample_rate(self) -> int:
        """Raise if sample_rate is not set; return it otherwise."""
        if self.sample_rate is None or self.sample_rate <= 0:
            raise ValueError(
                "sample_rate is required (a positive integer) for this "
                f"operation, got {self.sample_rate!r}. Set it via "
                "RecordData(..., sample_rate=fs) or rd.sample_rate = fs."
            )
        return self.sample_rate

    # ---- I/Q access ---------------------------------------------------

    @property
    def i_samples(self) -> np.ndarray:
        """In-phase samples."""
        return self._i

    @i_samples.setter
    def i_samples(self, value: np.ndarray) -> None:
        arr = np.asarray(value, dtype=self.dtype)
        if self._q.size != 0 and arr.shape != self._q.shape:
            raise ValueError("New I array shape does not match existing Q")
        self._i = arr
        self._x = None

    @property
    def q_samples(self) -> np.ndarray:
        """Quadrature samples."""
        return self._q

    @q_samples.setter
    def q_samples(self, value: np.ndarray) -> None:
        arr = np.asarray(value, dtype=self.dtype)
        if self._i.size != 0 and arr.shape != self._i.shape:
            raise ValueError("New Q array shape does not match existing I")
        self._q = arr
        self._x = None

    @property
    def samples(self) -> np.ndarray:
        """Complex-valued signal ``I + jQ`` at ``self.complex_dtype``.

        Defaults to complex128; complex64 when the record was decoded
        single-precision (``read_product(..., dtype=np.complex64)``).  Built
        directly at the target precision (no complex128 intermediate), and
        lossless either way -- the 2k+1 values are exact integers in float32.

        These are the recorded samples, unmodified.  No mean removal, no
        window, no scaling: on an open-loop recording the carrier can sit in
        the DC bin, where "harmless" DC removal would delete the signal.

        Built once and cached (5 ms per 1e6 samples at complex128), so
        repeated access is free.  The array is READ-ONLY for that reason:
        every caller shares it, and an in-place taper or mean removal would
        silently alter the record for everyone after.  Copy first if you
        must modify: ``x = rec.data.samples.copy()``.  Assigning
        ``i_samples`` / ``q_samples`` invalidates the cache.
        """
        if self._x is None:
            out = np.empty(self._i.shape[0], dtype=self.complex_dtype)
            out.real = self._i
            out.imag = self._q
            out.flags.writeable = False
            self._x = out
        return self._x

    # ---- scalar statistics --------------------------------------------

    def _sum_sq(self) -> float:
        # sum(I^2 + Q^2) in float64 without materialising float copies of I
        # and Q: einsum casts block-wise inside its loop. Exact for 2k+1
        # integers (|y| <= 65535, y^2 < 2^53) and ~7x faster than astype.
        return float(
            np.einsum("i,i->", self._i, self._i, dtype=np.float64)
            + np.einsum("i,i->", self._q, self._q, dtype=np.float64)
        )

    @property
    def power(self) -> float:
        """Average power of the complex signal."""
        if len(self) == 0:
            return float("nan")
        return self._sum_sq() / len(self)

    @property
    def power_db(self) -> float:
        """Average power in dB."""
        return 10.0 * np.log10(self.power + 1e-12)

    @property
    def energy(self) -> float:
        """Total energy (sum of |x|**2)."""
        return self._sum_sq()

    @property
    def energy_db(self) -> float:
        """Total energy in dB."""
        return 10.0 * np.log10(self.energy + 1e-12)

    @property
    def peak_amplitude(self) -> float:
        """Peak amplitude of the complex signal."""
        # max(I^2 + Q^2) in 64k-sample chunks: small int64 temps instead of
        # two full float64 copies. Values are exact integers.
        peak = 0
        for s in range(0, len(self), 65536):
            i = self._i[s : s + 65536]
            q = self._q[s : s + 65536]
            r2 = np.multiply(i, i, dtype=np.int64)
            r2 += np.multiply(q, q, dtype=np.int64)
            peak = max(peak, int(r2.max()))
        return float(np.sqrt(peak))

    @property
    def rms(self) -> float:
        """Root-mean-square amplitude."""
        return float(np.sqrt(self.power))

    # ---- saturation detection -----------------------------------------

    @property
    def is_saturated(self) -> bool:
        """
        Whether any I or Q sample is at the representable limit.

        A sample counts as saturated when its absolute value equals the
        maximum representable value for the current ``sample_size`` after
        the 2k+1 transform (§5.3.6).

        Returns
        -------
        bool
            True if at least one sample reaches the saturation threshold.

        Warns
        -----
        UserWarning
            For 1-bit samples, where saturation is undefined (every value
            is inherently ±1).  Returns False.
        """
        threshold = _SATURATION_THRESHOLD[self.sample_size]
        if self.sample_size == 1:
            warnings.warn(
                "Saturation is undefined for 1-bit samples "
                "(all values are inherently ±1)",
                stacklevel=2,
            )
            return False
        if len(self) == 0:
            return False
        # min/max reductions: no abs() copy of the arrays
        return bool(
            self._i.max() >= threshold or self._i.min() <= -threshold
            or self._q.max() >= threshold or self._q.min() <= -threshold
        )

    @property
    def saturation_ratio(self) -> float:
        """
        Fraction of complex samples where I or Q is at the limit.

        Returns
        -------
        float
            Value between 0.0 (no saturation) and 1.0 (all saturated).

        Warns
        -----
        UserWarning
            For 1-bit samples, where saturation is undefined.  Returns 0.0.
        """
        threshold = _SATURATION_THRESHOLD[self.sample_size]
        if self.sample_size == 1:
            warnings.warn(
                "Saturation is undefined for 1-bit samples "
                "(all values are inherently ±1)",
                stacklevel=2,
            )
            return 0.0
        n = len(self)
        if n == 0:
            return 0.0
        # Four 1-byte bool temps instead of two full-width abs() copies.
        hit = self._i >= threshold
        hit |= self._i <= -threshold
        hit |= self._q >= threshold
        hit |= self._q <= -threshold
        return float(np.count_nonzero(hit) / n)

    # ---- time-domain helpers ------------------------------------------

    @property
    def duration_s(self) -> float:
        """Duration in seconds (requires ``sample_rate``)."""
        return len(self) / self._require_sample_rate()

    @property
    def time_array(self) -> np.ndarray:
        """Sample time offsets in seconds (requires ``sample_rate``)."""
        return np.arange(len(self)) / self._require_sample_rate()


# =============================================================================
# Record
# =============================================================================


class Record:
    """
    Single RDEF record: one header plus one second of I/Q (§5.1).

    A record read from a file knows where its samples are on disk but does
    not hold them.  ``record.data`` reads and decodes them on each access;
    nothing keeps them, so they are freed as soon as you drop the result.
    ``load()`` keeps them in memory until ``release()``.

    Attributes
    ----------
    header : RecordHeader
    data : RecordData
        The samples -- from memory when loaded, else read from disk now.

    Notes
    -----
    On an unloaded record, every ``record.data`` access that finds no live
    result reads the file again: ``record.data.power`` then
    ``record.data.samples`` decodes twice (about 70 ms each for a 16-bit
    20 Msps record).  Take ``d = record.data`` once and reuse ``d``; while
    ``d`` is alive, further ``record.data`` accesses return the same object
    without touching the disk.  Arrays read this way are read-only -- to edit
    samples, ``load()`` first or assign ``record.data = new_data``.

    A record built in memory, ``Record(header, data)``, has no file behind
    it: it is always loaded and ``release()`` does nothing.
    """

    __slots__ = ("header", "_data", "_source", "_weak", "_index")

    header: RecordHeader
    _data: RecordData | None
    _source: Any  # reader._DiskSource, or None for a record built in memory
    _weak: weakref.ref | None
    _index: int | None

    def __init__(
        self,
        header: RecordHeader | dict | None = None,
        data: RecordData | None = None,
    ):
        """
        Parameters
        ----------
        header : RecordHeader, dict, or None
            Created empty if *None*, wrapped if *dict*.
        data : RecordData or None
            Defaults to an empty 16-bit RecordData.
        """
        if isinstance(header, RecordHeader):
            self.header = header
        elif header is None:
            self.header = RecordHeader()
        else:
            self.header = RecordHeader(header)

        self._data: RecordData | None = (
            data if data is not None else RecordData(sample_size=16)
        )
        # On-disk location of the samples (reader._DiskSource), or None for
        # a record built in memory.
        self._source = None
        # Weak reference to the last transient read: shared while anyone
        # holds it, freed when nobody does.
        self._weak: weakref.ref | None = None
        # Position within a parent ProductFile; set by ProductFile.__getitem__.
        self._index: int | None = None

    @classmethod
    def _on_disk(cls, header: RecordHeader, source) -> Record:
        """A record whose samples stay on disk until accessed or loaded."""
        rec = cls.__new__(cls)
        rec.header = header
        rec._data = None
        rec._source = source
        rec._weak = None
        rec._index = None
        return rec

    # ---- samples: on access, load, release --------------------------------

    @property
    def data(self) -> RecordData:
        """The samples: from memory when loaded, otherwise read from disk now."""
        if self._data is not None:
            return self._data
        live = self._weak() if self._weak is not None else None
        if live is not None:
            return live
        data = self._source.read(self.header, writeable=False)
        self._weak = weakref.ref(data)
        return data

    @data.setter
    def data(self, value: RecordData) -> None:
        # Assigned samples replace the file's: the record is now in memory.
        if not isinstance(value, RecordData):
            raise TypeError(f"data must be a RecordData, got {type(value).__name__}")
        self._data = value
        self._source = None
        self._weak = None

    def load(self) -> None:
        """Read the samples into memory and keep them until ``release()``."""
        if self._data is None:
            self._data = self._source.read(self.header, writeable=True)
            self._weak = None

    def release(self) -> None:
        """
        Drop loaded samples; later access reads them from disk again.

        Discards any in-memory edits.  No effect on a record built in memory,
        which has no file to re-read.
        """
        if self._source is not None:
            self._data = None

    def __len__(self) -> int:
        """Number of complex samples, without reading them from disk."""
        if self._data is not None:
            return len(self._data)
        return self._source.n_samples(self.header)

    def __bool__(self) -> bool:
        # A record is a record even with 0 samples; __len__ must not make
        # ``if record:`` false.
        return True

    def __getstate__(self):
        # The weak reference cannot be pickled or copied; nothing is decoded.
        return (self.header, self._data, self._source, self._index)

    def __setstate__(self, state) -> None:
        self.header, self._data, self._source, self._index = state
        self._weak = None

    def __repr__(self) -> str:
        return _display.record_repr(self)

    def __str__(self) -> str:
        return _display.record_str(self)

    def correct_header(self, **kwargs) -> None:
        """
        Update ``RecordHeader`` fields for this single record, in place.

        Parameters
        ----------
        **kwargs
            RecordHeader field names and their new values.

        Raises
        ------
        ValueError
            If an unknown field name is provided.

        Examples
        --------
        Mark a single record as invalid:

        >>> rec.correct_header(validity_flag=1)

        Notes
        -----
        For bulk edits across all records use ``ProductFile.correct_header()``.
        """
        for key in kwargs:
            if key not in _HEADER_FIELDS:
                raise ValueError(
                    f"Unknown header field: {key!r}. "
                    f"Valid fields: {list(_HEADER_FIELDS)}"
                )

        for key, value in kwargs.items():
            setattr(self.header, key, value)

    @property
    def downconversion_frequency(self) -> np.ndarray:
        """
        Total downconversion frequency for all samples in the record.

        Computes the full downconverter frequency as defined in CCSDS
        506.1-B-1 §5.2.6, combining fixed and variable components.

        **Fixed part (constant):**
            f_DC_fixed = rf_to_if_downconv + if_to_channel_downconv

        **Variable part (time-dependent):**
            f_DC_var(t) = c1 + 2·c2·(t-t0) + 3·c3·(t-t0)²

        **Total:**
            f_DC(t) = f_DC_fixed + f_DC_var(t)

        Returns
        -------
        np.ndarray
            Downconversion frequency in Hz for each sample. Shape matches
            the number of I/Q samples in the record.

        Notes
        -----
        - Time reference t0 is the start of the record (time_tag_second_of_day).
        - Sample times are measured in seconds from the record start.
        - Phase polynomial coefficients are in turns/s (Hz for c1), turns/s²
          for c2, and turns/s³ for c3.
        - Per §5.2.6, the fixed IF-to-channel downconverter is assumed to
          have integer phase at integer-second boundaries.

        References
        ----------
        CCSDS 506.1-B-1 (June 2013), §5.2.6: "Downconversion Reconstruction"
        """
        n_samples = len(self)
        if n_samples == 0:
            return np.array([], dtype=np.float64)
        fs = _header_sample_rate(self.header)
        return _downconv_hz(self.header, np.arange(n_samples, dtype=np.float64) / fs)

    @property
    def downconversion_phase(self) -> np.ndarray:
        """
        Downconversion phase for all samples in the record.

        Evaluates the phase polynomial from the record header as
        defined in CCSDS 506.1-B-1 §5.2.6:

            φ(t) = Φ + c0 + c1·t + c2·t² + c3·t³

        where ``Φ`` is CHANNEL ACCUMULATED PHASE (which already
        incorporates the fixed downconverter contribution), ``c0-c3``
        are the CHANNEL PHASE POLYNOMIAL COEFFICIENTS, and ``t`` is
        measured from the record start.

        The fixed-part phase ramp (from ``rf_to_if_downconv`` and
        ``if_to_channel_downconv``) is not computed separately because
        its absolute time reference is unknown (§5.2.6).

        Returns
        -------
        np.ndarray
            Downconversion phase in turns (cycles) for each sample.

        References
        ----------
        CCSDS 506.1-B-1 (June 2013), §5.2.6
        """
        n_samples = len(self)
        if n_samples == 0:
            return np.array([], dtype=np.float64)
        fs = _header_sample_rate(self.header)
        return _downconv_phase(self.header, np.arange(n_samples, dtype=np.float64) / fs)


def _load_records(records, *, progress: bool = False) -> None:
    """Load every not-yet-loaded record, one open handle per file, in order."""
    todo = [r for r in records if r._data is None]
    handles: dict = {}
    try:
        with _Progress(len(todo), progress) as pbar:
            for rec in todo:
                src = rec._source
                f = handles.get(src.path)
                if f is None:
                    f = handles[src.path] = src.open()
                rec._data = src.read(rec.header, writeable=True, f=f)
                rec._weak = None
                pbar.update(1)
    finally:
        for f in handles.values():
            f.close()


# =============================================================================
# ProductFile
# =============================================================================


class ProductFile:
    """
    Container for a complete RDEF product file (.prd).

    A product file holds one or more sequential one-second records
    (CCSDS 506.1-B-1 §5) for a single scan, channel, and station.

    Parameters
    ----------
    name : str
        Human-readable name (e.g. filename stem).  Defaults to ``""``.

    Attributes
    ----------
    records : list[Record]
        Ordered list of records.
    name : str
        Human-readable name for this product file.
    path : Path or None
        Absolute path of the file the records were read from; ``None`` for
        a product file built in memory or merged from different files.
        Kept by ``select``/``slice``/``copy``; not changed by
        ``write_product``.
    """

    def __init__(self, *, name: str = ""):
        self.records: list[Record] = []
        self.name: str = name
        self.path: Path | None = None

    def __len__(self) -> int:
        return len(self.records)

    @overload
    def __getitem__(self, idx: int) -> Record: ...
    @overload
    def __getitem__(self, idx: slice) -> list[Record]: ...

    def __getitem__(self, idx: int | slice) -> Record | list[Record]:
        result = self.records[idx]
        if isinstance(idx, int):
            result._index = idx + len(self.records) if idx < 0 else idx
        return result

    def __iter__(self) -> Iterator[Record]:
        """Iterate over records in order."""
        return iter(self.records)

    def __repr__(self) -> str:
        return _display.product_repr(self)

    def __str__(self) -> str:
        return _display.product_str(self)

    def info(self, buf: TextIO | None = None) -> None:
        """
        Print a summary of the product file.

        Same text as ``str(product_file)``: right-aligned ``label: value``
        rows, labelled by attribute name where one exists.  Derived rows:

        * ``first_timetag`` / ``last_timetag`` -- earliest and latest record
          START epochs (full ISO 8601 UTC, nanoseconds).
        * ``consecutive`` -- whether records are spaced exactly 1 s (§5.1).
        * ``band`` -- each record's fixed downconversion
          (``rf_to_if + if_to_channel``) classified into the nominal DSN
          S/X/Ka allocations; ``unknown`` if it matches none.
        * ``frequency_steering`` -- ``True`` when the variable downconversion
          is time-varying: any nonzero c2/c3, or c1 changing between records.

        Parameters
        ----------
        buf : text stream, optional
            Where to write (``sys.stdout`` when None), as in
            ``pandas.DataFrame.info``.
        """
        _display.print_info(self, buf)

    # ---- header editing -----------------------------------------------

    def correct_header(self, **kwargs) -> None:
        """
        Set header fields on every record, in place.

        Nothing guards against it: correcting a header is a deliberate act.

        Parameters
        ----------
        **kwargs
            RecordHeader field names and their new values.

        Raises
        ------
        ValueError
            If unknown field name is provided.

        Examples
        --------
        Bulk correction of common fields:

        >>> pf.correct_header(station_id=24, spacecraft_id=42)

        Mixed bulk and single-record editing:

        >>> pf.correct_header(station_id=24)               # bulk
        >>> pf.records[5].correct_header(validity_flag=1)  # single

        See Also
        --------
        Record.correct_header : Edit a single record header.
        """
        valid_fields = set(_HEADER_FIELDS)
        for key in kwargs:
            if key not in valid_fields:
                raise ValueError(
                    f"Unknown header field: {key!r}. "
                    f"Valid fields: {list(_HEADER_FIELDS)}"
                )

        for rec in self.records:
            rec.correct_header(**kwargs)

    def _check_constant_fields(self) -> None:
        """Warn when a field that must be file-constant varies between records."""
        varying_fields = set()
        for field in _FILE_CONSTANT_FIELDS:
            values = [
                getattr(r.header, field)
                for r in self.records
                if getattr(r.header, field) is not None
            ]
            if values and not all(v == values[0] for v in values):
                varying_fields.add(field)

        if varying_fields:
            warnings.warn(
                f"Metadata fields vary across records: {sorted(varying_fields)}; "
                f"a product file should be one channel of one station (§5).",
                RuntimeWarning,
                stacklevel=2,
            )

    # ---- concatenation ------------------------------------------------

    def concatenate(
        self,
        other: ProductFile,
        keep: str = "other",
        sort: bool = True,
    ) -> ProductFile:
        """
        Merge two product files of the same channel into a new one.

        Records from both files are merged and, by default, sorted
        chronologically by their time tags (year, DOY, second-of-day,
        picoseconds).  When two records share the same time tag the
        duplicate is reported and resolved according to *keep*.  The result
        holds independent copies of the records (headers and disk locations;
        samples only where loaded), so editing, loading or releasing it never
        touches *self* or *other*.

        Parameters
        ----------
        other : ProductFile
            The product file to concatenate with this one.
        keep : str
            Which duplicate to retain: ``"other"`` (default) keeps the
            record from *other*; ``"self"`` keeps the record from this
            file; ``"both"`` keeps both (no deduplication).  Within one
            input, a repeated time tag keeps its last record.
        sort : bool
            If ``True`` (default), sort the merged records by time tag.
            If ``False``, records from *self* appear first, followed by
            records from *other*, preserving their original order.

        Returns
        -------
        ProductFile
            New product file, named after *self*; its ``path`` is the
            inputs' common file, or ``None`` if they came from different files.

        Raises
        ------
        ValueError
            If *keep* is not one of ``"self"``, ``"other"``, or ``"both"``;
            or if the files disagree on a field that must be constant
            across a product file (station, spacecraft, agency, version,
            sample size, sample rate, fixed downconversion): a product file
            is one scan, one channel, one station (§5).

        Warns
        -----
        RuntimeWarning
            On duplicate time tags, within an input or between the two.
        """
        valid_keep = ("self", "other", "both")
        if keep not in valid_keep:
            raise ValueError(f"keep must be one of {valid_keep}, got {keep!r}")

        # Same channel? Compare every file-constant field across both inputs.
        clash = []
        for field in _FILE_CONSTANT_FIELDS:
            ours = {getattr(r.header, field) for r in self.records} - {None}
            theirs = {getattr(r.header, field) for r in other.records} - {None}
            if ours and theirs and ours != theirs:
                clash.append(f"{field}: {sorted(ours)} vs {sorted(theirs)}")
        if clash:
            raise ValueError(
                "Cannot concatenate product files of different channels "
                "(§5: one scan, one channel, one station per file); "
                "differing fields: " + "; ".join(clash)
            )

        def _sort_key(rec: Record) -> tuple:
            h = rec.header
            return (
                h.time_tag_year or 0,
                h.time_tag_doy or 0,
                h.time_tag_second_of_day or 0,
                h.time_tag_picoseconds_of_the_second or 0.0,
            )

        for label, recs in (("self", self.records), ("other", other.records)):
            n_dup = len(recs) - len({_sort_key(r) for r in recs})
            if n_dup:
                tail = "." if keep == "both" else "; keeping the last of each."
                warnings.warn(
                    f"{n_dup} duplicate time tag(s) within {label}{tail}",
                    RuntimeWarning,
                    stacklevel=2,
                )

        if keep == "both":
            merged = list(self.records) + list(other.records)
            n_dup = len(merged) - len({_sort_key(r) for r in merged})
            if n_dup:
                warnings.warn(
                    f"{n_dup} duplicate time tag(s) detected; "
                    f"keeping all records (keep='both').",
                    RuntimeWarning,
                    stacklevel=2,
                )
        else:
            self_by_key = {_sort_key(r): r for r in self.records}
            other_by_key = {_sort_key(r): r for r in other.records}
            duplicates = self_by_key.keys() & other_by_key.keys()
            if duplicates:
                warnings.warn(
                    f"{len(duplicates)} duplicate time tag(s) detected; "
                    f"keeping records from '{keep}'.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            if keep == "other":  # other overwrites self on collision
                merged = list({**self_by_key, **other_by_key}.values())
            else:  # self overwrites other on collision
                merged = list({**other_by_key, **self_by_key}.values())

        if sort:
            merged.sort(key=_sort_key)
        return self._new_from(merged, path=self.path if self.path == other.path else None)

    def _new_from(self, records: list[Record], path: Path | None = ...) -> ProductFile:
        """New product file named after self, holding independent copies of *records*."""
        pf = ProductFile(name=self.name)
        pf.path = self.path if path is ... else path
        # deepcopy copies headers and disk locations; samples only if loaded.
        pf.records = copy.deepcopy(records)
        pf._check_constant_fields()
        return pf

    # ---- slicing ------------------------------------------------------

    def slice(self, start: int, stop: int | None = None) -> ProductFile:
        """
        Records by index, as a new product file.

        ``slice(i)`` is the one record at *i* (``-1`` is the last);
        ``slice(i, j)`` is records ``i`` up to, not including, ``j``, with
        list-slice semantics.  To select by time, use :meth:`select`.

        Parameters
        ----------
        start : int
            Index of the first record.
        stop : int or None
            Index one past the last record; ``None`` takes only *start*.

        Returns
        -------
        ProductFile
            New product file with independent copies of the selected
            records (samples copied only where loaded); the copies still
            read unloaded samples from the original file.

        Raises
        ------
        IndexError
            If *stop* is None and *start* is out of range.
        TypeError
            If *start* or *stop* is not an integer.
        """
        start = operator.index(start)
        if stop is None:  # the one record at *start*; -1 is the last
            return self._new_from([self.records[start]])
        return self._new_from(self.records[start : operator.index(stop)])

    def select(
        self,
        *,
        filters: dict | None = None,
        min_downconv_hz: float | None = None,
        max_downconv_hz: float | None = None,
        start_epoch: datetime | np.datetime64 | str | None = None,
        stop_epoch: datetime | np.datetime64 | str | None = None,
    ) -> ProductFile:
        """
        Records matching all the given criteria, as a new product file.

        The original is untouched.  Same criteria, with the same meaning,
        as the filters of :func:`read_product`.

        Parameters
        ----------
        filters : dict, optional
            Header-field exact-match filters.  Keys are RecordHeader
            field names (e.g. ``"station_id"``); values are the
            expected values.  Records must match **all** filters.
        min_downconv_hz : float, optional
            Minimum total downconversion frequency in Hz.  Evaluated
            at the first sample of each record (t = 0) as
            ``rf_to_if_downconv + if_to_channel_downconv + c1``.
        max_downconv_hz : float, optional
            Maximum total downconversion frequency in Hz.
        start_epoch, stop_epoch : datetime, np.datetime64, str, or None, optional
            Keep records with ``start_epoch <= timetag < stop_epoch``
            (half-open, so consecutive windows tile without overlap or
            gap), compared on the full time tag, picoseconds included.  A
            naive ``datetime`` is UTC.

        Returns
        -------
        ProductFile
            New product file with independent copies of the kept records
            (samples copied only where loaded); the copies still read
            unloaded samples from the original file.

        Raises
        ------
        ValueError
            If a filter key is not a valid RecordHeader field name.

        Examples
        --------
        Records from station 24::

            >>> pf = read_product("scan.prd")
            >>> sub = pf.select(filters={"station_id": 24})

        A five-minute window within a downconversion-frequency band::

            >>> sub = pf.select(
            ...     start_epoch="2026-039T12:00:00",
            ...     stop_epoch="2026-039T12:05:00",
            ...     min_downconv_hz=8.4e9,
            ...     max_downconv_hz=8.5e9,
            ... )
        """
        if filters:
            valid_fields = set(RecordHeader.__slots__)
            for field in filters:
                if field not in valid_fields:
                    raise ValueError(
                        f"Unknown header field in filters: {field!r}. "
                        f"Valid fields: {sorted(valid_fields)}"
                    )

        t0 = _to_datetime64(start_epoch)
        t1 = _to_datetime64(stop_epoch)

        kept: list[Record] = []
        for rec in self.records:
            hdr = rec.header

            # Header field filtering
            if filters and any(getattr(hdr, f) != v for f, v in filters.items()):
                continue

            # Downconversion frequency filtering (at t=0)
            if min_downconv_hz is not None or max_downconv_hz is not None:
                f_dc = _downconv_hz(hdr, 0.0)
                if min_downconv_hz is not None and f_dc < min_downconv_hz:
                    continue
                if max_downconv_hz is not None and f_dc > max_downconv_hz:
                    continue

            # Time-window filtering
            if t0 is not None or t1 is not None:
                epoch = _record_epoch(hdr)
                if epoch is None:
                    continue
                if t0 is not None and epoch < t0:
                    continue
                if t1 is not None and epoch >= t1:
                    continue

            kept.append(rec)

        return self._new_from(kept)

    # ---- timetag ------------------------------------------------------

    @property
    def timetag_array(self) -> np.ndarray:
        """
        Per-record start epochs (one entry per record, header-only).

        Returns the time tag of each record -- its UTC start epoch -- as a
        ``numpy.datetime64[ns]`` array of length ``len(self.records)``.
        Header-only: no I/Q is decoded, so this stays cheap on a streamed or
        multi-gigabyte product file.  For per-sample times *within* one
        record, build them from ``Record.header.timetag`` and the record's
        sample rate.

        Returns
        -------
        np.ndarray
            ``datetime64[ns]`` record start epochs, one per record.
        """
        return np.array(
            [r.header.timetag for r in self.records], dtype="datetime64[ns]"
        )

    # ---- downconversion frequency -----------------------------------------

    @property
    def downconversion_frequency(self) -> np.ndarray:
        """
        Per-record downconversion frequency at each record's start (header-only).

        One value per record: the total downconversion frequency evaluated
        at the record start (t = 0),

            f = rf_to_if_downconv + if_to_channel_downconv + c1   [Hz]

        (§5.2.6), where ``c1`` is the linear phase-polynomial coefficient.
        Header-only -- no I/Q decode -- so it is cheap on a streamed or large
        product file.  For the full per-sample frequency *within* one record
        (including c2/c3 curvature) use ``Record.downconversion_frequency``.

        Returns
        -------
        np.ndarray
            Downconversion frequency in Hz, one per record (empty if none).

        References
        ----------
        CCSDS 506.1-B-1 (June 2013), §5.2.6: "Downconversion Reconstruction"
        """
        return np.array(
            [_downconv_hz(r.header, 0.0) for r in self.records], dtype=np.float64
        )

    # ---- downconversion phase ---------------------------------------------

    @property
    def downconversion_phase(self) -> np.ndarray:
        """
        Per-record downconversion phase at each record's start (header-only).

        One value per record: the downconversion phase at the record start
        (t = 0),

            φ = Φ + c0   [turns]

        (§5.2.6), where ``Φ`` is the channel accumulated phase and ``c0`` the
        constant phase-polynomial coefficient.  Header-only -- no I/Q decode.

        Returns
        -------
        np.ndarray
            Downconversion phase in turns (cycles), one per record (empty if
            none).

        References
        ----------
        CCSDS 506.1-B-1 (June 2013), §5.2.6
        """
        return np.array(
            [_downconv_phase(r.header, 0.0) for r in self.records], dtype=np.float64
        )

    # ---- samples: load / release -----------------------------------------

    def load(self, *, progress: bool = False) -> None:
        """
        Read every record's samples into memory and keep them.

        One sequential pass with a single open handle per file.  Records
        stay loaded until ``release()``.  Memory is the decoded samples of
        every record -- several times the file size -- so for long high-rate
        files prefer ``record.data`` one record at a time.

        Parameters
        ----------
        progress : bool
            Draw a progress bar on stderr (default False).
        """
        _load_records(self.records, progress=progress)

    def release(self) -> None:
        """Drop every record's loaded samples; access reads them from disk again."""
        for rec in self.records:
            rec.release()

    # ---- copy ---------------------------------------------------------

    def copy(self) -> ProductFile:
        """Return a deep copy of the product file."""
        pf = ProductFile()
        pf.records = copy.deepcopy(self.records)
        pf.name = self.name
        pf.path = self.path
        return pf

    # ---- text dump --------------------------------------------------------

    def dump_headers(self, dest: str | Path | TextIO | None = None) -> None:
        """
        Dump all record headers as human-readable text.

        Writes one block per record containing all scalar header fields
        (reserved byte fields are shown as length summaries).

        Parameters
        ----------
        dest : str, Path, text stream, or None
            ``None`` (default) prints to the terminal (``sys.stdout``); a
            path writes a new UTF-8 text file; an open text stream is
            written to and left open.

        Examples
        --------
        >>> pf.dump_headers()                 # terminal
        >>> pf.dump_headers("headers.txt")    # file
        """
        _display.dump_headers(self, dest)

    def dump(self, dest: str | Path | TextIO | None = None) -> None:
        """
        Dump the full product file (headers + data) as text.

        Writes header and I/Q data blocks interleaved in record order:
        header, data, header, data, etc.  I/Q samples are written as
        comma-separated ``I, Q`` pairs, one per line.

        Parameters
        ----------
        dest : str, Path, text stream, or None
            ``None`` (default) prints to the terminal (``sys.stdout``); a
            path writes a new UTF-8 text file; an open text stream is
            written to and left open.

        Notes
        -----
        One line per sample, so a whole file can run to millions of lines;
        to the terminal, dump a slice.

        Examples
        --------
        >>> pf.slice(0, 1).dump()             # first record, terminal
        >>> pf.dump("full_dump.txt")          # whole file, to a file
        """
        _display.dump(self, dest)
