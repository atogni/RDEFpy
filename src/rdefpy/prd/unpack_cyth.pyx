# cython: language_level=3
# cython: boundscheck=False, wraparound=False, nonecheck=False, initializedcheck=False, cdivision=True
# Copyright (c) 2026 Morehead State University
# Author: Andrea Togni
# SPDX-License-Identifier: MIT

"""
Cython-optimized I/Q sample unpacking and carrier-estimation kernels for RDEF files.

This module provides high-performance implementations for unpacking I/Q samples
from packed binary formats per CCSDS 506.1-B-1 Table 5-2.  All unpack functions
apply bias correction per §5.3.6:
output = 2*k + 1, where k is the two's complement interpretation of the raw bits.

Functions
---------
unpack_iq_cython      : Decode one data section (dispatches to the kernels)
unpack_iq_16_cython   : 16-bit samples (one 32-bit word per sample)
unpack_iq_8_cython    : 8-bit samples (one 16-bit halfword per sample)
unpack_iq_4_cython    : 4-bit samples (byte-table gather)
unpack_iq_2_cython    : 2-bit samples (byte-table gather)
unpack_iq_1_cython    : 1-bit samples (byte-table gather)
iq_to_complex         : I + jQ in one interleaving pass
unpack_complex_cython : Decode one data section straight into a complex array
spi_refine, frft_refine, frft_bank_fill, argmax_abs2 : carrier kernels for
                        ``rdefpy.carrier`` (NumPy reference there)

Notes
-----
Every unpacker must return arrays identical to its NumPy counterpart in
``_binary.py`` (the reference); ``tests/test_spec_prd_samples.py`` runs the
whole spec suite against both.  The sub-byte kernels copy their tables from
``_binary._BYTE_TABLES``, so the two paths share a single definition of the
Table 5-2 layout.  All kernels run without the GIL; large records are
split across threads (``rdefpy._utils._in_chunks``).
"""

import cython
import numpy as np
cimport numpy as cnp
from libc.math cimport M_PI, cos, floor, hypot, log, sin, fabs
from libc.stdint cimport int8_t, int16_t, int32_t, uint8_t, uint16_t, uint32_t
from libc.stdlib cimport free, malloc

from rdefpy._utils import _in_chunks
from rdefpy.prd._binary import _BYTE_TABLES

cnp.import_array()


# =============================================================================
# Sub-byte decode tables (copied from _binary._BYTE_TABLES at import)
# =============================================================================
# Row b of a table holds the 1, 2 or 4 int8 sample values of byte b, packed
# into one 8-, 16- or 32-bit entry.  A decode stores that entry straight into
# the int8 output, so the in-memory byte order is the table's own.  C arrays,
# read-only after import: safe for concurrent nogil decoding.

cdef uint8_t _T4I[256]
cdef uint8_t _T4Q[256]
cdef uint16_t _T2I[256]
cdef uint16_t _T2Q[256]
cdef uint32_t _T1I[256]
cdef uint32_t _T1Q[256]


def _load_tables():
    t4i, t4q = (t.view(np.uint8) for t in _BYTE_TABLES[4])
    t2i, t2q = (t.view(np.uint16) for t in _BYTE_TABLES[2])
    t1i, t1q = (t.view(np.uint32) for t in _BYTE_TABLES[1])
    cdef int b
    for b in range(256):
        _T4I[b] = t4i[b]
        _T4Q[b] = t4q[b]
        _T2I[b] = t2i[b]
        _T2Q[b] = t2q[b]
        _T1I[b] = t1i[b]
        _T1Q[b] = t1q[b]


_load_tables()


# The same tables as one int8 value per sample, for the fused decode to
# complex: _SI[t][b][s] is I of sample s in byte b, t = 0, 1, 2 for 4, 2,
# 1 bits (1, 2, 4 samples per byte; unused slots stay 0).
cdef int8_t _SI[3][256][4]
cdef int8_t _SQ[3][256][4]


def _load_sample_tables():
    cdef int t, b, s
    for t, bits in enumerate((4, 2, 1)):
        n = 8 // (2 * bits)
        ti, tq = (x.view(np.int8).reshape(256, n) for x in _BYTE_TABLES[bits])
        for b in range(256):
            for s in range(n):
                _SI[t][b][s] = ti[b, s]
                _SQ[t][b][s] = tq[b, s]


_load_sample_tables()


# Bounds checks are off in the kernels, so each one checks its outputs here:
# a direct call with a short output would write past it.
cdef int _check_out(Py_ssize_t need, Py_ssize_t ni, Py_ssize_t nq) except -1:
    if ni != need or nq != need:
        raise ValueError(f"output lengths {ni} and {nq}, need {need} each")
    return 0


# =============================================================================
# 16-bit samples: one little-endian 32-bit word per complex sample
# =============================================================================

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void unpack_iq_16_cython(
    const cnp.uint32_t[::1] words, int32_t[::1] i_view, int32_t[::1] q_view
):
    """
    Unpack 16-bit I/Q samples with bias correction (Table 5-2: MSB Q1 I1 LSB).

    Parameters
    ----------
    words : uint32[::1]
        The data section as little-endian 32-bit words, one per sample.
    i_view, q_view : int32[::1]
        Outputs, one per word: ``2k+1`` values in [-65535, 65535].
    """
    cdef Py_ssize_t n = words.shape[0]
    cdef Py_ssize_t k
    cdef uint32_t w
    _check_out(n, i_view.shape[0], q_view.shape[0])
    with nogil:
        for k in range(n):
            w = words[k]
            i_view[k] = 2 * <int32_t><int16_t>(w & 0xFFFF) + 1
            q_view[k] = 2 * <int32_t><int16_t>(w >> 16) + 1


# =============================================================================
# 8-bit samples: one little-endian 16-bit halfword per complex sample
# =============================================================================

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void unpack_iq_8_cython(
    const cnp.uint16_t[::1] halves, int16_t[::1] i_view, int16_t[::1] q_view
):
    """
    Unpack 8-bit I/Q samples with bias correction (Table 5-2: I in the low
    byte of each halfword, Q in the high byte).

    Parameters
    ----------
    halves : uint16[::1]
        The data section as little-endian 16-bit halfwords, one per sample.
    i_view, q_view : int16[::1]
        Outputs, one per halfword: ``2k+1`` values in [-255, 255].
    """
    cdef Py_ssize_t n = halves.shape[0]
    cdef Py_ssize_t k
    cdef uint16_t w
    _check_out(n, i_view.shape[0], q_view.shape[0])
    with nogil:
        for k in range(n):
            w = halves[k]
            i_view[k] = <int16_t>(2 * <int>(<int8_t>(w & 0xFF)) + 1)
            q_view[k] = <int16_t>(2 * <int>(<int8_t>(w >> 8)) + 1)


# =============================================================================
# 4-, 2-, 1-bit samples: one table gather per byte and lane
# =============================================================================

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void unpack_iq_4_cython(
    const cnp.uint8_t[::1] data, int8_t[::1] i_view, int8_t[::1] q_view
):
    """Unpack 4-bit I/Q samples (one complex sample per byte) into i/q_view."""
    cdef Py_ssize_t n = data.shape[0]
    cdef Py_ssize_t k
    _check_out(n, i_view.shape[0], q_view.shape[0])
    if n == 0:
        return
    cdef uint8_t* ip = <uint8_t*> &i_view[0]
    cdef uint8_t* qp = <uint8_t*> &q_view[0]
    with nogil:
        for k in range(n):
            ip[k] = _T4I[data[k]]
            qp[k] = _T4Q[data[k]]


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void unpack_iq_2_cython(
    const cnp.uint8_t[::1] data, int8_t[::1] i_view, int8_t[::1] q_view
):
    """Unpack 2-bit I/Q samples (two complex samples per byte) into i/q_view."""
    cdef Py_ssize_t n = data.shape[0]
    cdef Py_ssize_t k
    _check_out(2 * n, i_view.shape[0], q_view.shape[0])
    if n == 0:
        return
    cdef uint16_t* ip = <uint16_t*> &i_view[0]
    cdef uint16_t* qp = <uint16_t*> &q_view[0]
    with nogil:
        for k in range(n):
            ip[k] = _T2I[data[k]]
            qp[k] = _T2Q[data[k]]


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void unpack_iq_1_cython(
    const cnp.uint8_t[::1] data, int8_t[::1] i_view, int8_t[::1] q_view
):
    """Unpack 1-bit I/Q samples (four complex samples per byte) into i/q_view."""
    cdef Py_ssize_t n = data.shape[0]
    cdef Py_ssize_t k
    _check_out(4 * n, i_view.shape[0], q_view.shape[0])
    if n == 0:
        return
    cdef uint32_t* ip = <uint32_t*> &i_view[0]
    cdef uint32_t* qp = <uint32_t*> &q_view[0]
    with nogil:
        for k in range(n):
            ip[k] = _T1I[data[k]]
            qp[k] = _T1Q[data[k]]


# =============================================================================
# I/Q -> complex: one interleaving pass
# =============================================================================

ctypedef fused iq_t:
    int8_t
    int16_t
    int32_t

ctypedef fused re_t:
    float
    double


@cython.boundscheck(False)
@cython.wraparound(False)
def iq_to_complex(const iq_t[::1] i, const iq_t[::1] q, re_t[::1] out):
    """
    Fill *out*, a complex array viewed as its float parts, with ``I + jQ``.

    Writes ``out[2k] = i[k]`` and ``out[2k+1] = q[k]`` in one sequential
    pass: the same values ``x.real = i; x.imag = q`` stores (integers are
    exact in float32 and float64), with each cache line written once.
    Used on thread chunks of large records, where it beats NumPy's two
    strided stores; on one thread page faults dominate and it does not.
    """
    cdef Py_ssize_t n = i.shape[0]
    cdef Py_ssize_t k
    if q.shape[0] != n or out.shape[0] != 2 * n:
        raise ValueError("iq_to_complex: length mismatch")
    with nogil:
        for k in range(n):
            out[2 * k] = <re_t>i[k]
            out[2 * k + 1] = <re_t>q[k]


# =============================================================================
# Fused decode: packed bytes -> complex, no int I/Q arrays
# =============================================================================
# Each kernel writes I + jQ into *out*, a complex array viewed as its float
# parts.  The values are those of the int kernels above, cast to float: the
# 16- and 8-bit expressions are copied verbatim, the sub-byte ones read the
# same _BYTE_TABLES.  Every 2k+1 value is an integer below 2**24, exact in
# float32 and float64, so the result equals unpack-then-iq_to_complex bit for
# bit.

@cython.boundscheck(False)
@cython.wraparound(False)
def _complex_16(const cnp.uint32_t[::1] words, re_t[::1] out):
    cdef Py_ssize_t n = words.shape[0]
    cdef Py_ssize_t k
    cdef uint32_t w
    _check_out(2 * n, out.shape[0], out.shape[0])
    with nogil:
        for k in range(n):
            w = words[k]
            out[2 * k] = <re_t>(2 * <int32_t><int16_t>(w & 0xFFFF) + 1)
            out[2 * k + 1] = <re_t>(2 * <int32_t><int16_t>(w >> 16) + 1)


@cython.boundscheck(False)
@cython.wraparound(False)
def _complex_8(const cnp.uint16_t[::1] halves, re_t[::1] out):
    cdef Py_ssize_t n = halves.shape[0]
    cdef Py_ssize_t k
    cdef uint16_t w
    _check_out(2 * n, out.shape[0], out.shape[0])
    with nogil:
        for k in range(n):
            w = halves[k]
            out[2 * k] = <re_t>(<int16_t>(2 * <int>(<int8_t>(w & 0xFF)) + 1))
            out[2 * k + 1] = <re_t>(<int16_t>(2 * <int>(<int8_t>(w >> 8)) + 1))


@cython.boundscheck(False)
@cython.wraparound(False)
def _complex_sub(const cnp.uint8_t[::1] data, re_t[::1] out, int t, int per):
    """4-, 2-, 1-bit (t = 0, 1, 2): *per* complex samples per byte."""
    cdef Py_ssize_t n = data.shape[0]
    cdef Py_ssize_t k, m
    cdef int s
    cdef uint8_t b
    if not 0 <= t <= 2 or per != 1 << t:
        raise ValueError(
            f"_complex_sub: table {t} holds {1 << t if 0 <= t <= 2 else '?'} "
            f"samples per byte, got per={per}"
        )
    _check_out(2 * n * per, out.shape[0], out.shape[0])
    with nogil:
        for k in range(n):
            b = data[k]
            m = 2 * k * per
            for s in range(per):
                out[m + 2 * s] = <re_t>_SI[t][b][s]
                out[m + 2 * s + 1] = <re_t>_SQ[t][b][s]


_TABLE = {4: 0, 2: 1, 1: 2}


def unpack_complex_cython(data, int sample_size, out) -> None:
    """
    Decode one data section straight into *out* (complex64 or complex128).

    *out* must be 1-D, C-contiguous, writeable and hold exactly the
    section's complex samples; the caller checks that, and each kernel
    re-checks the length before writing.  Same values as
    ``unpack_iq_cython`` followed by ``I + jQ``.
    """
    if sample_size not in _VALID_SAMPLE_SIZES:
        raise ValueError(f"Unsupported sample_size: {sample_size}")
    count = len(data) // _UNIT[sample_size]
    src = np.frombuffer(data, dtype=_INPUT_DTYPE[sample_size], count=count)
    per = _OUTPUT[sample_size][1]
    if out.shape != (count * per,):
        raise ValueError(f"out has shape {out.shape}, need ({count * per},)")
    parts = out.view(out.real.dtype)
    if sample_size == 16:
        fill = lambda a, b: _complex_16(src[a:b], parts[2 * a : 2 * b])
    elif sample_size == 8:
        fill = lambda a, b: _complex_8(src[a:b], parts[2 * a : 2 * b])
    else:
        t = _TABLE[sample_size]
        fill = lambda a, b: _complex_sub(src[a:b], parts[2 * a * per : 2 * b * per], t, per)
    # Chunks fall on whole input elements; large records use threads.
    _in_chunks(count, per, fill)


# =============================================================================
# Dispatcher
# =============================================================================

_VALID_SAMPLE_SIZES = frozenset([1, 2, 4, 8, 16])

# Bytes per kernel input element; a trailing partial element is ignored,
# as in the NumPy reference.
_UNIT = {16: 4, 8: 2, 4: 1, 2: 1, 1: 1}
_INPUT_DTYPE = {16: "<u4", 8: "<u2", 4: np.uint8, 2: np.uint8, 1: np.uint8}
# Output dtype and complex samples per input element.
_OUTPUT = {16: (np.int32, 1), 8: (np.int16, 1), 4: (np.int8, 1), 2: (np.int8, 2), 1: (np.int8, 4)}
_KERNEL = {
    16: unpack_iq_16_cython,
    8: unpack_iq_8_cython,
    4: unpack_iq_4_cython,
    2: unpack_iq_2_cython,
    1: unpack_iq_1_cython,
}


def unpack_iq_cython(data, int sample_size) -> tuple:
    """
    Dispatch to the Cython I/Q unpacker for the given sample size.

    Parameters
    ----------
    data : bytes-like
        Raw binary data from one RDEF record's data section.
    sample_size : int
        Bits per sample component (1, 2, 4, 8, or 16).

    Returns
    -------
    i, q : np.ndarray
        In-phase and quadrature samples, bias-corrected.

    Raises
    ------
    ValueError
        If sample_size is not one of {1, 2, 4, 8, 16}.
    """
    if sample_size not in _VALID_SAMPLE_SIZES:
        raise ValueError(f"Unsupported sample_size: {sample_size}")
    count = len(data) // _UNIT[sample_size]
    src = np.frombuffer(data, dtype=_INPUT_DTYPE[sample_size], count=count)
    dtype, per = _OUTPUT[sample_size]
    i_arr = np.empty(count * per, dtype=dtype)
    q_arr = np.empty(count * per, dtype=dtype)
    kernel = _KERNEL[sample_size]
    # Chunks fall on whole input elements; large records use threads.
    _in_chunks(count, per, lambda a, b: kernel(
        src[a:b], i_arr[a * per : b * per], q_arr[a * per : b * per]
    ))
    return i_arr, q_arr


# =============================================================================
# Carrier estimation kernels (rdefpy.carrier): FrFT bank fill and argmax,
# and the SPI / log-SPI refinement, all without the GIL.
# =============================================================================
# Phasors run by recursion (two complex products per sample) and restart
# from exact sin/cos every _RESYNC samples, which bounds the rounding drift.
# Arithmetic is in explicit real/imaginary doubles: MSVC has no C99 complex.

cdef enum:
    _RESYNC = 1024

ctypedef fused bank_t:
    float complex
    double complex


cdef double _chirp_dtft_mag(const double complex* x, Py_ssize_t n, double mu, double f) noexcept nogil:
    """|sum_k x_k exp(-j pi mu (k-c)^2 - 2j pi f k)|, c = (n-1)/2 [cycles/sample, cycles/sample^2]."""
    cdef double c = (n - 1) / 2.0
    cdef double ar = 0.0, ai = 0.0, pr, pi_, qr, qi, wr, wi, tr, u, fk, ph
    cdef double xr, xi
    cdef Py_ssize_t k, k0 = 0, k1
    wr = cos(-2.0 * M_PI * mu)
    wi = sin(-2.0 * M_PI * mu)
    while k0 < n:
        k1 = k0 + _RESYNC
        if k1 > n:
            k1 = n
        u = k0 - c
        fk = f * k0
        fk -= floor(fk)  # f k0 reduced to [0, 1) cycles
        ph = -M_PI * mu * u * u - 2.0 * M_PI * fk
        pr = cos(ph)
        pi_ = sin(ph)
        ph = -M_PI * mu * (2.0 * u + 1.0) - 2.0 * M_PI * f  # phase step k0 -> k0 + 1
        qr = cos(ph)
        qi = sin(ph)
        for k in range(k0, k1):
            xr = x[k].real
            xi = x[k].imag
            ar += xr * pr - xi * pi_
            ai += xr * pi_ + xi * pr
            tr = pr * qr - pi_ * qi  # p *= q
            pi_ = pr * qi + pi_ * qr
            pr = tr
            tr = qr * wr - qi * wi  # q *= w: the step grows by -2 pi mu per sample
            qi = qr * wi + qi * wr
            qr = tr
        k0 = k1
    return hypot(ar, ai)


cdef double _spi(const double complex* x, Py_ssize_t n, double mu, double f, double m,
                 int max_iterations, double threshold) noexcept nogil:
    """SPI of |DTFT| of the block dechirped at mu, from seed f [cycles/sample]; m = padded FFT length."""
    cdef int i
    cdef double delta, f_prev, x1, x2, x3, denom
    for i in range(1, max_iterations + 1):
        f_prev = f
        delta = 1.0 / ((1 << i) * m)
        x1 = _chirp_dtft_mag(x, n, mu, f - delta)
        x2 = _chirp_dtft_mag(x, n, mu, f)
        x3 = _chirp_dtft_mag(x, n, mu, f + delta)
        if x2 < x3:  # re-centre so x2 is the largest of the three
            f += delta
            x1 = x2
            x2 = x3
            x3 = _chirp_dtft_mag(x, n, mu, f + delta)
        elif x2 < x1:
            f -= delta
            x3 = x2
            x2 = x1
            x1 = _chirp_dtft_mag(x, n, mu, f - delta)
        denom = x1 + x3 - 2.0 * x2
        if denom < 0.0:
            f -= 0.5 * delta * (x3 - x1) / denom
        if fabs(f - f_prev) < threshold:
            break
    return f


cdef inline double _log_power(const double complex* x, Py_ssize_t n, double mu, double f) noexcept nogil:
    return 2.0 * log(_chirp_dtft_mag(x, n, mu, f))


def spi_refine(const double complex[::1] x, double mu, double f, double m,
               int max_iterations=5, double threshold=1e-12):
    """SPI from seed *f* on the block dechirped at *mu*; returns f [cycles/sample]."""
    cdef Py_ssize_t n = x.shape[0]
    if n == 0:
        raise ValueError("spi_refine: empty x")
    with nogil:
        f = _spi(&x[0], n, mu, f, m, max_iterations, threshold)
    return f


def frft_refine(const double complex[::1] x, double mu, double f, double m, double step,
                int rate_iterations, int max_iterations=5, double threshold=1e-12):
    """Steps 2-3 of frft_frequency from the bank's (mu, f): returns (f_centre, mu).

    SPI at the coarse rate (seeded by the bank bin), log-SPI on the rate at
    that frequency, then SPI at the final rate (seeded by the padded-FFT bin
    nearest the first SPI).
    """
    cdef Py_ssize_t n = x.shape[0]
    cdef const double complex* p
    cdef int it
    cdef double nu0, h, y1, y2, y3, denom
    if n == 0:
        raise ValueError("frft_refine: empty x")
    p = &x[0]
    with nogil:
        nu0 = _spi(p, n, mu, f, m, max_iterations, threshold)
        for it in range(1, rate_iterations + 1):
            h = step / (1 << it)
            y1 = _log_power(p, n, mu - h, nu0)
            y2 = _log_power(p, n, mu, nu0)
            y3 = _log_power(p, n, mu + h, nu0)
            if y3 > y2:
                mu += h
                y1 = y2
                y2 = y3
                y3 = _log_power(p, n, mu + h, nu0)
            elif y1 > y2:
                mu -= h
                y3 = y2
                y2 = y1
                y1 = _log_power(p, n, mu - h, nu0)
            denom = y1 + y3 - 2.0 * y2
            if denom < 0:
                mu -= 0.5 * h * (y3 - y1) / denom
        # From the padded-FFT bin nearest nu0 (the NumPy path starts from a
        # new padded-FFT peak at the final rate): started on nu0 itself, the
        # first, widest step can fall under the threshold.
        f = _spi(p, n, mu, floor(nu0 * m + 0.5) / m, m, max_iterations, threshold)
    return f, mu


def frft_bank_fill(const float complex[::1] x, double rate0, double step, float complex[:, ::1] out):
    """out[i, k] = x[k] exp(-j pi (rate0 + i step) (k - c)^2), c = (n-1)/2."""
    cdef Py_ssize_t rows = out.shape[0], n = out.shape[1], i, k
    cdef double c = (n - 1) / 2.0, u, br, bi, sr, si, tr
    cdef double* st
    if x.shape[0] != n:
        raise ValueError("x and out rows differ in length")
    st = <double*> malloc(4 * n * sizeof(double))  # per sample: base phasor, step phasor
    if st == NULL:
        raise MemoryError()
    with nogil:
        for k in range(n):
            u = (k - c) * (k - c)
            st[4 * k] = cos(-M_PI * rate0 * u)
            st[4 * k + 1] = sin(-M_PI * rate0 * u)
            st[4 * k + 2] = cos(-M_PI * step * u)
            st[4 * k + 3] = sin(-M_PI * step * u)
        for i in range(rows):
            for k in range(n):
                br = st[4 * k]
                bi = st[4 * k + 1]
                out[i, k] = <float complex> ((x[k].real * br - x[k].imag * bi) + (x[k].real * bi + x[k].imag * br) * 1j)
                sr = st[4 * k + 2]
                si = st[4 * k + 3]
                tr = br * sr - bi * si
                st[4 * k + 1] = br * si + bi * sr
                st[4 * k] = tr
    free(st)


def argmax_abs2(bank_t[:, ::1] a):
    """(row, col, |a|^2) of the largest |a|^2, accumulated in double."""
    cdef Py_ssize_t rows = a.shape[0], cols = a.shape[1], i, k, bi_ = 0, bk = 0
    cdef double best = -1.0, v, re, im
    with nogil:
        for i in range(rows):
            for k in range(cols):
                re = a[i, k].real
                im = a[i, k].imag
                v = re * re + im * im
                if v > best:
                    best = v
                    bi_ = i
                    bk = k
    return bi_, bk, best
