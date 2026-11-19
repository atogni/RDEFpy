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

# Calls on at most this many samples keep the GIL: they take a few
# microseconds, less than handing the GIL to another thread and waiting to
# get it back.
cdef enum:
    _KEEP_GIL = 16384

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
    Also used on records of 501 to _KEEP_GIL samples, keeping the GIL.
    """
    cdef Py_ssize_t n = i.shape[0]
    cdef Py_ssize_t k
    if q.shape[0] != n or out.shape[0] != 2 * n:
        raise ValueError("iq_to_complex: length mismatch")
    if n <= _KEEP_GIL:  # a few microseconds: see _KEEP_GIL
        for k in range(n):
            out[2 * k] = <re_t>i[k]
            out[2 * k + 1] = <re_t>q[k]
        return
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


cdef void _chirp_dtft_mag3(const double complex* x, Py_ssize_t n, double mu0, double f0, double mu1, double f1,
                           double mu2, double f2, double* mag) noexcept nogil:
    """mag[j] = _chirp_dtft_mag(x, n, mu_j, f_j), j = 0, 1, 2, in one pass over x.

    Each point takes exactly the operations of _chirp_dtft_mag, in the same
    order, so the three values are the same bit for bit.  The phasor recursion
    is a serial dependency chain; three independent chains overlap in the CPU.
    """
    cdef double c = (n - 1) / 2.0
    cdef double u, fk, ph, tr, xr, xi
    cdef double ar0 = 0.0, ai0 = 0.0, pr0, pi0, qr0, qi0, wr0, wi0
    cdef double ar1 = 0.0, ai1 = 0.0, pr1, pi1, qr1, qi1, wr1, wi1
    cdef double ar2 = 0.0, ai2 = 0.0, pr2, pi2, qr2, qi2, wr2, wi2
    cdef Py_ssize_t k, k0 = 0, k1
    wr0 = cos(-2.0 * M_PI * mu0)
    wi0 = sin(-2.0 * M_PI * mu0)
    wr1 = cos(-2.0 * M_PI * mu1)
    wi1 = sin(-2.0 * M_PI * mu1)
    wr2 = cos(-2.0 * M_PI * mu2)
    wi2 = sin(-2.0 * M_PI * mu2)
    while k0 < n:
        k1 = k0 + _RESYNC
        if k1 > n:
            k1 = n
        u = k0 - c
        fk = f0 * k0
        fk -= floor(fk)
        ph = -M_PI * mu0 * u * u - 2.0 * M_PI * fk
        pr0 = cos(ph)
        pi0 = sin(ph)
        ph = -M_PI * mu0 * (2.0 * u + 1.0) - 2.0 * M_PI * f0
        qr0 = cos(ph)
        qi0 = sin(ph)
        fk = f1 * k0
        fk -= floor(fk)
        ph = -M_PI * mu1 * u * u - 2.0 * M_PI * fk
        pr1 = cos(ph)
        pi1 = sin(ph)
        ph = -M_PI * mu1 * (2.0 * u + 1.0) - 2.0 * M_PI * f1
        qr1 = cos(ph)
        qi1 = sin(ph)
        fk = f2 * k0
        fk -= floor(fk)
        ph = -M_PI * mu2 * u * u - 2.0 * M_PI * fk
        pr2 = cos(ph)
        pi2 = sin(ph)
        ph = -M_PI * mu2 * (2.0 * u + 1.0) - 2.0 * M_PI * f2
        qr2 = cos(ph)
        qi2 = sin(ph)
        for k in range(k0, k1):
            xr = x[k].real
            xi = x[k].imag
            ar0 += xr * pr0 - xi * pi0
            ai0 += xr * pi0 + xi * pr0
            tr = pr0 * qr0 - pi0 * qi0
            pi0 = pr0 * qi0 + pi0 * qr0
            pr0 = tr
            tr = qr0 * wr0 - qi0 * wi0
            qi0 = qr0 * wi0 + qi0 * wr0
            qr0 = tr
            ar1 += xr * pr1 - xi * pi1
            ai1 += xr * pi1 + xi * pr1
            tr = pr1 * qr1 - pi1 * qi1
            pi1 = pr1 * qi1 + pi1 * qr1
            pr1 = tr
            tr = qr1 * wr1 - qi1 * wi1
            qi1 = qr1 * wi1 + qi1 * wr1
            qr1 = tr
            ar2 += xr * pr2 - xi * pi2
            ai2 += xr * pi2 + xi * pr2
            tr = pr2 * qr2 - pi2 * qi2
            pi2 = pr2 * qi2 + pi2 * qr2
            pr2 = tr
            tr = qr2 * wr2 - qi2 * wi2
            qi2 = qr2 * wi2 + qi2 * wr2
            qr2 = tr
        k0 = k1
    mag[0] = hypot(ar0, ai0)
    mag[1] = hypot(ar1, ai1)
    mag[2] = hypot(ar2, ai2)


cdef double _spi(const double complex* x, Py_ssize_t n, double mu, double f, double m,
                 int max_iterations, double threshold) noexcept nogil:
    """SPI of |DTFT| of the block dechirped at mu, from seed f [cycles/sample]; m = padded FFT length."""
    cdef int i
    cdef double delta, f_prev, x1, x2, x3, denom
    cdef double mag[3]
    for i in range(1, max_iterations + 1):
        f_prev = f
        delta = 1.0 / ((1 << i) * m)
        _chirp_dtft_mag3(x, n, mu, f - delta, mu, f, mu, f + delta, mag)
        x1 = mag[0]
        x2 = mag[1]
        x3 = mag[2]
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


cdef void _refine(const double complex* p, Py_ssize_t n, double mu, double f, double m, double step,
                  int rate_iterations, int max_iterations, double threshold,
                  double* out_f, double* out_mu) noexcept nogil:
    cdef int it
    cdef double nu0, h, y1, y2, y3, denom
    cdef double mag[3]
    nu0 = _spi(p, n, mu, f, m, max_iterations, threshold)
    for it in range(1, rate_iterations + 1):
        h = step / (1 << it)
        _chirp_dtft_mag3(p, n, mu - h, nu0, mu, nu0, mu + h, nu0, mag)
        y1 = 2.0 * log(mag[0])
        y2 = 2.0 * log(mag[1])
        y3 = 2.0 * log(mag[2])
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
    out_f[0] = _spi(p, n, mu, floor(nu0 * m + 0.5) / m, m, max_iterations, threshold)
    out_mu[0] = mu


def frft_refine(const double complex[::1] x, double mu, double f, double m, double step,
                int rate_iterations, int max_iterations=5, double threshold=1e-12):
    """Steps 2-3 of frft_frequency from the bank's (mu, f): returns (f_centre, mu).

    SPI at the coarse rate (seeded by the bank bin), log-SPI on the rate at
    that frequency, then SPI at the final rate (seeded by the padded-FFT bin
    nearest the first SPI).
    """
    cdef Py_ssize_t n = x.shape[0]
    if n == 0:
        raise ValueError("frft_refine: empty x")
    with nogil:
        _refine(&x[0], n, mu, f, m, step, rate_iterations, max_iterations, threshold, &f, &mu)
    return f, mu


cdef void _bank_fill(const float complex* x, Py_ssize_t n, double rate0, double step,
                     float complex* out, Py_ssize_t rows) noexcept nogil:
    cdef Py_ssize_t i, k, m
    cdef double c = (n - 1) / 2.0, u, br, bi, sr = 1.0, si = 0.0, tr
    # Column by column: the phasor of sample k starts at the base rate and
    # advances one step per row.  (k - c)^2 is the same number, exactly, at
    # k and at m = n - 1 - k (k - c and m - c are exact and opposite), so
    # one cos/sin set serves both columns.
    for k in range((n + 1) // 2):
        m = n - 1 - k
        u = (k - c) * (k - c)
        br = cos(-M_PI * rate0 * u)
        bi = sin(-M_PI * rate0 * u)
        if rows > 1:  # the step phasor only takes a row to the next
            sr = cos(-M_PI * step * u)
            si = sin(-M_PI * step * u)
        for i in range(rows):
            out[i * n + k] = <float complex> ((x[k].real * br - x[k].imag * bi) + (x[k].real * bi + x[k].imag * br) * 1j)
            out[i * n + m] = <float complex> ((x[m].real * br - x[m].imag * bi) + (x[m].real * bi + x[m].imag * br) * 1j)
            if i + 1 < rows:
                tr = br * sr - bi * si
                bi = br * si + bi * sr
                br = tr


def frft_bank_fill(const float complex[::1] x, double rate0, double step, float complex[:, ::1] out):
    """out[i, k] = x[k] exp(-j pi (rate0 + i step) (k - c)^2), c = (n-1)/2."""
    cdef Py_ssize_t rows = out.shape[0], n = out.shape[1]
    if x.shape[0] != n:
        raise ValueError("x and out rows differ in length")
    if rows == 0 or n == 0:
        return
    with nogil:
        _bank_fill(&x[0], n, rate0, step, &out[0, 0], rows)


cdef void _argmax(const bank_t* a, Py_ssize_t rows, Py_ssize_t cols,
                  Py_ssize_t* row, Py_ssize_t* col, double* peak) noexcept nogil:
    cdef Py_ssize_t i, k, bi_ = 0, bk = 0
    cdef double best = -1.0, v, re, im
    for i in range(rows):
        for k in range(cols):
            re = a[i * cols + k].real
            im = a[i * cols + k].imag
            v = re * re + im * im
            if v > best:
                best = v
                bi_ = i
                bk = k
    row[0] = bi_
    col[0] = bk
    peak[0] = best


def argmax_abs2(bank_t[:, ::1] a):
    """(row, col, |a|^2) of the largest |a|^2, accumulated in double."""
    cdef Py_ssize_t rows = a.shape[0], cols = a.shape[1], bi_ = 0, bk = 0
    cdef double best = -1.0
    if rows and cols:
        with nogil:
            _argmax(&a[0, 0], rows, cols, &bi_, &bk, &best)
    return bi_, bk, best


# The same kernels over many blocks in one call (rdefpy.carrier._frft_batch):
# short blocks then cost one GIL release per batch, not one per block and step.
# Each block or segment gets exactly the single-block call above.

def frft_bank_fill_many(const float complex[:, ::1] x, const Py_ssize_t[::1] block, const double[::1] rate0,
                        const double[::1] step, const Py_ssize_t[::1] rows, float complex[:, ::1] out):
    """frft_bank_fill(x[block[s]], rate0[s], step[s]) into the next rows[s] rows of out, for each segment s."""
    cdef Py_ssize_t n = out.shape[1], segs = rows.shape[0], s, r = 0
    if x.shape[1] != n or n == 0:
        raise ValueError("x and out rows differ in length, or are empty")
    if block.shape[0] != segs or rate0.shape[0] != segs or step.shape[0] != segs:
        raise ValueError("one block, rate0, step and rows per segment")
    for s in range(segs):
        if rows[s] < 1 or not 0 <= block[s] < x.shape[0]:
            raise ValueError("segment rows must be positive and blocks inside x")
        r += rows[s]
    if r != out.shape[0]:
        raise ValueError(f"out has {out.shape[0]} rows, the segments need {r}")
    with nogil:
        r = 0
        for s in range(segs):
            _bank_fill(&x[block[s], 0], n, rate0[s], step[s], &out[r, 0], rows[s])
            r += rows[s]


def argmax_abs2_many(float complex[:, ::1] a, const Py_ssize_t[::1] rows):
    """argmax_abs2 of each run of rows[s] consecutive rows of a: arrays (row within the run, col, |a|^2)."""
    cdef Py_ssize_t cols = a.shape[1], segs = rows.shape[0], s, r = 0
    for s in range(segs):
        if rows[s] < 1:
            raise ValueError("segment rows must be positive")
        r += rows[s]
    if r != a.shape[0] or cols == 0:
        raise ValueError(f"a has {a.shape[0]} rows of {cols}, the segments need {r} non-empty rows")
    row, col, peak = np.empty(segs, np.intp), np.empty(segs, np.intp), np.empty(segs, np.float64)
    cdef Py_ssize_t[::1] row_v = row, col_v = col
    cdef double[::1] peak_v = peak
    with nogil:
        r = 0
        for s in range(segs):
            _argmax(&a[r, 0], rows[s], cols, &row_v[s], &col_v[s], &peak_v[s])
            r += rows[s]
    return row, col, peak


def spi_refine_many(const double complex[:, ::1] x, const Py_ssize_t[::1] block, const double[::1] f, double m):
    """spi_refine(x[block[s]], 0.0, f[s], m) for each s: array of f [cycles/sample]."""
    cdef Py_ssize_t n = x.shape[1], segs = block.shape[0], s
    if n == 0 or f.shape[0] != segs:
        raise ValueError("spi_refine_many: empty blocks, or not one f per block")
    for s in range(segs):
        if not 0 <= block[s] < x.shape[0]:
            raise ValueError("blocks must be inside x")
    out = np.empty(segs, np.float64)
    cdef double[::1] out_v = out
    with nogil:
        for s in range(segs):
            out_v[s] = _spi(&x[block[s], 0], n, 0.0, f[s], m, 5, 1e-12)
    return out


def frft_refine_many(const double complex[:, ::1] x, const Py_ssize_t[::1] block, const double[::1] mu,
                     const double[::1] f, double m, double step, int rate_iterations):
    """frft_refine(x[block[s]], mu[s], f[s], m, step, rate_iterations) for each s: arrays (f_centre, mu)."""
    cdef Py_ssize_t n = x.shape[1], segs = block.shape[0], s
    if n == 0 or f.shape[0] != segs or mu.shape[0] != segs:
        raise ValueError("frft_refine_many: empty blocks, or not one mu and f per block")
    for s in range(segs):
        if not 0 <= block[s] < x.shape[0]:
            raise ValueError("blocks must be inside x")
    out_f, out_mu = np.empty(segs, np.float64), np.empty(segs, np.float64)
    cdef double[::1] f_v = out_f, mu_v = out_mu
    with nogil:
        for s in range(segs):
            _refine(&x[block[s], 0], n, mu[s], f[s], m, step, rate_iterations, 5, 1e-12, &f_v[s], &mu_v[s])
    return out_f, out_mu
