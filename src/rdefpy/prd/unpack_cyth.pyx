# cython: language_level=3
# cython: boundscheck=False, wraparound=False, nonecheck=False, initializedcheck=False, cdivision=True
"""
Cython-optimized I/Q sample unpacking for RDEF files.

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
    Unpack 8-bit I/Q samples with bias correction (Table 5-2: MSB Q2 I2 Q1 I1 LSB).

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
