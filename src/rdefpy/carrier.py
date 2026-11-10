# Copyright (c) 2026 Morehead State University
# Author: Andrea Togni
# SPDX-License-Identifier: MIT

"""Open-loop carrier extraction: FrFT + Successive Parabolic Interpolation.

    from rdefpy import read_product, extract_carrier

    c = extract_carrier(read_product("scan.prd"), integration_time=1.0, max_rate_hz_s=10.0)
    c["time"], c["frequency_before_downconversion"]    # block midpoints, Hz; pd.DataFrame(c) for pandas

Per block of ``integration_time`` seconds, a fractional Fourier transform
(FrFT) bank over trial chirp rates finds the carrier and its rate, and
Successive Parabolic Interpolation (SPI) of the DTFT magnitude refines the
frequency. The samples go into the transforms untouched: no mean removal,
window, filter or normalization (the receiver may park the carrier in the
DC bin, where DC removal would delete it). Zero-padding is the only knob
on the data, and it alters no recorded sample.

References
----------
Togni, A., Zannoni, M., & Tortora, P. (2025). Accurate frequency
estimation through iterative parabolic interpolations. Signal
Processing, 237, 110094. https://doi.org/10.1016/j.sigpro.2025.110094
"""

import ctypes
import math
import os
import warnings
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from rdefpy import _utils
from rdefpy.prd.classes import _downconv_hz, _origin_offset, _record_epoch

# Optional compiled kernels (rdefpy.prd.unpack_cyth): the bank fill and
# argmax, and the SPI / log-SPI refinement without the GIL. The NumPy code
# below is the reference; the kernels take the same steps (phasors by
# recursion instead of exp; SPI seeds from the bank peak and the padded-FFT
# bin nearest the first SPI instead of new FFTs) and agree with it to 1e-7 Hz
# in tests/test_carrier.py. None: NumPy path.
try:
    from rdefpy.prd import unpack_cyth as _cy
except ImportError:
    _cy = None
_spi_refine = getattr(_cy, "spi_refine", None)
_frft_refine = getattr(_cy, "frft_refine", None)
_bank_fill = getattr(_cy, "frft_bank_fill", None)
_argmax_abs2 = getattr(_cy, "argmax_abs2", None)

__all__ = ["CARRIER_DTYPE", "extract_carrier", "frft_frequency", "spi_frequency"]

# Peak memory of one block in flight per block sample [bytes]. Measured with
# tracemalloc (frft_frequency, complex64 input of 8 B/sample): 96 for the
# tone model, 320 for the rate search up to 1 Msps, 128 from 4 to 20 Msps.
# Rounded up for the decode and what tracemalloc does not see. The whole
# extract_carrier, decode included, peaks at ~110-130 and ~340-350 per block
# sample per thread (docs/guide/carrier.rst, "Threads and memory").
_BYTES_PER_SAMPLE = 400

# frft_frequency widens the rate search while the winner is at or beyond this
# fraction of the current range.
_WIDEN_GATE = 0.75


def _free_bytes():
    """Memory this process can still take [bytes]: the smallest of free RAM,
    cgroup limit minus usage and RLIMIT_AS minus the address space in use;
    2 GiB if none can be read."""
    room = []

    def read(path):
        with open(path) as f:
            return f.read()

    if os.name == "nt":
        class Status(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        *((k, ctypes.c_ulonglong) for k in ("total", "avail", "tpf", "apf", "tv", "av", "aev"))]
        try:
            s = Status(dwLength=ctypes.sizeof(Status))
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)):
                room.append(s.avail)
        except (AttributeError, OSError):
            pass
    else:
        try:  # Linux
            room.append(int(read("/proc/meminfo").split("MemAvailable:")[1].split()[0]) * 1024)
        except (OSError, IndexError, ValueError):
            try:  # macOS and other POSIX: no free-RAM query, physical RAM
                room.append(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
            except (OSError, ValueError, AttributeError):
                pass
        for lim, use in (("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),  # cgroup v2, v1
                         ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes")):
            try:
                room.append(int(read(lim)) - int(read(use)))  # "max": ValueError, no limit
                break
            except (OSError, ValueError):
                pass
        try:
            import resource
            soft = resource.getrlimit(resource.RLIMIT_AS)[0]
            if soft != resource.RLIM_INFINITY:
                try:
                    used = int(read("/proc/self/statm").split()[0]) * os.sysconf("SC_PAGE_SIZE")
                except (OSError, ValueError):
                    used = 0
                room.append(soft - used)
        except (ImportError, OSError, ValueError):
            pass
    return max(min(room), 0) if room else 2 << 30

#: One row per block of ``extract_carrier``.
CARRIER_DTYPE = np.dtype([
    ("time", "datetime64[ns]"),  # block midpoint t0 + T/2
    ("frequency_before_downconversion", "f8"),  # Hz, downconversion_frequency + frequency
    ("frequency", "f8"),  # Hz, baseband carrier at the midpoint
    ("rate", "f8"),  # Hz/s, baseband carrier rate
    ("downconversion_frequency", "f8"),  # Hz, mean f_DC over the block (§5.2.6)
])

# 3-point Gauss-Legendre nodes and weights on [-1, 1].
_GL_X = np.array([-np.sqrt(0.6), 0.0, np.sqrt(0.6)])
_GL_W = np.array([5.0, 8.0, 5.0]) / 9.0


def _dtft_evaluator(samples):
    """Return ``value(f)``: the complex DTFT of *samples* at frequencies *f*
    [cycles/sample].

    Exact factorisation n = a*B + b: exp(-2j pi f n) = exp(-2j pi f a B)
    exp(-2j pi f b), so all frequencies come from one (A x B) @ (B x k)
    product plus exp() calls of length ~sqrt(n). The block is zero-padded up
    to A*B samples, which adds nothing to the sum.
    """
    samples = np.asarray(samples)
    n = len(samples)
    n_col = int(np.ceil(np.sqrt(n)))
    n_row = -(-n // n_col)
    block = np.zeros(n_row * n_col, dtype=np.result_type(samples.dtype, np.complex128))
    block[:n] = samples  # trailing zeros: no contribution to the sum
    block = block.reshape(n_row, n_col)
    a = np.arange(n_row) * n_col
    b = np.arange(n_col)

    def value(f):
        f = np.atleast_1d(np.asarray(f, dtype=float))
        # One column would dispatch to the slower threaded gemv: pad to two.
        fc = np.repeat(f, 2) if f.size == 1 else f
        inner = block @ np.exp(-2j * np.pi * np.outer(b, fc))
        out = (np.exp(-2j * np.pi * np.outer(a, fc)) * inner).sum(axis=0)
        return out[::2] if f.size == 1 else out

    return value


def spi_frequency(samples, sample_rate, zero_padding=3, max_iterations=5, convergence_threshold=1e-12):
    """Frequency of a tone in a complex baseband block [Hz]: FFT + SPI.

    Togni et al. (2025), Signal Processing 237, 110094,
    https://doi.org/10.1016/j.sigpro.2025.110094

    Zero-padded FFT peak (coarse), then Successive Parabolic Interpolation
    of the DTFT magnitude: delta halves each iteration from half a padded
    bin, and the three-point stencil is re-centred on the local maximum so
    the parabola interpolates, never extrapolates.

    :param samples: Complex baseband samples, used untouched.
    :param sample_rate: Sampling frequency [Hz].
    :param zero_padding: Zero-padding factor of the coarse FFT.
    :param max_iterations: SPI iterations.
    :param convergence_threshold: Stop when the update is below this
        [cycles/sample]; at the default all iterations normally run.
    :return: Frequency [Hz] relative to baseband 0 Hz, in [-fs/2, fs/2];
        NaN for an all-zero or non-finite block.
    :raises ValueError: Empty *samples*, or *zero_padding* not a positive
        integer.
    """
    n = len(samples)
    if n == 0:
        raise ValueError("samples array must not be empty")
    _check_padding(zero_padding)
    m = zero_padding * n

    # Coarse: zero-padded FFT peak; bins above m//2 are negative frequencies.
    spec = np.abs(np.fft.fft(samples, m))
    k_max = int(np.argmax(spec))
    if not spec[k_max] > 0:  # all-zero or non-finite block: no tone
        return np.nan
    f_est = k_max / m if k_max <= m // 2 else (k_max - m) / m
    if _spi_refine is not None:
        x = np.ascontiguousarray(samples, dtype=np.complex128)
        return _fold(_spi_refine(x, 0.0, f_est, float(m), max_iterations, convergence_threshold)) * sample_rate

    dtft_value = _dtft_evaluator(samples)

    def dtft_mag(*f):
        return np.abs(dtft_value(f))

    for i in range(1, max_iterations + 1):
        f_prev = f_est
        delta = 1.0 / (2**i * m)
        x1, x2, x3 = dtft_mag(f_est - delta, f_est, f_est + delta)
        # Re-centre so x2 is the largest of the three.
        if x2 < x3:
            f_est += delta
            x1, x2 = x2, x3
            x3 = dtft_mag(f_est + delta)[0]
        elif x2 < x1:
            f_est -= delta
            x3, x2 = x2, x1
            x1 = dtft_mag(f_est - delta)[0]
        denom = x1 + x3 - 2.0 * x2
        if denom < 0.0:  # a genuine peak, at any amplitude
            f_est -= 0.5 * delta * (x3 - x1) / denom
        if abs(f_est - f_prev) < convergence_threshold:
            break

    return _fold(f_est) * sample_rate


def _fold(f):
    """Frequency [cycles/sample] into [-1/2, 1/2): SPI can step past +-1/2."""
    return (f + 0.5) % 1.0 - 0.5


def _check_padding(zero_padding):
    if not isinstance(zero_padding, (int, np.integer)) or zero_padding < 1:
        raise ValueError(f"zero_padding must be a positive integer, got {zero_padding!r}")


def _bank_peak(x64, rates, step, k2, fft_len):
    """(peak, rate, bin) of the zero-padded |FFT| over the block dechirped at each rate.

    *rates* are evenly spaced by *step* (cycles/sample^2, either sign). Peak is |.|^2
    compiled, |.| in NumPy: compare peaks from the same path only.
    """
    n, best = len(x64), (-1.0, 0.0, 0)
    chunk = max(1, min(64, 2**24 // fft_len))  # <= 16 M complex64 per batch
    if _bank_fill is not None:
        buf = np.empty((min(chunk, len(rates)), n), dtype=np.complex64)
    else:
        step_chirp = np.exp(-1j * np.pi * step * k2).astype(np.complex64)
    for i0 in range(0, len(rates), chunk):
        m = min(chunk, len(rates) - i0)
        if _bank_fill is not None:
            c = buf[:m]
            _bank_fill(x64, float(rates[i0]), step, c)
            i, k, v = _argmax_abs2(np.fft.fft(c, fft_len, axis=-1))
        else:
            c = np.empty((m, n), dtype=np.complex64)
            c[0] = np.exp(-1j * np.pi * rates[i0] * k2)  # fresh base per chunk
            c[1:] = step_chirp
            np.cumprod(c, axis=0, out=c)  # base * step^i
            c *= x64
            p = np.abs(np.fft.fft(c, fft_len, axis=-1))  # |.|, not |.|^2: no overflow
            i, k = np.unravel_index(int(np.argmax(p)), p.shape)
            v = float(p[i, k])
        if v > best[0]:
            best = (v, float(rates[i0 + i]), int(k))
    return best


def frft_frequency(samples, sample_rate, max_rate_hz_s=None, rate_center_hz_s=0.0, rate_range_hz_s=2.0,
                   rate_step_cells=2.0, zero_padding=3, rate_iterations=3):
    """Frequency and rate of a linear-chirp block [Hz, Hz/s]: FrFT + SPI.

    Discrete-angle FrFT magnitude, implemented directly in (f, fdot):
    abs(FrFT_a) is abs(DTFT) of the block dechirped at rate -cot(a)
    (normalization aside). Dechirping is about the sample centre
    c = (n-1)/2; the chirp is part of the transform kernel, the samples are
    untouched.

    1. Coarse bank: trial rates rate_center +- rate_range at rate_step_cells
       cells (1 cell = 1/T^2 Hz/s), global abs(FFT) argmax over (rate, bin).
       While the winner is in the outer quarter of the range (at or
       beyond 3/4 of it), the range doubles
       (only the new outer rates are computed), up to rate_center +-
       max_rate (None: 0.75 fs/T); a winner still on the edge there warns.
       At least 3 trial rates (rate_center and +- one step) unless max_rate
       is under half a step. The widening follows the peak only where the
       carrier stands out of the noise inside the current range: at low SNR
       with a rate far beyond rate_range it can stop early (start with a
       larger rate_range_hz_s). An all-zero or non-finite block gives
       (NaN, NaN), also with ``max_rate_hz_s=0``.
       ``max_rate_hz_s=0`` skips steps 1-2: plain FFT + SPI (at
       rate_center, 0 by default), and the rate returned is rate_center.
    2. Rate: spi_frequency on the block dechirped at the coarse rate, then
       SPI on log abs(DTFT)^2 against the rate, half-steps rate_step_cells/2,
       /4, ... (near the peak the rate profile is Gaussian-like, hence log).
    3. Frequency: spi_frequency on the block dechirped at the final rate.

    Cost: the bank is ~2 R T^2 / rate_step_cells + 1 FFTs of
    zero_padding * n points (T = n / fs), R the final range, so it grows
    with T^2: +-2 Hz/s at T = 1 s is 3 FFTs, at T = 10 s about 200. rate_step_cells 2 loses
    <= 0.24 dB at the worst grid offset, 4 about 1 dB. The bank runs in
    complex64 (argmax only); steps 2-3 are complex128.

    :param samples: Complex baseband block, used untouched.
    :param sample_rate: Sampling frequency fs [Hz].
    :param max_rate_hz_s: Largest rate offset from *rate_center_hz_s*
        searched [Hz/s]. None (default): 0.75 fs/T. 0: no rate search.
    :param rate_center_hz_s: Centre of the rate search [Hz/s].
    :param rate_range_hz_s: Starting search range +- [Hz/s], doubled while
        the winner is in its outer quarter.
    :param rate_step_cells: Spacing of the trial rates [cells, 1/T^2 Hz/s].
    :param zero_padding: Zero-padding factor of the FFTs (positive integer).
    :param rate_iterations: Log-SPI iterations on the rate (step 2).
    :return: ``(f_hz, fdot_hz_s)``, f_hz at the block MIDPOINT t0 + T/2
        (centre estimate + fdot / (2 fs)).
    :raises ValueError: Empty *samples*, *zero_padding* not a positive
        integer, or *rate_step_cells* not positive.
    """
    x = np.asarray(samples, dtype=complex)
    n = len(x)
    if n == 0:
        raise ValueError("samples array must not be empty")
    _check_padding(zero_padding)
    if not rate_step_cells > 0:
        raise ValueError(f"rate_step_cells must be positive, got {rate_step_cells!r}")
    k2 = (np.arange(n) - (n - 1) / 2.0) ** 2
    if max_rate_hz_s == 0:  # no rate search: SPI at the given rate, one FFT
        mu = rate_center_hz_s / sample_rate**2
        f_centre = spi_frequency(x * np.exp(-1j * np.pi * mu * k2) if mu else x, sample_rate, zero_padding)
        if np.isnan(f_centre):  # all-zero or non-finite block
            return np.nan, np.nan
        return float(f_centre + 0.5 * mu * sample_rate), float(rate_center_hz_s)
    step = rate_step_cells / n**2  # cycles/sample^2
    mu0 = rate_center_hz_s / sample_rate**2
    # Cap: the given max_rate, else 0.75 fs / T (a chirp sweeping 3/4 of the band over the
    # block); beyond about 0.78 fs / T the widening can stop on a wrong interior peak.
    cap = round((abs(max_rate_hz_s) / sample_rate**2 if max_rate_hz_s is not None else 0.75 / n) / step)
    J = min(max(1, round(abs(rate_range_hz_s) / sample_rate**2 / step)), cap)  # branches mu0 + j step, |j| <= J
    fft_len = zero_padding * n
    x = np.ascontiguousarray(x)
    x64 = x.astype(np.complex64)
    best = _bank_peak(x64, mu0 + step * np.arange(-J, J + 1), step, k2, fft_len)
    # Winner in the outer part of the range: double it, search only the new outer branches.
    # Not only the outermost branch: a chirp beyond the range spreads its power, so at
    # moderate SNR the winner is often a noise-lifted branch just inside the edge.
    while J and abs(round((best[1] - mu0) / step)) >= _WIDEN_GATE * J and J < cap:
        Jn = min(2 * J, cap)
        for side in (-1, 1):
            p = _bank_peak(x64, mu0 + side * step * np.arange(J + 1, Jn + 1), side * step, k2, fft_len)
            if p[0] > best[0]:
                best = p
        J = Jn
    peak, mu, k_star = best
    if not peak > 0:  # all-zero or non-finite block: no carrier
        return np.nan, np.nan
    if J and abs(round((mu - mu0) / step)) == J:  # still on the edge at the cap
        warnings.warn(f"best trial rate is the outermost one, {rate_center_hz_s:g} +- {J * step * sample_rate**2:.4g} "
                      "Hz/s: the true rate may lie beyond max_rate_hz_s", RuntimeWarning, stacklevel=2)

    if _frft_refine is not None:  # same steps, compiled; FFTs stay NumPy
        f0 = k_star / fft_len if k_star <= fft_len // 2 else (k_star - fft_len) / fft_len
        nu, mu = _frft_refine(x, mu, f0, float(fft_len), step, rate_iterations)
        return float(nu * sample_rate + 0.5 * mu * sample_rate), float(mu * sample_rate**2)

    # Refine f before the rate step: a coarse bin up to 1/2 bin off costs up
    # to 3.9 dB of signal in the rate profile.
    nu0 = spi_frequency(x * np.exp(-1j * np.pi * mu * k2), 1.0, zero_padding)
    z = x * np.exp(-2j * np.pi * nu0 * np.arange(n))

    def log_power(r):
        return np.log(abs(np.dot(z, np.exp(-1j * np.pi * r * k2))) ** 2)

    for it in range(1, rate_iterations + 1):
        h = step / 2**it
        y1, y2, y3 = log_power(mu - h), log_power(mu), log_power(mu + h)
        if y3 > y2:
            mu += h
            y1, y2, y3 = y2, y3, log_power(mu + h)
        elif y1 > y2:
            mu -= h
            y3, y2, y1 = y2, y1, log_power(mu - h)
        denom = y1 + y3 - 2.0 * y2
        if denom < 0:
            mu -= 0.5 * h * (y3 - y1) / denom

    f_centre = spi_frequency(x * np.exp(-1j * np.pi * mu * k2), sample_rate, zero_padding)
    return float(f_centre + 0.5 * mu * sample_rate), float(mu * sample_rate**2)


def extract_carrier(pf, integration_time, max_rate_hz_s=None, rate_range_hz_s=2.0, zero_padding=3, start_epoch=None,
                    whole_second=False, memory_limit=None, **frft_kwargs):
    """Carrier frequency before downconversion of a product file, one value per block [Hz].

    FrFT + Successive Parabolic Interpolation (SPI): Togni, A., Zannoni, M.,
    & Tortora, P. (2025). Accurate frequency estimation through iterative
    parabolic interpolations. Signal Processing, 237, 110094.
    https://doi.org/10.1016/j.sigpro.2025.110094

    Splits ``pf.samples`` into blocks of ``integration_time`` seconds from
    the first sample (or ``start_epoch``) and runs ``frft_frequency`` on
    each, untouched. One pass over the file, each record decoded once.
    Peak memory grows with
    the threads: measured ~130 (max_rate_hz_s=0) to ~350 (rate search)
    bytes per block sample per thread (1 Msps, T = 1 s: 340 MB on 1
    thread, 1.3 GB on 4); set_threads and memory_limit bound it. Each
    block is tagged at its MIDPOINT t0 + T/2, and its frequency before
    downconversion (DSN radio science: sky frequency) is the
    baseband frequency plus the downconversion frequency f_DC (§5.2.6.1:
    phi_after = phi_before - phi_DC), the mean f_DC over the block. The
    result is the mean frequency before downconversion over the block,
    exact for any f_DC while the baseband frequency is linear in time over
    the block. The
    chirp fit returns the baseband frequency averaged with weight
    w(u) = 1 - (2u/T)^2 (u from the midpoint), so a baseband frequency
    with second derivative f_b'' leaves -f_b'' T^2 / 60 (receiver steering
    that tracks the carrier keeps f_b'' small). An all-zero or non-finite block gives
    NaN frequency and rate; downconversion_frequency stays finite. A block
    holding a record whose c1, c2 and c3 are NaN (DSN 0222 Millisecond
    Predict: the variable downconversion is not in the file) gives NaN
    downconversion_frequency and frequency_before_downconversion; its
    baseband frequency and rate are still estimated.

    A block that would span a time gap between records, or that holds a
    record flagged invalid (validity_flag nonzero, Table 5-1), gives NaN (no
    interpolation across gaps). Records are one second each, so blocks
    stay aligned to the records after a gap when the integration time
    divides 1 s or is a whole number of seconds. Samples after the last
    whole block are not used.

    Each block searches its baseband chirp rate (residual Doppler rate
    left after the downconversion) independently, from 0 Hz/s: +-
    *rate_range_hz_s* first, doubled while the best trial rate is in the
    outer quarter of the range, up to *max_rate_hz_s* (see ``frft_frequency``). No rate
    is carried between blocks, so all blocks run in parallel.

    :param pf: ``ProductFile`` (from ``read_product``).
    :param integration_time: Block length T [s]; T * sample_rate must be an
        integer.
    :param max_rate_hz_s: Largest rate offset from ``rate_center_hz_s`` (0
        by default) searched [Hz/s]. None (default): 0.75 fs / T (beyond,
        the chirp sweeps most of the band over the block and the search
        can lock on a wrong peak). 0: tone model, plain FFT + SPI.
    :param rate_range_hz_s: Starting search range +- [Hz/s]. The cost grows
        as range * T^2.
    :param zero_padding: Zero-padding factor of the FFTs.
    :param start_epoch: Start of the first block, in the time scale of the
        record time tags, in the forms ``read_product`` accepts: datetime64,
        datetime (naive = UTC) or an ISO-8601 or CCSDS string (e.g.
        "2026-039T12:00:00"), rounded to the nearest sample; it must fall inside
        a record. None (default): the first sample.
    :param whole_second: Tag the blocks on the whole-second grid (False by
        default). The start (the first sample, or ``start_epoch``) is
        rounded up to the next whole second and the first block starts at
        the sample nearest to it. No sample need fall exactly on the
        second, so the samples' own midpoint t_mid is off the grid by up to
        half a sample. Each block is therefore reported at the grid tag
        t_grid = second + T/2 + k T nearest to t_mid (x.500 for T = 1 s),
        moved there with the block's own estimate: with dt = t_grid -
        t_mid, ``frequency`` is f + rate * dt (exact for a baseband
        frequency linear over the block, the model of the fit; the error
        left is rate_error * dt + f_b'' dt^2 / 2), ``downconversion_frequency``
        is the mean f_DC over the interval shifted by dt (exact, from the
        polynomials: the block's two ends move by dt, record boundaries
        inside it stay; an end moved past its record's edge, by at most half
        a sample, uses that record's polynomial), and
        ``frequency_before_downconversion`` is their sum.
        The samples are neither shifted nor interpolated: only the
        estimates are carried over dt.
    :param memory_limit: Memory budget for the blocks in flight [bytes].
        Each costs about 400 bytes per block sample, so at most
        min(2 * threads, memory_limit // (400 n)) blocks run at once, at
        least 1. None (default): half of the memory the process can still
        take (the smallest of free RAM, cgroup limit minus usage and
        RLIMIT_AS minus the address space in use; 2 GiB if none can be
        read). A single block over the budget runs alone, with a
        RuntimeWarning. ``math.inf``: no limit, 2 * threads in flight, no
        warning. The results do not depend on it.
    :param frft_kwargs: Passed to ``frft_frequency``.
    :return: Structured array, one row per block, fields ``time``,
        ``frequency_before_downconversion``, ``frequency``, ``rate``,
        ``downconversion_frequency`` (``CARRIER_DTYPE``).
        ``pandas.DataFrame(c)`` makes it a DataFrame.
    :raises ValueError: No records; *integration_time* not finite and
        positive, or not a whole number of samples; records with different
        sample rates; *start_epoch* not inside a record. Bad
        *zero_padding* or *rate_step_cells* raise from ``frft_frequency``
        once a block is estimated.
    """
    if not pf.records:
        raise ValueError("product file has no records")
    if not np.isfinite(integration_time) or integration_time <= 0:
        raise ValueError(f"integration_time must be a positive number of seconds, got {integration_time}")
    rates = {rec.header.sample_rate for rec in pf.records}
    if len(rates) != 1:
        raise ValueError(f"records have different sample rates: {sorted(rates)}")
    fs = float(rates.pop())
    n = round(integration_time * fs)
    if n < 1 or abs(n - integration_time * fs) > 1e-9 * n:
        raise ValueError(f"integration_time * sample_rate = {integration_time * fs} is not a whole number of samples")

    recs, skip = pf.records, 0
    start_epoch = _utils._to_datetime64(start_epoch)  # same forms as read_product
    if whole_second:  # first whole second at or after the start
        start = recs[0].header.timetag if start_epoch is None else start_epoch
        sec = np.datetime64(start, "s")
        start_epoch = grid = sec if sec == start else sec + np.timedelta64(1, "s")
    if start_epoch is not None:  # drop whole records before it, then skip samples
        start = np.datetime64(start_epoch, "ns")
        for k0, rec in enumerate(recs):
            skip = round((start - rec.header.timetag) / np.timedelta64(1, "ns") * fs / 1e9)
            if 0 <= skip < len(rec):
                break
        else:
            raise ValueError(f"start_epoch {start} is not inside a record")
        recs = recs[k0:]
    # Blocks run on their own pool ("cpu" serves the decodes inside
    # read_samples: a block queued there could wait on its own decode).
    with _utils._pool_lock:
        ex = _utils._pools.get("carrier")
        if ex is None:
            ex = _utils._pools["carrier"] = ThreadPoolExecutor(_utils._THREADS)

    n_blocks = (sum(len(r) for r in recs) - skip) // n
    c = np.empty(n_blocks, dtype=CARRIER_DTYPE)
    for name in CARRIER_DTYPE.names[1:]:
        c[name] = np.nan

    def estimate(b, blk, dc, dt):  # one block; writes only row b; dt: tag shift [s] (whole_second)
        f, rate = frft_frequency(blk, fs, max_rate_hz_s, rate_range_hz_s=rate_range_hz_s, zero_padding=zero_padding,
                                 **frft_kwargs)
        f += rate * dt  # the fitted chirp, carried to the grid tag
        c[b] = c["time"][b], dc + f, f, rate, dc

    # At most 2 per thread in flight, fewer if the memory budget says so.
    cap = 2 * ex._max_workers
    if not (memory_limit is not None and math.isinf(memory_limit)):
        budget = _free_bytes() // 2 if memory_limit is None else memory_limit
        need = _BYTES_PER_SAMPLE * n
        cap = max(1, min(cap, int(budget // need)))
        if need > budget:
            warnings.warn(f"one block (integration_time {n / fs:g} s at {fs:g} Hz) needs about {need / 1e9:.2f} GB, over "
                          f"the memory budget of {budget / 1e9:.2f} GB: running one block at a time. Use a shorter "
                          f"integration_time, or memory_limit=math.inf to ignore the budget", RuntimeWarning,
                          stacklevel=2)
    pending = deque()
    # One pass, each record decoded once: blocks fill from consecutive records.
    parts, filled, gap, dc_int, b, T, dt = [], 0, False, 0.0, 0, n / fs, 0.0
    for k, rec in enumerate(recs):
        if filled and k:  # the open block continues into this record
            t0, t1 = _record_epoch(recs[k - 1].header), _record_epoch(rec.header)
            gap |= t0 is None or t1 is None or abs((t1 - t0) / np.timedelta64(1, "s") - 1.0) > 1e-6
        x = rec.read_samples()
        pos, skip = skip, 0
        while pos < len(x) and b < n_blocks:
            if not filled:  # block b starts here: tag its midpoint t0 + T/2
                c["time"][b] = rec.header.timetag + np.timedelta64(round((2 * pos + n) * 1e9 / (2 * fs)), "ns")
                if whole_second:  # nearest grid tag; dt = grid - samples' midpoint
                    rel = (c["time"][b] - grid) / np.timedelta64(1, "ns") - 0.5e9 * T  # from the grid start
                    k_grid = round(rel / (1e9 * T))
                    c["time"][b] = grid + np.timedelta64(round((k_grid + 0.5) * 1e9 * T), "ns")
                    dt = ((c["time"][b] - grid) / np.timedelta64(1, "ns") - 0.5e9 * T - rel) / 1e9
            take = min(n - filled, len(x) - pos)
            parts.append(x[pos : pos + take])
            gap |= bool(rec.header.validity_flag)  # nonzero: error code (Table 5-1): NaN
            # Integral of f_DC (§5.2.6) over this piece: 3-point Gauss-Legendre,
            # exact for f_DC (quadratic in record time); origin per pf.t0_offset.
            # whole_second: the block's two ends move by dt, record boundaries inside it stay
            ta = pos / fs + (dt if filled == 0 else 0.0)
            tb = (pos + take) / fs + (dt if filled + take == n else 0.0)
            tq = 0.5 * (ta + tb) + 0.5 * (tb - ta) * _GL_X + _origin_offset(rec.header, pf.t0_offset)  # from t0
            dc_int += 0.5 * (tb - ta) * float(np.sum(_GL_W * _downconv_hz(rec.header, tq)))
            filled, pos = filled + take, pos + take
            if filled == n:
                if not gap:  # no interpolation across a gap: NaN
                    blk = parts[0] if len(parts) == 1 else np.concatenate(parts)
                    pending.append(ex.submit(estimate, b, blk, dc_int / T, dt))  # mean f_DC
                    while len(pending) >= cap:
                        pending.popleft().result()
                parts, filled, gap, dc_int, b = [], 0, False, 0.0, b + 1
    for fut in pending:
        fut.result()
    return c
