"""extract_carrier / frft_frequency / spi_frequency on synthetic tones and chirps."""

import warnings

import numpy as np
import pytest

import rdef_spec as spec
from rdefpy import extract_carrier, read_product
from rdefpy.carrier import frft_frequency, spi_frequency

FS = 1000
F0, RATE = 123.4567, 2.5  # baseband carrier at the first sample [Hz], its rate [Hz/s]
# Steering on: c3 != 0 makes the block mean of f_DC differ from its midpoint value
# (by c3 T^2 / 4 = 0.19 Hz at T = 1 s), far above the tolerance below.
POLY = dict(rf_to_if_downconv=8.4e9, if_to_channel_downconv=1.0e6, channel_accumulated_phase=0.0,
            channel_phase_polynomial_coefficient_0=0.0, channel_phase_polynomial_coefficient_1=100.5,
            channel_phase_polynomial_coefficient_2=-3.25, channel_phase_polynomial_coefficient_3=0.75)


def _chirp_file(write_prd, seconds):
    """One 16-bit record per second in *seconds*; carrier phase continuous in absolute time."""
    recs = []
    for s in seconds:
        t = s + np.arange(FS) / FS
        ph = 2 * np.pi * (F0 * t + 0.5 * RATE * t**2)
        k_i, k_q = (np.round(12000 * f(ph)).astype(int).tolist() for f in (np.cos, np.sin))
        recs.append(spec.record_bytes(k_i, k_q, sample_size=16, sample_rate=FS,
                                      time_tag_second_of_day=43200 + s, **POLY))
    return write_prd(b"".join(recs))


def _mean_f_dc(t_start, T):
    """Mean §5.2.6 downconversion frequency over [t_start, t_start + T) s from the file start.

    f_DC = F + c1 + 2 c2 tau + 3 c3 tau^2, tau from each record's first sample (records are
    whole seconds here); dense midpoint-rule quadrature, independent of the code's.
    """
    t = t_start + (np.arange(200_000) + 0.5) * T / 200_000
    tau = t - np.floor(t)
    f = 100.5 + 2 * -3.25 * tau + 3 * 0.75 * tau**2  # variable part; fixed part added exactly
    return 8.4e9 + 1.0e6 + f.mean()


def test_spi_off_bin_tone():
    n = np.arange(1000)
    x = np.exp(2j * np.pi * (-0.123456789) * n)
    assert spi_frequency(x, 1.0) == pytest.approx(-0.123456789, abs=1e-10)


def test_frft_chirp_at_midpoint():
    t = np.arange(FS) / FS
    x = np.exp(2j * np.pi * (F0 * t + 0.5 * RATE * t**2))
    f, r = frft_frequency(x, FS, max_rate_hz_s=10.0)
    assert f == pytest.approx(F0 + RATE * 0.5, abs=1e-6)  # at t0 + T/2, not (n-1)/(2 fs)
    assert r == pytest.approx(RATE, abs=1e-4)


def test_max_rate_zero_is_plain_spi():
    n = np.arange(1000)
    x = np.exp(2j * np.pi * 0.123456789 * n)
    assert frft_frequency(x, 1000.0, 0.0) == (spi_frequency(x, 1000.0), 0.0)


@pytest.mark.parametrize("T", [1.0, 0.5, 2.0])
def test_frequency_before_downconversion_at_block_midpoints(T, write_prd):
    pf = read_product(_chirp_file(write_prd, [0, 1, 2, 3]))
    c = extract_carrier(pf, T, max_rate_hz_s=10.0)
    n_blocks = int(4 / T)
    t_mid = (np.arange(n_blocks) + 0.5) * T  # s from the first sample
    assert len(c["time"]) == n_blocks
    np.testing.assert_array_equal(c["time"] - c["time"][0], ((t_mid - t_mid[0]) * 1e9).astype("timedelta64[ns]"))
    assert c["time"][0] == pf.timetags[0] + np.timedelta64(round(T / 2 * 1e9), "ns")
    np.testing.assert_allclose(c["frequency"], F0 + RATE * t_mid, atol=1e-5)
    np.testing.assert_allclose(c["rate"], RATE, atol=1e-3)
    dc = np.array([_mean_f_dc(b * T, T) for b in range(n_blocks)])  # mean over the block, not at t_mid
    np.testing.assert_allclose(c["downconversion_frequency"], dc, rtol=0, atol=4e-6)  # float64 step at 8.4 GHz: 9.5e-7
    np.testing.assert_allclose(c["frequency_before_downconversion"], dc + F0 + RATE * t_mid, rtol=0, atol=1e-5)


@pytest.mark.parametrize("T", [1.0, 0.5])
def test_frequency_before_downconversion_is_the_mean_over_the_block(T, write_prd):
    # One sky signal with curvature (f'' = 0.9 Hz/s^2), recorded twice: steering that
    # tracks it (c1..c3 per record, baseband a pure tone) and a fixed LO (baseband carries
    # the curvature). Truth: mean frequency before downconversion = phase change / T.
    sky = lambda t: 100 * t + 1.25 * t**2 + 0.15 * t**3  # turns above F

    def recorded(steer):
        recs, cum = [], 0.0  # NCO phase continuous across records
        for s in range(2):
            c1, c2, c3 = steer(s)
            tau = np.arange(FS) / FS
            ph = 2 * np.pi * (sky(s + tau) - (cum + c1 * tau + c2 * tau**2 + c3 * tau**3))
            cum += c1 + c2 + c3
            k_i, k_q = (np.round(12000 * f(ph)).astype(int).tolist() for f in (np.cos, np.sin))
            poly = dict(POLY, **{f"channel_phase_polynomial_coefficient_{i}": c for i, c in zip((1, 2, 3), (c1, c2, c3))})
            recs.append(spec.record_bytes(k_i, k_q, sample_size=16, sample_rate=FS, time_tag_second_of_day=43200 + s,
                                          **poly))
        c = extract_carrier(read_product(write_prd(b"".join(recs))), T, max_rate_hz_s=20.0)
        return c["frequency_before_downconversion"] - (8.4e9 + 1.0e6)

    a = np.arange(int(2 / T)) * T
    mean = (sky(a + T) - sky(a)) / T
    tracked = recorded(lambda s: (100 + 2.5 * s + 0.45 * s**2, 1.25 + 0.45 * s, 0.15))
    fixed = recorded(lambda s: (100.0, 0.0, 0.0))
    np.testing.assert_allclose(tracked, mean, rtol=0, atol=1e-5)
    np.testing.assert_allclose(fixed, mean - 0.9 * T**2 / 60, rtol=0, atol=1e-4)  # chirp fit: -f_b'' T^2 / 60


def test_block_across_a_gap_is_nan(write_prd):
    pf = read_product(_chirp_file(write_prd, [0, 1, 3]))  # second 2 missing
    c = extract_carrier(pf, 1.5, max_rate_hz_s=10.0)  # blocks 0-1.5 s, 1.5-(gap)-3.5 s
    assert np.isfinite(c["frequency_before_downconversion"][0]) and np.isnan(c["frequency_before_downconversion"][1])
    assert c["frequency"][0] == pytest.approx(F0 + RATE * 0.75, abs=1e-5)


def test_integration_time_must_be_whole_samples(write_prd):
    pf = read_product(_chirp_file(write_prd, [0]))
    with pytest.raises(ValueError, match="whole number of samples"):
        extract_carrier(pf, 0.0005)


# ---- adaptive rate range ------------------------------------------------------


def _chirp(rate, n=1000):
    t = np.arange(n) / n  # fs = n: T = 1 s, 1 cell = 1 Hz/s
    return np.exp(2j * np.pi * (100 * t + 0.5 * rate * t**2))


def _bank_calls(monkeypatch):
    from rdefpy import carrier
    calls, real = [], carrier._bank_peak

    def spy(x64, rates, *a):
        calls.append(np.asarray(rates) * len(x64) ** 2)  # in cells
        return real(x64, rates, *a)

    monkeypatch.setattr(carrier, "_bank_peak", spy)
    return calls


def test_small_rate_needs_no_widening(monkeypatch):
    calls = _bank_calls(monkeypatch)
    f, r = frft_frequency(_chirp(0.7), 1000.0)
    np.testing.assert_allclose(calls[0], [-2, 0, 2])  # default +-2 Hz/s, 2-cell steps
    assert len(calls) == 1 and r == pytest.approx(0.7, abs=1e-4) and f == pytest.approx(100.35, abs=1e-6)


@pytest.mark.parametrize("rate", [37.0, -37.0])
def test_range_doubles_while_the_winner_is_on_the_edge(rate, monkeypatch):
    calls = _bank_calls(monkeypatch)
    f, r = frft_frequency(_chirp(rate), 1000.0)
    assert r == pytest.approx(rate, abs=1e-4) and f == pytest.approx(100 + rate / 2, abs=1e-6)
    # +-2, then the new outer branches out to +-4, 8, 16, 32, 64 (each computed once)
    edges = sorted(abs(c).max() for c in calls[1::2])
    assert edges == [4, 8, 16, 32, 64] and len(np.unique(np.concatenate(calls))) == 65


@pytest.mark.parametrize("seed", range(5))
def test_range_widens_when_the_winner_is_just_inside_the_edge(seed):
    # 10 s at 1 kHz, 3.5 Hz/s, ~9 dB per sample: the chirp beyond +-2 Hz/s spreads its
    # power, and a noise-lifted branch at 1.8-1.98 Hz/s must not win without widening.
    t = np.arange(10_000) / 1000.0
    rng = np.random.default_rng(seed)
    x = np.exp(2j * np.pi * (37.0 * t + 1.75 * t**2)) + 0.25 * (
        rng.standard_normal(t.size) + 1j * rng.standard_normal(t.size))
    assert frft_frequency(x, 1000.0)[1] == pytest.approx(3.5, abs=0.01)


def test_rate_outside_the_cap_warns():
    with pytest.warns(RuntimeWarning, match="outermost"):
        frft_frequency(_chirp(9.0), 1000.0, 3.0)


def test_extract_carrier_widens_per_block(write_prd):
    recs = []
    for s in range(3):
        t = s + np.arange(FS) / FS
        ph = 2 * np.pi * (F0 * t + 0.5 * -6.0 * t**2)
        k_i, k_q = (np.round(12000 * f(ph)).astype(int).tolist() for f in (np.cos, np.sin))
        recs.append(spec.record_bytes(k_i, k_q, sample_size=16, sample_rate=FS, time_tag_second_of_day=43200 + s,
                                      **dict(POLY, channel_phase_polynomial_coefficient_2=0.0,
                                             channel_phase_polynomial_coefficient_3=0.0)))
    c = extract_carrier(read_product(write_prd(b"".join(recs))), 1.0)
    np.testing.assert_allclose(c["rate"], -6.0, atol=1e-3)
    np.testing.assert_allclose(c["frequency"], F0 - 6.0 * (np.arange(3) + 0.5), atol=1e-5)


# ---- compiled kernels vs the NumPy reference ----------------------------------


def _numpy_only(monkeypatch):
    """Force the NumPy reference path of rdefpy.carrier for the rest of the test."""
    from rdefpy import carrier
    for name in ("_spi_refine", "_frft_refine", "_bank_fill", "_argmax_abs2"):
        monkeypatch.setattr(carrier, name, None)


# -37: the bank widens with a negative step. -0.8: the rate refinement re-centres
# downwards. +-250 Hz/s start: 251 trial rates, i.e. four batches of at most 64 in
# the bank, the winner in the third (71) or the second (-71).
@pytest.mark.parametrize("rate, rate_range", [(0.7, 2.0), (37.0, 2.0), (-37.0, 2.0), (-0.8, 2.0),
                                              (71.0, 250.0), (-71.0, 250.0)])
def test_compiled_kernels_match_numpy(rate, rate_range, monkeypatch):
    from rdefpy import carrier
    if carrier._frft_refine is None:
        pytest.skip("compiled kernels not built")
    rng = np.random.default_rng(3)
    x = _chirp(rate) + 0.3 * (rng.standard_normal(1000) + 1j * rng.standard_normal(1000))
    compiled = frft_frequency(x, 1000.0, rate_range_hz_s=rate_range), spi_frequency(x, 1000.0)
    _numpy_only(monkeypatch)
    calls = _bank_calls(monkeypatch)
    reference = frft_frequency(x, 1000.0, rate_range_hz_s=rate_range), spi_frequency(x, 1000.0)
    assert len(calls[0]) == 2 * round(rate_range / 2) + 1  # trial rates of the first bank
    np.testing.assert_allclose(compiled[0], reference[0], rtol=0, atol=1e-7)  # Hz, Hz/s
    assert compiled[1] == pytest.approx(reference[1], abs=1e-7)
    assert reference[0][1] == pytest.approx(rate, abs=0.2)  # both found the chirp, not the same wrong peak


def test_compiled_final_spi_starts_on_the_fft_grid(monkeypatch):
    """100 kHz, 1 s, noise seed 42: the compiled final SPI starts on the FFT grid
    and agrees with the NumPy path to 1e-7 Hz."""
    from rdefpy import carrier
    if carrier._frft_refine is None:
        pytest.skip("compiled kernels not built")
    n = 100_000
    t = np.arange(n) / n
    rng = np.random.default_rng(42)
    x = np.exp(2j * np.pi * (10_000.0 * t + 1.25 * t * t))  # 10 kHz at the first sample, 2.5 Hz/s
    x += (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
    compiled = frft_frequency(x, float(n))
    _numpy_only(monkeypatch)
    reference = frft_frequency(x, float(n))
    np.testing.assert_allclose(compiled, reference, rtol=0, atol=1e-7)  # Hz, Hz/s
    assert reference[0] == pytest.approx(10_001.25, abs=0.01)  # noise: 2e-3 Hz from the midpoint frequency


@pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "numpy"])
@pytest.mark.parametrize("rate", [-247.0, -71.0, 71.0, 247.0])
def test_bank_of_more_than_64_rates_finds_the_peak_in_any_batch(rate, compiled, monkeypatch):
    """The bank runs in batches of 64 trial rates, each from a fresh base chirp:
    251 rates are four batches, and these four chirps peak one in each."""
    from rdefpy import carrier
    if not compiled:
        _numpy_only(monkeypatch)
    elif carrier._frft_refine is None:
        pytest.skip("compiled kernels not built")
    calls = _bank_calls(monkeypatch)
    f, r = frft_frequency(_chirp(rate), 1000.0, rate_range_hz_s=250.0, max_rate_hz_s=250.0)
    assert len(calls) == 1 and len(calls[0]) == 251  # one bank, no widening
    assert r == pytest.approx(rate, abs=1e-4) and f == pytest.approx(100 + rate / 2, abs=1e-6)


@pytest.mark.parametrize("rate", [-0.8, 0.7, -0.3, 0.2])
def test_numpy_rate_refinement_recentres_on_either_side(rate, monkeypatch):
    """Coarse winner 0 Hz/s (2-cell grid: -2, 0, +2). The first half-step, 1
    cell, finds the neighbour at -1 or +1 higher when the rate is beyond +-0.5
    and moves the stencil there; within +-0.5 it stays. All four must converge."""
    _numpy_only(monkeypatch)
    f, r = frft_frequency(_chirp(rate), 1000.0)
    assert r == pytest.approx(rate, abs=1e-4) and f == pytest.approx(100 + rate / 2, abs=1e-6)


@pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "numpy"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, complex(0, np.nan)])
@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")  # NumPy's, on Inf
def test_block_with_a_non_finite_sample_gives_nan(bad, compiled, monkeypatch):
    """One NaN or Inf among 1000 good samples: no estimate, on every estimator
    (the sample is not skipped, zeroed or interpolated)."""
    from rdefpy import carrier
    if not compiled:
        _numpy_only(monkeypatch)
    elif carrier._frft_refine is None:
        pytest.skip("compiled kernels not built")
    x = _chirp(0.7)
    x[500] = bad
    assert np.isnan(frft_frequency(x, 1000.0)).all()  # rate search
    assert np.isnan(frft_frequency(x, 1000.0, 0.0)).all()  # tone model
    assert np.isnan(spi_frequency(x, 1000.0))


@pytest.mark.parametrize("empty", [np.zeros(0, complex), [], np.zeros(0, np.complex64)])
def test_empty_block_raises(empty):
    for call in (lambda: frft_frequency(empty, 1000.0), lambda: frft_frequency(empty, 1000.0, 0.0),
                 lambda: spi_frequency(empty, 1000.0)):
        with pytest.raises(ValueError, match="must not be empty"):
            call()


def test_compiled_kernels_refuse_an_empty_block():
    from rdefpy import carrier
    if carrier._frft_refine is None:
        pytest.skip("compiled kernels not built")
    empty = np.zeros(0, complex)
    with pytest.raises(ValueError, match="spi_refine: empty x"):
        carrier._spi_refine(empty, 0.0, 0.0, 1.0)
    with pytest.raises(ValueError, match="frft_refine: empty x"):
        carrier._frft_refine(empty, 0.0, 0.0, 1.0, 1.0, 1)


def test_invalid_record_gives_nan(write_prd):
    recs = []
    for s in range(3):
        t = s + np.arange(FS) / FS
        ph = 2 * np.pi * (F0 * t + 0.5 * RATE * t**2)
        k_i, k_q = (np.round(12000 * f(ph)).astype(int).tolist() for f in (np.cos, np.sin))
        recs.append(spec.record_bytes(k_i, k_q, sample_size=16, sample_rate=FS, time_tag_second_of_day=43200 + s,
                                      validity_flag=2 * (s == 1), **POLY))  # any nonzero code
    c = extract_carrier(read_product(write_prd(b"".join(recs))), 1.0)
    assert np.isfinite(c["frequency_before_downconversion"][[0, 2]]).all() and np.isnan(c["frequency_before_downconversion"][1])


def test_zero_block_gives_nan():
    assert np.isnan(frft_frequency(np.zeros(1000, complex), 1000.0)).all()
    assert np.isnan(frft_frequency(np.zeros(1000, complex), 1000.0, 0.0)).all()  # tone mode too
    assert np.isnan(spi_frequency(np.zeros(1000, complex), 1000.0))


def test_tone_near_minus_nyquist_stays_in_band():
    n = np.arange(1000)
    assert spi_frequency(np.exp(2j * np.pi * -0.4999 * n), 1000.0) == pytest.approx(-499.9, abs=1e-6)


@pytest.mark.parametrize("kw, match", [(dict(rate_step_cells=0), "rate_step_cells"),
                                       (dict(zero_padding=0), "zero_padding"),
                                       (dict(zero_padding=2.5), "zero_padding")])
def test_bad_arguments_raise(kw, match):
    with pytest.raises(ValueError, match=match):
        frft_frequency(_chirp(1.0), 1000.0, **kw)


@pytest.mark.parametrize("T", [float("nan"), float("inf"), -1.0])
def test_bad_integration_time_raises(T, write_prd):
    with pytest.raises(ValueError, match="integration_time"):
        extract_carrier(read_product(_chirp_file(write_prd, [0])), T)


def test_file_without_records_is_refused(write_prd):
    from rdefpy import ProductFile
    for empty in (ProductFile(), read_product(_chirp_file(write_prd, [0])).slice(0, 0)):
        with pytest.raises(ValueError, match="no records"):
            extract_carrier(empty, 1.0)


def test_integration_time_longer_than_the_file_gives_no_blocks(write_prd):
    """Samples after the last whole block are not used: with no whole block
    the result is empty, with the fields of any other result."""
    from rdefpy.carrier import CARRIER_DTYPE
    pf = read_product(_chirp_file(write_prd, [0, 1]))
    c = extract_carrier(pf, 2.5)
    assert c.shape == (0,) and c.dtype == CARRIER_DTYPE
    assert extract_carrier(pf, 2.0).shape == (1,)  # exactly the file: one block
    # No block, so no estimator call to notice a bad argument (as documented).
    assert extract_carrier(pf, 2.5, zero_padding=0).shape == (0,)


@pytest.mark.parametrize("kw, match", [(dict(zero_padding=0), "zero_padding must be a positive integer, got 0"),
                                       (dict(zero_padding=1.5), "zero_padding must be a positive integer"),
                                       (dict(rate_step_cells=-1), "rate_step_cells must be positive")])
def test_bad_estimator_arguments_raise_from_the_first_block(kw, match, write_prd):
    """They are checked where they are used, in frft_frequency on a worker
    thread: the error must still reach the caller."""
    pf = read_product(_chirp_file(write_prd, [0, 1]))
    with pytest.raises(ValueError, match=match):
        extract_carrier(pf, 1.0, **kw)


@pytest.mark.filterwarnings("ignore:Metadata fields vary:RuntimeWarning")
def test_records_with_different_sample_rates_are_refused(write_prd):
    recs = [spec.record_bytes([0] * rate, [0] * rate, sample_rate=rate, time_tag_second_of_day=43200 + s)
            for s, rate in enumerate((16, 32))]
    with pytest.raises(ValueError, match=r"different sample rates: \[16, 32\]"):
        extract_carrier(read_product(write_prd(b"".join(recs))), 1.0)


def test_start_epoch_as_datetime64_iso_string_or_naive_datetime(write_prd):
    """The three documented forms of one instant give the same blocks, bit for bit."""
    from datetime import datetime
    pf = read_product(_chirp_file(write_prd, [0, 1, 2, 3]))
    ref = extract_carrier(pf, 1.0, max_rate_hz_s=10.0, start_epoch=np.datetime64("2026-02-08T12:00:01", "ns"))
    assert len(ref) == 3 and ref["time"][0] == np.datetime64("2026-02-08T12:00:01.5")  # midpoint of [1 s, 2 s)
    np.testing.assert_allclose(ref["frequency"], F0 + RATE * (np.arange(3) + 1.5), atol=1e-5)
    for start in (np.datetime64("2026-02-08T12:00:01"), "2026-02-08T12:00:01", "2026-02-08T12:00:01.000000",
                  datetime(2026, 2, 8, 12, 0, 1)):
        np.testing.assert_array_equal(extract_carrier(pf, 1.0, max_rate_hz_s=10.0, start_epoch=start), ref)


def test_start_epoch_takes_the_forms_of_read_product_without_warnings(write_prd):
    """start_epoch goes through the parser of read_product / select: the CCSDS
    day-of-year string and timezone-aware datetimes (converted to UTC) give the
    same blocks as the calendar forms, bit for bit, and nothing warns."""
    from datetime import UTC, datetime, timedelta, timezone
    pf = read_product(_chirp_file(write_prd, [0, 1, 2]))
    ref = extract_carrier(pf, 1.0, max_rate_hz_s=0, start_epoch=np.datetime64("2026-02-08T12:00:01", "ns"))
    assert len(ref) == 2 and ref["time"][0] == np.datetime64("2026-02-08T12:00:01.5")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for start in ("2026-039T12:00:01", "2026-02-08T12:00:01", np.datetime64("2026-02-08T12:00:01"),
                      datetime(2026, 2, 8, 12, 0, 1), datetime(2026, 2, 8, 12, 0, 1, tzinfo=UTC),
                      datetime(2026, 2, 8, 14, 0, 1, tzinfo=timezone(timedelta(hours=2)))):
            np.testing.assert_array_equal(extract_carrier(pf, 1.0, max_rate_hz_s=0, start_epoch=start), ref)


@pytest.mark.parametrize("ps", [250_200_000_000, 50_000], ids=["ccsds-0.2502s", "dsn0222-50ns"])
def test_whole_second_tags_and_carries_the_estimates(ps, write_prd):
    # Records start ps into each second. 0.2502 s (any fraction of a second): the sample
    # nearest to the next whole second is 0.2 ms after it, so the samples' midpoints sit
    # 0.2 ms off the x.500 grid. 50 ns: inside DSN 0222's 0 to 100,000 ps, read without t0_offset.
    off, recs = ps * 1e-12, []
    for s in range(4):
        t = s + off + np.arange(FS) / FS  # s from 12:00:00
        ph = 2 * np.pi * (F0 * t + 0.5 * RATE * t**2)
        k_i, k_q = (np.round(12000 * f(ph)).astype(int).tolist() for f in (np.cos, np.sin))
        recs.append(spec.record_bytes(k_i, k_q, sample_size=16, sample_rate=FS, time_tag_second_of_day=43200 + s,
                                      time_tag_picoseconds_of_the_second=float(ps), **POLY))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # in 0222's range: no picoseconds warning
        kw = {} if ps <= 100_000 else {"t0_offset": 0.0}  # beyond it: spec origin, chosen explicitly
        pf = read_product(write_prd(b"".join(recs)), **kw)
    noon = np.datetime64(pf.timetags[0], "s")
    c = extract_carrier(pf, 1.0, max_rate_hz_s=10.0, whole_second=True)
    t_tag = np.array([1.5, 2.5, 3.5])  # whole seconds + T/2
    np.testing.assert_array_equal(c["time"], noon + (t_tag * 1e9).astype("timedelta64[ns]"))
    # Truth at the tags: rate * dt = 2.5 Hz/s * 0.2 ms = 0.5 mHz if not carried over dt
    np.testing.assert_allclose(c["frequency"], F0 + RATE * t_tag, rtol=0, atol=1e-5)
    t = t_tag[:, None] - 0.5 + (np.arange(200_000) + 0.5) / 200_000  # mean f_DC over the tagged interval
    tau = t - np.floor(t - off)  # from each record's SECOND OF DAY (t0_offset 0, the default)
    dc = 8.4e9 + 1.0e6 + (100.5 + 2 * -3.25 * tau + 3 * 0.75 * tau**2).mean(axis=1)
    np.testing.assert_allclose(c["downconversion_frequency"], dc, rtol=0, atol=4e-6)  # float64 step at 8.4 GHz: 9.5e-7
    np.testing.assert_array_equal(c["frequency_before_downconversion"], c["downconversion_frequency"] + c["frequency"])
    plain = extract_carrier(pf, 1.0, max_rate_hz_s=10.0, start_epoch=noon + np.timedelta64(1, "s"))
    shift_ns = round((off * FS % 1) / FS * 1e9)  # 200,000 (0.2 ms) or 50
    np.testing.assert_array_equal(plain["time"] - c["time"], np.timedelta64(shift_ns, "ns"))  # same samples
    np.testing.assert_allclose(plain["rate"], c["rate"], rtol=0, atol=0)  # same blocks


def test_start_epoch_shifts_the_blocks(write_prd):
    pf = read_product(_chirp_file(write_prd, [0, 1, 2, 3]))
    t0 = pf.timetags[0]
    c = extract_carrier(pf, 1.0, max_rate_hz_s=10.0, start_epoch=t0 + np.timedelta64(500_200, "us"))  # -> sample 500
    t_mid = np.array([1.0, 2.0, 3.0])  # rounded start 0.5 s + T/2
    np.testing.assert_array_equal(c["time"], t0 + (t_mid * 1e9).astype("timedelta64[ns]"))
    np.testing.assert_allclose(c["frequency"], F0 + RATE * t_mid, atol=1e-5)
    np.testing.assert_allclose(c["downconversion_frequency"], [_mean_f_dc(t - 0.5, 1.0) for t in t_mid], rtol=0, atol=1e-6)
    with pytest.raises(ValueError, match="not inside a record"):
        extract_carrier(pf, 1.0, start_epoch=t0 - np.timedelta64(1, "s"))


def test_memory_limit_one_block_in_flight_and_warns(write_prd, monkeypatch):
    import math, threading, time
    import rdefpy.carrier as carrier
    pf = read_product(_chirp_file(write_prd, range(6)))
    free = extract_carrier(pf, 0.5, max_rate_hz_s=10.0, memory_limit=math.inf)
    active, peak, lock, real = [0], [0], threading.Lock(), carrier.frft_frequency

    def counted(*a, **k):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.01)
        try:
            return real(*a, **k)
        finally:
            with lock:
                active[0] -= 1

    monkeypatch.setattr(carrier, "frft_frequency", counted)
    with pytest.warns(RuntimeWarning, match="memory budget"):
        tight = extract_carrier(pf, 0.5, max_rate_hz_s=10.0, memory_limit=1000)
    assert peak[0] == 1
    np.testing.assert_array_equal(tight, free)


def test_memory_limit_inf_never_warns(write_prd):
    import math, warnings
    pf = read_product(_chirp_file(write_prd, range(2)))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        extract_carrier(pf, 1.0, max_rate_hz_s=10.0, memory_limit=math.inf)


def test_default_memory_limit_is_half_the_free_memory(write_prd, monkeypatch):
    """memory_limit=None: budget = _free_bytes() // 2. Free memory for exactly 4
    blocks of 400 bytes per sample allows 2 in flight, on 4 threads (cap 8)."""
    import threading
    import rdefpy.carrier as carrier
    from rdefpy import set_threads
    assert carrier._free_bytes() > 0  # this machine's: free RAM (and cgroup, RLIMIT_AS on POSIX)
    pf = read_product(_chirp_file(write_prd, range(6)))
    free = extract_carrier(pf, 0.5, max_rate_hz_s=10.0, memory_limit=float("inf"))
    monkeypatch.setattr(carrier, "_free_bytes", lambda: 4 * carrier._BYTES_PER_SAMPLE * 500)  # T = 0.5 s at 1 kHz
    active, peak, lock, real = [0], [0], threading.Lock(), carrier.frft_frequency
    pair = threading.Barrier(2, timeout=30)  # 12 blocks pass in pairs only if 2 are in flight at once

    def counted(*a, **k):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        try:
            pair.wait()
            return real(*a, **k)
        finally:
            with lock:
                active[0] -= 1

    monkeypatch.setattr(carrier, "frft_frequency", counted)
    set_threads(4)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # 2 blocks fit: no over-budget warning
            c = extract_carrier(pf, 0.5, max_rate_hz_s=10.0)
    finally:
        set_threads()
    assert peak[0] == 2
    np.testing.assert_array_equal(c, free)


@pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "numpy"])
def test_rate_center_moves_the_search_and_the_tone_model(compiled, monkeypatch):
    """37 Hz/s chirp, truth at the midpoint 100 + 37/2 Hz. Tone model at rate_center
    37: no search, rate = the centre. Search around a centre: the trial rates are
    centre +- 2 Hz/s, and widen around the centre (30: +-2, 4, 8, 16 up to the cap 20)."""
    from rdefpy import carrier
    if not compiled:
        _numpy_only(monkeypatch)
    elif carrier._frft_refine is None:
        pytest.skip("compiled kernels not built")
    x = _chirp(37.0)
    f, r = frft_frequency(x, 1000.0, max_rate_hz_s=0, rate_center_hz_s=37.0)
    assert r == 37.0 and f == pytest.approx(118.5, abs=1e-6)
    calls = _bank_calls(monkeypatch)
    f, r = frft_frequency(x, 1000.0, rate_center_hz_s=36.6)
    np.testing.assert_allclose(calls[0], [34.6, 36.6, 38.6])  # winner 36.6, inside: no widening
    assert len(calls) == 1 and r == pytest.approx(37.0, abs=1e-4) and f == pytest.approx(118.5, abs=1e-6)
    calls.clear()
    f, r = frft_frequency(x, 1000.0, rate_center_hz_s=30.0, max_rate_hz_s=20.0)
    assert r == pytest.approx(37.0, abs=1e-4) and f == pytest.approx(118.5, abs=1e-6)
    np.testing.assert_allclose(calls[0], [28, 30, 32])
    np.testing.assert_allclose(sorted(abs(c - 30).max() for c in calls[1:]), [4, 4, 8, 8, 16, 16])


def test_whole_second_on_a_whole_second_start_changes_nothing(write_prd):
    """Records at picoseconds 0: the first sample is on the grid, so whole_second gives
    the blocks of the default, first tag x.500 of the first second, none dropped."""
    pf = read_product(_chirp_file(write_prd, [0, 1, 2, 3]))
    assert pf[0].header.time_tag_picoseconds_of_the_second == 0.0
    c = extract_carrier(pf, 1.0, max_rate_hz_s=10.0, whole_second=True)
    assert len(c) == 4 and c["time"][0] == np.datetime64("2026-02-08T12:00:00.5")
    np.testing.assert_array_equal(c, extract_carrier(pf, 1.0, max_rate_hz_s=10.0))


@pytest.mark.parametrize("s", [2.5, 2.9994, 4.0, 10.0], ids=["gap", "gap-end", "file-end", "after"])
def test_start_epoch_in_a_gap_or_after_the_file_raises(s, write_prd):
    """Records at 0, 1 and 3 s: a start between 2 s and 3 s - half a sample, or at
    or after 4 s, has no sample to round to."""
    pf = read_product(_chirp_file(write_prd, [0, 1, 3]))
    with pytest.raises(ValueError, match="not inside a record"):
        extract_carrier(pf, 1.0, start_epoch=pf.timetags[0] + np.timedelta64(round(s * 1e9), "ns"))


@pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "numpy"])
def test_iteration_counts_are_used(compiled, monkeypatch):
    """spi: 1 iteration leaves ~4e-7 cycles/sample, the default 5 under 1e-12; a
    threshold above the first update stops there. frft: 0 rate iterations return
    the coarse grid rate (0 of -2, 0, 2 Hz/s) exactly."""
    from rdefpy import carrier
    if not compiled:
        _numpy_only(monkeypatch)
    elif carrier._frft_refine is None:
        pytest.skip("compiled kernels not built")
    x = np.exp(2j * np.pi * -0.123456789 * np.arange(1000))
    one = spi_frequency(x, 1.0, max_iterations=1)
    assert 1e-7 < abs(one + 0.123456789) < 1e-6 and abs(spi_frequency(x, 1.0) + 0.123456789) < 1e-12
    assert spi_frequency(x, 1.0, convergence_threshold=1.0) == one
    assert frft_frequency(_chirp(0.7), 1000.0, rate_iterations=0)[1] == 0.0
    assert abs(frft_frequency(_chirp(0.7), 1000.0, rate_iterations=1)[1] - 0.7) > 1e-4  # default 3: 1e-9
