"""pf.samples / pf.timetags: per-sample values sliced across record boundaries."""

from fractions import Fraction

import numpy as np
import pytest

import rdef_spec as spec
from rdefpy import read_product

RATE = 1000  # 1 kHz: 1000 samples per record, so 1024 crosses a boundary


def _file(write_prd, seconds, bits=16, rate=RATE, **hdr):
    """One record per entry of *seconds* (second of day offsets), random codes."""
    rng = np.random.default_rng(len(seconds) + bits)
    lo, hi = spec.code_range(bits)[0], spec.code_range(bits)[-1]
    raw = b"".join(
        spec.record_bytes(rng.integers(lo, hi + 1, rate).tolist(), rng.integers(lo, hi + 1, rate).tolist(),
                          sample_size=bits, sample_rate=rate, time_tag_second_of_day=43200 + s, **hdr)
        for s in seconds)
    return write_prd(raw)


def _expected_tags(seconds, rate, ps=0.0):
    """Sample times from first principles: second of day + ps + k/rate, exact, rounded to ns."""
    day = np.datetime64("2026-02-08T00:00:00", "ns")  # 2026, day 39 (rdef_spec defaults)
    out = []
    for s in seconds:
        for k in range(rate):
            ns = round(Fraction((43200 + s) * 10**12 + int(ps), 1000) + Fraction(k * 10**9, rate))
            out.append(day + np.timedelta64(ns, "ns"))
    return np.array(out, dtype="datetime64[ns]")


def _concat(pf):
    return np.concatenate([r.data.samples for r in pf])


def _spec_samples(path):
    """Every sample of the file as I + jQ, decoded by rdef_spec alone (no rdefpy)."""
    raw, pos, out = path.read_bytes(), 0, []
    while pos < len(raw):
        hdr = spec.parse_header(raw[pos : pos + spec.HEADER_SIZE])
        i, q = spec.decode_values(raw[pos + spec.HEADER_SIZE : pos + hdr["record_length"]], hdr["sample_size"])
        out.append(np.array(i) + 1j * np.array(q))
        pos += hdr["record_length"]
    return np.concatenate(out)


SLICES = [
    slice(None), slice(0, 0), slice(999, 1001), slice(1000, 2000), slice(500, 2500),
    slice(-1024, None), slice(10, 2900, 7), slice(2900, 10, -3), slice(None, None, -1),
    slice(5000, 6000),
]


@pytest.mark.parametrize("bits", [16, 8, 4, 2])
def test_first_1024_cross_a_record(bits, write_prd, decode_path):
    pf = read_product(_file(write_prd, [0, 1, 2], bits))
    x = pf.samples[:1024]
    assert x.shape == (1024,) and x.dtype == np.complex128
    assert x.tobytes() == _concat(pf)[:1024].tobytes()  # bit for bit


@pytest.mark.parametrize("idx", SLICES)
def test_slices_match_numpy(idx, write_prd, decode_path):
    pf = read_product(_file(write_prd, [0, 1, 2]))
    assert pf.samples[idx].tobytes() == _concat(pf)[idx].tobytes()


@pytest.mark.parametrize("idx", SLICES)
def test_slices_of_on_disk_records_match_the_spec(idx, write_prd, decode_path):
    """The same slices with no record in memory (read_product keeps 1 kHz
    records; release() puts them back on disk): whole records are decoded
    straight into the result, partial ones through a read of their own."""
    path = _file(write_prd, [0, 1, 2])
    pf = read_product(path)
    pf.release()
    assert all(r._data is None for r in pf.records)
    assert pf.samples[idx].tobytes() == _spec_samples(path)[idx].tobytes()
    np.testing.assert_array_equal(pf.timetags[idx], _expected_tags([0, 1, 2], RATE)[idx])
    assert all(r._data is None for r in pf.records)  # read, not kept


@pytest.mark.parametrize("bits", [16, 8, 4, 2, 1])
def test_on_disk_slice_across_records_at_every_sample_size(bits, write_prd, decode_path):
    rate = 1008  # whole 32-bit words at 1 bit too; above the 1 kHz auto-load
    path = _file(write_prd, [0, 1, 2], bits, rate=rate)
    pf = read_product(path, dtype=np.complex64)
    assert all(r._data is None for r in pf.records)
    x = pf.samples[1000:2100]  # end of record 0, all of 1, start of 2
    assert x.dtype == np.complex64
    assert x.tobytes() == _spec_samples(path)[1000:2100].astype(np.complex64).tobytes()


def test_len_index_and_repr(write_prd):
    pf = read_product(_file(write_prd, [0, 1, 2]))
    ref = _concat(pf)
    assert len(pf.samples) == 3000
    assert pf.samples[1000] == ref[1000] and pf.samples[-1] == ref[-1]
    with pytest.raises(IndexError):
        pf.samples[3000]
    assert "3,000 complex samples in 3 records" in repr(pf.samples)


def test_complex64(write_prd):
    path = _file(write_prd, [0, 1])
    x = read_product(path, dtype=np.complex64).samples[:1024]
    assert x.dtype == np.complex64
    assert x.tobytes() == _concat(read_product(path))[:1024].astype(np.complex64).tobytes()


def test_gap_raises_only_when_crossed(write_prd):
    pf = read_product(_file(write_prd, [0, 1, 3]))  # second 2 missing
    ref = _concat(pf)
    assert pf.samples[500:1500].tobytes() == ref[500:1500].tobytes()  # 0 -> 1: fine
    assert pf.samples[2000:3000].tobytes() == ref[2000:3000].tobytes()  # inside record 2
    with pytest.raises(ValueError, match="records 1 and 2"):
        pf.samples[1500:2500]


def test_rate_change_raises_when_crossed(write_prd):
    # Only in memory: a file read from disk has one sample rate.
    pf = read_product(_file(write_prd, [0]))
    pf.records.append(read_product(_file(write_prd, [1], rate=2000)).records[0])
    assert pf.samples[0:1000].tobytes() == pf[0].data.samples.tobytes()
    with pytest.raises(ValueError, match="sample rates"):
        pf.samples[990:1010]


def test_strided_gap_raises(write_prd):
    pf = read_product(_file(write_prd, [0, 1, 3]))
    assert pf.samples[0:2000:7].tobytes() == _concat(pf)[0:2000:7].tobytes()
    with pytest.raises(ValueError, match="records 1 and 2"):
        pf.samples[::7]
    with pytest.raises(ValueError, match="records 1 and 2"):
        pf.samples[::-2500]  # 2999 and 499: nothing from record 1, still crosses


def test_strided_reads_one_record_at_a_time(write_prd):
    import tracemalloc
    rate = 4_000_000
    k = list(range(-8, 8)) * (rate // 16)
    raw = b"".join(spec.record_bytes(k, k, sample_size=4, sample_rate=rate, time_tag_second_of_day=43200 + s)
                   for s in range(4))
    pf = read_product(write_prd(raw))
    def peak(view, n):
        tracemalloc.start()
        try:
            x = view[: n * rate : 1000]
            return tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

    for view in (pf.samples, pf.timetags):
        assert len(view[::1000]) == 4 * rate // 1000
        # Flat in the span: 4 records cost what 1 does (each read is freed).
        assert peak(view, 4) < 1.2 * peak(view, 1)
    # ... and the right ones: sample n holds code k[n % 16] in I and in Q (§5.3.6: 2k+1).
    n = np.arange(0, 4 * rate, 1000)
    v = 2 * (n % 16 - 8) + 1
    np.testing.assert_array_equal(pf.samples[::1000], v + 1j * v)
    noon = np.datetime64("2026-02-08T12:00:00", "ns")  # rdef_spec defaults: 2026, day 39, 43200 s
    np.testing.assert_array_equal(pf.timetags[::1000], noon + n * np.timedelta64(250, "ns"))  # 1 / 4 MHz


def test_asarray_reads_once(write_prd, monkeypatch):
    pf = read_product(_file(write_prd, [0, 1]))
    want_x, want_t = _concat(pf), pf.timetags[:]
    calls = []
    for view in (pf.samples, pf.timetags):  # count reads: one for the whole span, not one per element
        real = type(view)._read
        monkeypatch.setattr(type(view), "_read", lambda self, a, b, real=real: calls.append((a, b)) or real(self, a, b))
    assert np.asarray(pf.samples).tobytes() == want_x.tobytes()
    assert np.array_equal(np.asarray(pf.timetags), want_t)
    assert calls == [(0, 2 * RATE)] * 2
    monkeypatch.undo()
    assert np.asarray(pf.samples, dtype=np.complex64).dtype == np.complex64
    if np.lib.NumpyVersion(np.__version__) >= "2.0.0":  # asarray(copy=) is NumPy 2 only
        with pytest.raises(ValueError, match="copy"):
            np.asarray(pf.samples, copy=False)


def test_unloaded_records_are_not_kept(write_prd, decode_path):
    from rdefpy.prd import reader

    # Above the 1 kHz auto-load, so samples stay on disk; 4-bit is one byte per
    # complex sample, so each data section is exactly the read-ahead threshold.
    rate = reader._READ_AHEAD_MIN_BYTES
    rng = np.random.default_rng(4)
    datas = [rng.integers(0, 256, rate, dtype=np.uint8) for _ in range(4)]
    pf = read_product(write_prd(*(
        spec.header_bytes(sample_size=4, sample_rate=rate, time_tag_second_of_day=43200 + s) + d.tobytes()
        for s, d in enumerate(datas))))
    # Table 5-2, 4-bit: I in the low nibble, Q in the high one; §5.3.6: 2k+1.
    b = np.concatenate(datas).astype(np.int64)
    nib = lambda c: 2 * np.where(c >= 8, c - 16, c) + 1  # noqa: E731
    want = nib(b & 15) + 1j * nib(b >> 4)
    i, q = spec.decode_values(datas[1][:8].tobytes(), 4)  # the shortcut above, held to rdef_spec
    assert want[rate : rate + 8].real.tolist() == i and want[rate : rate + 8].imag.tolist() == q

    # End of record 0 to start of record 3: records 2 and 3 arrive by read-ahead.
    x = pf.samples[rate - 3 : 3 * rate + 3]
    np.testing.assert_array_equal(x, want[rate - 3 : 3 * rate + 3])
    assert all(r._data is None and (r._weak is None or r._weak() is None) for r in pf.records)


def test_views_of_an_empty_product_file():
    from rdefpy import ProductFile

    pf = ProductFile()
    for view, dtype in ((pf.samples, np.complex128), (pf.timetags, "datetime64[ns]"),
                        (pf.downconversion_frequency, np.float64), (pf.downconversion_phase, np.float64)):
        assert len(view) == 0
        assert view[:].shape == (0,) and view[:].dtype == np.dtype(dtype)
        assert np.asarray(view).shape == (0,)
        with pytest.raises(IndexError, match="out of range for 0 samples"):
            view[0]
    assert repr(pf.samples) == "<0 complex samples in 0 records>"


def test_leap_second_record_joins_the_second_before_it_only(write_prd):
    """SECOND OF DAY 86400 (Table 5-1) is one second after 86399, so those two
    join. NumPy has no leap second: 86400 and the next day's second 0 get the
    same datetime64, dt = 0, and samples across them are refused."""
    k = [0] * 16
    path = write_prd(*(spec.record_bytes(k, k, time_tag_doy=doy, time_tag_second_of_day=sod)
                       for doy, sod in ((181, 86399), (181, 86400), (182, 0))))
    pf = read_product(path)
    tags = pf.timetag_array
    assert tags[1] - tags[0] == np.timedelta64(1, "s") and tags[2] == tags[1]
    assert [r.header.time_tag_second_of_day for r in pf] == [86399, 86400, 0]  # the fields keep them apart
    assert pf.samples[8:24].tobytes() == _spec_samples(path)[8:24].tobytes()  # 86399 -> 86400
    with pytest.raises(ValueError, match=r"records 1 and 2, which are not 1 s apart \(dt = 0.0 s\)"):
        pf.samples[24:40]
    np.testing.assert_array_equal(pf.timetags[16:32], pf.timetags[32:48])  # the tags repeat, nothing raised


# ---- pf.timetags -----------------------------------------------------------


@pytest.mark.filterwarnings("ignore:.*outside the allowed range:UserWarning")
def test_timetags_of_a_record_without_a_usable_time_tag_raise(write_prd):
    """Day of year 400 (Table 5-1: 1-366) is read with a warning; it has no
    datetime64, so its sample times are an error, not a wrapped date."""
    k = [0] * 16
    pf = read_product(write_prd(spec.record_bytes(k, k), spec.record_bytes(k, k, time_tag_doy=400)))
    assert len(pf.timetags[:16]) == 16  # record 0 alone: fine
    with pytest.raises(ValueError, match="record 1 has no usable time tag"):
        pf.timetags[:]
    with pytest.raises(ValueError, match="records 0 and 1, which are not 1 s apart"):
        pf.samples[:]  # no time tag: cannot be shown contiguous
    assert pf.samples[16:].tobytes() == pf[1].data.samples.tobytes()  # within the record: fine


def test_timetags_cross_a_record(write_prd):
    pf = read_product(_file(write_prd, [0, 1, 2]))
    t = pf.timetags[:1024]
    assert t.dtype == np.dtype("datetime64[ns]") and t.shape == (1024,)
    np.testing.assert_array_equal(t, _expected_tags([0, 1, 2], RATE)[:1024])
    assert t[1000] - t[999] == np.timedelta64(1, "ms")  # boundary: no jump
    assert len(pf.timetags) == len(pf.samples) == 3000
    # Tags come from the headers: records above 1 kHz stay on disk, unread.
    fast = read_product(_file(write_prd, [0, 1], rate=2000))
    assert fast.timetags[1999:2001].tolist() == _expected_tags([0, 1], 2000)[1999:2001].tolist()
    assert all(r._data is None and r._weak is None for r in fast.records)


def test_timetags_exact_at_a_rate_that_does_not_divide_1e9(write_prd):
    rate = 3000  # 1/3000 s = 333333.33 ns: float k/fs would jitter by 1 ns
    pf = read_product(_file(write_prd, [0, 1], rate=rate))
    np.testing.assert_array_equal(pf.timetags[:], _expected_tags([0, 1], rate))


def test_timetags_include_the_picoseconds(write_prd):
    ps = 250_000.0  # 250 ns after the second
    pf = read_product(_file(write_prd, [0, 1], time_tag_picoseconds_of_the_second=ps), t0_offset=0.0)
    np.testing.assert_array_equal(pf.timetags[995:1005], _expected_tags([0, 1], RATE, ps)[995:1005])
    assert pf.timetags[0] == pf.records[0].header.timetag


@pytest.mark.parametrize("idx", [slice(None), slice(999, 1001), slice(10, 2900, 7), slice(None, None, -1), 1500, -1])
def test_timetags_index_like_numpy(idx, write_prd):
    pf = read_product(_file(write_prd, [0, 1, 2]))
    np.testing.assert_array_equal(pf.timetags[idx], _expected_tags([0, 1, 2], RATE)[idx])


def test_timetags_show_a_gap_instead_of_raising(write_prd):
    pf = read_product(_file(write_prd, [0, 1, 3]))
    t = pf.timetags[1998:2002]
    assert t[2] - t[1] == np.timedelta64(1001, "ms")  # 1.999 s -> 3.000 s: second 2 missing
    np.testing.assert_array_equal(t, _expected_tags([0, 1, 3], RATE)[1998:2002])


# ---- pf.downconversion_frequency / pf.downconversion_phase -----------------

POLY = dict(rf_to_if_downconv=8.4e9, if_to_channel_downconv=1.0e6, channel_accumulated_phase=12345.25,
            channel_phase_polynomial_coefficient_0=0.125, channel_phase_polynomial_coefficient_1=100.5,
            channel_phase_polynomial_coefficient_2=-3.25, channel_phase_polynomial_coefficient_3=0.75)


@pytest.mark.parametrize("attr", ["downconversion_frequency", "downconversion_phase"])
@pytest.mark.parametrize("idx", [slice(None), slice(990, 1034), slice(10, 2900, 7), slice(None, None, -1), 1500, -1])
def test_downconversion_matches_records(attr, idx, write_prd):
    pf = read_product(_file(write_prd, [0, 1, 2], **POLY))
    ref = np.concatenate([getattr(r, attr) for r in pf])
    got = getattr(pf, attr)[idx]
    assert np.asarray(got).tobytes() == np.asarray(ref[idx]).tobytes()  # bit for bit


def test_downconversion_is_header_only_and_per_record_values_kept(write_prd, monkeypatch):
    from rdefpy.prd import reader
    pf = read_product(_file(write_prd, [0, 1, 2], rate=4_000_000 // 1000 * 1000, **POLY))
    monkeypatch.setattr(reader, "_decode_iq", lambda *a: pytest.fail("decoded samples"))
    monkeypatch.setattr(reader, "_decode_complex", lambda *a: pytest.fail("decoded samples"))
    f = pf.downconversion_frequency[3_999_990:4_000_010]
    assert f.shape == (20,) and len(pf.downconversion_phase) == len(pf.samples)
    np.testing.assert_array_equal(pf.record_downconversion_frequency,
                                  [pf.downconversion_frequency[k * 4_000_000] for k in range(3)])
    np.testing.assert_array_equal(pf.record_downconversion_phase,
                                  [pf.downconversion_phase[k * 4_000_000] for k in range(3)])
