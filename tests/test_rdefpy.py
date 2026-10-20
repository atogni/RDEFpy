"""Tests for rdefpy -- CCSDS 506.1-B-1 RDEF reader/writer.

The decisive test here is `test_decode_matches_reference_digests`. The
low-level decode must stay bit-exact: REFERENCE below holds SHA-256
digests of the decoded I/Q of each fixture file, so any change to the
decoded values fails it.

If one of those digests ever fails, the decode drifted. Do not update the
digest to make it pass -- find out what changed in the decode.

Fixtures in tests/data/ are simulated example products, one per sample
size (16, 8, 4, 2, 1 bits), two records each at 2000 sps.
"""

import copy
import gc
import hashlib
import io
import os
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pytest

from rdefpy import (
    ProductFile,
    Record,
    RecordData,
    RecordHeader,
    read_headers,
    read_product,
    scan_product,
    write_product,
)
from rdefpy.prd import _binary as binary

DATA = Path(__file__).parent / "data"

# Reference digests of the decoded I/Q. See module docstring.
REFERENCE = {
    "EXMPn001tSsDS24r01c01-26039120000.prd": dict(
        sample_size=16, sample_rate=2000, n_records=2,
        dtype="int32", n_samples=4000,
        sha256="183d838b1e2f62948b8cbfcd70973036dfd895d5abd36cae6708d469422eb8d0",
        i_head=[65535, 62327, 53019, 38521, 20251, 1],
        q_head=[1, 20251, 38521, 53019, 62327, 65535],
    ),
    "EXMPn001tSsDS24r01c02-26039120000.prd": dict(
        sample_size=8, sample_rate=2000, n_records=2,
        dtype="int16", n_samples=4000,
        sha256="a28deb832aeaa65cfe3b4440aacfd39afa92eea1a9e973438504d979da41e02d",
        i_head=[255, 243, 207, 149, 79, 1],
        q_head=[1, 79, 149, 207, 243, 255],
    ),
    "EXMPn001tSsDS24r01c03-26039120000.prd": dict(
        sample_size=4, sample_rate=2000, n_records=2,
        dtype="int8", n_samples=4000,
        sha256="24fefdc0e7a3e9a579dd5a3815e74521761bb195af96a72523f690fc3e2bb40b",
        i_head=[15, 15, 13, 9, 5, 1],
        q_head=[1, 5, 9, 13, 15, 15],
    ),
    "EXMPn001tSsDS24r01c04-26039120000.prd": dict(
        sample_size=2, sample_rate=2000, n_records=2,
        dtype="int8", n_samples=4000,
        sha256="1c4f0ef92fe2ebcff193ccec08a6a3df6f3ef1c567dca9bf0d2b8d88ef1e4053",
        i_head=[3, 3, 3, 1, 1, 1],
        q_head=[1, 1, 1, 3, 3, 3],
    ),
    "EXMPn001tSsDS24r01c05-26039120000.prd": dict(
        sample_size=1, sample_rate=2000, n_records=2,
        dtype="int8", n_samples=4000,
        sha256="83600bd386a23e2dd74e60707b9a77de3930704568f74f1eecb3493a1894631c",
        i_head=[1, 1, 1, 1, 1, 1],
        q_head=[1, 1, 1, 1, 1, 1],
    ),
}
FIXTURES = sorted(REFERENCE)
PRD_16 = DATA / FIXTURES[0]


# Integer width the digests are taken at, per sample size. 4- and 2-bit
# samples are stored as int8 (the smallest exact type) but hashed as int16,
# so the pin is about VALUES, not storage.
DIGEST_DTYPE = {16: np.int32, 8: np.int16, 4: np.int16, 2: np.int16, 1: np.int8}


def iq_digest(pf):
    """sha256 over every I and Q array in the file, in record order."""
    h = hashlib.sha256()
    n = 0
    for rec in pf.records:
        dt = DIGEST_DTYPE[rec.data.sample_size]
        h.update(rec.data.i_samples.astype(dt).tobytes())
        h.update(rec.data.q_samples.astype(dt).tobytes())
        n += len(rec.data)
    return h.hexdigest(), n


# =============================================================================
# The decode is pinned
# =============================================================================


@pytest.mark.parametrize("name", FIXTURES)
def test_decode_matches_reference_digests(name):
    """Decoded I/Q matches the reference digests, for all five sample sizes.

    Covers the §5.3 bit unpacking and the §5.3.6 ``2k+1`` transform
    together: a sign-extension or bit-offset error in any of the five
    packings changes the digest.
    """
    ref = REFERENCE[name]
    pf = read_product(DATA / name)

    assert len(pf.records) == ref["n_records"]
    hdr = pf.records[0].header
    assert hdr.sample_size == ref["sample_size"]
    assert hdr.sample_rate == ref["sample_rate"]
    assert str(pf.records[0].data.i_samples.dtype) == ref["dtype"]

    sha, n = iq_digest(pf)
    assert n == ref["n_samples"]
    assert sha == ref["sha256"], (
        f"{name}: decoded I/Q differs from the reference digest. "
        "The low-level decode drifted -- do not update this digest."
    )
    assert pf.records[0].data.i_samples[:6].tolist() == ref["i_head"]
    assert pf.records[0].data.q_samples[:6].tolist() == ref["q_head"]


@pytest.mark.parametrize("name", FIXTURES)
def test_every_sample_is_an_odd_integer(name):
    """§5.3.6 encodes ``y = 2k+1``, so every decoded sample is odd.

    A scaling, centring, or float conversion sneaking into the decode path
    breaks this immediately.
    """
    pf = read_product(DATA / name)
    for rec in pf.records:
        assert np.all(rec.data.i_samples % 2 != 0)
        assert np.all(rec.data.q_samples % 2 != 0)


@pytest.mark.parametrize("bits,lo,hi", [(16, -32768, 32767), (8, -128, 127),
                                        (4, -8, 7), (2, -2, 1)])
def test_unpack_covers_the_full_signed_range(bits, lo, hi):
    """Sign extension is right at both ends of each bit depth.

    Builds every representable code for the small depths (and the extremes
    for 16-bit) and checks the decode lands on ``2k+1``.
    """
    ks = np.array([lo, lo + 1, -1, 0, 1, hi], dtype=np.int64)
    if bits == 16:
        raw = np.empty(2 * ks.size, dtype="<i2")
        raw[0::2] = ks
        raw[1::2] = ks[::-1]
        i, q = binary.unpack_iq_16(raw.tobytes())
    elif bits == 8:
        raw = np.empty(2 * ks.size, dtype="<i1")
        raw[0::2] = ks
        raw[1::2] = ks[::-1]
        i, q = binary.unpack_iq_8(raw.tobytes())
    elif bits == 4:
        # one I/Q pair per byte: I low nibble, Q high nibble
        i_n = np.where(ks < 0, ks + 16, ks).astype(np.uint8)
        q_n = np.where(ks[::-1] < 0, ks[::-1] + 16, ks[::-1]).astype(np.uint8)
        i, q = binary.unpack_iq_4((i_n | (q_n << 4)).astype(np.uint8).tobytes())
    else:  # 2-bit, 4 pairs per 16-bit word
        ks = np.array([lo, -1, 0, hi], dtype=np.int64)
        i_c = (np.where(ks < 0, ks + 4, ks).astype(np.uint16)) & 0x3
        q_c = i_c[::-1]
        word = np.uint16(0)
        for s in range(4):
            word |= i_c[s] << (4 * s)
            word |= q_c[s] << (4 * s + 2)
        i, q = binary.unpack_iq_2(np.array([word], dtype="<u2").tobytes())

    assert i.tolist() == (2 * ks + 1).tolist()
    assert q.tolist() == (2 * ks[::-1] + 1).tolist()


def test_unpack_1bit_maps_bits_to_plus_minus_one():
    """§5.3.6 for 1-bit: bit 0 -> +1, bit 1 -> -1 (LSB-first in the byte)."""
    # 0b01010101 -> alternating bits, LSB first: 1,0,1,0,...
    i, q = binary.unpack_iq_1(bytes([0b01010101]))
    assert i.tolist() == [-1, -1, -1, -1]
    assert q.tolist() == [1, 1, 1, 1]
    assert i.dtype == np.int8


def test_empty_data_returns_empty_arrays_of_the_right_dtype():
    for fn, dt in ((binary.unpack_iq_16, np.int32), (binary.unpack_iq_8, np.int16),
                   (binary.unpack_iq_4, np.int8), (binary.unpack_iq_2, np.int8),
                   (binary.unpack_iq_1, np.int8)):
        i, q = fn(b"")
        assert i.size == 0 and q.size == 0
        assert i.dtype == dt and q.dtype == dt


# =============================================================================
# Round trip: writer inverts reader
# =============================================================================


@pytest.mark.parametrize("loaded", [True, False], ids=["packed", "verbatim"])
@pytest.mark.parametrize("name", FIXTURES)
def test_write_then_read_is_lossless(name, loaded, tmp_path):
    """``pack_record_data`` is the exact inverse of ``unpack_iq_N``.

    The §5.3.6 bias inversion and the bit packing have to round-trip for
    every sample size, or a corrected file silently loses samples.  Loaded
    records go through the packer; unloaded ones are copied verbatim.
    """
    pf = read_product(DATA / name)
    if loaded:
        pf.load()
    out = tmp_path / name
    write_product(out, pf)

    back = read_product(out)
    assert iq_digest(back) == iq_digest(pf)
    assert out.stat().st_size == (DATA / name).stat().st_size


@pytest.mark.parametrize("loaded", [True, False], ids=["packed", "verbatim"])
@pytest.mark.parametrize("name", FIXTURES)
def test_written_bytes_match_the_original_file(name, loaded, tmp_path):
    """A read/write cycle reproduces the file byte for byte, header included."""
    pf = read_product(DATA / name)
    if loaded:
        pf.load()
    out = tmp_path / "rt.prd"
    write_product(out, pf)
    assert out.read_bytes() == (DATA / name).read_bytes()


def test_verbatim_write_keeps_header_edits_and_checks(tmp_path, decodes):
    """Unloaded records are copied without decoding; header edits still apply."""
    pf = read_product(PRD_16)
    pf.correct_header(validity_flag=1)
    out = tmp_path / "v.prd"
    write_product(out, pf)
    assert decodes == []
    back = read_product(out)
    assert all(r.header.validity_flag == 1 for r in back)
    assert iq_digest(back) == iq_digest(read_product(PRD_16))

    pf[0].header.sample_rate = 1000  # samples no longer match the header
    with pytest.raises(ValueError, match="does not match"):
        write_product(tmp_path / "bad.prd", pf)


def test_write_over_its_own_on_disk_source(tmp_path):
    """Records still read from the target while it is replaced (temp + os.replace)."""
    path = tmp_path / PRD_16.name
    path.write_bytes(PRD_16.read_bytes())
    write_product(path, read_product(path))
    assert path.read_bytes() == PRD_16.read_bytes()
    assert [p.name for p in tmp_path.iterdir()] == [path.name]  # no temp left


def test_progress_bars_run(tmp_path):
    pf = read_product(PRD_16)
    pf.load(progress=True)
    write_product(tmp_path / "p.prd", pf, progress=True)
    assert (tmp_path / "p.prd").read_bytes() == PRD_16.read_bytes()


def test_write_refuses_a_header_only_product(tmp_path):
    pf = read_headers(PRD_16)
    with pytest.raises(ValueError, match="no I/Q data"):
        write_product(tmp_path / "x.prd", pf)


def test_pack_header_rejects_non_finite_downconversion():
    """§5.2 float64 fields carry no NaN/Inf; a bad value must not reach disk."""
    from rdefpy.prd.writer import pack_record_header

    pf = read_product(PRD_16)
    hdr = pf.records[0].header
    hdr.rf_to_if_downconv = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        pack_record_header(hdr)


# =============================================================================
# Reader modes agree
# =============================================================================


def test_on_access_decodes_the_same_samples_as_loaded():
    loaded = read_product(PRD_16)
    loaded.load()
    on_access = read_product(PRD_16)
    assert iq_digest(on_access) == iq_digest(loaded)


def test_records_hold_no_iq_until_loaded():
    pf = read_product(PRD_16)
    rec = pf.records[0]
    assert rec._data is None
    _ = rec.data.samples          # read on access ...
    assert rec._data is None      # ... and not kept
    rec.load()
    assert rec._data is not None
    rec.release()  # disk-backed, so it can re-derive
    assert rec._data is None
    assert len(rec.data) == 2000


@pytest.fixture
def decodes(monkeypatch):
    """Count sample decodes from disk."""
    from rdefpy.prd import reader

    calls = []
    real = reader._decode_iq
    monkeypatch.setattr(reader, "_decode_iq", lambda *a: calls.append(1) or real(*a))
    return calls


def test_on_disk_read_is_shared_while_held_and_freed_after(decodes):
    rec = read_product(PRD_16).records[0]  # 2 kHz: on access
    d = rec.data
    assert rec.data is d and len(decodes) == 1
    del d
    gc.collect()
    assert rec._weak() is None and rec._data is None
    rec.data
    assert len(decodes) == 2


def test_on_disk_samples_are_read_only_and_loaded_ones_writeable():
    rec = read_product(PRD_16).records[0]
    d = rec.data
    for arr in (d.i_samples, d.q_samples):
        with pytest.raises(ValueError, match="read-only"):
            arr[0] = 5
    rec.load()
    rec.data.i_samples[0] = 5
    rec.data.q_samples[0] = 5
    assert rec.data.i_samples[0] == rec.data.q_samples[0] == 5


def test_len_and_bool_do_not_read_the_disk(decodes):
    rec = read_product(PRD_16).records[0]
    assert bool(rec) and len(rec) == 2000
    assert decodes == [] and rec._data is None


@pytest.mark.parametrize("how", ["copy", "deepcopy", "pickle"])
def test_copying_on_disk_records_decodes_nothing(how, decodes):
    pf = read_product(PRD_16)
    held = pf.records[0].data  # a live weak reference must not be copied
    decodes.clear()
    dup = {"copy": pf.copy,
           "deepcopy": lambda: copy.deepcopy(pf),
           "pickle": lambda: pickle.loads(pickle.dumps(pf))}[how]()
    assert decodes == []
    assert all(r._data is None for r in pf.records + dup.records)
    assert all(r._weak is None for r in dup.records)
    assert iq_digest(dup) == iq_digest(pf)
    del held


def test_copy_of_a_loaded_record_owns_its_samples():
    pf = read_product(PRD_16)
    pf.load()
    dup = pf.copy()
    dup.records[0].data.i_samples[0] = 5
    assert pf.records[0].data.i_samples[0] != 5


def test_read_headers_skips_data():
    pf = read_headers(PRD_16)
    assert len(pf.records) == 2
    assert pf.records[0].header.sample_rate == 2000
    assert len(pf.records[0].data) == 0


def test_scan_product_counts_records():
    count, consistent = scan_product(PRD_16)
    assert count == 2
    assert consistent is True


def test_complex64_is_lossless_for_2k_plus_1():
    """complex64 halves bandwidth; the 2k+1 integers stay exact in float32."""
    c128 = read_product(PRD_16)
    c64 = read_product(PRD_16, dtype=np.complex64)
    x64 = c64.records[0].data.samples
    x128 = c128.records[0].data.samples
    assert x64.dtype == np.complex64
    assert np.array_equal(x64.astype(np.complex128), x128)


def test_bad_dtype_rejected_at_call_time():
    with pytest.raises(ValueError, match="complex64 or np.complex128"):
        read_product(PRD_16, dtype=np.float64)


# =============================================================================
# Filtering
# =============================================================================


def test_filter_on_header_field():
    sid = read_headers(PRD_16).records[0].header.station_id
    assert len(read_product(PRD_16, filters={"station_id": sid}).records) == 2
    assert len(read_product(PRD_16, filters={"station_id": sid + 1}).records) == 0


def test_unknown_filter_field_rejected():
    with pytest.raises(ValueError, match="Unknown header field"):
        read_product(PRD_16, filters={"not_a_field": 1})


@pytest.mark.parametrize("progress", [False, True])
def test_stop_epoch_ends_the_read_early(progress):
    """stop_epoch breaks out mid-file; the early exit must not crash.

    Regression: the early-break path filled the progress bar via tqdm-only
    attributes, so any stop_epoch raised AttributeError whether or not a bar
    was drawn.
    """
    t = read_headers(PRD_16).timetag_array
    pf = read_product(PRD_16, stop_epoch=t[1], progress=progress)
    assert len(pf.records) == 1
    assert pf.records[0].header.timetag == t[0]


def test_epoch_window_is_half_open():
    """start_epoch <= timetag < stop_epoch: the start boundary is kept, the stop one is not."""
    t = read_headers(PRD_16).timetag_array
    assert len(read_product(PRD_16, start_epoch=t[0]).records) == 2
    assert len(read_product(PRD_16, start_epoch=t[1]).records) == 1
    assert len(read_product(PRD_16, stop_epoch=t[1]).records) == 1
    assert len(read_product(PRD_16, start_epoch=t[0], stop_epoch=t[0]).records) == 0
    # One nanosecond past the first record drops it.
    after = t[0] + np.timedelta64(1, "ns")
    assert len(read_product(PRD_16, start_epoch=after).records) == 1


def test_read_product_and_select_agree_on_epoch_bounds():
    """Same keywords, same meaning: tiled windows partition the file exactly."""
    pf = read_product(PRD_16)
    t = pf.timetag_array
    for lo, hi in ((t[0], t[1]), (t[1], t[1] + np.timedelta64(1, "s")), (t[0], t[0])):
        read = read_product(PRD_16, start_epoch=lo, stop_epoch=hi)
        sel = pf.select(start_epoch=lo, stop_epoch=hi)
        assert [r.header.timetag for r in read] == [r.header.timetag for r in sel]
    first = pf.select(stop_epoch=t[1])
    rest = pf.select(start_epoch=t[1])
    assert len(first) + len(rest) == len(pf)


def test_epoch_window_accepts_strings_and_datetimes():
    """CCSDS day-of-year, ISO-8601 and datetime all resolve to the same window."""
    from datetime import UTC, datetime as dt

    assert len(read_product(PRD_16, stop_epoch="2026-039T12:00:01").records) == 1
    assert len(read_product(PRD_16, stop_epoch="2026-02-08T12:00:01").records) == 1
    assert len(
        read_product(PRD_16, stop_epoch=dt(2026, 2, 8, 12, 0, 1, tzinfo=UTC)).records
    ) == 1


def test_num_records_caps_the_read():
    assert len(read_product(PRD_16, num_records=1).records) == 1


def test_band_and_explicit_window_are_mutually_exclusive():
    with pytest.raises(ValueError, match="not both"):
        read_product(PRD_16, band="X", min_downconv_hz=1.0)


def test_unknown_band_rejected():
    with pytest.raises(ValueError, match="Unknown band"):
        read_product(PRD_16, band="Q")


def test_select_returns_a_new_file_and_leaves_the_original():
    pf = read_product(PRD_16)
    sid = pf.records[0].header.station_id
    sub = pf.select(filters={"station_id": sid + 99})
    assert sub is not pf
    assert len(sub.records) == 0
    assert len(pf.records) == 2
    kept = pf.select(filters={"station_id": sid})
    assert len(kept) == 2 and kept.records[0] is not pf.records[0]


def test_slice_is_by_index_only():
    pf = read_product(PRD_16)
    assert len(pf.slice(0).records) == 1
    assert len(pf.slice(0, 2).records) == 2
    assert len(pf.slice(np.int64(1)).records) == 1
    with pytest.raises(TypeError):
        pf.slice(pf.records[0].header.timetag)
    with pytest.raises(TypeError):
        pf.slice("2026-039T12:00:00")


# =============================================================================
# Header correction
# =============================================================================


def test_correct_header_propagates():
    pf = read_product(PRD_16)
    pf.correct_header(station_id=42, spacecraft_id=7)
    assert all(r.header.station_id == 42 for r in pf.records)
    assert all(r.header.spacecraft_id == 7 for r in pf.records)


def test_correct_header_rejects_unknown_field():
    pf = read_product(PRD_16)
    with pytest.raises(ValueError, match="Unknown header field"):
        pf.correct_header(nonsense=1)


def test_single_record_edit():
    pf = read_product(PRD_16)
    pf.records[0].correct_header(validity_flag=1)
    assert pf.records[0].header.validity_flag == 1


def test_timetag_is_nanosecond_datetime64():
    pf = read_headers(PRD_16)
    tt = pf.timetag_array
    assert tt.dtype == np.dtype("datetime64[ns]")
    assert len(tt) == 2
    # Records are one second apart (§5.1).
    assert (tt[1] - tt[0]) == np.timedelta64(1, "s")


# =============================================================================
# Downconversion reconstruction (§5.2.6)
# =============================================================================


def test_record_downconversion_frequency_follows_the_polynomial():
    """f_DC(t) = (rf_to_if + if_to_chan) + c1 + 2*c2*t + 3*c3*t^2 (§5.2.6)."""
    rec = Record(
        RecordHeader({
            "sample_rate": 4,
            "sample_size": 16,
            "rf_to_if_downconv": 8.4e9,
            "if_to_channel_downconv": 1.0e6,
            "channel_phase_polynomial_coefficient_1": 100.0,
            "channel_phase_polynomial_coefficient_2": 10.0,
            "channel_phase_polynomial_coefficient_3": 1.0,
        }),
        RecordData(16, 4, np.ones(4, np.int32), np.ones(4, np.int32)),
    )
    t = np.arange(4) / 4.0
    expected = 8.4e9 + 1.0e6 + 100.0 + 2 * 10.0 * t + 3 * 1.0 * t**2
    assert np.allclose(rec.downconversion_frequency, expected)


def test_record_downconversion_phase_follows_the_polynomial():
    """phi(t) = Phi + c0 + c1*t + c2*t^2 + c3*t^3 (§5.2.6)."""
    rec = Record(
        RecordHeader({
            "sample_rate": 4,
            "sample_size": 16,
            "channel_accumulated_phase": 5.0,
            "channel_phase_polynomial_coefficient_0": 0.25,
            "channel_phase_polynomial_coefficient_1": 100.0,
            "channel_phase_polynomial_coefficient_2": 10.0,
            "channel_phase_polynomial_coefficient_3": 1.0,
        }),
        RecordData(16, 4, np.ones(4, np.int32), np.ones(4, np.int32)),
    )
    t = np.arange(4) / 4.0
    expected = 5.0 + 0.25 + 100.0 * t + 10.0 * t**2 + 1.0 * t**3
    assert np.allclose(rec.downconversion_phase, expected)


def test_productfile_downconversion_arrays_are_header_only(monkeypatch):
    """Per-record values must not read any samples from disk."""
    from rdefpy.prd import reader

    pf = read_product(PRD_16)
    calls = []
    real = reader._decode_iq
    monkeypatch.setattr(reader, "_decode_iq", lambda *a: calls.append(1) or real(*a))
    f = pf.downconversion_frequency
    p = pf.downconversion_phase
    assert f.shape == (2,) and p.shape == (2,)
    assert calls == []
    assert all(r._data is None for r in pf.records)


# =============================================================================
# RecordData
# =============================================================================


def test_saturation_detected_at_the_bit_depth_limit():
    d = RecordData(8, 4, np.array([255, 1, 1, 1], np.int16), np.ones(4, np.int16))
    assert d.is_saturated is True
    assert d.saturation_ratio == pytest.approx(0.25)


def test_saturation_is_undefined_for_one_bit():
    d = RecordData(1, 2, np.array([1, -1], np.int8), np.array([1, -1], np.int8))
    with pytest.warns(UserWarning, match="undefined for 1-bit"):
        assert d.is_saturated is False
    with pytest.warns(UserWarning, match="undefined for 1-bit"):
        assert d.saturation_ratio == 0.0


def test_power_and_rms_agree():
    d = RecordData(16, 4, np.full(4, 3, np.int32), np.full(4, 4, np.int32))
    assert d.power == pytest.approx(25.0)
    assert d.rms == pytest.approx(5.0)
    assert d.energy == pytest.approx(100.0)


def test_samples_is_exactly_i_plus_jq():
    d = RecordData(16, 2, np.array([1, 3], np.int32), np.array([5, 7], np.int32))
    assert np.array_equal(d.samples, np.array([1 + 5j, 3 + 7j]))


def test_operations_needing_sample_rate_say_so():
    d = RecordData(16, None, np.ones(2, np.int32), np.ones(2, np.int32))
    with pytest.raises(ValueError, match="sample_rate is required"):
        _ = d.duration_s


def test_mismatched_iq_shapes_rejected():
    with pytest.raises(ValueError, match="same shape"):
        RecordData(16, 2, np.ones(2, np.int32), np.ones(3, np.int32))


def test_invalid_sample_size_rejected():
    with pytest.raises(ValueError, match=r"1, 2, 4, 8, 16"):
        RecordData(sample_size=3)


# =============================================================================
# Reporting helpers do not crash
# =============================================================================


def test_summaries_render():
    pf = read_product(PRD_16)
    assert "ProductFile" in str(pf)
    assert "band" in str(pf)
    assert "timetag" in str(pf[0])
    assert "RecordHeader" in str(pf.records[0].header)
    assert "RecordData" in str(pf.records[0].data)


def test_printing_an_on_disk_file_decodes_nothing(decodes):
    pf = read_product(PRD_16)
    rec = pf[0]
    for obj in (pf, rec, rec.header):
        str(obj), repr(obj)
    assert decodes == [] and rec._data is None


def test_titles_are_class_name_and_identifier():
    pf = read_product(PRD_16)
    first = lambda obj: str(obj).splitlines()[0].strip()  # noqa: E731
    assert first(pf) == f"ProductFile {pf.name}"
    assert first(pf[1]) == "Record 1"
    assert first(pf[0].header) == "RecordHeader"
    assert first(pf[0].data) == "RecordData"


def test_record_summary_reports_samples_and_loaded_state():
    rec = read_product(PRD_16)[0]
    rows = lambda: dict(ln.strip().split(": ", 1) for ln in str(rec).splitlines()[1:])  # noqa: E731
    assert rows()["samples"] == "2,000" and rows()["loaded"] == "False"
    rec.load()
    assert rows()["loaded"] == "True"


def _every_display_object():
    pf = read_product(PRD_16)
    return [pf, pf[0], pf[0].header, pf[0].data, ProductFile()]


def test_str_and_repr_are_ascii():
    """Printing must work on any console or redirected stream codec.

    The path row is the user's own file system, not library text, so it is
    left out: a checkout under a non-ASCII directory must not fail this.
    """
    for obj in _every_display_object():
        for text in (str(obj), repr(obj)):
            text = "\n".join(ln for ln in text.splitlines() if not ln.strip().startswith("path:"))
            assert text.isascii(), (type(obj).__name__, text)


def test_record_data_summary_makes_no_pass_over_the_samples(monkeypatch):
    def boom(self):
        raise AssertionError("str(RecordData) touched the samples")

    for name in ("samples", "power", "power_db", "energy", "energy_db", "peak_amplitude",
                 "rms", "is_saturated", "saturation_ratio", "time_array"):
        monkeypatch.setattr(RecordData, name, property(boom))
    d = RecordData(16, 4, np.ones(4, np.int32), np.ones(4, np.int32))
    rows = [ln.split(":")[0].strip() for ln in str(d).splitlines()[1:]]
    assert rows == ["samples", "sample_size", "sample_rate", "duration_s", "dtype", "complex_dtype"]


@pytest.mark.parametrize("value,shown", [
    (1e-7, "1e-07"), (-1e-7, "-1e-07"), (2.5e-7, "2.5e-07"),
    (0.0, "0"), (-0.0, "0"), (1e-6, "0.000001"), (1234.5, "1,234.5"),
    (123456.1234567, "123,456.123457"),
])
def test_header_shows_small_nonzero_values(value, shown):
    hdr = read_product(PRD_16)[0].header
    hdr.channel_phase_polynomial_coefficient_2 = value
    row = [ln for ln in str(hdr).splitlines() if "coefficient_2:" in ln][0]
    assert row.split(": ", 1)[1] == shown + " turns/s^2"


# Table 5-1 units, shown after the value in str(RecordHeader). Table 5-1 gives
# the phase items "in 'turns', i.e., rad/2π" (§5.2.6.2 calls them cycles).
HEADER_UNITS = {
    "record_length": "bytes",
    "sample_size": "bit",
    "sample_rate": "Hz",
    "rf_to_if_downconv": "Hz",
    "if_to_channel_downconv": "Hz",
    "time_tag_second_of_day": "s",
    "time_tag_picoseconds_of_the_second": "ps",
    "channel_accumulated_phase": "turns",
    "channel_phase_polynomial_coefficient_0": "turns",
    "channel_phase_polynomial_coefficient_1": "turns/s",
    "channel_phase_polynomial_coefficient_2": "turns/s^2",
    "channel_phase_polynomial_coefficient_3": "turns/s^3",
}


def test_header_rows_carry_table_5_1_units_and_dump_does_not():
    pf = read_product(PRD_16)
    rows = dict(ln.strip().split(": ", 1) for ln in str(pf[0].header).splitlines()[1:])
    for field, unit in HEADER_UNITS.items():
        assert rows[field].endswith(" " + unit), (field, rows[field])
    for field in ("station_id", "time_tag_year", "time_tag_doy"):  # no unit
        assert rows[field].replace(",", "").isdigit(), (field, rows[field])
    buf = io.StringIO()
    pf.dump_headers(buf)
    assert "  sample_rate: 2000\n" in buf.getvalue()
    assert not any(ln.rstrip().endswith((" Hz", " bit", " ps", " s")) for ln in buf.getvalue().splitlines())


def test_dump_headers_writes_a_file(tmp_path):
    pf = read_headers(PRD_16)
    out = tmp_path / "hdr.txt"
    pf.dump_headers(out)
    text = out.read_text()
    assert "Record 0" in text
    assert "sample_rate: 2000" in text


@pytest.mark.parametrize("method", ["dump", "dump_headers"])
def test_dump_targets_agree(method, tmp_path, capsys):
    """File, stdout (default) and an open text stream get the same text."""
    pf = read_product(PRD_16)
    out = tmp_path / "dump.txt"
    getattr(pf, method)(out)
    want = out.read_text(encoding="utf-8")
    assert want.startswith("RDEF ")

    getattr(pf, method)()
    assert capsys.readouterr().out == want

    buf = io.StringIO()
    getattr(pf, method)(buf)
    assert not buf.closed  # caller's stream is left open
    assert buf.getvalue() == want


def test_empty_product_summary_does_not_crash():
    assert "empty" in str(ProductFile())


def test_concatenate_dedupes_on_time_tag():
    a = read_product(PRD_16)
    b = read_product(PRD_16)
    with pytest.warns(RuntimeWarning, match="duplicate time tag"):
        merged = a.concatenate(b)
    assert len(merged.records) == 2  # same epochs, deduped

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        assert len(a.concatenate(b, keep="both").records) == 4


def test_path_is_absolute_and_follows_derived_files(tmp_path, monkeypatch):
    monkeypatch.chdir(DATA)  # read via relative names
    for pf in (read_product(PRD_16.name), read_headers(PRD_16.name)):
        assert pf.path.is_absolute() and pf.path == PRD_16.resolve()
    pf = read_product(PRD_16.name)
    assert pf.select().path == pf.slice(0).path == pf.copy().path == pf.path
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        assert pf.concatenate(read_product(PRD_16)).path == pf.path  # same file
        other = tmp_path / "other.prd"
        other.write_bytes(PRD_16.read_bytes())
        assert pf.concatenate(read_product(other)).path is None
    assert ProductFile().path is None


def test_concatenate_rejects_a_bad_keep():
    pf = read_product(PRD_16)
    with pytest.raises(ValueError, match="keep must be one of"):
        pf.concatenate(pf, keep="neither")


# ---------------------------------------------------------------------------
# Cython fast path agrees with the NumPy reference
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("threaded", [False, True], ids=["1thread", "threads"])
@pytest.mark.parametrize("bits", [16, 8, 4, 2, 1])
def test_cython_unpack_matches_numpy(bits, threaded, monkeypatch):
    """Whole, or split into thread chunks."""
    from rdefpy.prd import reader

    if reader._unpack_fast is None:
        pytest.skip("unpack_cyth extension not built")
    if threaded:
        from rdefpy import _utils

        monkeypatch.setattr(_utils, "_THREADED_MIN_SAMPLES", 1)
    raw = np.random.default_rng(bits).integers(0, 256, 4100, dtype=np.uint8).tobytes()
    i_ref, q_ref = reader._UNPACKERS[bits](raw)
    i_cy, q_cy = reader._unpack_fast(raw, bits)
    assert i_cy.dtype == i_ref.dtype and q_cy.dtype == q_ref.dtype
    np.testing.assert_array_equal(i_cy, i_ref)
    np.testing.assert_array_equal(q_cy, q_ref)


@pytest.mark.parametrize("kernel", [True, False], ids=["cython", "numpy"])
@pytest.mark.parametrize("threaded", [False, True], ids=["1thread", "threads"])
@pytest.mark.parametrize("cdt", [np.complex64, np.complex128])
@pytest.mark.parametrize("bits", [16, 8, 4, 2, 1])
def test_samples_are_i_plus_jq_on_every_path(bits, cdt, threaded, kernel, monkeypatch):
    """``samples`` equals ``real = I; imag = Q`` whole or in thread chunks,
    with or without the Cython interleave."""
    from rdefpy import _utils
    from rdefpy.prd import classes, reader

    if kernel:
        if reader._unpack_fast is None:
            pytest.skip("unpack_cyth extension not built")
        from conftest import _require_fresh_build

        _require_fresh_build()
        assert classes._iq_to_complex is not None  # built, so it must be there
    else:
        monkeypatch.setattr(classes, "_iq_to_complex", None)
    if threaded:
        monkeypatch.setattr(_utils, "_THREADED_MIN_SAMPLES", 1)
    dt = {16: np.int32, 8: np.int16}.get(bits, np.int8)
    info = np.iinfo(dt)
    rng = np.random.default_rng(bits)
    i, q = (rng.integers(info.min, info.max, 4099, endpoint=True).astype(dt)
            for _ in range(2))
    got = RecordData(bits, 4099, i, q, complex_dtype=cdt).samples
    ref = np.empty(4099, cdt)
    ref.real = i
    ref.imag = q
    assert got.dtype == ref.dtype
    np.testing.assert_array_equal(got.view(got.real.dtype), ref.view(ref.real.dtype))


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
def test_threaded_decode_works_in_a_forked_child(monkeypatch):
    """The thread pool does not survive fork(); the child must start its own."""
    from rdefpy import _utils
    from rdefpy.prd import reader

    monkeypatch.setattr(_utils, "_THREADED_MIN_SAMPLES", 1)
    raw = bytes(range(256)) * 16
    reader._decode_iq(raw, 2)  # the parent's pool now exists
    pid = os.fork()
    if pid == 0:  # child: decode once more, then leave without pytest teardown
        try:
            reader._decode_iq(raw, 2)
            os._exit(0)
        finally:
            os._exit(1)
    for _ in range(200):  # a hung child would wait forever on its pool
        done, status = os.waitpid(pid, os.WNOHANG)
        if done:
            break
        time.sleep(0.05)
    else:
        os.kill(pid, 9)
        pytest.fail("forked child hung on the inherited thread pool")
    assert os.waitstatus_to_exitcode(status) == 0


def test_samples_are_cached_and_read_only():
    """One complex build per record, shared by every caller, so nobody may
    edit it in place; assigning I or Q invalidates the cache."""
    pf = read_product(PRD_16)
    d = pf.records[0].data
    x = d.samples
    assert d.samples is x
    with pytest.raises(ValueError, match="read-only"):
        x[0] = 0
    d.i_samples = d.i_samples.copy()
    assert d.samples is not x
    assert np.array_equal(d.samples, x)
