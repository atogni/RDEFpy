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
    assert [p.name for p in tmp_path.iterdir()] == ["v.prd"]  # no bad.prd, no temp file


def test_write_over_its_own_on_disk_source(tmp_path):
    """Records still read from the target while it is replaced (temp + os.replace)."""
    path = tmp_path / PRD_16.name
    path.write_bytes(PRD_16.read_bytes())
    write_product(path, read_product(path))
    assert path.read_bytes() == PRD_16.read_bytes()
    assert [p.name for p in tmp_path.iterdir()] == [path.name]  # no temp left


@pytest.mark.parametrize("loaded", [True, False], ids=["packed", "verbatim"])
def test_failed_write_leaves_the_existing_target_untouched(loaded, tmp_path):
    """The write goes to a temporary file that replaces the target only when
    every record is on disk: a record refused part-way leaves the old file
    byte for byte, and no temporary file beside it."""
    target = tmp_path / "target.prd"
    target.write_bytes(b"an existing file, not even RDEF")
    pf = read_product(PRD_16)
    if loaded:
        pf.load()
    pf[1].header.station_id = 70000  # no 2-byte field holds it; record 0 is written first
    with pytest.raises(ValueError, match="station_id"):
        write_product(target, pf)
    assert target.read_bytes() == b"an existing file, not even RDEF"
    assert [p.name for p in tmp_path.iterdir()] == ["target.prd"]


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_written_file_mode_is_the_usual_one_then_the_targets(tmp_path):
    """The temporary file is created owner-only; the file left behind is not:
    a new file gets 0o666 less the umask, an overwritten one keeps its mode."""
    import stat

    pf = read_product(PRD_16)
    out = tmp_path / "mode.prd"
    old = os.umask(0o027)
    try:
        write_product(out, pf)
        assert stat.S_IMODE(out.stat().st_mode) == 0o640  # 0o666 & ~0o027
        os.chmod(out, 0o604)
        write_product(out, pf)
        assert stat.S_IMODE(out.stat().st_mode) == 0o604
    finally:
        os.umask(old)
    assert out.read_bytes() == PRD_16.read_bytes()


def test_progress_bars_run(tmp_path, capsys):
    """Each bar is drawn on stderr, to the full 30 stars, under its scale."""
    pf = read_product(PRD_16, progress=True)
    pf.load(progress=True)
    write_product(tmp_path / "p.prd", pf, progress=True)
    assert (tmp_path / "p.prd").read_bytes() == PRD_16.read_bytes()
    read_headers(PRD_16, progress=True)
    scan_product(PRD_16, progress=True)
    out, err = capsys.readouterr()
    assert out == ""
    bars = err.splitlines()
    assert bars == [" " * 8 + "0%" + " " * 24 + "100%", " " * 8 + "*" * 30] * 5

    read_product(PRD_16)  # off by default: nothing drawn
    assert capsys.readouterr().err == ""


def test_progress_bar_is_full_when_the_file_ends_before_num_records(capsys):
    """num_records = 10 on a 2-record file: the bar counts records, and EOF fills it."""
    assert len(read_product(PRD_16, num_records=10, progress=True)) == 2
    assert capsys.readouterr().err.splitlines() == [" " * 8 + "0%" + " " * 24 + "100%", " " * 8 + "*" * 30]
    read_product(PRD_16, num_records=1, progress=True)  # stops at num_records: full too
    assert capsys.readouterr().err.splitlines()[1] == " " * 8 + "*" * 30


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


def test_assigning_data_replaces_the_samples_on_disk(tmp_path, decodes):
    """``record.data = new`` puts the record in memory: the file's samples are
    no longer its samples, so release() must not bring them back, and
    write_product writes the new ones."""
    pf = read_product(PRD_16)
    rec = pf.records[0]
    assert rec._data is None and rec._source is not None  # 2 kHz: on disk
    held = rec.data  # a live transient read must not come back either
    new = RecordData(16, 2000, np.full(2000, 3, np.int32), np.full(2000, -5, np.int32))
    rec.data = new
    assert rec.data is new and rec._source is None and rec._weak is None
    rec.release()
    pf.release()  # nothing to re-read: stays in memory
    assert rec.data is new and len(rec) == 2000
    del held

    decodes.clear()
    out = tmp_path / "new.prd"
    write_product(out, pf)
    assert decodes == []  # record 0 packed from memory, record 1 copied verbatim
    raw, src = out.read_bytes(), PRD_16.read_bytes()
    n = len(raw) // 2
    assert raw[:176] == src[:176]  # header untouched
    assert raw[176:n] == np.tile(np.array([1, -3], "<i2"), 2000).tobytes()  # 2k+1 = 3, -5 (§5.3.6)
    assert raw[n:] == src[n:]

    for bad in (np.zeros(3), None, [1, 3]):
        with pytest.raises(TypeError, match="must be a RecordData"):
            rec.data = bad
    assert rec.data is new


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
def test_stop_epoch_drops_later_records_with_or_without_a_progress_bar(progress, capsys):
    """stop_epoch keeps the records before it; the whole file is still walked
    (records need not be in time order), so a progress bar runs to its end.
    Works whether or not a bar is drawn.
    """
    t = read_headers(PRD_16).timetag_array
    pf = read_product(PRD_16, stop_epoch=t[1], progress=progress)
    assert len(pf.records) == 1
    assert pf.records[0].header.timetag == t[0]
    assert capsys.readouterr().err.count("*") == (30 if progress else 0)  # 30 stars: 100%


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
    assert len(read_product(PRD_16, stop_epoch="2026039T12:00:01").records) == 1


def test_epochs_with_a_utc_offset_are_converted_and_naive_ones_are_utc():
    """One instant, 12:00:01 UTC, written five ways: each gives the same window."""
    from datetime import datetime as dt, timedelta, timezone

    from rdefpy._utils import _to_datetime64

    want = np.datetime64("2026-02-08T12:00:01", "ns")
    plus2 = timezone(timedelta(hours=2))
    for epoch in (dt(2026, 2, 8, 12, 0, 1),                    # naive: UTC, not local time
                  dt(2026, 2, 8, 14, 0, 1, tzinfo=plus2),       # aware: converted
                  "2026-02-08T14:00:01+02:00", "2026-02-08T09:30:01-02:30",
                  np.datetime64("2026-02-08T12:00:01")):         # second precision -> ns
        got = _to_datetime64(epoch)
        assert got == want and got.dtype == np.dtype("datetime64[ns]"), epoch
        assert len(read_product(PRD_16, stop_epoch=epoch).records) == 1
        assert len(read_product(PRD_16).select(start_epoch=epoch).records) == 1
    assert _to_datetime64(None) is None


@pytest.mark.parametrize("text", ["2026-02-08T12:00:00.000000250", "2026-02-08T12:00:00.000000250Z",
                                  "2026-02-08T14:00:00.000000250+02:00", "2026-02-08T12:00:00.0000002509"])
def test_epoch_strings_keep_nanoseconds(text):
    """250 ns past the first record (12:00:00) drops it: datetime alone would truncate to 12:00:00."""
    from rdefpy import extract_carrier

    assert len(read_product(PRD_16, start_epoch=text)) == 1
    assert len(read_product(PRD_16).select(start_epoch=text)) == 1
    # whole_second: a start 250 ns past 12:00:00 moves to the next whole second
    c = extract_carrier(read_product(PRD_16), 1.0, max_rate_hz_s=0, start_epoch=text, whole_second=True)
    assert c["time"][0] == np.datetime64("2026-02-08T12:00:01.5")


@pytest.mark.parametrize("text, ns", [
    ("2026-02-08T12:00:00.000000250", 250), ("2026-02-08T12:00:00.0000002", 200),  # 7 digits: 200 ns, not 2
    ("2026-02-08T12:00:00.00000025", 250), ("2026-02-08T12:00:00.1234567891", 123_456_789),  # 10th digit dropped
    ("2026-02-08T14:00:00.000000250+02:00", 250), ("2026-02-08T12:00:00.000000250Z", 250),
    ("2026-02-08T12:00:00,000000250", 250), ("2026-02-08T12:00:00.5", 500_000_000),
    # CCSDS 301.0-B-4 §3.5.1.2, ASCII Time Code B: "as many 'd' characters ... as required"
    ("2026-039T12:00:00.5", 500_000_000), ("2026-039T12:00:00.000000250", 250),
    ("2026039T12:00:00.0000002Z", 200), ("2026-039T12:00:00.1234567891Z", 123_456_789),
])
def test_epoch_strings_to_the_nanosecond(text, ns):
    from rdefpy._utils import _to_datetime64

    got = _to_datetime64(text)
    assert got == np.datetime64("2026-02-08T12:00:00", "ns") + np.timedelta64(ns, "ns")
    assert got.dtype == np.dtype("datetime64[ns]")


# "Invalid isoformat string": no CCSDS match, so the stdlib's datetime.fromisoformat
# refuses it (no ordinal dates in ISO mode, checked on Python 3.11 and 3.15).
_NOT_CCSDS = "Invalid isoformat string"


@pytest.mark.parametrize("text, match", [
    ("2026-039T12:00:00.", _NOT_CCSDS), ("2026-039T12:00:60.5", "field out of range"),
    ("2026-039T12:00:00,5", _NOT_CCSDS), ("2026-039T12:00:00.5.5", _NOT_CCSDS),
    ("2026-366T12:00:00.5", "day of year out of range"),
])
def test_ccsds_epoch_with_a_bad_fraction_raises(text, match):
    """Period separator and digits only (§3.5.1.2); field ranges checked as without a fraction."""
    from rdefpy._utils import _to_datetime64

    with pytest.raises(ValueError, match=match):
        _to_datetime64(text)


@pytest.mark.parametrize("epoch", [1770552001, 1770552001.0, b"2026-039T12:00:01", ("2026", "039")])
def test_epoch_of_another_type_is_a_type_error(epoch):
    """A number could be seconds, days or nanoseconds from any origin: refused."""
    with pytest.raises(TypeError, match="Expected datetime, np.datetime64, str, or None"):
        read_product(PRD_16, start_epoch=epoch)
    with pytest.raises(TypeError, match="Expected datetime, np.datetime64, str, or None"):
        read_product(PRD_16).select(stop_epoch=epoch)


@pytest.mark.parametrize("text, match", [
    ("2026-400T00:00:00", "field out of range"), ("2026-000T00:00:00", "field out of range"),
    ("2026-366T00:00:00", "day of year out of range"), ("2026-039T25:00:00", "field out of range"),
    ("2026039T12:61:00", "field out of range"), ("2026-039T12:00:60", "field out of range"),
    ("2026--01T12:00:00", _NOT_CCSDS), ("2026- 39T 1:00:00", _NOT_CCSDS),
])
def test_epoch_out_of_range_raises(text, match):
    """A bad epoch field raises instead of rolling over into another time."""
    pf = read_product(PRD_16)
    with pytest.raises(ValueError, match=match):
        pf.select(start_epoch=text)
    with pytest.raises(ValueError, match=match):
        read_product(PRD_16, stop_epoch=text)


def test_num_records_caps_the_read():
    assert len(read_product(PRD_16, num_records=1).records) == 1


def test_band_and_explicit_window_are_mutually_exclusive():
    with pytest.raises(ValueError, match="not both"):
        read_product(PRD_16, band="X", min_downconv_hz=1.0)


def test_unknown_band_rejected():
    with pytest.raises(ValueError, match="Unknown band"):
        read_product(PRD_16, band="Q")


@pytest.mark.parametrize("band", [5, b"X", ["X"]])
def test_band_that_is_not_a_string_is_an_unknown_band(band):
    with pytest.raises(ValueError, match="Unknown band"):
        read_product(PRD_16, band=band)


def test_band_keeps_the_records_of_its_allocation(write_prd):
    """One file per band (fixed downconversion is file-constant), each read with every
    band name: kept only by its own, in any case (S 2.29-2.30, X 8.40-8.45, Ka 31.8-32.3 GHz)."""
    import rdef_spec as spec

    files = {name: write_prd(spec.record_bytes([0] * 16, [0] * 16, rf_to_if_downconv=hz, if_to_channel_downconv=0.0))
             for name, hz in (("S", 2.295e9), ("X", 8.42e9), ("Ka", 32.0e9))}
    for band in ("S", "s", "X", "x", "Ka", "ka", "KA", "kA"):
        kept = [name for name, path in files.items() if len(read_product(path, band=band)) == 1]
        assert kept == [band.upper().replace("KA", "Ka")], band


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
    np.testing.assert_allclose(rec.downconversion_frequency, expected, rtol=0, atol=4e-6)  # ~4 float64 steps at 8.4 GHz


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
    np.testing.assert_allclose(rec.downconversion_phase, expected, rtol=0, atol=1e-9)


def test_mean_downconversion_frequency_is_the_time_average_over_the_record():
    """The mean over the record's second is the phase advance over that second
    (first sample at tau = 0.3 s from the origin, TIME TAG SECOND OF DAY).
    Expected values from rdef_spec and the polynomial written out, not from rdefpy."""
    import rdef_spec as spec

    fields = {
        "sample_rate": 4, "sample_size": 16,
        "rf_to_if_downconv": 8.4e9, "if_to_channel_downconv": 1.0e6,
        "time_tag_picoseconds_of_the_second": 0.3e12,  # first sample 0.3 s after the second of day (t0_offset 0, the default)
        "channel_phase_polynomial_coefficient_1": 100.0,
        "channel_phase_polynomial_coefficient_2": 10.0,
        "channel_phase_polynomial_coefficient_3": 1.0,
    }
    pf = ProductFile()
    pf.records = [Record(RecordHeader(fields), RecordData(16, 4, np.ones(4, np.int32), np.ones(4, np.int32)))]
    (mean,) = pf.mean_downconversion_frequency
    # float64 spacing at 8.4 GHz is 2**-20 Hz = 9.5e-7 Hz: allow 2 spacings
    # (the closed form and these sums round differently).
    tol = 2 * np.spacing(8.4e9)
    assert mean == pytest.approx(spec.dc_mean(fields), abs=tol)
    t = 0.3 + (np.arange(1_000_000) + 0.5) / 1e6  # midpoint rule, independent of the closed form
    assert mean == pytest.approx(8.4e9 + 1.0e6 + (100.0 + 2 * 10.0 * t + 3 * 1.0 * t**2).mean(), abs=tol)
    assert abs(mean - (8.4e9 + 1.0e6 + 100.0 + 2 * 10.0 * 0.8 + 3 * 1.0 * 0.8**2)) > 0.1  # c3 != 0: not the midpoint value


def test_productfile_downconversion_arrays_are_header_only(monkeypatch):
    """Per-record values must not read any samples from disk."""
    from rdefpy.prd import reader

    pf = read_product(PRD_16)
    calls = []
    real = reader._decode_iq
    monkeypatch.setattr(reader, "_decode_iq", lambda *a: calls.append(1) or real(*a))
    f = pf.record_downconversion_frequency
    p = pf.record_downconversion_phase
    m = pf.mean_downconversion_frequency
    fs, ps = pf.downconversion_frequency[:], pf.downconversion_phase[:]
    assert f.shape == (2,) and p.shape == (2,) and m.shape == (2,)
    assert fs.shape == ps.shape == (len(pf.samples),)
    assert calls == []
    assert all(r._data is None for r in pf.records)


# =============================================================================
# RecordData
# =============================================================================


def test_full_scale_fraction_at_the_bit_depth_limit():
    d = RecordData(8, 4, np.array([255, 1, 1, 1], np.int16), np.ones(4, np.int16))
    assert d.full_scale_fraction == pytest.approx(0.25)


def test_full_scale_fraction_is_undefined_for_one_bit():
    d = RecordData(1, 2, np.array([1, -1], np.int8), np.array([1, -1], np.int8))
    with pytest.warns(UserWarning, match="undefined for 1-bit"):
        assert d.full_scale_fraction == 0.0


def test_full_scale_fraction_of_an_empty_record_is_nan():
    assert np.isnan(RecordData(8).full_scale_fraction)


def test_i_without_q_is_an_error_not_a_zero_q():
    iq = np.array([1, 3], np.int16)
    for kw in (dict(i_samples=iq), dict(q_samples=iq)):  # refused at construction
        with pytest.raises(ValueError, match="both provided or both None"):
            RecordData(8, 2, **kw)
    for first in ("i_samples", "q_samples"):  # set one at a time: usable once both are there
        d = RecordData(8, 2)
        setattr(d, first, iq)
        for name in ("samples", "full_scale_fraction", "power", "rms", "power_db"):
            with pytest.raises(ValueError, match="set both"):
                getattr(d, name)
        setattr(d, "q_samples" if first == "i_samples" else "i_samples", iq)
        assert d.samples.tolist() == [1 + 1j, 3 + 3j] and d.power == 10.0


@pytest.mark.parametrize("name", ["power", "rms", "power_db"])
def test_q_without_i_is_an_error_not_an_empty_record(name):
    d = RecordData(8, 2)
    d.q_samples = np.array([1, 3], np.int16)
    with pytest.raises(ValueError, match="set both"):
        getattr(d, name)


@pytest.mark.parametrize("first", ["i_samples", "q_samples"])
def test_one_of_i_and_q_has_no_length_and_is_not_written(first, tmp_path):
    """With only I or only Q set, len, duration_s, time_array and the writer
    raise as power does: no length, nothing written."""
    from rdefpy import pack_record_data

    d = RecordData(8, 2)
    setattr(d, first, np.array([1, 3], np.int16))
    rec = Record(RecordHeader({"sample_size": 8, "sample_rate": 2}), d)
    pf = ProductFile()
    pf.records = [rec]
    for call in (lambda: len(d), lambda: d.duration_s, lambda: d.time_array, lambda: len(rec),
                 lambda: pack_record_data(rec), lambda: write_product(tmp_path / "x.prd", pf)):
        with pytest.raises(ValueError, match="set both"):
            call()
    # Displaying the record still works, so the mismatch can be seen.
    want = "I 2 / Q 0" if first == "i_samples" else "I 0 / Q 2"
    assert want in repr(d) and want in str(d)
    str(rec), repr(rec)


def _on_disk_record():
    return read_product(PRD_16)[0]


def _iq(n=2):
    return RecordData(8, n, np.ones(n, np.int16), np.ones(n, np.int16))


def _hand_built(**header):
    return Record(RecordHeader(header), RecordData(16, 4, np.ones(4, np.int32), np.ones(4, np.int32)))


@pytest.mark.parametrize("call, error, match", [
    (lambda: setattr(_iq(), "i_samples", np.ones(3, np.int16)), ValueError, "New I array shape does not match existing Q"),
    (lambda: setattr(_iq(), "q_samples", np.ones(3, np.int16)), ValueError, "New Q array shape does not match existing I"),
    (lambda: RecordData(8, complex_dtype=np.float64), ValueError, "complex_dtype must be np.complex64 or np.complex128"),
    (lambda: RecordHeader({"nope": 1}), KeyError, "Unknown header field: nope"),
    (lambda: Record({"nope": 1}), KeyError, "Unknown header field: nope"),
    (lambda: read_product(PRD_16).select(filters={"nope": 1}), ValueError, "Unknown header field in filters: 'nope'"),
    (lambda: _on_disk_record().correct_header(nope=1), ValueError, "Unknown header field: 'nope'"),
    (lambda: _hand_built(sample_rate=4).downconversion_frequency, ValueError, "must both be set"),
    (lambda: _hand_built(sample_rate=4, rf_to_if_downconv=8.4e9).downconversion_frequency, ValueError,
     "must both be set"),
    (lambda: _hand_built(rf_to_if_downconv=8.4e9, if_to_channel_downconv=0.0).downconversion_frequency, ValueError,
     "sample_rate is required"),
    (lambda: _hand_built(sample_rate=0, rf_to_if_downconv=8.4e9, if_to_channel_downconv=0.0).downconversion_phase,
     ValueError, "Invalid sample rate"),
    (lambda: RecordHeader({"agency_flag": 3, "empty_fields_internal_agency_use": bytes(39)}).agency_fields(),
     ValueError, "holds 39 bytes, expected 40"),
], ids=["i-size", "q-size", "complex_dtype", "header-key", "record-dict-key", "select-filter", "correct_header",
        "fixed-unset", "if-unset", "rate-unset", "rate-0", "agency-39-bytes"])
def test_bad_input_raises_and_names_the_problem(call, error, match):
    with pytest.raises(error, match=match):
        call()


def test_unknown_field_in_correct_header_changes_nothing():
    rec = _on_disk_record()
    with pytest.raises(ValueError, match="Unknown header field"):
        rec.correct_header(validity_flag=1, nope=1)  # checked before any field is set
    assert rec.header.validity_flag == 0


def test_agency_fields_of_an_unset_block_are_all_zero():
    """An unset block decodes as the 40 zero bytes write_product writes for it."""
    dsn = RecordHeader({"agency_flag": 3}).agency_fields()
    assert dsn.dtype.names[0] == "predict_pass_number" and dsn.tobytes() == bytes(40)
    raw = RecordHeader({"agency_flag": 1}).agency_fields()
    assert raw.dtype.names == ("raw",) and raw.tobytes() == bytes(40)


def test_samples_must_be_1d():
    two_d = np.ones((2, 3), np.int32)
    with pytest.raises(ValueError, match="1-D"):
        RecordData(16, 3, two_d, two_d)
    d = RecordData(16, 3, np.ones(3, np.int32), np.ones(3, np.int32))
    with pytest.raises(ValueError, match="1-D"):
        d.i_samples = two_d


def test_decode_refuses_a_partial_32_bit_word():
    """§5.3.2: the data section is whole 32-bit words, for every sample size."""
    from rdefpy.prd.reader import _decode_iq
    for bits in (16, 8, 4, 2, 1):
        with pytest.raises(ValueError, match="32-bit words"):
            _decode_iq(bytes(6), bits)


def test_power_and_rms_agree():
    d = RecordData(16, 4, np.full(4, 3, np.int32), np.full(4, 4, np.int32))
    assert d.power == pytest.approx(25.0)
    assert d.rms == pytest.approx(5.0)
    assert d.power_db == pytest.approx(10 * np.log10(25.0))


def test_power_db_has_no_floor():
    zeros = np.zeros(2, np.int16)
    assert RecordData(8, 2, zeros, zeros).power_db == -np.inf
    assert np.isnan(RecordData(8).power_db)


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


def _rows(obj):
    """str(obj) as {label: value}, title line dropped."""
    return dict(ln.strip().split(": ", 1) for ln in str(obj).splitlines()[1:])


def test_summaries_render():
    """Every row of every summary, for a file whose values are known (tests/data)."""
    pf = read_product(PRD_16)
    assert _rows(pf) == {
        "path": str(PRD_16.resolve()), "records": "2",
        "first_timetag": "2026-02-08T12:00:00.000000000Z", "last_timetag": "2026-02-08T12:00:01.000000000Z",
        "consecutive": "True", "sample_rate": "2,000 Hz", "sample_size": "16 bit", "spacecraft_id": "1",
        "station_id": "24", "band": "X", "frequency_steering": "False",
    }
    assert _rows(pf[1]) == {
        "timetag": "2026-02-08T12:00:01.000000000Z", "sample_rate": "2,000 Hz", "sample_size": "16 bit",
        "samples": "2,000", "loaded": "False", "spacecraft_id": "1", "station_id": "24", "band": "X",
        "fixed_downconv_hz": "8,405,000,000", "variable_downconv_hz": "0", "phase_coef_0": "0 turns",
        "phase_coef_1": "0 turns/s", "phase_coef_2": "0 turns/s^2", "phase_coef_3": "0 turns/s^3",
    }
    assert _rows(pf[0].data) == {"samples": "2,000", "sample_size": "16 bit", "sample_rate": "2,000 Hz",
                                 "duration_s": "1", "dtype": "int32", "complex_dtype": "complex128"}
    hdr = _rows(pf[0].header)
    assert list(hdr) == [*binary._FIELDS, "timetag"]  # Table 5-1 order, then the derived time tag
    assert hdr["record_length"] == "8,176 bytes" and hdr["timetag"] == "2026-02-08T12:00:00.000000000Z"
    assert repr(pf) == f"<ProductFile {pf.name!r} 2 records>"
    assert repr(pf[0]) == "<Record 2026-02-08T12:00:00.000000000Z st=24 sc=1 16 bit 2,000 Hz>"
    assert repr(pf[0].header) == "<RecordHeader 2026-02-08T12:00:00.000000000Z st=24 sc=1 16 bit 2,000 Hz>"
    assert repr(pf[0].data) == "<RecordData 2,000 samples 16 bit 2,000 Hz>"


@pytest.mark.filterwarnings("ignore:Metadata fields vary:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:.*duplicate time tag:RuntimeWarning")
def test_summary_counts_the_records_of_each_value_that_varies():
    """A field that differs between records lists each value with its record
    count, in first-seen order; a fixed downconversion in no band is 'unknown'."""
    pf = read_product(PRD_16).concatenate(read_product(PRD_16), keep="both", sort=False)  # 4 records
    pf.records[2].header.station_id = 26
    pf.records[3].header.sample_rate = 4000
    pf.records[3].header.rf_to_if_downconv = 2.0  # + 5 MHz: no downlink band
    rows = _rows(pf)
    assert rows["station_id"] == "24 (3 rec), 26 (1 rec)"
    assert rows["sample_rate"] == "2,000 Hz (3 rec), 4,000 Hz (1 rec)"
    assert rows["band"] == "X (3 rec), unknown (1 rec)"
    assert rows["spacecraft_id"] == "1" and rows["sample_size"] == "16 bit"  # constant: no count
    assert rows["consecutive"] == "False (min dt -1.000000 s, max dt 1.000000 s)"
    assert _rows(pf[3])["band"] == "unknown" and _rows(pf[3])["fixed_downconv_hz"] == "5,000,002"
    assert _rows(pf[0])["band"] == "X"


@pytest.mark.parametrize("rf, band", [(2.295e9, "S"), (8.425e9, "X"), (32.0e9, "Ka"), (1.2e9, "unknown"),
                                      (2.0, "unknown"), (5.0e9, "unknown"), (None, "unknown")])
def test_band_row_classifies_the_fixed_downconversion(rf, band):
    """Nominal downlink allocations, 1 GHz guard each side (as read_product(band=...))."""
    rec = Record(RecordHeader({"rf_to_if_downconv": rf, "if_to_channel_downconv": 0.0}))
    assert _rows(rec)["band"] == band
    assert _rows(rec)["fixed_downconv_hz"] == ("?" if rf is None else f"{rf:,.0f}")


def test_info_prints_the_summary(capsys):
    pf = read_product(PRD_16)
    pf.info()
    assert capsys.readouterr().out == str(pf) + "\n"
    buf = io.StringIO()
    pf.info(buf)
    assert buf.getvalue() == str(pf) + "\n" and capsys.readouterr().out == ""


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


def test_slice_of_records_prints_one_per_line():
    pf = read_product(PRD_16)
    recs = pf[0:2]
    assert isinstance(recs, list) and recs[0] is pf.records[0]
    assert repr(recs).splitlines() == ["[", f"  {pf.records[0]!r},", f"  {pf.records[1]!r},", "]"]
    assert repr(pf[5:5]) == "[]"


def test_record_summary_reports_samples_and_loaded_state():
    rec = read_product(PRD_16)[0]
    rows = lambda: dict(ln.strip().split(": ", 1) for ln in str(rec).splitlines()[1:])  # noqa: E731
    assert rows()["samples"] == "2,000" and rows()["loaded"] == "False"
    rec.load()
    assert rows()["loaded"] == "True"


@pytest.mark.parametrize("c,shown", [
    ((1234.5, -0.01, 1e-6), "1,234.5 - 0.02*(t-t0) + 0.000003*(t-t0)^2"),
    ((-123456.78901, -0.31415, 0.0), "-123,457 - 0.6283*(t-t0)"),
    ((0.0, -0.5, 0.0), "-1*(t-t0)"),
    ((0.0, 0.0, 2.0), "6*(t-t0)^2"),
    ((1234567.0, 0.0, 0.0), "1,234,567"),
    ((0.0, -0.0, 0.0), "0"),
    ((1e-300, 0.0, 0.0), "1e-300"),
])
def test_record_summary_shows_variable_downconversion(c, shown):
    """§5.2.6.3: f_var = c1 + 2*c2*(t-t0) + 3*c3*(t-t0)^2, factors multiplied in."""
    rec = read_product(PRD_16)[0]
    h = rec.header
    (h.channel_phase_polynomial_coefficient_1, h.channel_phase_polynomial_coefficient_2,
     h.channel_phase_polynomial_coefficient_3) = c
    rows = dict(ln.strip().split(": ", 1) for ln in str(rec).splitlines()[1:])
    assert rows["variable_downconv_hz"] == shown
    assert rows["phase_coef_1"].endswith(" turns/s")  # raw coefficients shown too


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


def test_dump_and_dump_headers():
    pf = read_product(PRD_16).slice(0, 1)
    rec = pf[0]
    for data in (True, False):
        buf = io.StringIO()
        (pf.dump if data else pf.dump_headers)(buf)
        lines = buf.getvalue().splitlines()
        assert lines[0] == f"RDEF {'Full' if data else 'Header'} Dump -- 1 records"
        assert lines[3] == ("Record 0 -- Header" if data else "Record 0")
        assert "  empty_fields_future_extension: 36 bytes, all zero" in lines
        assert f"  station_id: {rec.header.station_id}" in lines
        iq = f"  {rec.data.i_samples[0]}, {rec.data.q_samples[0]}"
        assert (iq in lines) is data
    assert len(lines) == 5 + 21 + 1  # title, rule, blank, label, rule; 21 fields; blank


def test_record_data_summary_makes_no_pass_over_the_samples(monkeypatch):
    def boom(self):
        raise AssertionError("str(RecordData) touched the samples")

    for name in ("samples", "power", "power_db",
                 "rms", "full_scale_fraction", "time_array"):
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


@pytest.mark.parametrize("method", ["dump", "dump_headers"])
def test_dump_refuses_to_overwrite_its_source(method, tmp_path):
    src = tmp_path / "src.prd"
    src.write_bytes(Path(PRD_16).read_bytes())
    before = src.read_bytes()
    pf = read_product(src)
    with pytest.raises(ValueError, match="overwrite"):
        getattr(pf, method)(src)
    assert src.read_bytes() == before


def test_failed_dump_leaves_no_file(tmp_path):
    src = tmp_path / "src.prd"
    src.write_bytes(Path(PRD_16).read_bytes())
    pf = read_product(src)  # 2 kHz: samples stay on disk
    with src.open("ab") as f:
        f.write(b"\0")  # the file changed since it was read
    out = tmp_path / "dump.txt"
    with pytest.raises(RuntimeError):
        pf.dump(out)
    assert not out.exists()


def test_dump_that_cannot_open_its_target_deletes_nothing(tmp_path, monkeypatch):
    out = tmp_path / "dump.txt"
    out.write_text("keep")
    pf = read_product(PRD_16)

    def refuse(self, *args, **kwargs):
        raise PermissionError(self)

    monkeypatch.setattr(Path, "open", refuse)  # e.g. a read-only file on POSIX
    with pytest.raises(PermissionError):
        pf.dump(out)
    assert os.path.getsize(out) == 4


def test_record_number_follows_position():
    pf = read_product(PRD_16)
    _ = pf[1]
    title = lambda rec: str(rec).splitlines()[0]  # noqa: E731
    assert title(pf.slice(1).records[0]) == "Record"  # the copy drops index 1
    assert [title(r) for r in pf.slice(1)] == ["Record 0"]
    assert [title(r) for r in pf] == [f"Record {i}" for i in range(len(pf))]


def test_display_never_raises_on_bad_values():
    assert "duration_s" not in str(RecordData(16, -1))
    pf = read_product(PRD_16)
    pf.records[0].header.channel_phase_polynomial_coefficient_1 = "abc"
    assert "frequency_steering: ?" in str(pf)


def test_unset_values_show_as_question_mark():
    assert repr(RecordHeader()) == "<RecordHeader ? st=? sc=? ? ?>"
    assert "None" not in str(RecordHeader())
    pf = read_product(PRD_16)
    pf.path = None
    assert "path: ?" in str(pf)


@pytest.mark.parametrize("name,value,shown", [
    ("record_length", np.int64(1234567), "1,234,567 bytes"),
    ("rf_to_if_downconv", np.float32(8.4e9), "8,400,000,000 Hz"),  # exact in float32
    ("sample_rate", np.int64(2000), "2,000 Hz"),
    ("sample_rate", 1000.0, "1,000 Hz"),
    ("rf_to_if_downconv", 1e300, "1e+300 Hz"),
])
def test_header_numbers_of_any_type(name, value, shown):
    hdr = read_product(PRD_16)[0].header
    setattr(hdr, name, value)
    rows = dict(ln.strip().split(": ", 1) for ln in str(hdr).splitlines()[1:])
    assert rows[name] == shown


def test_rate_rejects_bool():
    from rdefpy.prd._display import _fmt_rate
    assert _fmt_rate(True) == "?"


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


@pytest.mark.filterwarnings("ignore:.*duplicate time tag:RuntimeWarning")
def test_concatenating_a_file_with_itself_gives_independent_records():
    pf = read_product(PRD_16)
    both = pf.concatenate(pf, keep="both", sort=False)
    assert len(both) == 4
    both.records[2].header.station_id = 26
    assert [r.header.station_id for r in both] == [24, 24, 26, 24]
    both.records[0].load()  # 2000 Hz: on disk; loading one copy loads only that one
    assert [r._data is None for r in both] == [False, True, True, True] and pf.records[0]._data is None
    # (the _DiskSource each copy holds is never changed after the read: sharing it would be harmless)
    pf.load()  # samples in memory: they are copied too
    for m in (pf.concatenate(pf, keep="both"), pf.concatenate(pf.slice(0, 1), keep="both")):
        objs = [o for r in m for o in (r, r.header, r._data)] + [o for r in pf for o in (r, r.header, r._data)]
        assert len({id(o) for o in objs}) == len(objs)  # all distinct, and none from pf


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


@pytest.mark.parametrize("strided", ["i", "q", "both"])
@pytest.mark.parametrize("cdt", [np.complex64, np.complex128])
def test_strided_i_and_q_give_the_same_samples_in_thread_chunks(cdt, strided, monkeypatch):
    """The Cython interleave takes contiguous I and Q only. Arrays given as
    strided views (kept uncopied) must go through the NumPy stores instead,
    chunk by chunk, with the same values."""
    from rdefpy import _utils

    monkeypatch.setattr(_utils, "_THREADED_MIN_SAMPLES", 1)
    rng = np.random.default_rng(5)
    wide_i, wide_q = (2 * rng.integers(-32768, 32768, 2 * 4099).astype(np.int32) + 1 for _ in range(2))
    i = wide_i[::2] if strided in ("i", "both") else wide_i[:4099].copy()
    q = wide_q[1::2] if strided in ("q", "both") else wide_q[:4099].copy()
    d = RecordData(16, 4099, i, q, complex_dtype=cdt)
    assert d.i_samples is i and d.q_samples is q  # not copied: still strided
    assert not (i.flags.c_contiguous and q.flags.c_contiguous)
    ref = np.empty(4099, cdt)
    ref.real = i
    ref.imag = q
    got = d.samples
    assert got.dtype == ref.dtype and got.tobytes() == ref.tobytes()
    assert RecordData(16, 4099, i.copy(), q.copy(), complex_dtype=cdt).samples.tobytes() == ref.tobytes()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
def test_threaded_decode_works_in_a_forked_child(monkeypatch):
    """The thread pool does not survive fork(); the child must start its own."""
    from rdefpy import _utils
    from rdefpy.prd import reader

    monkeypatch.setattr(_utils, "_THREADED_MIN_SAMPLES", 1)
    raw = bytes(range(256)) * 16
    _utils._in_chunks(4, 1, lambda a, b: None)  # on either decoder: _decode_iq uses no pool on NumPy
    assert _utils._pools  # the parent's pool now exists
    pid = os.fork()
    if pid == 0:  # child: use the pool and decode once more, then leave without pytest teardown
        try:
            _utils._in_chunks(4, 1, lambda a, b: None)
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


def test_set_threads_resizes_the_pool_not_the_values(monkeypatch):
    """Any thread count decodes the same samples; the pool follows the setting."""
    from rdefpy import _utils, set_threads
    from rdefpy.prd import reader

    monkeypatch.setattr(_utils, "_THREADED_MIN_SAMPLES", 1)
    raw = np.random.default_rng(0).integers(0, 256, 4100, dtype=np.uint8).tobytes()
    try:
        ref = reader._decode_iq(raw, 2)
        for n in (1, 3, 32):
            set_threads(n)
            got = reader._decode_iq(raw, 2)
            _utils._in_chunks(n, 1, lambda a, b: None)
            assert _utils._pools["cpu"]._max_workers == n
            np.testing.assert_array_equal(got[0], ref[0])
            np.testing.assert_array_equal(got[1], ref[1])
        for bad in (0, -2, 2.0, True, "4"):
            with pytest.raises(ValueError, match="positive int"):
                set_threads(bad)
    finally:
        set_threads()
    assert _utils._THREADS == _utils._DEFAULT_THREADS


def test_fork_hooks_leave_no_lock_held(monkeypatch):
    """A lock held by another thread at fork() stays held in the child: the
    child hooks must start fresh ones (simulated: Windows has no fork)."""
    import threading

    from rdefpy import _utils
    from rdefpy.prd import reader

    # Stand-ins, so the real locks are never left held; restored after.
    monkeypatch.setattr(_utils, "_pool_lock", threading.Lock())
    monkeypatch.setattr(_utils, "_pools", {"cpu": object()})
    monkeypatch.setattr(reader, "_buffers", reader._Buffers())
    monkeypatch.setattr(reader, "_readahead", reader._ReadAhead())
    reader._buffers.give(np.empty(4, np.uint8))
    reader._readahead._last = reader._readahead._next = ("stale",)
    _utils._pool_lock.acquire()
    reader._buffers._lock.acquire()
    reader._readahead._lock.acquire()
    _utils._drop_pool()
    reader._buffers.__init__()
    reader._readahead.__init__()
    assert not _utils._pool_lock.locked() and not reader._buffers._lock.locked()
    assert not reader._readahead._lock.locked()
    # ... and nothing of the parent's is carried over: no pool, buffer, thread or pending read.
    assert _utils._pools == {} and reader._buffers._free == []
    ra = reader._readahead
    assert ra._thread is None and ra._last is None and ra._next is None


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
def test_fork_hooks_are_registered():
    """The same, through a real fork(): the three hooks must be registered with
    os.register_at_fork, or the child inherits locks that nothing will release."""
    import threading

    from rdefpy import _utils
    from rdefpy.prd import reader

    held, done = threading.Event(), threading.Event()

    def hold():
        with _utils._pool_lock, reader._buffers._lock, reader._readahead._lock:
            held.set()
            done.wait(30)

    holder = threading.Thread(target=hold)
    holder.start()
    try:
        assert held.wait(30)
        pid = os.fork()
        if pid == 0:  # child: the holder thread does not exist here
            try:
                locks = (_utils._pool_lock, reader._buffers._lock, reader._readahead._lock)
                os._exit(sum(lock.locked() << n for n, lock in enumerate(locks)))  # 0: all three fresh
            finally:
                os._exit(64)
    finally:
        done.set()
        holder.join()
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0  # bit 0: pool lock, 1: buffers, 2: read-ahead


@pytest.mark.parametrize("bits,per", [(16, 1), (8, 1), (4, 1), (2, 2), (1, 4)])
def test_cython_kernels_refuse_short_outputs(bits, per):
    """Bounds checks are off in the kernels: a short output must raise, not overrun."""
    from rdefpy.prd import reader

    if reader._unpack_fast is None:
        pytest.skip("unpack_cyth extension not built")
    from rdefpy.prd import unpack_cyth as u

    src = np.zeros(4, {16: np.uint32, 8: np.uint16}.get(bits, np.uint8))
    out = {16: np.int32, 8: np.int16}.get(bits, np.int8)
    kernel = getattr(u, f"unpack_iq_{bits}_cython")
    big = np.zeros(8 * per, out)
    with pytest.raises(ValueError, match="need"):
        kernel(src, big[:1], big[:1])
    assert not big.any()  # nothing written past the view
    kernel(src, np.zeros(4 * per, out), np.zeros(4 * per, out))  # exact length works


def test_samples_always_match_i_and_q():
    """Built on every access: in-place I/Q edits show up, and editing the
    returned array leaves the record unchanged."""
    pf = read_product(PRD_16)
    pf.load()
    d = pf.records[0].data
    x = d.samples
    x[0] = 0
    assert d.samples[0] != 0
    d.i_samples[0] = 5
    d.q_samples[0] = 7
    assert d.samples[0] == 5 + 7j


def test_write_refuses_an_empty_product_file(tmp_path):
    """§5.1.1: a product file consists of records; none is not a file."""
    out = tmp_path / "x.prd"
    with pytest.raises(ValueError, match="5.1.1"):
        write_product(out, read_product(PRD_16).slice(0, 0))
    assert not out.exists()


def test_write_into_missing_directory_names_the_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing"):
        write_product(tmp_path / "missing" / "x.prd", read_product(PRD_16))


def test_epoch_accepts_trailing_z():
    assert len(read_product(PRD_16, stop_epoch="2026-039T12:00:01Z").records) == 1


def test_version_matches_metadata():
    from importlib.metadata import version

    import rdefpy

    assert rdefpy.__version__ == version("rdefpy")
