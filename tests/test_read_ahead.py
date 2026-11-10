"""Read-ahead of large data sections (reader._ReadAhead): every record gets its own bytes.

A record whose data section is at least ``reader._READ_AHEAD_MIN_BYTES``, read
in file order, has the NEXT record's section read on a background thread while
it decodes. That is a guess (same size, next in the file): these tests check
that a right guess returns that record's bytes, a wrong one is thrown away, and
the file's end and a file changed on disk are respected. Expected samples come
from the bytes written (Table 5-2, §5.3.6), never from rdefpy.
"""

from __future__ import annotations

import functools
import os
import threading

import numpy as np
import pytest

import rdef_spec as spec
from rdefpy import read_product
from rdefpy.prd import reader

N = reader._READ_AHEAD_MIN_BYTES  # smallest data section that is read ahead
HDR = spec.HEADER_SIZE


def _file(write_prd, sizes, bits=16, seed=0, invalid=()):
    """One record per data-section size in bytes, distinct random bytes each, the
    records numbered in *invalid* flagged (validity_flag 1): (path, [data section])."""
    rng = np.random.default_rng(seed)
    datas = [rng.integers(0, 256, n, dtype=np.uint8).tobytes() for n in sizes]
    recs = [spec.header_bytes(sample_size=bits, sample_rate=len(d) * 8 // (2 * bits),
                              time_tag_second_of_day=43200 + k, validity_flag=int(k in invalid)) + d
            for k, d in enumerate(datas)]
    return write_prd(*recs), datas


@functools.lru_cache(maxsize=16)  # rdef_spec decodes 512 KiB of 2-bit in ~0.5 s: once per section
def _want(data: bytes, bits: int = 16) -> np.ndarray:
    """Data section -> I + jQ. 16-bit here (Table 5-2: I in the low half of each
    word; §5.3.6: 2k+1), other sizes by rdef_spec."""
    if bits != 16:
        i, q = spec.decode_values(data, bits)
        return np.array(i) + 1j * np.array(q)
    v = 2 * np.frombuffer(data, "<i2").astype(np.int64) + 1
    return v[0::2] + 1j * v[1::2]


def _offsets(sizes) -> list[int]:
    """File offset of each record's data section."""
    out, pos = [], 0
    for n in sizes:
        out.append(pos + HDR)
        pos += HDR + n
    return out


@pytest.fixture(autouse=True)
def readahead(monkeypatch):
    """A fresh read-ahead state per test: the module's is one per process."""
    ra = reader._ReadAhead()
    monkeypatch.setattr(reader, "_readahead", ra)
    yield ra
    if ra._thread is not None:
        ra._thread.shutdown(wait=True)


@pytest.fixture
def ahead_reads(monkeypatch):
    """Offsets of the data sections read on another thread than the test's."""
    me, seen, real = threading.get_ident(), [], reader._DiskSource._read_now

    def spy(self, f=None):
        if threading.get_ident() != me:
            seen.append(self.offset)
        return real(self, f)

    monkeypatch.setattr(reader._DiskSource, "_read_now", spy)
    return seen


def _settle(ra) -> None:
    """Wait for a read-ahead left pending, so the spy has seen it."""
    if ra._next is not None:
        ra._next[1].result()


def test_reference_decode_agrees_with_the_transcription():
    """_want is the test's own shortcut for 16-bit: hold it to rdef_spec."""
    data = np.random.default_rng(1).integers(0, 256, 4 * 64, dtype=np.uint8).tobytes()
    i, q = spec.decode_values(data, 16)
    assert _want(data).real.tolist() == i and _want(data).imag.tolist() == q


def test_thresholds_are_the_measured_ones():
    """512 KiB read-ahead, 8 MiB parallel read: changing either is a deliberate act."""
    assert reader._READ_AHEAD_MIN_BYTES == 512 * 1024
    assert reader._PARALLEL_READ_MIN_BYTES == 8 * 1024 * 1024
    assert reader._READ_AHEAD_MIN_BYTES < reader._PARALLEL_READ_MIN_BYTES


# =============================================================================
# A right guess: the bytes read ahead are that record's
# =============================================================================


def _by_read_samples(pf):
    return [r.read_samples() for r in pf]


def _by_refilled_buffer(pf):
    out, buf = [], None
    for r in pf:
        buf = r.read_samples(out=buf)  # one array, overwritten per record
        out.append(buf.copy())
    return out


def _by_data(pf):
    return [r.data.samples for r in pf]


def _by_load(pf):
    pf.load()
    return [r.data.samples for r in pf]


@pytest.mark.parametrize("bits", [16, 2])
@pytest.mark.parametrize("read", [_by_read_samples, _by_refilled_buffer, _by_data, _by_load],
                         ids=["read_samples", "out=", "data", "load"])
def test_each_record_read_in_order_gets_its_own_samples(read, bits, decode_path, write_prd, ahead_reads):
    sizes = [N] * 5
    path, datas = _file(write_prd, sizes, bits=bits)
    got = read(read_product(path))
    assert len(got) == len(datas)
    for x, d in zip(got, datas, strict=True):
        np.testing.assert_array_equal(x, _want(d, bits))
    # Records 0 and 1 are read directly (two reads in order start it), 2-4 ahead.
    assert ahead_reads == _offsets(sizes)[2:]


def test_a_record_skipped_by_a_filter_is_a_wrong_guess(decode_path, write_prd, readahead):
    """Record 3 is flagged and filtered out: reading record 2 reads 3 ahead, and
    record 4, asked for next, must get its own bytes, not the guess."""
    path, datas = _file(write_prd, [N] * 6, invalid=(3,))
    pf = read_product(path, filters={"validity_flag": 0})
    assert [r.header.time_tag_second_of_day - 43200 for r in pf] == [0, 1, 2, 4, 5]
    for rec, k in zip(pf, (0, 1, 2, 4, 5), strict=True):
        np.testing.assert_array_equal(rec.read_samples(), _want(datas[k]), err_msg=f"record {k}")
    _settle(readahead)


def test_the_guess_past_num_records_is_not_handed_to_another_record(decode_path, write_prd, readahead):
    """num_records=3 keeps records 0-2 of 5; reading 2 reads record 3 of the file
    ahead. Going back to record 0 must give record 0's bytes."""
    path, datas = _file(write_prd, [N] * 5)
    pf = read_product(path, num_records=3)
    assert len(pf) == 3
    for k in (0, 1, 2, 0):
        np.testing.assert_array_equal(pf[k].read_samples(), _want(datas[k]), err_msg=f"record {k}")
    _settle(readahead)


# =============================================================================
# Where it starts and stops
# =============================================================================


@pytest.mark.parametrize("size, ahead", [(N, True), (N - 4, False)], ids=["512KiB", "512KiB-4"])
def test_read_ahead_starts_at_exactly_the_threshold(size, ahead, write_prd, ahead_reads, readahead):
    sizes = [size] * 4
    path, datas = _file(write_prd, sizes)
    for rec, d in zip(read_product(path), datas, strict=True):
        np.testing.assert_array_equal(rec.read_samples(), _want(d))
    assert ahead_reads == (_offsets(sizes)[2:] if ahead else [])
    assert readahead._next is None


def test_last_record_is_never_read_ahead_past_the_end_of_the_file(write_prd, ahead_reads, readahead):
    sizes = [N] * 3
    path, datas = _file(write_prd, sizes)
    pf = read_product(path)
    for rec in pf:
        rec.read_samples()
    assert ahead_reads == [_offsets(sizes)[2]]  # the last record, and nothing after it
    assert readahead._next is None  # nothing left pending, no buffer held
    assert max(ahead_reads) + N == path.stat().st_size


def test_one_read_or_a_backward_walk_starts_nothing(write_prd, ahead_reads, readahead):
    """Two reads in file order start it: random access and single reads do not."""
    path, datas = _file(write_prd, [N] * 4)
    pf = read_product(path)
    for k in (3, 2, 1, 0, 2, 0):
        np.testing.assert_array_equal(pf[k].read_samples(), _want(datas[k]))
    assert ahead_reads == [] and readahead._next is None


# =============================================================================
# A wrong guess is thrown away
# =============================================================================


def test_out_of_order_access_gets_each_records_own_samples(decode_path, write_prd, readahead):
    path, datas = _file(write_prd, [N] * 5)
    pf = read_product(path)
    for k in (0, 1, 3, 4, 2, 1, 0):  # 0, 1 read 2 ahead; 3 is asked for instead
        np.testing.assert_array_equal(pf[k].read_samples(), _want(datas[k]), err_msg=f"record {k}")
    _settle(readahead)


@pytest.mark.filterwarnings("ignore:Metadata fields vary:RuntimeWarning")
def test_records_of_other_sizes_get_their_own_samples(decode_path, write_prd, readahead):
    """The guess is "same size as this one" (one rate per file, §5.1.3): a file
    that breaks it still decodes every record from its own bytes."""
    path, datas = _file(write_prd, [N, N, 2 * N, N, N // 2, N])
    pf = read_product(path)
    assert [len(r) for r in pf] == [len(d) // 4 for d in datas]
    for k, (rec, d) in enumerate(zip(pf, datas, strict=True)):
        np.testing.assert_array_equal(rec.read_samples(), _want(d), err_msg=f"record {k}")
    _settle(readahead)


def test_two_files_read_alternately_do_not_cross(decode_path, write_prd, readahead):
    path_a, a = _file(write_prd, [N] * 4, seed=1)
    path_b, b = _file(write_prd, [N] * 4, seed=2)
    pa, pb = read_product(path_a), read_product(path_b)
    np.testing.assert_array_equal(pa[0].read_samples(), _want(a[0]))
    np.testing.assert_array_equal(pa[1].read_samples(), _want(a[1]))  # a[2] now read ahead
    for rec, d in zip(pb, b, strict=True):  # same offsets and sizes, another file
        np.testing.assert_array_equal(rec.read_samples(), _want(d))
    np.testing.assert_array_equal(pa[2].read_samples(), _want(a[2]))
    np.testing.assert_array_equal(pa[3].read_samples(), _want(a[3]))
    _settle(readahead)


# =============================================================================
# The file changed while a section was read ahead
# =============================================================================


@pytest.mark.parametrize("change", ["size", "mtime"])
def test_bytes_read_ahead_of_a_changed_file_are_not_returned(change, write_prd, readahead):
    """Record 2 is in memory, read ahead, when the file is replaced: asking for
    it must raise as any other read of a changed file, not hand back stale samples."""
    path, datas = _file(write_prd, [N] * 4)
    pf = read_product(path)
    pf[0].read_samples()
    pf[1].read_samples()
    _settle(readahead)  # record 2 read from the original file, handle closed
    assert readahead._next is not None
    st = path.stat()
    other = bytes(len(path.read_bytes()))  # same layout, different bytes
    if change == "size":
        path.write_bytes(other + bytes(4))
    else:
        path.write_bytes(other)
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
        assert path.stat().st_size == st.st_size
    with pytest.raises(RuntimeError, match="has changed"):
        pf[2].read_samples()
    # That call queued record 3 behind it; its read fails on the same check on
    # the other thread, and asking for record 3 raises here as a direct read does.
    with pytest.raises(RuntimeError, match="has changed"):
        pf[3].read_samples()


# =============================================================================
# Reused raw buffers (reader._Buffers) never back the samples handed out
# =============================================================================


def test_loaded_records_do_not_share_the_reused_read_buffer(decode_path, write_prd, monkeypatch):
    """Sections read by the parallel-read path land in a buffer that is reused
    for the next record. With both thresholds lowered every record takes that
    path (as one slice: 4096 bytes is below the threaded minimum; several
    slices: the next test): all five held at once must still be five
    different records."""
    size, bits = 4096, 8
    monkeypatch.setattr(reader, "_PARALLEL_READ_MIN_BYTES", size)
    monkeypatch.setattr(reader, "_READ_AHEAD_MIN_BYTES", size)
    monkeypatch.setattr(reader, "_buffers", reader._Buffers())
    path, datas = _file(write_prd, [size] * 5, bits=bits)
    want = [spec.decode_values(d, bits) for d in datas]

    pf = read_product(path)
    assert all(r._data is None for r in pf.records)  # 2048 Hz: on disk
    pf.load()
    for rec, (i, q) in zip(pf.records, want, strict=True):
        assert rec.data.i_samples.tolist() == i and rec.data.q_samples.tolist() == q
    assert 1 <= len(reader._buffers._free) <= 2  # the path under test was taken, and is bounded

    pf.release()
    held = [r.data for r in pf.records]  # transient reads, all alive together
    complexes = [r.read_samples() for r in pf.records]
    for d, x, (i, q) in zip(held, complexes, want, strict=True):
        assert d.i_samples.tolist() == i and d.q_samples.tolist() == q
        assert x.real.tolist() == i and x.imag.tolist() == q


@pytest.mark.parametrize("read", [_by_read_samples, _by_data, _by_load], ids=["read_samples", "data", "load"])
def test_sections_read_in_several_slices_get_their_own_bytes(read, decode_path, write_prd, monkeypatch, readahead):
    """A section at or above _PARALLEL_READ_MIN_BYTES is read in one slice per
    thread, each through its own handle at offset + a. Thresholds lowered so a
    4096-byte section splits in 4: every slice, a > 0 included, must land at its
    own place, in this record and in the ones read ahead."""
    from rdefpy import _utils, set_threads

    size, bits = 4096, 8
    monkeypatch.setattr(reader, "_PARALLEL_READ_MIN_BYTES", size)
    monkeypatch.setattr(reader, "_READ_AHEAD_MIN_BYTES", size)
    monkeypatch.setattr(_utils, "_THREADED_MIN_SAMPLES", 1)
    monkeypatch.setattr(reader, "_buffers", reader._Buffers())
    slices, real = [], reader._in_chunks

    def spy(n, per, fn, *, pool="cpu"):
        if pool != "io":
            return real(n, per, fn, pool=pool)
        return real(n, per, lambda a, b: slices.append((a, b)) or fn(a, b), pool=pool)

    monkeypatch.setattr(reader, "_in_chunks", spy)
    path, datas = _file(write_prd, [size] * 5, bits=bits)
    want = [spec.decode_values(d, bits) for d in datas]
    set_threads(4)
    try:
        got = read(read_product(path))
        _settle(readahead)
    finally:
        set_threads()
    for x, (i, q) in zip(got, want, strict=True):
        assert x.real.tolist() == i and x.imag.tolist() == q
    # Every record read this way, each in the same 4 slices: [0, 1024), ..., [3072, 4096)
    assert sorted(set(slices)) == [(0, 1024), (1024, 2048), (2048, 3072), (3072, 4096)]
    assert len(slices) == 4 * 5
