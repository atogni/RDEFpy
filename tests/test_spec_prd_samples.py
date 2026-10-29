"""Product file data section against CCSDS 506.1-B-1 §5.3 and Table 5-2.

Decoding is checked against ``rdef_spec`` (a bit-by-bit transcription of
Table 5-2 and §5.3.6), and against literal bytes worked out by hand from
Table 5-2, so a bug shared by the library and the transcription still shows.
Every decode test runs once per decoder (``decode_path``: Cython and NumPy)
and through both read paths: loaded (``load()``) and read on access.

The data contract, in one line: raw code k on disk <-> sample value 2k+1 in
memory, with nothing else done to it. No scaling, no centring, no clipping.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import rdef_spec as spec
from rdefpy import ProductFile, Record, RecordData, RecordHeader, read_product, write_product

DATA = Path(__file__).parent / "data"
FIXTURES = sorted(DATA.glob("*.prd"))
BITS = spec.VALID_SAMPLE_SIZES
STORAGE = {16: np.int32, 8: np.int16, 4: np.int8, 2: np.int8, 1: np.int8}


def _open(path, stream, **kw):
    """read_product, then load() -- or, for *stream*, release() so every
    record is read on access (read_product auto-loads records <= 1 kHz)."""
    pf = read_product(path, **kw)
    if stream:
        pf.release()
    else:
        pf.load()
    return pf


def _read(path, stream):
    pf = _open(path, stream)
    i = np.concatenate([r.data.i_samples for r in pf.records])
    q = np.concatenate([r.data.q_samples for r in pf.records])
    return pf, i, q


def _raw_record(data: bytes, bits: int, sod: int = 43200) -> bytes:
    """A spec header for *data* (whole 32-bit words) followed by *data*."""
    rate = len(data) * 8 // (2 * bits)
    return spec.header_bytes(sample_size=bits, sample_rate=rate,
                             time_tag_second_of_day=sod) + data


def _spec_parse_file(raw: bytes):
    """Walk a product file with the spec: [(header dict, I values, Q values)]."""
    out, pos = [], 0
    while pos < len(raw):
        hdr = spec.parse_header(raw[pos : pos + spec.HEADER_SIZE])
        data = raw[pos + spec.HEADER_SIZE : pos + hdr["record_length"]]
        out.append((hdr, *spec.decode_values(data, hdr["sample_size"])))
        pos += hdr["record_length"]
    assert pos == len(raw)
    return out


def _product(records) -> ProductFile:
    """ProductFile from [(header overrides, I values, Q values)]."""
    pf = ProductFile()
    for overrides, v_i, v_q in records:
        f = spec.default_fields(**overrides)
        model = {k: f[k] for k in spec.MODEL_FIELDS}
        pf.records.append(Record(RecordHeader(model),
                                 RecordData(f["sample_size"], f["sample_rate"], v_i, v_q)))
    return pf


# =============================================================================
# Table 5-2 worked by hand: one 32-bit word per sample size
# =============================================================================

GOLDEN = {
    # 16-bit: MSB Q1 I1 LSB. I1 = 0x8000 (k=-32768), Q1 = 0x7FFF (k=32767).
    16: (bytes([0x00, 0x80, 0xFF, 0x7F]), [-65535], [65535]),
    # 8-bit: MSB Q2 I2 Q1 I1 LSB, so bytes are I1 Q1 I2 Q2.
    8: (bytes([0x01, 0xFF, 0x80, 0x7F]), [3, -255], [-1, 255]),
    # 4-bit: one complex sample per byte, I in the low nibble.
    #   0x8F: I1=F(-1) Q1=8(-8) | 0x10: I2=0 Q2=1 | 0x7E: I3=E(-2) Q3=7 | 0x01: I4=1 Q4=0
    4: (bytes([0x8F, 0x10, 0x7E, 0x01]), [-1, 1, -3, 3], [-15, 3, 15, 1]),
    # 2-bit: two complex samples per byte, I in the lower 2 bits.
    #   0xE4 = 11 10 01 00: I1=00 Q1=01 I2=10 Q2=11
    #   0x1B = 00 01 10 11: I3=11 Q3=10 I4=01 Q4=00
    #   0x00: all code 0 | 0xFF: all code -1
    2: (bytes([0xE4, 0x1B, 0x00, 0xFF]),
        [1, -3, -1, 3, 1, 1, -1, -1], [3, -1, -3, 1, 1, 1, -1, -1]),
    # 1-bit: [Q16,I16] ... [Q1,I1], LSB first; bit 0 -> +1, bit 1 -> -1.
    #   0x06: bit1 (Q1) and bit2 (I2) set | 0x80 in byte 4: bit 31 = Q16
    1: (bytes([0x06, 0x00, 0x00, 0x80]),
        [1, -1] + [1] * 14, [-1, 1] + [1] * 13 + [-1]),
}


@pytest.mark.parametrize("stream", [False, True], ids=["loaded", "on_access"])
@pytest.mark.parametrize("bits", BITS)
def test_table_5_2_golden_word(bits, stream, decode_path, write_prd):
    data, want_i, want_q = GOLDEN[bits]
    _, i, q = _read(write_prd(_raw_record(data, bits)), stream)
    assert i.tolist() == want_i
    assert q.tolist() == want_q


def test_golden_words_agree_with_the_transcription():
    """The hand-worked words and rdef_spec must tell the same story."""
    for bits, (data, want_i, want_q) in GOLDEN.items():
        assert spec.decode_values(data, bits) == (want_i, want_q)
        assert spec.encode_values(want_i, want_q, bits) == data


@pytest.mark.parametrize("bits", BITS)
def test_writer_produces_the_golden_word(bits, tmp_path):
    data, v_i, v_q = GOLDEN[bits]
    out = tmp_path / "g.prd"
    write_product(out, _product([(dict(sample_size=bits, sample_rate=len(v_i)), v_i, v_q)]))
    assert out.read_bytes() == _raw_record(data, bits)


# =============================================================================
# One slot at a time: catches I/Q swaps and sample-order errors (§5.3.3-§5.3.4)
# =============================================================================


def _one_hot_cases():
    for bits in BITS:
        mask = (1 << bits) - 1
        msb = 1 << (bits - 1)
        for word in (0, 1):
            for slot in range(spec.per_word(bits)):
                for comp in ("I", "Q"):
                    for code in sorted({mask, msb}):
                        yield pytest.param(bits, word, slot, comp, code,
                                           id=f"{bits}b-w{word}-s{slot}-{comp}-{code:#x}")


@pytest.mark.parametrize("bits,word,slot,comp,code", list(_one_hot_cases()))
def test_one_slot_set(bits, word, slot, comp, code, decode_path, write_prd):
    """Set one I or Q slot of one word; exactly that sample must change."""
    shift = 2 * bits * slot + (bits if comp == "Q" else 0)  # Table 5-2
    words = [0, 0]
    words[word] = code << shift
    data = b"".join(w.to_bytes(4, "little") for w in words)
    _, i, q = _read(write_prd(_raw_record(data, bits)), stream=False)

    k = code - (1 << bits) if code >> (bits - 1) else code  # two's complement
    n = word * spec.per_word(bits) + slot  # time index (§5.3.4: LSB first)
    want_i = np.ones(2 * spec.per_word(bits), dtype=int)
    want_q = want_i.copy()
    (want_i if comp == "I" else want_q)[n] = 2 * k + 1
    assert i.tolist() == want_i.tolist()
    assert q.tolist() == want_q.tolist()


# =============================================================================
# Every code, every byte value
# =============================================================================


@pytest.mark.parametrize("bits", BITS)
def test_every_code_decodes_to_2k_plus_1(bits, decode_path, write_prd):
    """§5.3.6 over the whole b-bit two's-complement range, I and Q independently."""
    codes = list(spec.code_range(bits))
    n_per = spec.per_word(bits)
    codes += [0] * (-len(codes) % n_per)  # whole words
    k_i, k_q = codes, codes[::-1]
    path = write_prd(spec.record_bytes(k_i, k_q, sample_size=bits, sample_rate=len(codes)))
    _, i, q = _read(path, stream=False)
    assert i.tolist() == [2 * k + 1 for k in k_i]
    assert q.tolist() == [2 * k + 1 for k in k_q]
    assert i.min() == -(2**bits - 1) and i.max() == 2**bits - 1


@pytest.mark.parametrize("bits", BITS)
def test_every_byte_value_at_every_byte_position(bits, decode_path, write_prd):
    """All 256 byte values in each of the four byte positions of a word."""
    data = b"".join(bytes((v + s) % 256 for v in range(256)) for s in range(4))
    _, i, q = _read(write_prd(_raw_record(data, bits)), stream=False)
    want_i, want_q = spec.decode_values(data, bits)
    assert i.tolist() == want_i
    assert q.tolist() == want_q


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("stream", [False, True], ids=["loaded", "on_access"])
@pytest.mark.parametrize("bits", BITS)
def test_random_multi_record_file_matches_the_spec(bits, stream, seed, decode_path, write_prd):
    rng = np.random.default_rng(1000 * bits + seed)
    raw = b"".join(
        _raw_record(rng.integers(0, 256, 4 * 24, dtype=np.uint8).tobytes(), bits, sod=s)
        for s in range(3)
    )
    pf, _, _ = _read(write_prd(raw), stream)
    ref = _spec_parse_file(raw)
    assert len(pf.records) == len(ref)
    for rec, (_hdr, want_i, want_q) in zip(pf.records, ref, strict=True):
        assert rec.data.i_samples.tolist() == want_i
        assert rec.data.q_samples.tolist() == want_q
        assert len(rec.data) == rec.header.sample_rate  # §5.1.1: one second


@pytest.mark.parametrize("bits", BITS)
def test_decoded_values_are_odd_nonzero_and_in_range(bits, decode_path, write_prd):
    """§5.3.6 notes: zero is absent; values are symmetric about zero."""
    raw = np.random.default_rng(bits).integers(0, 256, 4 * 256, dtype=np.uint8).tobytes()
    _, i, q = _read(write_prd(_raw_record(raw, bits)), stream=False)
    for x in (i, q):
        assert x.dtype == STORAGE[bits]
        assert np.all(x % 2 != 0)
        assert np.all(np.abs(x) <= 2**bits - 1)


# =============================================================================
# Encoding: values -> bytes (the inverse direction)
# =============================================================================


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("bits", BITS)
def test_writer_packs_values_exactly_as_the_spec(bits, seed, tmp_path):
    rng = np.random.default_rng(seed)
    lo, hi = spec.code_range(bits)[0], spec.code_range(bits)[-1]
    rate = 4 * spec.per_word(bits)
    recs, raw = [], b""
    for s in range(3):
        k_i = rng.integers(lo, hi + 1, rate).tolist()
        k_q = rng.integers(lo, hi + 1, rate).tolist()
        hdr = dict(sample_size=bits, sample_rate=rate, time_tag_second_of_day=s)
        recs.append((hdr, [2 * k + 1 for k in k_i], [2 * k + 1 for k in k_q]))
        raw += spec.record_bytes(k_i, k_q, **hdr)
    out = tmp_path / "w.prd"
    write_product(out, _product(recs))
    assert out.read_bytes() == raw


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("stream", [False, True], ids=["loaded", "on_access"])
@pytest.mark.parametrize("bits", BITS)
def test_any_bytes_survive_read_then_write(bits, stream, seed, decode_path, write_prd, tmp_path):
    """Every bit pattern is a valid sample, so decode/encode is a bijection."""
    rng = np.random.default_rng(seed + 10 * bits)
    raw = b"".join(
        _raw_record(rng.integers(0, 256, 4 * 32, dtype=np.uint8).tobytes(), bits, sod=s)
        for s in range(4)
    )
    out = tmp_path / "rt.prd"
    write_product(out, _open(write_prd(raw), stream))
    assert out.read_bytes() == raw


@pytest.mark.parametrize("bits", BITS)
def test_any_valid_values_survive_write_then_read(bits, decode_path, tmp_path):
    rng = np.random.default_rng(bits)
    lo, hi = spec.code_range(bits)[0], spec.code_range(bits)[-1]
    rate = 8 * spec.per_word(bits)
    v_i = (2 * rng.integers(lo, hi + 1, rate) + 1).tolist()
    v_q = (2 * rng.integers(lo, hi + 1, rate) + 1).tolist()
    out = tmp_path / "v.prd"
    write_product(out, _product([(dict(sample_size=bits, sample_rate=rate), v_i, v_q)]))
    _, i, q = _read(out, stream=False)
    assert i.tolist() == v_i and q.tolist() == v_q


@pytest.mark.parametrize("bits", BITS)
def test_writer_refuses_values_with_no_code(bits, tmp_path):
    """§5.3.6: only odd values within ±(2**bits - 1) have a code.  The writer
    refuses the rest instead of clipping or rounding them."""
    lim = 2**bits - 1
    rate = spec.per_word(bits)
    ok = np.ones(rate, dtype=STORAGE[bits])
    for bad in (lim + 2, -lim - 2, 0):
        v = ok.copy()
        v[-1] = bad
        pf = _product([(dict(sample_size=bits, sample_rate=rate), ok, v)])
        with pytest.raises(ValueError, match="2k\\+1"):
            write_product(tmp_path / "x.prd", pf)


@pytest.mark.parametrize("bits", BITS)
def test_record_data_refuses_a_lossy_cast(bits):
    """Samples are stored exactly as given or not at all: no wrap, no
    truncation."""
    wraps = int(np.iinfo(STORAGE[bits]).max) + 2
    for bad in ([wraps], [2.5], [np.nan]):
        with pytest.raises(ValueError):
            RecordData(bits, 1, bad, [1])
        rd = RecordData(bits, 1, [1], [1])
        with pytest.raises(ValueError):
            rd.i_samples = bad
        with pytest.raises(ValueError):
            rd.q_samples = bad


def test_record_data_keeps_storage_dtype_input_uncopied():
    v = np.ones(4, dtype=np.int32)
    rd = RecordData(16, 4, v, v)
    assert rd.i_samples is v and rd.q_samples is v


def test_16_bit_full_code_range_both_directions(decode_path, write_prd, tmp_path):
    """All 65536 sixteen-bit codes, for I and (reversed) for Q."""
    k = np.arange(-32768, 32768, dtype=np.int64)
    k_q = k[::-1]
    raw = spec.header_bytes(sample_size=16, sample_rate=k.size)
    words = (k.astype("<i2").view("<u2").astype(np.uint32)
             | (k_q.astype("<i2").view("<u2").astype(np.uint32) << 16))
    raw += words.astype("<u4").tobytes()
    _, i, q = _read(write_prd(raw), stream=False)
    np.testing.assert_array_equal(i, 2 * k + 1)
    np.testing.assert_array_equal(q, 2 * k_q + 1)
    out = tmp_path / "16.prd"
    write_product(out, read_product(write_prd(raw)))
    assert out.read_bytes() == raw


# =============================================================================
# The samples handed to the caller are the recorded ones
# =============================================================================


@pytest.mark.parametrize("dtype", [np.complex128, np.complex64])
@pytest.mark.parametrize("bits", BITS)
def test_complex_samples_are_exactly_i_plus_jq(bits, dtype, decode_path, write_prd):
    raw = np.random.default_rng(bits).integers(0, 256, 4 * 16, dtype=np.uint8).tobytes()
    want_i, want_q = spec.decode_values(raw, bits)
    for stream in (False, True):
        rec = _open(write_prd(_raw_record(raw, bits)), stream,
                    dtype=dtype).records[0]
        x = rec.data.samples
        assert x.dtype == dtype
        assert x.real.tolist() == want_i  # no mean removal, scaling or window
        assert x.imag.tolist() == want_q


# =============================================================================
# The real fixtures, walked independently
# =============================================================================


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.name)
def test_fixture_decodes_as_the_spec_says(path, decode_path):
    raw = path.read_bytes()
    ref = _spec_parse_file(raw)
    for stream in (False, True):
        pf = _open(path, stream)
        assert len(pf.records) == len(ref)
        for rec, (hdr, want_i, want_q) in zip(pf.records, ref, strict=True):
            for name in spec.MODEL_FIELDS:
                got = getattr(rec.header, name)
                assert (bytes(got) if isinstance(got, bytes | bytearray) else got) == hdr[name]
            assert rec.data.i_samples.tolist() == want_i
            assert rec.data.q_samples.tolist() == want_q
        assert hdr["record_label"] == b"RDEF" and hdr["end_label"] == -99999


@pytest.mark.parametrize("stream", [False, True], ids=["loaded", "on_access"])
@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.name)
def test_fixture_read_write_is_byte_identical(path, stream, decode_path, tmp_path):
    out = tmp_path / path.name
    write_product(out, _open(path, stream))
    assert out.read_bytes() == path.read_bytes()


# =============================================================================
# Loaded vs on-access state (read_product auto-loads records <= 1 kHz)
# =============================================================================


_MIXED_RATES = pytest.mark.filterwarnings("ignore:Metadata fields vary:RuntimeWarning")


def _rate_file(write_prd, *rates, seed=0):
    """One 16-bit record per rate, random codes; returns (path, [(I, Q)])."""
    rng = np.random.default_rng(seed)
    codes = spec.code_range(16)
    recs, want = [], []
    for sod, rate in enumerate(rates):
        k_i, k_q = (rng.integers(codes.start, codes.stop, rate).tolist() for _ in "iq")
        recs.append(spec.record_bytes(k_i, k_q, sample_size=16, sample_rate=rate,
                                      time_tag_second_of_day=sod))
        want.append(([spec.value_of(k) for k in k_i], [spec.value_of(k) for k in k_q]))
    return write_prd(*recs), want


def _clobber(path):
    """Replace the file with different bytes of a different size."""
    path.write_bytes(b"\0" * (path.stat().st_size + 4))


@_MIXED_RATES
def test_auto_load_threshold_is_inclusive_at_1000_hz(decode_path, write_prd):
    """<= 1000 Hz is in memory after read_product; 1001 Hz still reads from disk."""
    path, want = _rate_file(write_prd, 1000, 1001)
    slow, fast = read_product(path).records
    _clobber(path)
    assert slow.data.i_samples.tolist() == want[0][0]  # never touches the disk
    assert slow.data.q_samples.tolist() == want[0][1]
    with pytest.raises(RuntimeError):
        fast.data  # on access: sees the changed file and refuses


@_MIXED_RATES
def test_release_returns_an_auto_loaded_file_to_on_access(decode_path, write_prd):
    path, want = _rate_file(write_prd, 16, 1000)
    pf = read_product(path)
    pf.release()
    for rec, (wi, wq) in zip(pf.records, want, strict=True):
        assert rec.data.i_samples.tolist() == wi  # same values from disk
        assert rec.data.q_samples.tolist() == wq
    _clobber(path)
    for rec in pf.records:
        with pytest.raises(RuntimeError):
            rec.data


@_MIXED_RATES
def test_load_pins_a_fast_record_and_values_match_on_access(decode_path, write_prd):
    path, want = _rate_file(write_prd, 1001, 2000)
    pf = read_product(path)
    on_access = [(r.data.i_samples.tolist(), r.data.q_samples.tolist()) for r in pf.records]
    pf.load()
    _clobber(path)
    loaded = [(r.data.i_samples.tolist(), r.data.q_samples.tolist()) for r in pf.records]
    assert on_access == loaded == want
