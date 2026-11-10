"""Product file record framing and header against CCSDS 506.1-B-1 §5.1-§5.2.

Every expectation comes from ``rdef_spec`` (the Blue Book transcribed into
code), never from rdefpy's own struct or constants. The library is checked in
both directions: bytes the spec defines must read back to the right fields,
and fields the library writes must land on the right bytes.
"""

from __future__ import annotations

import io
import math
import os
import struct
import warnings

import numpy as np
import pytest

import rdef_spec as spec
from conftest import _clobber
from rdefpy import (
    ProductFile,
    Record,
    RecordData,
    RecordHeader,
    pack_record_header,
    read_headers,
    read_product,
    scan_product,
    write_product,
)

# Three header value sets: every field distinct, every field at its maximum,
# every field at its minimum. Together they catch a swapped, shifted, signed
# or wrongly sized field anywhere in Table 5-1.
DISTINCT = dict(
    record_version_id=1,
    station_id=0xA1B2,
    spacecraft_id=0xC3D4,
    sample_size=16,
    sample_rate=258,
    validity_flag=0xE5F6,
    agency_flag=2,
    rf_to_if_downconv=8.4e9 + 0.5,
    if_to_channel_downconv=-1.25e8,
    time_tag_year=2028,
    time_tag_doy=366,
    time_tag_second_of_day=86400,
    time_tag_picoseconds_of_the_second=987654321012.0,
    channel_accumulated_phase=123456789.0,
    channel_phase_polynomial_coefficient_0=0.123456789,
    channel_phase_polynomial_coefficient_1=-2.5e5,
    channel_phase_polynomial_coefficient_2=1.0e-3,
    channel_phase_polynomial_coefficient_3=-7.0e-9,
    empty_fields_future_extension=bytes(range(1, 37)),
    empty_fields_internal_agency_use=bytes(range(100, 140)),
)
MAXIMA = dict(
    record_version_id=1,
    station_id=0xFFFF,
    spacecraft_id=0xFFFF,
    sample_size=16,
    sample_rate=16,
    validity_flag=0xFFFF,
    agency_flag=3,
    rf_to_if_downconv=1.7976931348623157e308,
    if_to_channel_downconv=-1.7976931348623157e308,
    time_tag_year=0xFFFF,
    time_tag_doy=366,
    time_tag_second_of_day=86400,
    time_tag_picoseconds_of_the_second=999999999999.0,
    channel_accumulated_phase=1.7976931348623157e308,
    channel_phase_polynomial_coefficient_0=5e-324,
    channel_phase_polynomial_coefficient_1=-5e-324,
    channel_phase_polynomial_coefficient_2=2.2250738585072014e-308,
    channel_phase_polynomial_coefficient_3=-2.2250738585072014e-308,
    empty_fields_future_extension=b"\xff" * 36,
    empty_fields_internal_agency_use=b"\xff" * 40,
)
MINIMA = dict(
    record_version_id=1,
    station_id=0,
    spacecraft_id=0,
    sample_size=16,
    sample_rate=1,  # 1 * 2 * 16 = 32: the smallest legal record
    validity_flag=0,
    agency_flag=0,
    rf_to_if_downconv=0.0,
    if_to_channel_downconv=0.0,
    time_tag_year=0,
    time_tag_doy=1,
    time_tag_second_of_day=0,
    time_tag_picoseconds_of_the_second=0.0,
)
VALUE_SETS = {"distinct": DISTINCT, "maxima": MAXIMA, "minima": MINIMA}


def _zero_record(**overrides) -> bytes:
    """A spec-built record whose samples are all code 0 (value +1)."""
    f = spec.default_fields(**overrides)
    n = f["sample_rate"]
    return spec.record_bytes([0] * n, [0] * n, **overrides)


def _model(**overrides) -> dict:
    """Spec header fields as a RecordHeader dict (label/end label are implicit)."""
    f = spec.default_fields(**overrides)
    return {k: f[k] for k in spec.MODEL_FIELDS}


def _open(path, stream, **kw):
    """read_product, then load() -- or, for *stream*, release() so every
    record is read on access (read_product auto-loads records <= 1 kHz)."""
    pf = read_product(path, t0_offset=0.0, **kw)  # explicit: no picoseconds advisory
    if stream:
        pf.release()
    else:
        pf.load()
    return pf


READERS = {
    "loaded": lambda p: _open(p, stream=False),
    "on_access": lambda p: _open(p, stream=True),
    "headers": lambda p: read_headers(p, t0_offset=0.0),
}


# =============================================================================
# The transcription itself (guards rdef_spec against a typo)
# =============================================================================


def test_layout_tiles_the_176_byte_header_in_table_5_1_order():
    """Table 5-1: 23 rows (the 2 empty fields included), contiguous, 176 bytes."""
    end = 0
    for _name, off, size, _kind in spec.HEADER_LAYOUT:
        assert off == end
        end = off + size
    assert end == spec.HEADER_SIZE == 176
    assert len(spec.HEADER_LAYOUT) == 23
    sizes = {n: s for n, _, s, _ in spec.HEADER_LAYOUT}
    assert sizes["empty_fields_future_extension"] == 36
    assert sizes["empty_fields_internal_agency_use"] == 40
    assert all(sizes[n] == 8 for n in spec.FLOAT_FIELDS)
    assert len(spec.FLOAT_FIELDS) == 8


def test_model_carries_every_table_5_1_field():
    """RecordHeader exposes every Table 5-1 field (label/end label implicit)."""
    hdr = RecordHeader()
    for name in spec.MODEL_FIELDS:
        assert hasattr(hdr, name), name


# =============================================================================
# Encoding: fields land on the right bytes (§5.1.5, §5.2, Table 5-1)
# =============================================================================


@pytest.mark.parametrize("values", VALUE_SETS.values(), ids=VALUE_SETS.keys())
def test_pack_header_is_byte_exact(values):
    """Every field at its Table 5-1 offset, little-endian, binary64 floats."""
    packed = pack_record_header(RecordHeader(_model(**values)))
    assert len(packed) == spec.HEADER_SIZE
    assert packed == spec.header_bytes(**values)


def test_label_and_end_label_literal_bytes():
    """RECORD LABEL 'RDEF' at bytes 0-3, END LABEL -99999 (int32 LE) at 172-175."""
    packed = pack_record_header(RecordHeader(_model()))
    assert packed[0:4] == b"RDEF"
    assert packed[172:176] == bytes([0x61, 0x79, 0xFE, 0xFF])  # -99999, LE


def test_multi_byte_values_are_little_endian():
    """§5.1.5: least significant byte first, checked on literal bytes."""
    packed = pack_record_header(
        RecordHeader(_model(sample_rate=0x01020304, station_id=0x0A0B,
                            rf_to_if_downconv=1.0))
    )
    assert packed[16:20] == bytes([0x04, 0x03, 0x02, 0x01])  # SAMPLE RATE
    assert packed[10:12] == bytes([0x0B, 0x0A])  # STATION ID
    assert packed[24:32] == bytes([0, 0, 0, 0, 0, 0, 0xF0, 0x3F])  # 1.0 binary64


@pytest.mark.parametrize("bits", spec.VALID_SAMPLE_SIZES)
@pytest.mark.parametrize("multiple", [1, 2, 7])
def test_record_length_follows_table_5_1(bits, multiple, tmp_path):
    """RECORD LENGTH = 2*rate*size/8 + 176, and it is the bytes actually written."""
    rate = multiple * spec.per_word(bits)  # smallest word-aligned rates
    packed = pack_record_header(RecordHeader(_model(sample_size=bits, sample_rate=rate)))
    declared = struct.unpack_from("<I", packed, 4)[0]
    assert declared == spec.record_length(rate, bits)

    ones = np.ones(rate, dtype=np.int32)
    pf = ProductFile()
    pf.records.append(
        Record(RecordHeader(_model(sample_size=bits, sample_rate=rate)),
               RecordData(bits, rate, ones, ones))
    )
    out = tmp_path / "r.prd"
    write_product(out, pf)
    assert out.stat().st_size == declared


# =============================================================================
# Decoding: bytes read back to the right fields
# =============================================================================


@pytest.mark.parametrize("how", READERS)
@pytest.mark.parametrize("values", VALUE_SETS.values(), ids=VALUE_SETS.keys())
def test_reader_decodes_every_field_from_its_offset(values, how, write_prd):
    path = write_prd(_zero_record(**values))
    hdr = READERS[how](path).records[0].header
    expected = spec.default_fields(**values)
    for name in spec.MODEL_FIELDS:
        got = getattr(hdr, name)
        want = expected[name]
        if isinstance(want, bytes):
            assert bytes(got) == want, name
        else:
            assert got == want, name
            assert type(got) is type(want), f"{name}: {type(got).__name__}"


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("values", VALUE_SETS.values(), ids=VALUE_SETS.keys())
def test_header_read_write_is_byte_exact(values, stream, write_prd, tmp_path):
    """Spec bytes -> read -> write -> the same bytes, reserved fields included."""
    raw = _zero_record(**values) + _zero_record(
        **{**values, "time_tag_second_of_day": 1}
    )
    path = write_prd(raw)
    out = tmp_path / "rt.prd"
    write_product(out, _open(path, stream))
    assert out.read_bytes() == raw


def test_reserved_fields_are_passed_through_untouched(write_prd, tmp_path):
    """Table 5-1: the 76 reserved bytes belong to the agency; never zero them."""
    raw = _zero_record(**DISTINCT)
    pf = read_product(write_prd(raw), t0_offset=0.0)
    hdr = pf.records[0].header
    assert bytes(hdr.empty_fields_future_extension) == DISTINCT["empty_fields_future_extension"]
    assert (
        bytes(hdr.empty_fields_internal_agency_use)
        == DISTINCT["empty_fields_internal_agency_use"]
    )
    write_product(tmp_path / "o.prd", pf)
    assert (tmp_path / "o.prd").read_bytes()[96:172] == raw[96:172]


def test_records_with_different_layouts_in_one_file(write_prd):
    """§5.1.4: each record's own header fully determines its data length."""
    a = _zero_record(sample_size=16, sample_rate=3)
    b = _zero_record(sample_size=2, sample_rate=16, time_tag_second_of_day=1)
    c = _zero_record(sample_size=1, sample_rate=64, time_tag_second_of_day=2)
    assert len({len(a), len(b), len(c)}) == 3
    path = write_prd(a, b, c)
    assert scan_product(path) == (3, False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # metadata varies on purpose
        for how in READERS:
            pf = READERS[how](path)
            assert [r.header.sample_size for r in pf.records] == [16, 2, 1]
            assert [r.header.record_length for r in pf.records] == [
                len(a), len(b), len(c)
            ]
        assert [len(r.data) for r in read_product(path).records] == [3, 16, 64]


def test_scan_product_counts_and_checks_lengths(write_prd):
    path = write_prd(*[_zero_record(time_tag_second_of_day=s) for s in range(5)])
    assert scan_product(path) == (5, True)


# =============================================================================
# Framing violations the reader must refuse
# =============================================================================


def _bad(**overrides) -> bytes:
    """A header with *overrides*, followed by as many data bytes as it declares."""
    f = spec.default_fields(**overrides)
    return spec.header_bytes(**overrides) + bytes(max(0, f["record_length"] - 176))


FATAL = {
    "label RDEX": _bad(record_label=b"RDEX"),
    "label rdef": _bad(record_label=b"rdef"),
    "end label -99998": _bad(end_label=-99998),
    "end label +99999": _bad(end_label=99999),
    "end label 0": _bad(end_label=0),
    "sample size 0": _bad(sample_size=0, record_length=176),
    "sample size 3": _bad(sample_size=3, record_length=176 + 12),
    "sample size 32": _bad(sample_size=32, record_length=176 + 128),
    "1-bit not word aligned": _bad(sample_size=1, sample_rate=8),
    "2-bit not word aligned": _bad(sample_size=2, sample_rate=4),
    "4-bit not word aligned": _bad(sample_size=4, sample_rate=2),
    "8-bit not word aligned": _bad(sample_size=8, sample_rate=1),
    "length +1": _bad(record_length=spec.record_length(16, 16) + 1),
    "length -1": _bad(record_length=spec.record_length(16, 16) - 1),
    "length header only": _bad(record_length=176),
}


@pytest.mark.parametrize("how", READERS)
@pytest.mark.parametrize("raw", FATAL.values(), ids=FATAL.keys())
def test_reader_refuses_a_malformed_record(raw, how, write_prd):
    with pytest.raises(ValueError):
        READERS[how](write_prd(raw))


@pytest.mark.parametrize("how", READERS)
@pytest.mark.parametrize("raw", FATAL.values(), ids=FATAL.keys())
def test_reader_refuses_a_malformed_second_record(raw, how, write_prd):
    """A good first record does not let a bad second one through."""
    with pytest.raises(ValueError):
        READERS[how](write_prd(_zero_record(), raw))


def test_scan_product_refuses_a_bad_label_and_a_short_length(write_prd):
    with pytest.raises(ValueError):
        scan_product(write_prd(_bad(record_label=b"XDEF")))
    with pytest.raises(ValueError):
        scan_product(write_prd(_bad(record_length=100)))


@pytest.mark.parametrize("how", READERS)
def test_truncated_header_is_an_error(how, write_prd):
    path = write_prd(_zero_record(), _zero_record()[:100])
    with pytest.raises(EOFError):
        READERS[how](path)


def test_scan_product_refuses_a_truncated_prefix(write_prd):
    """scan_product reads only the 8-byte prefix, so that is all it can check."""
    with pytest.raises(EOFError):
        scan_product(write_prd(_zero_record(), _zero_record()[:5]))


@pytest.mark.parametrize("how", READERS)
def test_truncated_data_section_is_an_error(how, write_prd):
    """§5.1.4: the header fixes the data length; a short file is broken."""
    path = write_prd(_zero_record(), _zero_record()[:-1])
    with pytest.raises(EOFError):
        READERS[how](path)


def test_empty_file_reads_as_no_records(write_prd):
    path = write_prd(b"")
    for how in READERS:
        assert len(READERS[how](path).records) == 0
    assert scan_product(path)[0] == 0


# =============================================================================
# Out-of-range values: readable, but reported (Table 5-1 allowed values)
# =============================================================================

SOFT = {
    "version 0": ("record_version_id", dict(record_version_id=0)),
    "version 2": ("record_version_id", dict(record_version_id=2)),
    "agency 4": ("agency_flag", dict(agency_flag=4)),
    "doy 0": ("time_tag_doy", dict(time_tag_doy=0)),
    "doy 367": ("time_tag_doy", dict(time_tag_doy=367)),
    "sod 86401": ("time_tag_second_of_day", dict(time_tag_second_of_day=86401)),
    "picoseconds -1": ("time_tag_picoseconds", dict(time_tag_picoseconds_of_the_second=-1.0)),
    "picoseconds 1e12": ("time_tag_picoseconds", dict(time_tag_picoseconds_of_the_second=1e12)),
    "sample rate 0": ("sample_rate", dict(sample_rate=0)),
    **{f"{f} {b}": (f, {f: v}) for f in spec.FLOAT_FIELDS
       for b, v in (("nan", math.nan), ("-inf", -math.inf))},
}


def _reported(capsys, caught) -> str:
    """Everything the reader said: stdout plus warning messages."""
    return capsys.readouterr().out + "".join(str(w.message) for w in caught)


@pytest.mark.parametrize("label,overrides", SOFT.values(), ids=SOFT.keys())
def test_out_of_range_field_is_read_and_reported(label, overrides, write_prd, capsys):
    path = write_prd(_zero_record(**overrides))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pf = read_product(path)
    assert len(pf.records) == 1
    (name, value), = overrides.items()
    got = getattr(pf.records[0].header, name)
    assert got == value or (math.isnan(value) and math.isnan(got))  # read as is
    mine = [w for w in caught if w.category is UserWarning]
    assert mine and all(os.path.samefile(w.filename, __file__) for w in mine)  # points at the caller
    assert label in _reported(capsys, caught)


@pytest.mark.parametrize("field", spec.FLOAT_FIELDS)
def test_negative_zero_is_read_as_zero_and_reported_once(field, write_prd):
    """§5.2.4: '-0' is not supported; read as +0.0, one warning for the file."""
    path = write_prd(*[_zero_record(**{field: -0.0})] * 3)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default")
        pf = read_product(path)
    assert all(math.copysign(1.0, getattr(r.header, field)) == 1.0 for r in pf.records)
    (w,) = caught
    assert f"{field}: -0.0" in str(w.message) and os.path.samefile(w.filename, __file__)


def test_out_of_range_reported_once_even_if_value_varies(write_prd):
    path = write_prd(*[_zero_record(time_tag_picoseconds_of_the_second=-p) for p in (1.0, 2.0, 3.0)])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default")
        read_product(path)
    (w,) = caught
    assert "time_tag_picoseconds" in str(w.message)


IN_RANGE = {
    "doy 1": dict(time_tag_doy=1),
    "doy 366": dict(time_tag_doy=366),
    "sod 0": dict(time_tag_second_of_day=0),
    "sod 86400": dict(time_tag_second_of_day=86400),
    "picoseconds 1e12 - 1": dict(time_tag_picoseconds_of_the_second=1e12 - 1),
    **{f"agency {a}": dict(agency_flag=a) for a in range(4)},
    "validity 65535": dict(validity_flag=65535),
}


@pytest.mark.parametrize("overrides", IN_RANGE.values(), ids=IN_RANGE.keys())
def test_in_range_boundaries_are_silent(overrides, write_prd, capsys):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # t0_offset passed: picoseconds 1e12 - 1 is in Table 5-1's range but
        # above DSN 0222's, which would get a separate advisory warning.
        read_product(write_prd(_zero_record(**overrides)), t0_offset=0.0)
    assert _reported(capsys, caught) == ""


# =============================================================================
# Values the writer must never put on disk (§5.2.3-§5.2.4, Table 5-1)
# =============================================================================


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf], ids=["nan", "+inf", "-inf"])
@pytest.mark.parametrize("field", spec.FLOAT_FIELDS)
def test_writer_rejects_non_finite_floats(field, bad):
    """§5.2.4: NaN, -Inf and +Inf are not supported."""
    with pytest.raises(ValueError):
        pack_record_header(RecordHeader(_model(**{field: bad})))


@pytest.mark.parametrize("field", spec.FLOAT_FIELDS)
def test_writer_writes_negative_zero_as_positive_zero(field):
    """§5.2.4: '-0' is not supported; the writer packs the same value, +0.0."""
    packed = pack_record_header(RecordHeader(_model(**{field: -0.0})))
    off = spec.LAYOUT[field][0]
    assert packed[off : off + 8] == struct.pack("<d", 0.0)


@pytest.mark.parametrize("label,overrides", list(SOFT.values())[:8], ids=list(SOFT)[:8])
def test_writer_rejects_out_of_range_fields(label, overrides):
    """What the reader only reports (Table 5-1 allowed values), the writer refuses."""
    with pytest.raises(ValueError, match=label):
        pack_record_header(RecordHeader(_model(**overrides)))


@pytest.mark.parametrize(
    "overrides",
    [
        dict(sample_size=0),
        dict(sample_size=3),
        dict(sample_size=32),
        dict(sample_size=1, sample_rate=8),
        dict(sample_size=2, sample_rate=4),
        dict(sample_size=4, sample_rate=2),
        dict(sample_size=8, sample_rate=1),
        dict(sample_rate=0),
    ],
    ids=lambda d: ",".join(f"{k}={v}" for k, v in d.items()),
)
def test_writer_rejects_sizes_and_rates_the_spec_forbids(overrides):
    """Table 5-1: SAMPLE SIZE in {1,2,4,8,16}; rate*2*size a multiple of 32."""
    m = _model()
    m.update(overrides)
    with pytest.raises(ValueError):
        pack_record_header(RecordHeader(m))


@pytest.mark.parametrize("overrides", [
    dict(station_id=70000), dict(validity_flag=70000), dict(time_tag_year=70000),
    dict(sample_rate=2**32), dict(sample_size=16, sample_rate=2**30),
], ids=lambda d: ",".join(f"{k}={v}" for k, v in d.items()))
def test_writer_rejects_integers_wider_than_their_field(overrides):
    """Table 5-1 widths: 2 or 4 bytes unsigned; a ValueError, not struct.error."""
    m = _model()
    m.update(overrides)
    with pytest.raises(ValueError, match="sample_rate|station_id|validity_flag|time_tag_year"):
        pack_record_header(RecordHeader(m))


@pytest.mark.parametrize("overrides", [
    dict(rf_to_if_downconv="8.1e9"), dict(empty_fields_future_extension="x" * 36),
    dict(empty_fields_internal_agency_use="x" * 40),
], ids=lambda d: next(iter(d)))
def test_writer_rejects_strings_naming_the_field(overrides):
    with pytest.raises(ValueError, match=next(iter(overrides))):
        pack_record_header(RecordHeader(_model(**overrides)))


@pytest.mark.parametrize("field,size", [("empty_fields_future_extension", 36),
                                        ("empty_fields_internal_agency_use", 40)])
@pytest.mark.parametrize("delta", [-1, 1])
def test_writer_rejects_wrong_size_reserved_fields(field, size, delta):
    with pytest.raises(ValueError):
        pack_record_header(RecordHeader(_model(**{field: bytes(size + delta)})))


@pytest.mark.parametrize("field", [f for f in spec.MODEL_FIELDS
                                   if f != "record_length" and not f.startswith("empty")])
def test_writer_rejects_a_missing_mandatory_field(field):
    """Table 5-1 marks these mandatory; None must not be packed as anything."""
    m = _model()
    m[field] = None
    with pytest.raises(ValueError):
        pack_record_header(RecordHeader(m))


def test_writer_rejects_sample_count_not_matching_the_rate(tmp_path):
    """§5.1.1: one record is exactly one second, i.e. SAMPLE RATE samples."""
    ones = np.ones(8, dtype=np.int32)
    pf = ProductFile()
    pf.records.append(Record(RecordHeader(_model(sample_rate=16)),
                             RecordData(16, 16, ones, ones)))
    with pytest.raises(ValueError):
        write_product(tmp_path / "x.prd", pf)


_FILLED_BY_THE_WRITER = ("record_length", "empty_fields_future_extension", "empty_fields_internal_agency_use")


def _mandatory_only(**overrides) -> dict:
    """A header as a caller builds one by hand: the mandatory fields, nothing else."""
    return {k: v for k, v in _model(**overrides).items() if k not in _FILLED_BY_THE_WRITER}


@pytest.mark.parametrize("values", VALUE_SETS.values(), ids=VALUE_SETS.keys())
def test_writer_fills_record_length_and_zeroes_unset_reserved_fields(values, tmp_path):
    """Table 5-1: RECORD LENGTH follows from rate and size, and the 76 reserved
    bytes of a header that sets none are zero. The bytes are the spec's."""
    values = {k: v for k, v in values.items() if k not in _FILLED_BY_THE_WRITER}
    hdr = RecordHeader(_mandatory_only(**values))
    assert all(getattr(hdr, f) is None for f in _FILLED_BY_THE_WRITER)
    want = spec.header_bytes(**values)  # rdef_spec defaults: computed length, zero bytes
    assert want[96:172] == bytes(76)
    assert pack_record_header(hdr) == want

    rate = values["sample_rate"]
    v_i, v_q = [2 * k + 1 for k in range(rate)], [-(2 * k + 1) for k in range(rate)]
    pf = ProductFile()
    pf.records.append(Record(hdr, RecordData(16, rate, v_i, v_q)))
    out = tmp_path / "minimal.prd"
    write_product(out, pf)
    assert out.read_bytes() == want + spec.encode_values(v_i, v_q, 16)
    assert all(getattr(hdr, f) is None for f in _FILLED_BY_THE_WRITER)  # the caller's header is left as built


@pytest.mark.parametrize("wrong", [0, 175, 176, 239, 241, 2**32 - 1])
def test_writer_refuses_a_record_length_that_disagrees_with_the_data(wrong, tmp_path):
    """A header that states a RECORD LENGTH other than 176 + data bytes (here
    176 + 2*16*16/8 = 240) is refused, not silently corrected; nothing is written."""
    ones = np.ones(16, dtype=np.int32)
    pf = ProductFile()
    pf.records.append(Record(RecordHeader(_model(record_length=wrong)), RecordData(16, 16, ones, ones)))
    with pytest.raises(ValueError, match=rf"record_length={wrong} does not match computed length 240"):
        write_product(tmp_path / "x.prd", pf)
    assert list(tmp_path.iterdir()) == []
    pf[0].header.record_length = 240
    write_product(tmp_path / "x.prd", pf)
    assert (tmp_path / "x.prd").stat().st_size == 240


# =============================================================================
# Time tag (§5.2.5, Table 5-1 TIME TAG fields)
# =============================================================================


@pytest.mark.parametrize(
    "year,doy,sod,ps,iso",
    [
        (2026, 1, 0, 0.0, "2026-01-01T00:00:00"),
        (2026, 39, 43200, 0.0, "2026-02-08T12:00:00"),
        (2024, 366, 86399, 0.0, "2024-12-31T23:59:59"),
        (2025, 365, 86399, 0.0, "2025-12-31T23:59:59"),
        (2026, 39, 43200, 5.0e11, "2026-02-08T12:00:00.500"),
        (2026, 39, 43200, 1.0e3, "2026-02-08T12:00:00.000000001"),
        (2026, 39, 43200, 999999999000.0, "2026-02-08T12:00:00.999999999"),
        (2026, 39, 43200, 123456789000.0, "2026-02-08T12:00:00.123456789"),
        (2026, 39, 43200, np.nextafter(123456789000.0, np.inf), "2026-02-08T12:00:00.123456789"),
        # rounding up to a whole second carries into the next second, day, year
        (2026, 39, 43200, 999999999500.0, "2026-02-08T12:00:01"),
        (2026, 39, 43200, 999999999999.0, "2026-02-08T12:00:01"),
        (2025, 365, 86399, 999999999999.0, "2026-01-01T00:00:00"),
    ],
)
def test_timetag_composes_year_doy_sod_picoseconds(year, doy, sod, ps, iso, write_prd):
    path = write_prd(_zero_record(time_tag_year=year, time_tag_doy=doy,
                                  time_tag_second_of_day=sod,
                                  time_tag_picoseconds_of_the_second=ps))
    for how in READERS:
        assert READERS[how](path).records[0].header.timetag == np.datetime64(iso, "ns")


@pytest.mark.parametrize("year,doy,sod", [(2263, 1, 0), (1677, 365, 0), (2026, 400, 0), (2026, 0, 0), (2026, 1, 86401)])
def test_timetag_out_of_range_raises_instead_of_wrapping(year, doy, sod):
    hdr = RecordHeader({"time_tag_year": year, "time_tag_doy": doy, "time_tag_second_of_day": sod})
    with pytest.raises(ValueError, match="outside"):
        hdr.timetag


@pytest.mark.parametrize(
    "ps,ns",
    [
        (np.nextafter(123456789000.0, 0.0), 123456789),  # one ulp short of a whole ns
        (123456789500.0, 123456790),  # half-ns ties round to even
        (123456788500.0, 123456788),
    ],
)
def test_timetag_rounds_picoseconds_to_the_nearest_ns(ps, ns, write_prd):
    path = write_prd(_zero_record(time_tag_picoseconds_of_the_second=ps))
    tag = read_headers(path, t0_offset=0.0).records[0].header.timetag
    assert tag.astype(np.int64) % 10**9 == ns


def test_one_record_per_second(write_prd):
    """§5.1.1: each record is exactly one second."""
    path = write_prd(*[_zero_record(time_tag_second_of_day=100 + s) for s in range(4)])
    tt = read_headers(path).timetag_array
    assert np.all(np.diff(tt) == np.timedelta64(1, "s"))


@pytest.mark.parametrize(
    "year,last_doy",
    [(2025, 365), (2024, 366)],
    ids=["non-leap", "leap"],
)
def test_one_record_per_second_across_a_year_boundary(year, last_doy, write_prd):
    """§5.1.1 across a year rollover: DOY last/86399 -> next year DOY 1/0 is 1 s."""
    path = write_prd(
        _zero_record(time_tag_year=year, time_tag_doy=last_doy,
                     time_tag_second_of_day=86399),
        _zero_record(time_tag_year=year + 1, time_tag_doy=1,
                     time_tag_second_of_day=0),
    )
    pf = read_headers(path)
    assert np.diff(pf.timetag_array)[0] == np.timedelta64(1, "s")
    consecutive = [ln for ln in str(pf).splitlines() if "consecutive:" in ln]
    assert consecutive and consecutive[0].rstrip().endswith("True"), consecutive


# =============================================================================
# Downconversion reference time (§5.2.6.2, Table 5-1; DSN 0222 Table 3-1, §3.4)
# =============================================================================
#
# Both standards refer c0..c3 to TIME TAG SECOND OF DAY. RDEFpy's origin is
# TIME TAG SECOND OF DAY + t0_offset (read-time keyword, default 0.0): sample k
# at tau = ps*1e-12 - t0_offset + k/fs.

_FS = 4
_F = 8.1e9 + 3.0e8  # rf_to_if + if_to_channel of spec.default_fields
_PHI, _C0, _C1, _C2, _C3 = 5.0, 0.25, 100.0, -0.125, 0.75
_POLY = dict(channel_accumulated_phase=_PHI, channel_phase_polynomial_coefficient_0=_C0,
             channel_phase_polynomial_coefficient_1=_C1, channel_phase_polynomial_coefficient_2=_C2,
             channel_phase_polynomial_coefficient_3=_C3)
# Frequencies near 8.4 GHz: float64 spacing 2**-20 Hz = 9.5e-7 Hz; the library
# and these expressions sum the terms in different orders: allow about 4 spacings.
_F_TOL = 4e-6


def _ps_record(ps):
    hdr = RecordHeader(_model(sample_rate=_FS, sample_size=16, time_tag_picoseconds_of_the_second=ps, **_POLY))
    ones = np.ones(_FS, dtype=np.int32)
    return Record(hdr, RecordData(16, _FS, ones, ones))


def _ps_path(write_prd, ps, n=2, sod=43200):
    return write_prd(*[_zero_record(sample_rate=_FS, time_tag_second_of_day=sod + s,
                                    time_tag_picoseconds_of_the_second=ps, **_POLY) for s in range(n)])


def _ps_file(write_prd, ps, n=2, t0_offset=0.0, **kw):
    """Read with an explicit t0_offset (no picoseconds warning)."""
    return read_product(_ps_path(write_prd, ps, n), t0_offset=t0_offset, **kw)


def _tau(ps, t0_offset):
    return ps * 1e-12 - t0_offset + np.arange(_FS) / _FS


def _phase(tau):
    return _PHI + _C0 + _C1 * tau + _C2 * tau**2 + _C3 * tau**3


def _freq(tau):
    return _F + _C1 + 2 * _C2 * tau + 3 * _C3 * tau**2


def _mean(a):
    b = a + 1.0
    return _F + _C1 + _C2 * (a + b) + _C3 * (a * a + a * b + b * b)


@pytest.mark.parametrize("ps", [0.0, 0.25e12])
def test_standalone_record_origin_is_the_second_of_day(ps):
    """A standalone record uses t0 = TIME TAG SECOND OF DAY: tau = ps*1e-12 + k/fs."""
    rec = _ps_record(ps)
    tau = _tau(ps, 0.0)
    np.testing.assert_allclose(rec.downconversion_phase, _phase(tau), rtol=0, atol=1e-9)
    np.testing.assert_allclose(rec.downconversion_frequency, _freq(tau), rtol=0, atol=_F_TOL)


def test_default_t0_offset_is_zero(write_prd):
    assert ProductFile().t0_offset == 0.0
    assert read_product(_ps_path(write_prd, 0.0)).t0_offset == 0.0
    assert read_headers(write_prd(_zero_record())).t0_offset == 0.0


@pytest.mark.parametrize("t0_offset", [0.0, 0.875, 1.0, -0.5, 1, np.float32(0.875), np.int64(1)])
def test_t0_offset_gives_the_analytic_values(t0_offset, write_prd):
    """First sample 0.875 s after the second of day: any offset gives the
    analytic values of its own origin (0.875 = the first sample, 1.0 = next second)."""
    ps = 0.875e12
    pf = _ps_file(write_prd, ps, t0_offset=t0_offset)
    assert pf.t0_offset == float(t0_offset) and type(pf.t0_offset) is float
    tau = _tau(ps, float(t0_offset))
    a = tau[0]
    for rec in pf:
        np.testing.assert_allclose(rec.downconversion_frequency, _freq(tau), rtol=0, atol=_F_TOL)
        np.testing.assert_allclose(rec.downconversion_phase, _phase(tau), rtol=0, atol=1e-9)
    np.testing.assert_allclose(pf.downconversion_frequency[:], np.tile(_freq(tau), 2), rtol=0, atol=_F_TOL)
    np.testing.assert_allclose(pf.downconversion_phase[:], np.tile(_phase(tau), 2), rtol=0, atol=1e-9)
    np.testing.assert_allclose(pf.mean_downconversion_frequency, [_mean(a)] * 2, rtol=0, atol=_F_TOL)
    np.testing.assert_allclose(pf.record_downconversion_frequency, [_freq(a)] * 2, rtol=0, atol=_F_TOL)
    np.testing.assert_allclose(pf.record_downconversion_phase, [_phase(a)] * 2, rtol=0, atol=1e-9)


def test_offsets_one_second_apart_differ_by_the_derivative_terms(write_prd):
    """ps = 0.875e12, offsets 0 and 1: the mean f_DC differs by exactly
    2*c2 + c3*(b0**3 - a0**3 - b1**3 + a1**3) = 2*c2 + 6*a*c3 (a = 0.875)."""
    s, n = (_ps_file(write_prd, 0.875e12, t0_offset=o) for o in (0.0, 1.0))
    d = s.mean_downconversion_frequency - n.mean_downconversion_frequency
    np.testing.assert_allclose(d, 2 * _C2 + 6 * 0.875 * _C3, rtol=0, atol=2 * _F_TOL)
    assert abs(d[0]) > 0.5


def test_first_sample_offset_puts_origin_on_first_sample(write_prd):
    """t0_offset = ps*1e-12 puts the origin on the first sample: tau = k/fs."""
    ps = 0.875e12
    pf = _ps_file(write_prd, ps, t0_offset=ps * 1e-12)
    np.testing.assert_allclose(pf[0].downconversion_frequency, _freq(np.arange(_FS) / _FS), rtol=0, atol=_F_TOL)
    np.testing.assert_allclose(pf.mean_downconversion_frequency, [_mean(0.0)] * 2, rtol=0, atol=_F_TOL)


def test_t0_offset_and_agency_layout_are_read_only(write_prd):
    pf = _ps_file(write_prd, 0.0)
    with pytest.raises(AttributeError):
        pf.t0_offset = 1.0
    with pytest.raises(AttributeError):
        pf.agency_layout = "dsn"


def test_t0_offset_is_kept_by_slice_copy_select_and_concatenate(write_prd):
    pf = _ps_file(write_prd, 0.875e12, n=3, t0_offset=1.0)
    other = read_product(_ps_path(write_prd, 0.875e12, n=1, sod=43205), t0_offset=0.0)
    for sub in (pf.slice(0, 2), pf.copy(), pf.select(filters={"station_id": 24}), pf.concatenate(other)):
        assert sub.t0_offset == 1.0
        assert all(r.header._t0_offset == 1.0 for r in sub.records)
        np.testing.assert_array_equal(sub.mean_downconversion_frequency,
                                      np.full(len(sub), pf.mean_downconversion_frequency[0]))
    assert other.concatenate(pf).t0_offset == 0.0


def test_extract_carrier_follows_t0_offset(write_prd):
    """extract_carrier's f_DC integral uses the file's origin: 1-s blocks equal
    mean_downconversion_frequency under each offset, and the offsets differ."""
    from rdefpy import extract_carrier

    out = {}
    for t0_offset in (0.0, 1.0):
        pf = _ps_file(write_prd, 0.875e12, n=3, t0_offset=t0_offset)
        out[t0_offset] = extract_carrier(pf, 1.0, max_rate_hz_s=0)["downconversion_frequency"]
        np.testing.assert_allclose(out[t0_offset], pf.mean_downconversion_frequency, rtol=0, atol=_F_TOL)
    assert np.abs(out[0.0] - out[1.0]).min() > 0.5


def test_picoseconds_above_the_0222_range_warn_once_without_t0_offset(write_prd):
    path = _ps_path(write_prd, 0.875e12, n=3)
    other = read_product(_ps_path(write_prd, 0.875e12, n=1, sod=43210), t0_offset=0.0)
    for read in (read_product, read_headers):
        with pytest.warns(RuntimeWarning, match=r"DSN 0222.*Pass t0_offset.*including 0\.0") as w:
            pf = read(path)
        assert len(w) == 1 and w[0].filename == __file__
        with warnings.catch_warnings():  # no warning on property access or copies
            warnings.simplefilter("error")
            pf.mean_downconversion_frequency, pf[0].downconversion_frequency
            pf.slice(0, 2), pf.copy(), pf.select(), pf.concatenate(other)


@pytest.mark.parametrize("ps, t0_offset", [(1e5, None), (0.0, None), (0.875e12, 0.0), (0.875e12, 1.0)])
def test_no_picosecond_warning_within_range_or_with_t0_offset(ps, t0_offset, write_prd):
    path = write_prd(_zero_record(time_tag_picoseconds_of_the_second=ps))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        read_product(path, t0_offset=t0_offset)
        read_headers(path, t0_offset=t0_offset)


def test_record_moved_by_hand_keeps_its_offset(write_prd):
    """Records are stamped at read: one moved into a file read with another
    offset keeps its own; file-level arrays use the file's."""
    ps = 0.875e12
    a = _ps_file(write_prd, ps, n=1, t0_offset=1.0)
    b = _ps_file(write_prd, ps, n=1, t0_offset=0.0)
    b.records.append(a.records[0])
    np.testing.assert_allclose(b[1].downconversion_frequency, _freq(_tau(ps, 1.0)), rtol=0, atol=_F_TOL)
    np.testing.assert_allclose(b.mean_downconversion_frequency, [_mean(_tau(ps, 0.0)[0])] * 2, rtol=0, atol=_F_TOL)


@pytest.mark.parametrize("bad", [True, False, np.bool_(True), math.nan, math.inf, -math.inf,
                                 "0", "second", b"0", [0.0], 1j])
def test_invalid_t0_offset_raises(bad, write_prd):
    path = write_prd(_zero_record())
    with pytest.raises(ValueError, match="t0_offset must be a finite number of seconds"):
        read_product(path, t0_offset=bad)
    with pytest.raises(ValueError, match="t0_offset must be a finite number of seconds"):
        read_headers(path, t0_offset=bad)


def test_downconversion_filter_uses_t0_offset(write_prd):
    """The min/max filter compares the frequency at the first sample, origin t0."""
    ps = 0.875e12
    path = write_prd(_zero_record(sample_rate=_FS, time_tag_picoseconds_of_the_second=ps, **_POLY))
    f0, f1 = _freq(_tau(ps, 0.0))[0], _freq(_tau(ps, 1.0))[0]
    lo = min(f0, f1) + abs(f0 - f1) / 2  # between the two
    keep = 0.0 if f0 > lo else 1.0
    for t0_offset in (0.0, 1.0):
        assert len(read_product(path, t0_offset=t0_offset, min_downconv_hz=lo)) == (t0_offset == keep)
        assert len(read_product(path, t0_offset=t0_offset).select(min_downconv_hz=lo)) == (t0_offset == keep)


# =============================================================================
# NaN c1..c3: DSN 0222 Millisecond Predict mode (Table 3-1, §2.2)
# =============================================================================

_NAN_BITS = struct.pack("<Q", 0x7FFFFFFFFFFFFFFF)  # the 0222 NaN, "7fffffffffffffff"


def _with_nan_coeffs(rec: bytes, which=(1, 2, 3)) -> bytes:
    buf = bytearray(rec)
    for k in which:
        off = spec.LAYOUT[f"channel_phase_polynomial_coefficient_{k}"][0]
        buf[off : off + 8] = _NAN_BITS
    return bytes(buf)


def test_all_nan_c123_reads_without_warning_and_round_trips(write_prd, tmp_path):
    raw = b"".join(_with_nan_coeffs(_zero_record(time_tag_second_of_day=43200 + s)) for s in range(2))
    path = write_prd(raw)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pf = read_product(path)
        pf.info(buf=io.StringIO())
    out = tmp_path / "out.prd"
    write_product(out, pf)
    assert out.read_bytes() == raw
    assert np.isnan(pf.mean_downconversion_frequency).all()
    assert np.isnan(pf.record_downconversion_frequency).all()
    assert np.isnan(pf[0].downconversion_frequency).all()
    assert np.isnan(pf.downconversion_phase[:]).all()
    # band / min-max filters compare the fixed part (8.4 GHz) for these records
    assert len(read_product(path, band="X")) == 2
    assert len(pf.select(min_downconv_hz=8.3e9, max_downconv_hz=8.5e9)) == 2


def test_nan_c123_written_as_the_0222_bit_pattern(tmp_path):
    """float('nan') in memory (payload 0x7ff8...) is written as 7fffffffffffffff."""
    hdr = RecordHeader(_model(sample_rate=_FS, **{f"channel_phase_polynomial_coefficient_{k}": float("nan")
                                                 for k in (1, 2, 3)}))
    pf = ProductFile()
    pf.records = [Record(hdr, RecordData(16, _FS, np.ones(_FS, np.int32), np.ones(_FS, np.int32)))]
    out = tmp_path / "nan.prd"
    write_product(out, pf)
    raw = out.read_bytes()
    for k in (1, 2, 3):
        off = spec.LAYOUT[f"channel_phase_polynomial_coefficient_{k}"][0]
        assert raw[off : off + 8] == bytes.fromhex("ffffffffffffff7f"), k
    write_product(tmp_path / "again.prd", read_product(out))  # read from disk, written again
    assert (tmp_path / "again.prd").read_bytes() == raw


@pytest.mark.parametrize("which", [(1,), (2, 3), (1, 3)])
def test_partial_nan_c123_still_warns_and_is_refused(which, write_prd, tmp_path):
    path = write_prd(_with_nan_coeffs(_zero_record(), which))
    with pytest.warns(UserWarning, match="§5.2.4"):
        pf = read_product(path)
    with pytest.raises(ValueError, match="§5.2.4"):
        write_product(tmp_path / "out.prd", pf)


@pytest.mark.parametrize("field", ["channel_phase_polynomial_coefficient_1", "channel_accumulated_phase",
                                   "rf_to_if_downconv"])
def test_inf_still_warns_and_is_refused_with_nan_c123(field, write_prd, tmp_path):
    raw = bytearray(_with_nan_coeffs(_zero_record()))
    off = spec.LAYOUT[field][0]
    raw[off : off + 8] = struct.pack("<d", math.inf)
    with pytest.warns(UserWarning, match=field):
        pf = read_product(write_prd(bytes(raw)))
    with pytest.raises(ValueError, match=field):
        write_product(tmp_path / "out.prd", pf)


def test_extract_carrier_gives_nan_downconversion_for_nan_c123(write_prd):
    from rdefpy import extract_carrier

    fs = 64
    n = np.arange(fs)
    k = np.round(100 * np.cos(2 * np.pi * 5 * n / fs)).astype(int).tolist()
    k_q = np.round(100 * np.sin(2 * np.pi * 5 * n / fs)).astype(int).tolist()
    raw = spec.record_bytes(k, k_q, sample_rate=fs, sample_size=16)
    pf = read_product(write_prd(_with_nan_coeffs(raw)))
    c = extract_carrier(pf, 1.0, max_rate_hz_s=0)
    assert np.isnan(c["downconversion_frequency"]).all()
    assert np.isnan(c["frequency_before_downconversion"]).all()
    assert c["frequency"][0] == pytest.approx(5.0, abs=1e-3)


# =============================================================================
# Internal-agency block (header bytes 132-171; DSN 0222 Table 3-1 / Figure 3-4)
# =============================================================================

_DSN_VALUES = (4321, 2, 3, 1, 17, 33, 1, -42.5, 1234.5625, 19)
_DSN_BLOCK = struct.pack("<HBBBBBBfdB19x", *_DSN_VALUES)
_DSN_NAMES = ("predict_pass_number", "uplink_band", "downlink_band", "track_mode", "uplink_dss_id",
              "olr_id", "olr_software_version", "channel_power_calibration_factor",
              "total_frequency_offset", "channel_number")


def _agency_file(write_prd, agency_flag, n=2, **kw):
    recs = [_zero_record(agency_flag=agency_flag, time_tag_second_of_day=43200 + s,
                         empty_fields_internal_agency_use=_DSN_BLOCK) for s in range(n)]
    return read_headers(write_prd(*recs), **kw)


def _assert_dsn(row):
    for name, v in zip(_DSN_NAMES, _DSN_VALUES, strict=True):
        assert row[name] == v, name


def test_dsn_agency_dtype_is_40_bytes():
    from rdefpy import DSN_AGENCY_DTYPE

    assert DSN_AGENCY_DTYPE.itemsize == 40
    assert DSN_AGENCY_DTYPE.names[: len(_DSN_NAMES)] == _DSN_NAMES


def test_auto_layout_decodes_dsn_on_agency_flag_3(write_prd):
    pf = _agency_file(write_prd, 3)
    assert pf.agency_layout == "auto"
    rows = pf.agency_fields()
    assert rows.shape == (2,)
    for row in rows:
        _assert_dsn(row)
    _assert_dsn(pf[0].header.agency_fields())


def test_auto_layout_returns_raw_bytes_on_other_agency_flags(write_prd):
    pf = _agency_file(write_prd, 1)
    rows = pf.agency_fields()
    assert rows.dtype.names == ("raw",) and rows.dtype.itemsize == 40
    assert [bytes(r["raw"]) for r in rows] == [r.header.empty_fields_internal_agency_use for r in pf.records]
    assert bytes(pf[0].header.agency_fields()["raw"]) == _DSN_BLOCK


def test_auto_layout_with_mixed_agency_flags_warns_and_returns_raw(write_prd):
    recs = [_zero_record(agency_flag=f, time_tag_second_of_day=43200 + s,
                         empty_fields_internal_agency_use=_DSN_BLOCK) for s, f in enumerate((3, 1))]
    path = write_prd(*recs)
    with pytest.warns(RuntimeWarning, match="agency_flag"):  # the file-constant check
        pf = read_headers(path)
    with pytest.warns(RuntimeWarning, match="agency_flag differs across records.*raw bytes") as w:
        rows = pf.agency_fields()
    assert len(w) == 1 and w[0].filename == __file__
    assert rows.dtype.names == ("raw",)
    assert [bytes(r) for r in rows["raw"]] == [_DSN_BLOCK] * 2
    with pytest.warns(RuntimeWarning, match="agency_flag"):  # read again with a layout
        pf = read_headers(path, agency_layout="dsn")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for row in pf.agency_fields():
            _assert_dsn(row)


def test_forced_dsn_layout_on_agency_flag_1(write_prd):
    pf = _agency_file(write_prd, 1, agency_layout="dsn")
    for row in pf.agency_fields():
        _assert_dsn(row)
    _assert_dsn(pf[1].header.agency_fields())


@pytest.mark.parametrize("layout", ["<u2,u1,u1,u1,u1,u1,u1,<f4,<f8,u1,V19",
                                    [("a", "<u2"), ("rest", "V38")]])
def test_custom_agency_layout(layout, write_prd):
    pf = _agency_file(write_prd, 2, agency_layout=layout)
    rows = pf.agency_fields()
    assert rows.dtype == np.dtype(layout)
    assert rows[0][0] == 4321
    assert pf.slice(0, 1).agency_fields().dtype == np.dtype(layout)  # kept by slice


def test_read_product_agency_layout_keyword(write_prd):
    recs = [_zero_record(agency_flag=0, empty_fields_internal_agency_use=_DSN_BLOCK)]
    pf = read_product(write_prd(*recs), agency_layout="dsn")
    assert pf.agency_layout == "dsn"
    _assert_dsn(pf.agency_fields()[0])
    assert pf.copy().agency_layout == "dsn"
    assert pf.select().agency_layout == "dsn"


@pytest.mark.parametrize("layout", ["<u2,V37", [("a", "<f8")], "V41"])
def test_agency_layout_wrong_size_raises(layout, write_prd):
    with pytest.raises(ValueError, match="40"):
        _agency_file(write_prd, 3, agency_layout=layout)


@pytest.mark.parametrize("layout", ["DSN", "Auto", "raw", "not a dtype", 3.5, None])
def test_invalid_agency_layout_raises_value_error(layout, write_prd):
    path = write_prd(_zero_record())
    msg = 'agency_layout must be "auto", "dsn" or a 40-byte dtype'
    with pytest.raises(ValueError, match=msg):
        read_product(path, agency_layout=layout)
    with pytest.raises(ValueError, match=msg):
        read_headers(path, agency_layout=layout)


def test_agency_layout_with_object_fields_raises(write_prd):
    path = write_prd(_zero_record())
    with pytest.raises(ValueError, match="without object fields"):
        # 40 bytes on any platform: an object field is 4 bytes on 32-bit, 8 on 64-bit
        read_product(path, agency_layout=[("a", "O"), ("rest", f"V{40 - np.dtype('O').itemsize}")])


def test_agency_fields_of_a_written_dsn_block(tmp_path):
    from rdefpy import DSN_AGENCY_DTYPE

    block = np.zeros(1, DSN_AGENCY_DTYPE)
    block["predict_pass_number"], block["downlink_band"], block["total_frequency_offset"] = 77, 2, -3.5
    hdr = RecordHeader(_model(sample_rate=_FS))
    pf = ProductFile()
    pf.records = [Record(hdr, RecordData(16, _FS, np.ones(_FS, np.int32), np.ones(_FS, np.int32)))]
    pf.correct_header(empty_fields_internal_agency_use=block.tobytes())
    write_product(tmp_path / "a.prd", pf)
    row = read_headers(tmp_path / "a.prd").agency_fields()[0]
    assert (row["predict_pass_number"], row["downlink_band"], row["total_frequency_offset"]) == (77, 2, -3.5)


# =============================================================================
# One channel per file (§5): concatenate() and slice()
# =============================================================================
# A product file holds one scan, one channel, one station, so merging two
# files that differ in a file-constant field must fail. slice() and
# concatenate() return independent copies: nothing done to the result may
# reach the parent.

CHANNEL_CLASH = dict(
    station_id=17,
    spacecraft_id=7,
    agency_flag=1,
    sample_size=8,
    sample_rate=32,
    rf_to_if_downconv=8.2e9,
    if_to_channel_downconv=3.1e8,
)


def _coded_record(code: int, sod: int, **overrides) -> bytes:
    """A record whose samples are all *code*, so its source is recognisable."""
    f = spec.default_fields(time_tag_second_of_day=sod, **overrides)
    n = f["sample_rate"]
    return spec.record_bytes([code] * n, [code] * n,
                             time_tag_second_of_day=sod, **overrides)


def _first_i(pf) -> list[int]:
    return [int(r.data.i_samples[0]) for r in pf.records]


@pytest.mark.parametrize("field,value", list(CHANNEL_CLASH.items()), ids=list(CHANNEL_CLASH))
def test_concatenate_refuses_a_different_channel(field, value, write_prd):
    assert spec.default_fields()[field] != value
    a = read_product(write_prd(_coded_record(0, 0)))
    b = read_product(write_prd(_coded_record(0, 1, **{field: value})))
    for x, y in ((a, b), (b, a)):
        with pytest.raises(ValueError, match=field):
            x.concatenate(y)


def test_concatenate_same_channel_merges_in_time_order(write_prd):
    a = read_product(write_prd(_coded_record(2, 2), _coded_record(0, 0)))
    b = read_product(write_prd(_coded_record(1, 1)))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no duplicates, so no warning
        merged = a.concatenate(b)
    assert [r.header.time_tag_second_of_day for r in merged.records] == [0, 1, 2]
    assert _first_i(merged) == [spec.value_of(k) for k in (0, 1, 2)]


def test_concatenate_warns_on_a_duplicate_within_one_input_and_keeps_the_last(write_prd):
    a = read_product(write_prd(_coded_record(3, 0), _coded_record(5, 0)))  # same tag
    b = read_product(write_prd(_coded_record(1, 1)))
    with pytest.warns(RuntimeWarning, match="within self"):
        merged = a.concatenate(b)
    assert _first_i(merged) == [spec.value_of(5), spec.value_of(1)]


@pytest.mark.parametrize("keep, kept", [("self", 0), ("other", 7)])
def test_concatenate_unsorted_puts_self_first(write_prd, keep, kept):
    a = read_product(write_prd(_coded_record(2, 2), _coded_record(0, 0)))
    b = read_product(write_prd(_coded_record(7, 0), _coded_record(1, 1)))  # 7 collides with 0
    with pytest.warns(RuntimeWarning, match="duplicate"):
        merged = a.concatenate(b, keep=keep, sort=False)
    assert _first_i(merged) == [spec.value_of(k) for k in (2, kept, 1)]


def test_slice_negative_index_and_out_of_range(write_prd):
    pf = read_product(write_prd(*(_coded_record(k, k) for k in range(3))))
    assert _first_i(pf.slice(-1)) == [spec.value_of(2)]
    assert _first_i(pf.slice(0)) == [spec.value_of(0)]
    for bad in (3, -4):
        with pytest.raises(IndexError):
            pf.slice(bad)


def _derived(pf, how):
    if how == "slice":
        return pf.slice(0, len(pf.records))
    if how == "select":
        return pf.select(filters={"station_id": 24})  # matches every record
    other = ProductFile(name="empty")
    other.records = []
    return pf.concatenate(other)


@pytest.mark.parametrize("how", ["slice", "concatenate", "select"])
def test_editing_the_result_leaves_the_parent_unchanged(how, write_prd):
    pf = read_product(write_prd(_coded_record(0, 0)))  # 16 Hz: auto-loaded
    child = _derived(pf, how)
    child.records[0].header.station_id = 99
    child.records[0].data.i_samples[0] = 77
    assert pf.records[0].header.station_id == 24
    assert _first_i(pf) == [spec.value_of(0)]


@pytest.mark.parametrize("how", ["slice", "concatenate", "select"])
def test_releasing_the_result_leaves_the_parent_loaded(how, write_prd):
    path = write_prd(_coded_record(4, 0))  # 16 Hz: auto-loaded
    pf = read_product(path)
    child = _derived(pf, how)
    child.release()
    _clobber(path)
    assert _first_i(pf) == [spec.value_of(4)]  # parent still in memory
    with pytest.raises(RuntimeError):
        child.records[0].data  # child went back to the (changed) disk


@pytest.mark.parametrize("how", ["slice", "concatenate", "select"])
def test_loading_the_result_leaves_the_parent_on_access(how, write_prd):
    path = write_prd(_coded_record(4, 0, sample_rate=1001))  # > 1 kHz: on access
    pf = read_product(path)
    child = _derived(pf, how)
    child.load()
    _clobber(path)
    assert _first_i(child) == [spec.value_of(4)]  # child pinned
    with pytest.raises(RuntimeError):
        pf.records[0].data  # parent still reads from disk


# =============================================================================
# Edge cases: unordered or corrupt headers, truncation, writer input
# =============================================================================


def test_stop_epoch_keeps_in_window_records_after_a_later_one(write_prd):
    """Records need not be in time order: no early stop at stop_epoch."""
    path = write_prd(_zero_record(time_tag_second_of_day=43201),
                     _zero_record(time_tag_second_of_day=43200))
    stop = "2026-02-08T12:00:01"
    got = [r.header.time_tag_second_of_day for r in read_product(path, stop_epoch=stop)]
    assert got == [43200]
    assert got == [r.header.time_tag_second_of_day
                   for r in read_product(path).select(stop_epoch=stop)]


def test_unusable_time_tag_fails_any_time_window(write_prd):
    path = write_prd(_zero_record(time_tag_picoseconds_of_the_second=math.nan), _zero_record())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the NaN is reported; not the point here
        kw = dict(start_epoch="2026-02-08T11:00:00")
        assert len(read_product(path, **kw)) == 1
        assert len(read_product(path).select(**kw)) == 1


@pytest.mark.parametrize("window", [dict(band="X"), dict(min_downconv_hz=0.0),
                                    dict(max_downconv_hz=1e12),
                                    dict(min_downconv_hz=0.0, max_downconv_hz=1e12)])
def test_nan_downconversion_is_outside_every_window(window, write_prd):
    path = write_prd(_zero_record(rf_to_if_downconv=math.nan), _zero_record())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert len(read_product(path, **window)) == 1
        if "band" not in window:  # select() has no band shortcut
            assert len(read_product(path).select(**window)) == 1


def test_varying_field_warning_points_at_the_caller(write_prd):
    path = write_prd(_zero_record(station_id=24, time_tag_second_of_day=43200),
                     _zero_record(station_id=26, time_tag_second_of_day=43201))
    for read in (read_product, read_headers):
        with pytest.warns(RuntimeWarning, match=r"Metadata fields vary across records: \['station_id'\]") as w:
            read(path)
        assert len(w) == 1 and w[0].filename == __file__


def test_nan_field_is_not_reported_as_varying(write_prd):
    path = write_prd(_zero_record(rf_to_if_downconv=math.nan),
                     _zero_record(rf_to_if_downconv=math.nan))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pf = read_product(path)
    assert not [w for w in caught if "vary across records" in str(w.message)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # duplicate time tags: not the point
        assert len(pf.concatenate(read_product(path), keep="both")) == 4  # same channel
        assert "frequency_steering: False" in str(read_product(
            write_prd(_zero_record(channel_phase_polynomial_coefficient_1=math.nan),
                      _zero_record(channel_phase_polynomial_coefficient_1=math.nan))))


def test_scan_product_reports_a_truncated_last_record(write_prd):
    path = write_prd(_zero_record(), _zero_record()[:-1])
    with pytest.raises(EOFError):
        scan_product(path)


@pytest.mark.parametrize("field,bad", [("station_id", 2.5), ("sample_rate", 16.5),
                                       ("validity_flag", True), ("spacecraft_id", -1),
                                       ("time_tag_doy", math.nan), ("agency_flag", "1")])
def test_writer_refuses_non_integers_in_integer_fields(field, bad):
    """Table 5-1: these fields are UNSIGNED INTEGER; int() would truncate 2.5."""
    with pytest.raises(ValueError, match=field):
        pack_record_header(RecordHeader(_model(**{field: bad})))


@pytest.mark.parametrize("value", [24, 24.0, np.uint16(24), np.int64(24)])
def test_writer_accepts_exact_integers_of_any_type(value):
    packed = pack_record_header(RecordHeader(_model(station_id=value)))
    assert pack_record_header(RecordHeader(_model(station_id=24))) == packed


def test_pack_record_data_checks_rate_and_bit_depth():
    from rdefpy import pack_record_data

    ones = np.ones(3, np.int8)
    with pytest.raises(ValueError, match="multiple of 32"):  # 2*3*2 bits: no whole words
        pack_record_data(Record(RecordHeader(_model(sample_size=2, sample_rate=3)),
                                RecordData(2, 3, ones, ones)))
    ones = np.ones(16, np.int8)
    with pytest.raises(ValueError, match="2-bit"):
        pack_record_data(Record(RecordHeader(_model(sample_size=4, sample_rate=16)),
                                RecordData(2, 16, ones, ones)))
