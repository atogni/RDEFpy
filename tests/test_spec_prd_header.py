"""Product file record framing and header against CCSDS 506.1-B-1 §5.1-§5.2.

Every expectation comes from ``rdef_spec`` (the Blue Book transcribed into
code), never from rdefpy's own struct or constants. The library is checked in
both directions: bytes the spec defines must read back to the right fields,
and fields the library writes must land on the right bytes.

Tests marked ``xfail(strict=True)`` are confirmed deviations from the
standard. They XPASS (and fail the run) once fixed. Remove the marker then.
"""

from __future__ import annotations

import math
import struct
import warnings

import numpy as np
import pytest

import rdef_spec as spec
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
    pf = read_product(path, **kw)
    if stream:
        pf.release()
    else:
        pf.load()
    return pf


READERS = {
    "loaded": lambda p: _open(p, stream=False),
    "on_access": lambda p: _open(p, stream=True),
    "headers": lambda p: read_headers(p),
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
    pf = read_product(write_prd(raw))
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
    "version 0": ("RECORD VERSION ID", dict(record_version_id=0)),
    "version 2": ("RECORD VERSION ID", dict(record_version_id=2)),
    "agency 4": ("AGENCY FLAG", dict(agency_flag=4)),
    "doy 0": ("DOY", dict(time_tag_doy=0)),
    "doy 367": ("DOY", dict(time_tag_doy=367)),
    "sod 86401": ("SECOND OF DAY", dict(time_tag_second_of_day=86401)),
    "picoseconds -1": ("PICOSECONDS", dict(time_tag_picoseconds_of_the_second=-1.0)),
    "picoseconds 1e12": ("PICOSECONDS", dict(time_tag_picoseconds_of_the_second=1e12)),
    "sample rate 0": ("SAMPLE RATE", dict(sample_rate=0)),
    **{f"{f} {b}": (f"{f}: {v!r}", {f: v}) for f in spec.FLOAT_FIELDS
       for b, v in (("nan", math.nan), ("-inf", -math.inf), ("-0", -0.0))},
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
    assert mine and all(w.filename == __file__ for w in mine)  # points at the caller
    assert label in _reported(capsys, caught)


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
        read_product(write_prd(_zero_record(**overrides)))
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
    tag = read_headers(path).records[0].header.timetag
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
# Downconversion reference time (§5.2.6.2, Table 5-1)
# =============================================================================
#
# c0..c3 are "referred to the second boundary, as provided by item TIME TAG
# SECOND OF DAY", and the picoseconds field is the delay of the FIRST SAMPLE
# after that boundary. So sample k sits at t - t0 = ps*1e-12 + k/fs in the
# phase polynomial. The library uses k/fs, which is the same only when ps = 0.

_PS = 0.25e12  # first sample 0.25 s after the second boundary
_FS = 4


def _ps_record():
    hdr = RecordHeader(_model(
        sample_rate=_FS, sample_size=16,
        time_tag_picoseconds_of_the_second=_PS,
        channel_accumulated_phase=5.0,
        channel_phase_polynomial_coefficient_0=0.25,
        channel_phase_polynomial_coefficient_1=100.0,
        channel_phase_polynomial_coefficient_2=10.0,
        channel_phase_polynomial_coefficient_3=1.0,
    ))
    ones = np.ones(_FS, dtype=np.int32)
    return Record(hdr, RecordData(16, _FS, ones, ones))


def _tau():
    return _PS * 1e-12 + np.arange(_FS) / _FS


def test_downconversion_polynomials_at_zero_picoseconds():
    """With ps = 0 the second boundary is the first sample: plain k/fs."""
    rec = _ps_record()
    rec.header.time_tag_picoseconds_of_the_second = 0.0
    t = np.arange(_FS) / _FS
    fixed = rec.header.rf_to_if_downconv + rec.header.if_to_channel_downconv
    np.testing.assert_allclose(rec.downconversion_phase,
                               5.0 + 0.25 + 100 * t + 10 * t**2 + t**3, rtol=0, atol=1e-9)
    np.testing.assert_allclose(rec.downconversion_frequency,
                               fixed + 100 + 20 * t + 3 * t**2, rtol=0, atol=1e-3)


def test_downconversion_phase_counts_the_picosecond_offset():
    t = _tau()
    np.testing.assert_allclose(_ps_record().downconversion_phase,
                               5.0 + 0.25 + 100 * t + 10 * t**2 + t**3, rtol=0, atol=1e-9)


def test_downconversion_frequency_counts_the_picosecond_offset():
    rec = _ps_record()
    t = _tau()
    fixed = rec.header.rf_to_if_downconv + rec.header.if_to_channel_downconv
    np.testing.assert_allclose(rec.downconversion_frequency,
                               fixed + 100 + 20 * t + 3 * t**2, rtol=0, atol=1e-3)


# =============================================================================
# One channel per file (§5): concatenate() and slice()
# =============================================================================
# A product file holds one scan, one channel, one station, so merging two
# files that differ in a file-constant field must fail. slice() and
# concatenate() return independent copies: nothing done to the result may
# reach the parent.

CHANNEL_CLASH = dict(
    station_id=25,
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


def _clobber(path):
    """Replace the file with different bytes of a different size."""
    path.write_bytes(b"\0" * (path.stat().st_size + 4))


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
