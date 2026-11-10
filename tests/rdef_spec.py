"""CCSDS 506.1-B-1 (June 2013) written out as code: the reference the spec tests check rdefpy against.

Everything here is transcribed from the Blue Book, NOT from rdefpy, and this
module must never import rdefpy. If the library's struct, lookup tables or bit
shifts change, the tests built on this module still hold the library to the
standard. Copying an expectation from the implementation would make the test
prove nothing.

The code is deliberately naive: plain integers, one field and one sample at a
time. It is slow and obviously correct, and it should stay that way.

Product file (§5)
    HEADER_LAYOUT     Table 5-1 field order, byte offsets and types
    header_bytes()    176-byte record header built field by field at its offset
    parse_header()    the inverse
    encode_codes()    raw two's-complement codes -> data section (Table 5-2)
    decode_values()   data section -> 2k+1 sample values (§5.3.6)
    record_bytes()    one full record, header + data
    dc_first_sample() / dc_mean() / dc_phase_first_sample()
                      §5.2.6 downconversion at a record's first sample and
                      over its second

DSN 820-013 0222-Science Rev. A (DSN's version of the same record)
    DSN_AGENCY_LAYOUT the 40-byte internal-agency block, Table 3-1 / Fig. 3-4
    parse_dsn_agency() the block decoded field by field

File names (§6)
    FILENAME_RE       the §6.2.2 pattern
"""

from __future__ import annotations

import re
import struct

# =============================================================================
# §5.2 / Table 5-1 / Figure 5-3: the record header
# =============================================================================

HEADER_SIZE = 176  # Figure 5-2: "HEADER SECTION [176 Bytes]"
RECORD_LABEL = b"RDEF"  # Table 5-1, RECORD LABEL
END_LABEL = -99999  # Table 5-1, END LABEL ("Shall be equal to -99999")

# (field, byte offset, byte length, type), in Table 5-1 order. Each offset is
# the running sum of the Table 5-1 byte counts; Figure 5-3 draws the same rows.
# Types: "a" ASCII, "u" unsigned integer, "i" signed integer,
#        "f" IEEE 754 binary64 (§5.2.3), "x" raw reserved bytes.
# The names are rdefpy's attribute names, so the tests can compare directly;
# label and end label are not rdefpy attributes.
HEADER_LAYOUT = (
    ("record_label", 0, 4, "a"),
    ("record_length", 4, 4, "u"),
    ("record_version_id", 8, 2, "u"),
    ("station_id", 10, 2, "u"),
    ("spacecraft_id", 12, 2, "u"),
    ("sample_size", 14, 2, "u"),
    ("sample_rate", 16, 4, "u"),
    ("validity_flag", 20, 2, "u"),
    ("agency_flag", 22, 2, "u"),
    ("rf_to_if_downconv", 24, 8, "f"),
    ("if_to_channel_downconv", 32, 8, "f"),
    ("time_tag_year", 40, 2, "u"),
    ("time_tag_doy", 42, 2, "u"),
    ("time_tag_second_of_day", 44, 4, "u"),
    ("time_tag_picoseconds_of_the_second", 48, 8, "f"),
    ("channel_accumulated_phase", 56, 8, "f"),
    ("channel_phase_polynomial_coefficient_0", 64, 8, "f"),
    ("channel_phase_polynomial_coefficient_1", 72, 8, "f"),
    ("channel_phase_polynomial_coefficient_2", 80, 8, "f"),
    ("channel_phase_polynomial_coefficient_3", 88, 8, "f"),
    ("empty_fields_future_extension", 96, 36, "x"),
    ("empty_fields_internal_agency_use", 132, 40, "x"),
    ("end_label", 172, 4, "i"),
)
LAYOUT = {name: (off, size, kind) for name, off, size, kind in HEADER_LAYOUT}

FLOAT_FIELDS = tuple(n for n, _, _, k in HEADER_LAYOUT if k == "f")
# Every field rdefpy carries on RecordHeader (label and end label are implicit).
MODEL_FIELDS = tuple(n for n, *_ in HEADER_LAYOUT if n not in ("record_label", "end_label"))

VALID_SAMPLE_SIZES = (1, 2, 4, 8, 16)  # Table 5-1, SAMPLE SIZE allowed values

# §1.3.3 / §5.1.5: multi-byte values are little-endian.
_STRUCT = {("u", 2): "<H", ("u", 4): "<I", ("i", 4): "<i", ("f", 8): "<d"}


def record_length(sample_rate: int, sample_size: int) -> int:
    """Table 5-1 RECORD LENGTH: 2*(SAMPLE RATE*SAMPLE SIZE)/8 + 176."""
    return 2 * sample_rate * sample_size // 8 + HEADER_SIZE


def default_fields(**overrides) -> dict:
    """A valid header; *overrides* replace fields. RECORD LENGTH follows the rate."""
    f = {
        "record_label": RECORD_LABEL,
        "record_version_id": 1,
        "station_id": 24,
        "spacecraft_id": 0,
        "sample_size": 16,
        "sample_rate": 16,
        "validity_flag": 0,
        "agency_flag": 3,
        "rf_to_if_downconv": 8.1e9,
        "if_to_channel_downconv": 3.0e8,
        "time_tag_year": 2026,
        "time_tag_doy": 39,
        "time_tag_second_of_day": 43200,
        "time_tag_picoseconds_of_the_second": 0.0,
        "channel_accumulated_phase": 0.0,
        "channel_phase_polynomial_coefficient_0": 0.0,
        "channel_phase_polynomial_coefficient_1": 0.0,
        "channel_phase_polynomial_coefficient_2": 0.0,
        "channel_phase_polynomial_coefficient_3": 0.0,
        "empty_fields_future_extension": bytes(36),
        "empty_fields_internal_agency_use": bytes(40),
        "end_label": END_LABEL,
    }
    f.update(overrides)
    if "record_length" not in overrides:
        f["record_length"] = record_length(f["sample_rate"], f["sample_size"])
    return f


def header_bytes(**overrides) -> bytes:
    """Build a 176-byte header one field at a time, each at its Table 5-1 offset."""
    fields = default_fields(**overrides)
    buf = bytearray(HEADER_SIZE)
    for name, off, size, kind in HEADER_LAYOUT:
        value = fields[name]
        if kind in ("a", "x"):
            raw = bytes(value)
            assert len(raw) == size, f"{name} must be {size} bytes"
        else:
            raw = struct.pack(_STRUCT[kind, size], value)
        buf[off : off + size] = raw
    return bytes(buf)


def parse_header(buf: bytes) -> dict:
    """Read every Table 5-1 field from its offset."""
    assert len(buf) >= HEADER_SIZE
    out = {}
    for name, off, size, kind in HEADER_LAYOUT:
        raw = bytes(buf[off : off + size])
        out[name] = raw if kind in ("a", "x") else struct.unpack(_STRUCT[kind, size], raw)[0]
    return out


# =============================================================================
# §5.3 / Table 5-2: the data section
# =============================================================================
#
# §5.3.2  samples are packed into 32-bit words (little-endian, §5.1.5);
# §5.3.3  the I and Q of one time sample are adjacent, 1-16 complex samples
#         per word;
# §5.3.4  time runs from the LSB to the MSB;
# Table 5-2  within each complex sample, I takes the lower bits and Q the
#         higher ("MSB ... Q2 I2 Q1 I1 ... LSB").
#
# So with b bits per component, complex sample n sits in word n // (16/b),
# slot p = n % (16/b): I at bit 2*b*p, Q at bit 2*b*p + b.
#
# §5.3.6  each component is the two's-complement code k of b bits, and its
#         value is 2k + 1.


def per_word(bits: int) -> int:
    """Complex samples per 32-bit word (Table 5-2)."""
    return 16 // bits


def slot_offsets(bits: int, slot: int) -> tuple[int, int]:
    """Bit offsets of (I, Q) for *slot* within a 32-bit word (Table 5-2)."""
    return 2 * bits * slot, 2 * bits * slot + bits


def code_range(bits: int) -> range:
    """All b-bit two's-complement codes k."""
    return range(-(1 << (bits - 1)), 1 << (bits - 1))


def value_of(k: int) -> int:
    """§5.3.6: the sample value of code k."""
    return 2 * k + 1


def code_of(value: int) -> int:
    """Inverse of §5.3.6 (value must be odd)."""
    assert value % 2 != 0, f"{value} is not a 2k+1 value"
    return (value - 1) // 2


def encode_codes(k_i, k_q, bits: int) -> bytes:
    """Pack raw codes into the data section, one sample at a time."""
    assert len(k_i) == len(k_q)
    n_per = per_word(bits)
    assert len(k_i) % n_per == 0, "a record holds whole 32-bit words"
    mask = (1 << bits) - 1
    lo, hi = code_range(bits)[0], code_range(bits)[-1]
    words = [0] * (len(k_i) // n_per)
    for n, (ki, kq) in enumerate(zip(k_i, k_q)):
        assert lo <= ki <= hi and lo <= kq <= hi, (ki, kq, bits)
        off_i, off_q = slot_offsets(bits, n % n_per)
        words[n // n_per] |= (ki & mask) << off_i
        words[n // n_per] |= (kq & mask) << off_q
    return b"".join(w.to_bytes(4, "little") for w in words)


def encode_values(v_i, v_q, bits: int) -> bytes:
    """Pack 2k+1 sample values into the data section."""
    return encode_codes([code_of(v) for v in v_i], [code_of(v) for v in v_q], bits)


def decode_values(data: bytes, bits: int) -> tuple[list[int], list[int]]:
    """Unpack a data section to 2k+1 values (I list, Q list)."""
    assert len(data) % 4 == 0, "data section is whole 32-bit words"
    mask = (1 << bits) - 1
    sign = 1 << (bits - 1)
    out_i, out_q = [], []
    for w0 in range(0, len(data), 4):
        word = int.from_bytes(data[w0 : w0 + 4], "little")
        for slot in range(per_word(bits)):
            for off, out in zip(slot_offsets(bits, slot), (out_i, out_q)):
                c = (word >> off) & mask
                k = c - (1 << bits) if c & sign else c
                out.append(value_of(k))
    return out_i, out_q


def record_bytes(k_i, k_q, **header_overrides) -> bytes:
    """One full record: header + data section from raw codes."""
    hdr = default_fields(**header_overrides)
    data = encode_codes(k_i, k_q, hdr["sample_size"])
    return header_bytes(**header_overrides) + data


# =============================================================================
# §5.2.6 downconversion (also DSN 0222 §3.4, Table 3-1)
# =============================================================================
# §5.2.6.2  phi_var(t) = Phi + c0 + c1 (t-t0) + c2 (t-t0)^2 + c3 (t-t0)^3  [turns]
# §5.2.6.3  f_var(t)   = c1 + 2 c2 (t-t0) + 3 c3 (t-t0)^2                  [Hz]
# §5.2.6.1  f_DC = RF_TO_IF DOWNCONV + IF_TO_CHANNEL DOWNCONV + f_var
# t0 is TIME TAG SECOND OF DAY (§5.2.6.2, Table 5-1; 0222 §3.4, Table 3-1); the
# first sample is TIME TAG PICOSECONDS OF THE SECOND after it.


def first_sample_tau(hdr: dict) -> float:
    """t - t0 of the record's first sample, s."""
    return hdr["time_tag_picoseconds_of_the_second"] * 1e-12


def dc_first_sample(hdr: dict) -> float:
    """f_DC at the first sample, Hz."""
    a = first_sample_tau(hdr)
    c1, c2, c3 = (hdr[f"channel_phase_polynomial_coefficient_{k}"] for k in (1, 2, 3))
    return hdr["rf_to_if_downconv"] + hdr["if_to_channel_downconv"] + (c1 + 2 * c2 * a + 3 * c3 * a * a)


def dc_mean(hdr: dict) -> float:
    """Mean f_DC over the record's second (§5.1.1), [first sample, first
    sample + 1 s), Hz: the variable phase advance from t - t0 = a to b = a + 1,
    plus the fixed part."""
    a = first_sample_tau(hdr)
    b = a + 1.0
    c1, c2, c3 = (hdr[f"channel_phase_polynomial_coefficient_{k}"] for k in (1, 2, 3))
    var = c1 * (b - a) + c2 * (b * b - a * a) + c3 * (b * b * b - a * a * a)
    return hdr["rf_to_if_downconv"] + hdr["if_to_channel_downconv"] + var


def dc_phase_first_sample(hdr: dict) -> float:
    """phi_var at the first sample, turns (fixed part not included, §5.2.6.1)."""
    a = first_sample_tau(hdr)
    c = [hdr[f"channel_phase_polynomial_coefficient_{k}"] for k in range(4)]
    return hdr["channel_accumulated_phase"] + c[0] + c[1] * a + c[2] * a * a + c[3] * a * a * a


# =============================================================================
# DSN 820-013 0222-Science Rev. A: internal-agency block (header bytes 132-171)
# =============================================================================
# Table 3-1 / Figure 3-4, fields "only defined for the DSN OLR"; little-endian.
# (field, offset within the block, struct format)
DSN_AGENCY_LAYOUT = (
    ("predict_pass_number", 0, "<H"),
    ("uplink_band", 2, "<B"),  # 0 unknown/n.a., 1 S, 2 X, 3 Ka, 4 Ku, 5 L
    ("downlink_band", 3, "<B"),  # 0 unknown, 1 S, 2 X, 3 Ka, 4 Ku, 5 L
    ("track_mode", 4, "<B"),
    ("uplink_dss_id", 5, "<B"),
    ("olr_id", 6, "<B"),  # 31 OLR1 ... 38 OLR8
    ("olr_software_version", 7, "<B"),
    ("channel_power_calibration_factor", 8, "<f"),  # dBfs -> dBm at the A/D input
    ("total_frequency_offset", 12, "<d"),  # Hz, channel centre minus predict
    ("channel_number", 20, "<B"),  # 0 to 127
)
DSN_AGENCY_SPARE = (21, 19)  # offset, length: internal agency use


def parse_dsn_agency(block: bytes) -> dict:
    """The 40-byte block decoded field by field (DSN_AGENCY_LAYOUT)."""
    assert len(block) == 40
    out = {name: struct.unpack_from(fmt, block, off)[0] for name, off, fmt in DSN_AGENCY_LAYOUT}
    off, n = DSN_AGENCY_SPARE
    out["spare"] = block[off : off + n]
    return out


def dsn_agency_bytes(**fields) -> bytes:
    """A 40-byte block with *fields* at their offsets, spare bytes zero."""
    buf = bytearray(40)
    for name, off, fmt in DSN_AGENCY_LAYOUT:
        struct.pack_into(fmt, buf, off, fields[name])
    return bytes(buf)


# =============================================================================
# §6.2 file names
# =============================================================================

# §6.2.2  MMMMnNNNtTsSSSSrRRcCC-YYDDDHHMMSS.XXX
# §6.2.3  character IDs are uppercase alphanumeric
# The pattern is 37 characters long (Table 4-2 says 39, which does not add up).
FILENAME_RE = re.compile(
    r"^(?P<mission>[A-Z0-9]{4})"
    r"n(?P<scan>\d{3})"
    r"t(?P<type>[ISQ])"
    r"s(?P<station>[A-Z0-9]{4})"
    r"r(?P<receiver>[A-Z0-9]{2})"
    r"c(?P<channel>\d{2})"
    r"-(?P<yy>\d{2})(?P<ddd>\d{3})(?P<hh>\d{2})(?P<mm>\d{2})(?P<ss>\d{2})"
    r"\.(?P<ext>obs|prd)$"
)
FILENAME_LENGTH = 37
