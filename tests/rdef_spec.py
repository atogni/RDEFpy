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


def word_aligned(sample_rate: int, sample_size: int) -> bool:
    """Table 5-1 SAMPLE RATE: SAMPLE RATE * 2 * SAMPLE SIZE is a multiple of 32."""
    return (sample_rate * 2 * sample_size) % 32 == 0


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
