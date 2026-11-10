"""Write text renderings of the test product files, for checking other decoders.

For each ``tests/data/*.prd`` this writes four files to ``tests/reference/``,
named by bits per sample (``sample_16bit`` ... ``sample_1bit``):

- ``.prd``  the product file itself, copied unchanged
- ``.hdr``  every Table 5-1 header field of every record
- ``.tab``  the decoded I/Q samples (§5.3.6 values 2k+1)
- ``.hex``  a hex dump of the whole file

It also builds the downconversion / agency-block reference cases (``CASES``):
synthetic 16-bit, 1 kHz, three-record files written by ``rdef_spec`` into
``tests/reference/`` with the same four files plus

- ``.dc``      per record: f_DC at the first sample, mean f_DC over the
               record, phase at the first sample (§5.2.6)
- ``.agency``  (``dsn_agency_block`` only) the DSN 0222 block, field by field

Everything is computed by ``rdef_spec`` (the Blue Book transcribed as plain
code), never by rdefpy, so rdefpy is checked against it like any other
decoder (``test_reference.py``). Run from the repository root:

    python tests/make_reference.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rdef_spec as spec  # noqa: E402

DATA = HERE / "data"
OUT = HERE / "reference"
PAIRS_PER_LINE = 10


def records(raw: bytes):
    """Yield (header dict, data bytes) per record, walking RECORD LENGTH."""
    pos = 0
    while pos < len(raw):
        hdr = spec.parse_header(raw[pos:])
        end = pos + hdr["record_length"]
        assert end <= len(raw), "truncated record"
        yield hdr, raw[pos + spec.HEADER_SIZE : end]
        pos = end


def _field(value) -> str:
    if isinstance(value, bytes):
        if value == b"RDEF":
            return "RDEF"
        return f"{len(value)} bytes, all zero" if not any(value) else value.hex(" ")
    return repr(value)  # floats: shortest repr that reads back to the same binary64


def hdr_text(path: Path) -> str:
    raw = path.read_bytes()
    recs = list(records(raw))
    lines = [f"{_label(path)}.prd: record headers (CCSDS 506.1-B-1 Table 5-1)"]
    pos = 0
    for n, (hdr, _) in enumerate(recs, 1):
        lines += ["", f"Record {n} of {len(recs)}, file offset {pos}"]
        lines += [f"  {name:<40s} = {_field(hdr[name])}" for name, *_ in spec.HEADER_LAYOUT]
        pos += hdr["record_length"]
    return "\n".join(lines) + "\n"


def tab_text(path: Path) -> str:
    """One line per 10 samples: record (from 1), first sample index (from 0), then I Q pairs."""
    lines = []
    for n, (hdr, data) in enumerate(records(path.read_bytes()), 1):
        i, q = spec.decode_values(data, hdr["sample_size"])
        for s in range(0, len(i), PAIRS_PER_LINE):
            pairs = "".join(f" {a:6d} {b:6d}" for a, b in zip(i[s : s + PAIRS_PER_LINE], q[s : s + PAIRS_PER_LINE]))
            lines.append(f"{n:4d} {s:8d}{pairs}")
    return "\n".join(lines) + "\n"


def hex_text(path: Path) -> str:
    """16 bytes per line, offset in hex; no lines skipped."""
    raw = path.read_bytes()
    return "".join(f"{o:08x}  {raw[o : o + 16].hex(' ')}\n" for o in range(0, len(raw), 16))


RENDER = {".hdr": hdr_text, ".tab": tab_text, ".hex": hex_text}


def name(prd: Path) -> str:
    """``sample_<bits>bit``, from the first record's SAMPLE SIZE."""
    return f"sample_{spec.parse_header(prd.read_bytes())['sample_size']}bit"


def _label(prd: Path) -> str:
    """Name shown in a .hdr title: the case name in OUT, else ``name()``."""
    return prd.stem if prd.parent == OUT else name(prd)


# =============================================================================
# Downconversion / agency-block reference cases (synthetic)
# =============================================================================

CASE_FS = 1000  # samples per second
CASE_BITS = 16
CASE_RECORDS = 3
CASE_SOD = 43200  # second of day of the first record

# Steering: one cubic phase in T = seconds from the first record's second of
# day, phi(T) = P0 + P1 T + P2 T^2 + P3 T^3 turns. Record n (second of day
# CASE_SOD + n) gets the Taylor coefficients about T = n, so phase and rate are
# continuous from one record to the next (Table 5-1 note on the phase fields).
# Binary fractions: every coefficient is exact in float64.
P0, P1, P2, P3 = 100.25, 1234.5, -0.125, -0.0009765625


def _steering(n: int) -> dict:
    phi = P0 + P1 * n + P2 * n**2 + P3 * n**3
    return {
        "channel_accumulated_phase": float(math.floor(phi)),
        "channel_phase_polynomial_coefficient_0": phi - math.floor(phi),
        "channel_phase_polynomial_coefficient_1": P1 + 2 * P2 * n + 3 * P3 * n**2,
        "channel_phase_polynomial_coefficient_2": P2 + 3 * P3 * n,
        "channel_phase_polynomial_coefficient_3": P3,
    }


# DSN 0222 Table 3-1 values: nonzero and distinct in every field.
DSN_AGENCY_FIELDS = dict(
    predict_pass_number=4321, uplink_band=1, downlink_band=2, track_mode=3,
    uplink_dss_id=17, olr_id=34, olr_software_version=4,
    channel_power_calibration_factor=-42.5, total_frequency_offset=1234.5625,
    channel_number=19,
)

# case name -> header fields that differ from rdef_spec.default_fields
CASES = {
    "steered_ps0": dict(time_tag_picoseconds_of_the_second=0.0, agency_flag=0),
    "dsn_agency_block": dict(
        time_tag_picoseconds_of_the_second=50_000.0,
        agency_flag=3,
        empty_fields_internal_agency_use=spec.dsn_agency_bytes(**DSN_AGENCY_FIELDS),
    ),
}


def case_bytes(case: str) -> bytes:
    """The case's product file, built by rdef_spec record by record."""
    out = []
    for n in range(CASE_RECORDS):
        # 16-bit codes: steps 7*257 and 11*257 wrap the full code range many
        # times per record, so both bytes and the sign bit change.
        k_i = [(1799 * s + 3 * n) % 65536 - 32768 for s in range(CASE_FS)]
        k_q = [(2827 * s + 5 * n + 16384) % 65536 - 32768 for s in range(CASE_FS)]
        out.append(spec.record_bytes(
            k_i, k_q, sample_size=CASE_BITS, sample_rate=CASE_FS,
            time_tag_second_of_day=CASE_SOD + n, **_steering(n), **CASES[case],
        ))
    return b"".join(out)


def dc_text(path: Path) -> str:
    """Per record: f_DC at the first sample, mean f_DC, phase at the first sample."""
    lines = [
        f"# {_label(path)}.prd: downconversion (CCSDS 506.1-B-1 sec. 5.2.6) per record, from rdef_spec",
        "# t0 = TIME TAG SECOND OF DAY (Table 5-1, sec. 5.2.6.2; DSN 0222 Table 3-1, sec. 3.4);",
        "#   first sample at t - t0 = picoseconds*1e-12",
        "# f_first: total f_DC at the record's first sample [Hz]",
        "# f_mean:  total f_DC averaged over [first sample, first sample + 1 s) [Hz]",
        "# phase_first: variable-part phase at the first sample, Phi + c0 + ... [turns]",
        "# record f_first f_mean phase_first",
    ]
    for n, (hdr, _) in enumerate(records(path.read_bytes()), 1):
        lines.append(f"{n} {spec.dc_first_sample(hdr)!r} {spec.dc_mean(hdr)!r} "
                     f"{spec.dc_phase_first_sample(hdr)!r}")
    return "\n".join(lines) + "\n"


def agency_text(path: Path) -> str:
    """Every record's internal-agency block decoded with the DSN 0222 layout."""
    lines = [f"# {_label(path)}.prd: internal-agency block (header bytes 132-171, 0-based;",
             "# 0222 numbers bytes from 1: 133-172), DSN 820-013",
             "# 0222-Science Rev. A Table 3-1 / Figure 3-4 layout, little-endian, from rdef_spec",
             "# record field = value"]
    for n, (hdr, _) in enumerate(records(path.read_bytes()), 1):
        for field, value in spec.parse_dsn_agency(hdr["empty_fields_internal_agency_use"]).items():
            lines.append(f"{n} {field} = {_field(value)}")
    return "\n".join(lines) + "\n"


CASE_RENDER = {**RENDER, ".dc": dc_text}
AGENCY_CASES = ("dsn_agency_block",)


def case_render(case: str) -> dict:
    return {**CASE_RENDER, ".agency": agency_text} if case in AGENCY_CASES else CASE_RENDER


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for prd in sorted(DATA.glob("*.prd")):
        stem = OUT / name(prd)
        stem.with_suffix(".prd").write_bytes(prd.read_bytes())
        for ext, render in RENDER.items():
            stem.with_suffix(ext).write_text(render(prd), encoding="ascii", newline="\n")
        print(prd.name, "->", stem.name)
    for case in CASES:
        prd = (OUT / case).with_suffix(".prd")
        prd.write_bytes(case_bytes(case))
        for ext, render in case_render(case).items():
            prd.with_suffix(ext).write_text(render(prd), encoding="utf-8", newline="\n")
        print(case)
