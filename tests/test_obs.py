"""Observation file reader (CCSDS 506.1-B-1 §4)."""

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from rdefpy import read_observation

DATA = Path(__file__).parent / "data"

# Two-way session, spacecraft scan then quasar scan, log line.
OBS = """\
# Observation File
V VERSION = 1
R STATION = DS24
T STATION = DS25
Z
# SCAN_NUM  SRC_ID  START_TIME         STOP_TIME          RA          DEC         TFREQ
S 001       EXMP    2026-039T12:00:00  2026-039T12:00:02  999         999         8400000000.0000
D EXMPn001tSsDS24r01c01-26039120000.prd   T   0       0.0        0
D EXMPn001tSsDS24r01c02-26039120000.prd   T   1/440   0.0        -1
D EXMPn001tSsDS24r01c03-26039120000.prd   F   0       375000.0   2
Z
S 002       Q_TEST  2026-039T12:01:00  2026-039T12:02:00  60.797422   26.005385   0.0000
D EXMPn002tQsDS24r01c01-26039120100.prd   T   0       375000.0   0
Z
F LOGfile session.log
E *=END=*
"""


def _write(tmp_path: Path, text: str, newline: str = "\n") -> Path:
    p = tmp_path / "EXMPn000tIsDS24r01c00-26039120000.obs"
    p.write_bytes(text.replace("\n", newline).encode("latin-1"))
    return p


@pytest.mark.parametrize("newline", ["\n", "\r", "\r\n", "\n\r"])  # §4.1.3 g
def test_read_observation(tmp_path, newline):
    obs = read_observation(_write(tmp_path, OBS, newline))
    assert (obs.version, obs.receive_station, obs.transmit_station) == (1, "DS24", "DS25")
    assert obs.logs == ["LOGfile session.log"]
    assert len(obs) == 2

    s, q = obs
    assert q.src_id == "Q_TEST" and q.tfreq == 0.0
    assert (q.ra, q.dec) == (60.797422, 26.005385)
    assert s.ra is None and s.dec is None
    assert s.start_time == np.datetime64("2026-02-08T12:00:00")
    assert s.stop_time - s.start_time == np.timedelta64(2, "s")
    assert s.tfreq == 8.4e9

    e = s.products[1]
    assert e.datafile == tmp_path / "EXMPn001tSsDS24r01c02-26039120000.prd"
    assert (e.coherent, e.dor_mult, e.harmonic) == (True, Fraction(1, 440), -1)
    assert s.products[2].coherent is False and s.products[2].fsub == 375000.0


def test_one_way_and_limits(tmp_path):
    text = (OBS.replace("T STATION = DS25\n", "")
               .replace("# Observation File", "#" + "x" * 179)   # exactly 180
               .replace("12:00:02", "12:00:02Z"))                 # time code B Z
    obs = read_observation(_write(tmp_path, text))
    assert obs.transmit_station is None
    assert obs[0].stop_time == np.datetime64("2026-02-08T12:00:02")


def test_lenient_where_spec_allows(tmp_path):
    text = (OBS.replace("F LOGfile", "FLOGfile")                        # §4.4.2
               .replace("T   1/440   0.0", "T   +1/440   x")             # FSUB any value
               .replace("F   0       375000.0", "F   1e-3    375000.0"))  # any number
    lines = text.splitlines(keepends=True)
    lines[1] = lines[1].replace("\n", "\r\n")                           # mixed endings
    obs = read_observation(_write(tmp_path, "".join(lines)))
    assert obs.logs == ["LOGfile session.log"]
    e1, e2 = obs[0].products[1:]
    assert e1.dor_mult == Fraction(1, 440) and np.isnan(e1.fsub)
    assert e2.dor_mult == Fraction(1, 1000)


def test_last_line_needs_terminator(tmp_path):
    with pytest.raises(ValueError, match="no line terminator"):
        read_observation(_write(tmp_path, OBS.removesuffix("\n")))


def test_str_repr(tmp_path):
    obs = read_observation(_write(tmp_path, OBS))
    assert repr(obs) == "<ObservationFile 'EXMPn000tIsDS24r01c00-26039120000' 2 scans>"
    text = str(obs)
    assert text.startswith("ObservationFile EXMPn000tIsDS24r01c00-26039120000\n")
    assert "  transmit_station: DS25" in text
    assert "scan 002: Q_TEST (quasar) 2026-02-08T12:01:00Z to 2026-02-08T12:02:00Z, " \
           "1 products" in text
    assert "log: LOGfile session.log" in text

    s = obs[0]
    assert repr(s) == "<Scan 001 EXMP 2026-02-08T12:00:00Z 3 products>"
    text = str(s)
    assert text.startswith("Scan 001\n")
    assert "     tfreq: 8,400,000,000 Hz" in text
    assert "product 2: EXMPn001tSsDS24r01c02-26039120000.prd  T  DOR_MULT 1/440  " \
           "harmonic -1" in text
    assert "product 3: EXMPn001tSsDS24r01c03-26039120000.prd  F  FSUB 375,000 Hz  " \
           "harmonic +2" in text


def test_datafile_reads(tmp_path):
    # Scan times match the product file the entry names.
    read_product = pytest.importorskip("rdefpy.prd").read_product
    obs = read_observation(_write(tmp_path, OBS))
    entry = obs[0].products[0]
    pf = read_product(DATA / entry.datafile.name)
    assert pf[0].header.timetag == obs[0].start_time


@pytest.mark.parametrize("old, new, msg", [
    # §4.1.3
    ("# Observation File", "#" + "x" * 180, "limit 180"),
    ("V VERSION", "V\tVERSION", "non-printable"),
    ("Q_TEST", "Q_TÉST", "non-ASCII"),
    ("Z\nS 002", "Z\n\nS 002", "blank line"),
    ("V VERSION", "VX VERSION", "unknown line type"),
    # §4.1.1, §4.1.4 structure
    ("R STATION = DS24\n", "", "no version or receive"),
    ("V VERSION = 1\n", "V VERSION = 1\nV VERSION = 2\n", "repeated"),
    ("T STATION = DS25\n", "T STATION = DS25\nT STATION = DS26\n", "repeated"),
    ("T STATION = DS25\nZ\n", "T STATION = DS25\n", "header section"),
    ("375000.0   0\nZ\nF", "375000.0   0\nF", "not allowed here"),
    ("Z\nS 002", "Z\nF early\nS 002", "not allowed here"),
    ("D EXMPn002tQsDS24r01c01-26039120100.prd   T   0       375000.0   0\n", "",
     "no product file"),
    ("S 002", "R STATION = DS24\nS 002", "not allowed here"),
    ("E *=END=*\n", "", "incomplete"),
    ("E *=END=*\n", "E *=END=*\n# trailing\n", "after end line"),
    ("E *=END=*", "E END", "end line"),
    # §4.2
    ("V VERSION = 1", "V VERSION=1", "VERSION = <value>"),
    ("V VERSION = 1", "V VERSION = 1.0", "not an integer"),
    ("R STATION = DS24", "R STATION = DSS24", "4 characters"),
    # Table 4-1
    ("S 002", "S 2  ", "001-999"),
    ("S 002", "S 001", "not increasing"),
    ("Q_TEST", "Q_" + "X" * 15, "up to 16"),
    ("EXMP    2026", "EXMPL   2026", "4 characters for a spacecraft"),
    ("2026-039T12:01:00", "2026-39T12:01:00", "YYYY-DDDThh:mm:ss"),
    ("2026-039T12:01:00", "2026-366T12:01:00", "day of year"),
    ("2026-039T12:01:00", "2026-039T12:01:00.5", "YYYY-DDDThh:mm:ss"),
    ("60.797422", "361", "RA"),
    ("26.005385", "-91", "DEC"),
    ("60.797422", "nan", "decimal notation"),
    ("8400000000.0000", "-1", "negative"),
    ("8400000000.0000", "8400000000.00000000", "16 significant"),
    ("12:02:00  60.797422", "12:02:00  60.797422 x", "7 items"),
    # Table 4-2
    ("D EXMPn002", "D ../EXMPn002", "bare file name"),
    ("T   1/440", "X   1/440", "not T or F"),
    ("T   1/440", "T   1.5", "fraction of two integers"),
    ("T   1/440", "T   1/0", "zero denominator"),
    ("F   0", "F   abc", "not a number"),
    ("F   0", "F   1/0", "zero denominator"),
    ("D EXMPn002tQsDS24r01c01-26039120100.prd", "D ..", "bare file name"),
    ("D EXMPn002tQsDS24r01c01-26039120100.prd", r"D a\b.prd", "bare file name"),
    ("2026-039T12:01:00", "2026-039T24:01:00", "out of range"),
    ("2026-039T12:01:00", "2026-039T12:01:60", "out of range"),
    ("0.0        -1", "0.0        1_0", "not an integer"),
    ("0.0        -1", "0.0        -1 x", "5 items"),
])
def test_malformed(tmp_path, old, new, msg):
    assert old in OBS
    with pytest.raises(ValueError, match=msg):
        read_observation(_write(tmp_path, OBS.replace(old, new, 1)))
