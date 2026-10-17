# Copyright (c) 2026 Morehead State University
# Author: Andrea Togni
# SPDX-License-Identifier: MIT

"""Observation file reader (CCSDS 506.1-B-1 §4).

Strict: a file that breaks a "shall" of §4 raises ``ValueError`` naming
the file and line.  Cross-checks against §6.2 file names are not made.
"""

from __future__ import annotations

import re
from datetime import datetime
from fractions import Fraction
from pathlib import Path

import numpy as np

from rdefpy.obs.classes import ObservationFile, ProductEntry, Scan

_DECIMAL = re.compile(r"[+-]?(\d+\.?\d*|\.\d+)")
_TIME = re.compile(r"\d{4}-\d{3}T\d{2}:\d{2}:\d{2}Z?")  # ASCII time code B, 1 s
_NOT_PROVIDED = 999.0  # RA/DEC "not provided" (Table 4-1)


def _decimal(s: str, what: str) -> float:
    # Decimal notation, no more than 16 significant digits (Tables 4-1, 4-2).
    if not _DECIMAL.fullmatch(s):
        raise ValueError(f"{what} {s!r} is not decimal notation")
    if len(re.sub(r"\D", "", s).lstrip("0")) > 16:
        raise ValueError(f"{what} {s!r} has more than 16 significant digits")
    return float(s)


def _time(s: str) -> np.datetime64:
    # Leap second :60 is valid time code B but not representable; rejected.
    if not _TIME.fullmatch(s):
        raise ValueError(f"time {s!r} is not YYYY-DDDThh:mm:ss")
    try:
        t = datetime.strptime(s.removesuffix("Z"), "%Y-%jT%H:%M:%S")
    except ValueError:
        raise ValueError(f"time {s!r}: field out of range") from None
    if t.year != int(s[:4]):  # strptime rolls day 366 of a common year over
        raise ValueError(f"time {s!r}: day of year out of range")
    return np.datetime64(t, "s")


def _angle(s: str, what: str, lo: float, hi: float) -> float | None:
    v = _decimal(s, what)
    if v == _NOT_PROVIDED:
        return None
    if not lo <= v <= hi:
        raise ValueError(f"{what} {s!r} outside {lo:g} to {hi:g}")
    return v


def _dor_mult(s: str, coherent: bool) -> Fraction:
    # Coherent: a fraction of two integers (a bare integer, e.g. 0, allowed).
    # Not coherent: "any number is allowed" (Table 4-2).
    if coherent and not re.fullmatch(r"[+-]?\d+(/\d+)?", s):
        raise ValueError(f"DOR_MULT {s!r} is not a fraction of two integers")
    try:
        return Fraction(s)
    except ValueError:
        raise ValueError(f"DOR_MULT {s!r} is not a number") from None
    except ZeroDivisionError:
        raise ValueError(f"DOR_MULT {s!r} has a zero denominator") from None


def _fsub(s: str, coherent: bool) -> float:
    # Coherent: "any value is allowed" (Table 4-2), kept if numeric, else NaN.
    if not coherent:
        return _decimal(s, "FSUB")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _assignment(parts: list[str], key: str) -> str:
    # 'V VERSION = 1', 'R STATION = DS24'
    if len(parts) != 4 or parts[1:3] != [key, "="]:
        raise ValueError(f"expected '{parts[0]} {key} = <value>'")
    return parts[3]


def _station(parts: list[str]) -> str:
    s = _assignment(parts, "STATION")
    if len(s) != 4:
        raise ValueError(f"station {s!r} is not 4 characters")
    return s


def _lines(path: Path) -> list[str]:
    # §4.1.3 g: each line ends in LF, CR, CR/LF or LF/CR.  No blank lines
    # are allowed, so a mix of terminators still splits unambiguously.
    lines = re.split(r"\r\n|\n\r|\n|\r", path.read_bytes().decode("latin-1"))
    if lines[-1]:
        raise ValueError(f"{path.name}:{len(lines)}: last line has no line terminator")
    return lines[:-1]


def read_observation(path: str | Path) -> ObservationFile:
    """
    Read an RDEF observation file (CCSDS 506.1-B-1 §4).

    The file is checked strictly against §4: section order and ``Z``
    terminators, single version/station lines, field formats and ranges
    (Tables 4-1, 4-2), line length and characters (§4.1.3).  Any of the
    four line terminators of §4.1.3 g is accepted, also mixed.

    Parameters
    ----------
    path : str or Path
        Path to the ``.obs`` file.

    Returns
    -------
    ObservationFile
        Header fields, scans with their product file entries, and log lines.

    Raises
    ------
    ValueError
        The file breaks a rule of §4.  The message names the file and line,
        the rule broken, and quotes the line.
    OSError
        The file cannot be read.

    Notes
    -----
    Not checked: §6.2 file names (pattern, scan number or station in the
    name against the file contents), and whether the product files exist.
    A leap second (``hh:mm:60``) is valid time code but is rejected, as
    ``numpy.datetime64`` cannot hold it.

    Examples
    --------
    >>> obs = read_observation("session.obs")
    >>> for scan in obs:
    ...     for entry in scan.products:
    ...         print(scan.scan_num, entry.datafile.name, entry.harmonic)
    """
    path = Path(path)
    version = receive = transmit = None
    scans: list[Scan] = []
    logs: list[str] = []
    # header -> (scan_start -> scan)+ -> ending -> done   (§4.1.1, §4.1.4)
    state = "header"

    for n, line in enumerate(_lines(path), 1):
        kind = line[:1]
        parts = line.split()
        try:
            # §4.1.3 a-c
            if len(line) > 180:
                raise ValueError(f"line is {len(line)} characters, limit 180")
            if not all(" " <= c <= "~" for c in line):
                raise ValueError("non-printable or non-ASCII character")
            if not parts:
                raise ValueError("blank line")
            if state == "done":
                raise ValueError("line after end line")
            if kind == "#":
                continue
            # Z (§4.1.4) and F (§4.4.2) lines need only start with their letter.
            if kind not in ("Z", "F") and parts[0] != kind:
                raise ValueError(f"unknown line type {parts[0]!r}")

            if state == "header":
                if kind == "V" and version is None:
                    v = _assignment(parts, "VERSION")
                    if not v.isdigit():
                        raise ValueError(f"version {v!r} is not an integer")
                    version = int(v)
                elif kind == "R" and receive is None:
                    receive = _station(parts)
                elif kind == "T" and transmit is None:
                    transmit = _station(parts)
                elif kind == "Z":
                    if version is None or receive is None:
                        raise ValueError("header has no version or receive station line")
                    state = "scan_start"
                else:
                    raise ValueError("unexpected or repeated line in header section")

            elif state == "scan_start" and kind == "S":
                if len(parts) != 8:
                    raise ValueError("scan line needs 7 items")
                num, src, t0, t1, ra, dec, tfreq = parts[1:]
                if not re.fullmatch(r"\d{3}", num) or num == "000":
                    raise ValueError(f"scan number {num!r} is not 001-999")
                if scans and int(num) <= scans[-1].scan_num:
                    raise ValueError(f"scan number {num} is not increasing")
                f = _decimal(tfreq, "TFREQ")
                if f < 0:
                    raise ValueError(f"TFREQ {tfreq!r} is negative")
                if len(src) > 16 or (f != 0 and len(src) != 4):
                    raise ValueError(f"source ID {src!r}: 4 characters for a "
                                     "spacecraft, up to 16 for a quasar")
                scans.append(Scan(int(num), src, _time(t0), _time(t1),
                                  _angle(ra, "RA", 0, 360),
                                  _angle(dec, "DEC", -90, 90), f))
                state = "scan"

            elif state == "scan" and kind == "D":
                if len(parts) != 6:
                    raise ValueError("product file line needs 5 items")
                name, coh, mult, fsub, harm = parts[1:]
                if re.search(r"[/\\:]", name) or name in (".", ".."):
                    raise ValueError(f"datafile {name!r} is not a bare file name")
                if coh not in ("T", "F"):
                    raise ValueError(f"coherence flag {coh!r} is not T or F")
                if not re.fullmatch(r"[+-]?\d+", harm):
                    raise ValueError(f"harmonic {harm!r} is not an integer")
                scans[-1].products.append(ProductEntry(
                    path.parent / name, coh == "T", _dor_mult(mult, coh == "T"),
                    _fsub(fsub, coh == "T"), int(harm)))

            elif state == "scan" and kind == "Z":
                if not scans[-1].products:
                    raise ValueError("scan section has no product file line")
                state = "scan_start"

            elif state in ("scan_start", "ending") and kind in "FE" and scans:
                if kind == "F":
                    logs.append(line[1:].strip())
                    state = "ending"
                elif parts == ["E", "*=END=*"]:
                    state = "done"
                else:
                    raise ValueError("end line is not 'E *=END=*'")

            else:
                raise ValueError(f"{kind!r} line not allowed here ({state})")
        except ValueError as e:
            raise ValueError(f"{path.name}:{n}: {e}: {line!r}") from None

    if state != "done":
        raise ValueError(f"{path.name}: incomplete, ends in {state} "
                         "(missing 'Z' or 'E *=END=*')")
    assert version is not None and receive is not None  # header Z checked both
    return ObservationFile(path, version, receive, transmit, scans, logs)
