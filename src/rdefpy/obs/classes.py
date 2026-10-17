# Copyright (c) 2026 Morehead State University
# Author: Andrea Togni
# SPDX-License-Identifier: MIT

"""Observation file containers (CCSDS 506.1-B-1 §4)."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import TextIO

import numpy as np

from rdefpy._utils import _align_rows


def _num(v: float) -> str:
    # Full precision (shortest round-trip), thousands separators.
    return f"{v:,}".removesuffix(".0")


def _angle(v: float | None) -> str:
    return "999 (not provided)" if v is None else f"{_num(v)} deg"


@dataclass
class ProductEntry:
    """
    One product file line (``D`` line) of a scan (CCSDS 506.1-B-1 Table 4-2).

    Each entry names the product file recorded for one frequency channel and
    says how to compute that channel's tone frequency.

    Attributes
    ----------
    datafile : Path
        Product file name joined to the observation file's directory, so
        ``read_product(entry.datafile)`` opens it wherever the caller runs.
    coherent : bool
        Coherence flag (``T``/``F``).  True: the tone frequency follows from
        *dor_mult* and *harmonic*.  False: from *fsub* and *harmonic*.
    dor_mult : Fraction
        Fundamental tone as a fraction of the scan's TFREQ, e.g. ``1/440``,
        when *coherent*.  Any number is allowed otherwise.
    fsub : float
        Fundamental subcarrier frequency in Hz, when not *coherent*.  When
        *coherent* the spec allows any value; a non-numeric one is NaN.
    harmonic : int
        Subcarrier or coherent tone harmonic number.
    """

    datafile: Path
    coherent: bool
    dor_mult: Fraction
    fsub: float
    harmonic: int


@dataclass(repr=False)
class Scan:
    """
    One scan section: the scan line (``S`` line, CCSDS 506.1-B-1 Table 4-1)
    and its product file entries.

    ``print(scan)`` shows every field and one line per product file.

    Attributes
    ----------
    scan_num : int
        Scan number, 1-999, increasing through the file.
    src_id : str
        Source: spacecraft ID (4 characters) or quasar ID (up to 16).
    start_time, stop_time : numpy.datetime64
        Nominal scan start and stop, UTC, 1 s precision.
    ra, dec : float or None
        Source right ascension (0-360) and declination (-90 to +90) in
        degrees, true equator and equinox of date, corrected for aberration.
        None when the file gives 999 (not provided).
    tfreq : float
        Transmitted frequency in Hz for a spacecraft; 0 for a quasar.
    products : list[ProductEntry]
        One entry per frequency channel, in file order.
    """

    scan_num: int
    src_id: str
    start_time: np.datetime64
    stop_time: np.datetime64
    ra: float | None
    dec: float | None
    tfreq: float
    products: list[ProductEntry] = field(default_factory=list)

    @property
    def kind(self) -> str:
        """``"quasar"`` when TFREQ is 0 (Table 4-1), else ``"spacecraft"``."""
        return "quasar" if self.tfreq == 0 else "spacecraft"

    def __repr__(self) -> str:
        return (f"<Scan {self.scan_num:03d} {self.src_id} {self.start_time}Z "
                f"{len(self.products)} products>")

    def __str__(self) -> str:
        rows = [
            ("src_id", f"{self.src_id} ({self.kind})"),
            ("start_time", f"{self.start_time}Z"),
            ("stop_time", f"{self.stop_time}Z"),
            ("ra", _angle(self.ra)),
            ("dec", _angle(self.dec)),
            ("tfreq", f"{_num(self.tfreq)} Hz"),
        ]
        for i, e in enumerate(self.products, 1):
            tone = (f"T  DOR_MULT {e.dor_mult}" if e.coherent
                    else f"F  FSUB {_num(e.fsub)} Hz")
            rows.append((f"product {i}",
                         f"{e.datafile.name}  {tone}  harmonic {e.harmonic:+d}"))
        return f"Scan {self.scan_num:03d}\n" + _align_rows(rows, indent="  ")


@dataclass(repr=False)
class ObservationFile:
    """
    Parsed RDEF observation file (.obs, CCSDS 506.1-B-1 §4).

    Iterate, index and ``len()`` over its scans; ``print(obs)`` or
    ``obs.info()`` shows a summary.  Build one with :func:`read_observation`.

    Attributes
    ----------
    path : Path
        File the observation was read from.
    version : int
        Format version (``V`` line).  §4.2.4 requires it to match the
        product files' version; that is not checked here.
    receive_station : str
        Receiving station ID, 4 characters (``R`` line).
    transmit_station : str or None
        Transmitting station ID (``T`` line); None for one-way data.
    scans : list[Scan]
        Scan sections in file order.
    logs : list[str]
        Text of the log lines (``F`` lines) after the letter ``F``: how to
        retrieve receiver messages, data and logs.
    """

    path: Path
    version: int
    receive_station: str
    transmit_station: str | None
    scans: list[Scan]
    logs: list[str]

    def __iter__(self) -> Iterator[Scan]:
        return iter(self.scans)

    def __len__(self) -> int:
        return len(self.scans)

    def __getitem__(self, i: int) -> Scan:
        return self.scans[i]

    def __repr__(self) -> str:
        return f"<ObservationFile {self.path.stem!r} {len(self.scans):,} scans>"

    def __str__(self) -> str:
        rows = [
            ("path", str(self.path.resolve())),
            ("version", str(self.version)),
            ("receive_station", self.receive_station),
            ("transmit_station", self.transmit_station or "None (one-way)"),
            ("scans", f"{len(self.scans):,}"),
            ("products", f"{sum(len(s.products) for s in self.scans):,}"),
        ]
        for s in self.scans:
            rows.append((f"scan {s.scan_num:03d}",
                         f"{s.src_id} ({s.kind}) {s.start_time}Z to {s.stop_time}Z, "
                         f"{len(s.products)} products"))
        rows += [("log", log) for log in self.logs]
        return f"ObservationFile {self.path.stem}\n" + _align_rows(rows, indent="  ")

    def info(self, buf: TextIO | None = None) -> None:
        """Print a summary of the observation file (same text as ``str``)."""
        print(self, file=buf)
