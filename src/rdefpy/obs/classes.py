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
    """Product file line (Table 4-2).

    ``datafile`` is resolved against the observation file's directory.
    """

    datafile: Path
    coherent: bool
    dor_mult: Fraction
    fsub: float
    harmonic: int


@dataclass(repr=False)
class Scan:
    """Scan line (Table 4-1) and its product file lines.

    ``ra``/``dec`` are degrees, ``None`` when the file gives 999.
    ``tfreq`` is Hz, 0 for a quasar.
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
    """Parsed observation file."""

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
