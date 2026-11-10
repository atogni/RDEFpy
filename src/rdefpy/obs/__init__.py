# Copyright (c) 2026 Morehead State University
# Author: Andrea Togni
# SPDX-License-Identifier: MIT

"""
Observation file (.obs) sub-package for CCSDS 506.1-B-1 §4.

An observation file is ASCII: a header section (version, receive and
optional transmit station), one scan section per scan (a scan line and one
product file line per channel), and an ending section (optional log lines,
then the end line).  The first character of each line gives its type.

    from rdefpy import read_observation, read_product

    obs = read_observation("session.obs")
    for scan in obs:
        for entry in scan.products:
            pf = read_product(entry.datafile)

Classes
-------
ObservationFile
    Parsed observation file: header fields, scans, and log lines.
Scan
    Scan line (Table 4-1) and its product file entries.
ProductEntry
    Product file line (Table 4-2).

I/O Functions
-------------
read_observation
    Parse an observation file.
"""

from rdefpy.obs.classes import (
    ObservationFile,
    ProductEntry,
    Scan,
)
from rdefpy.obs.reader import read_observation

__all__ = [
    "ObservationFile",
    "ProductEntry",
    "Scan",
    "read_observation",
]
