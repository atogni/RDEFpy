# Copyright (c) 2026 Morehead State University
# Author: Andrea Togni
# SPDX-License-Identifier: MIT

"""CCSDS Delta-DOR Raw Data Exchange Format (RDEF) product-file reader and writer.

Implements the product files (.prd) of CCSDS 506.1-B-1 (June 2013), §5:
binary one-second records of complex I/Q with full downconversion
metadata, and reads the ASCII observation files (§4) that index them.

    from rdefpy import read_product

    pf = read_product("scan.prd")
    pf.info()

Product files run to gigabytes. ``read_product`` loads records at or below
1 kHz (small, the common case) and leaves faster ones on disk: their samples
are read when accessed and freed when no longer used, so a loop holds one
record at a time; ``load()`` keeps them, ``release()`` drops them:

    pf = read_product("scan.prd")
    for rec in pf:
        print(rec.header.timetag, rec.data.power_db)
    pf.load()                                          # keep all in memory

Scope
-----
This is a format layer: it decodes bytes into samples with correct time
tags, and encodes them back, and depends on NumPy alone. One estimator is
included, ``extract_carrier`` (FrFT + SPI open-loop carrier frequency per
block, see ``rdefpy.carrier``); other spectra, tracking and simulation
belong to the caller.

Low-level decoding is meant to stay as it is: the ``2k+1`` transform
(§5.3.6), the bit offsets, and the header struct ARE the format.
``tests/test_rdefpy.py`` pins all five sample sizes.
"""

from importlib.metadata import version as _version

from rdefpy._utils import set_threads
from rdefpy.browse import browse
from rdefpy.carrier import extract_carrier
from rdefpy.obs import ObservationFile, ProductEntry, Scan, read_observation
from rdefpy.prd import (
    DSN_AGENCY_DTYPE,
    ProductFile,
    Record,
    RecordData,
    RecordHeader,
    pack_record_data,
    pack_record_header,
    read_headers,
    read_product,
    scan_product,
    write_product,
)

__version__ = _version("rdefpy")

__all__ = [
    # Observation files (.obs, §4)
    "ObservationFile",
    "ProductEntry",
    "Scan",
    "read_observation",
    # Product files (.prd, §5)
    "ProductFile",
    "Record",
    "RecordData",
    "RecordHeader",
    "pack_record_data",
    "pack_record_header",
    "read_headers",
    "read_product",
    "scan_product",
    "write_product",
    "DSN_AGENCY_DTYPE",
    # Carrier extraction (FrFT + SPI)
    "extract_carrier",
    # Record browser (needs matplotlib)
    "browse",
    # Threads for large records
    "set_threads",
]
