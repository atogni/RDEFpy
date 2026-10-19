# Copyright (c) 2026 Morehead State University
# Author: Andrea Togni
# SPDX-License-Identifier: MIT

"""
Product file (.prd) sub-package for CCSDS 506.1-B-1 §5.

Product files are binary: a run of one-second records, each a 176-byte
header (Table 5-1) followed by packed complex I/Q at 1, 2, 4, 8, or 16
bits per component (Table 5-2).  They reach multiple gigabytes, so
``read_product`` loads only records at or below 1 kHz and leaves faster
ones on disk:

    from rdefpy import read_product

    pf = read_product("scan.prd")    # <= 1 kHz loaded; faster stays on disk
    x = pf[0].data.samples           # from memory, or read now and freed when dropped
    pf.load()                        # or keep every record's samples in memory
    pf.release()                     # ... until released

Filtering (time window, header fields, downlink band) skips a record
without reading its data.

Classes
-------
ProductFile
    Container for a complete product file: its name and a list
    of ``Record`` objects.
Record
    Single RDEF record combining a parsed header and I/Q sample data.
RecordHeader
    Parsed 176-byte RDEF record header (Table 5-1).
RecordData
    I/Q sample container with level and saturation checks.

I/O Functions
-------------
read_product
    Decode a product file, eagerly or lazily, with optional filtering.
read_headers
    Read only record headers, skipping every data section.
scan_product
    Fast scan to count records and check length consistency.
write_product
    Write a ``ProductFile`` to disk.
"""

from rdefpy.prd.classes import (
    ProductFile,
    Record,
    RecordData,
    RecordHeader,
)
from rdefpy.prd.reader import (
    read_headers,
    read_product,
    scan_product,
)
from rdefpy.prd.writer import (
    pack_record_data,
    pack_record_header,
    write_product,
)

__all__ = [
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
]
