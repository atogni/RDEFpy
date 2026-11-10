"""Work through a big file without loading it: header-only scans, lazy iteration, threads.

Writes ten seconds of 100 kHz, 16-bit noise (4 MB) to a temporary file. The calls are
the same for gigabyte files: scan_product and read_headers never touch the samples,
and iterating a product file decodes one record at a time.

    python examples/stream_large_file.py
"""

import tempfile
from pathlib import Path

import numpy as np

from rdefpy import (
    ProductFile, Record, RecordData, RecordHeader, read_headers, read_product, scan_product, set_threads, write_product,
)

FS = 100_000
rng = np.random.default_rng(1)

pf = ProductFile(name="noise")
for s in range(10):
    i, q = (2 * rng.integers(-8000, 8000, FS) + 1 for _ in range(2))  # 2k+1 codes (§5.3.6)
    header = RecordHeader(dict(
        record_version_id=1, station_id=24, spacecraft_id=1,
        sample_size=16, sample_rate=FS, validity_flag=0, agency_flag=0,
        rf_to_if_downconv=8.4e9, if_to_channel_downconv=5.0e6,
        time_tag_year=2026, time_tag_doy=39, time_tag_second_of_day=43200 + s,
        time_tag_picoseconds_of_the_second=0.0, channel_accumulated_phase=0.0,
        channel_phase_polynomial_coefficient_0=0.0, channel_phase_polynomial_coefficient_1=0.0,
        channel_phase_polynomial_coefficient_2=0.0, channel_phase_polynomial_coefficient_3=0.0,
    ))
    pf.records.append(Record(header, RecordData(16, FS, i, q)))

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "big.prd"
    write_product(path, pf)

    count, consistent = scan_product(path)  # reads 8 bytes per record
    print(f"scan_product: {count} records, consistent length: {consistent}")

    hdrs = read_headers(path)  # every header field, no samples
    print(f"read_headers: {hdrs[0].header.timetag} to {hdrs[-1].header.timetag}")

    set_threads(4)  # threads per read; set_threads(None) restores the default
    lazy = read_product(path)  # 100 kHz records stay on disk
    buf = None
    for rec in lazy:  # one record in memory at a time
        buf = rec.read_samples(out=buf)
        print(f"{rec.header.timetag}  {10 * np.log10(np.mean(np.abs(buf) ** 2)):.2f} dB")
    set_threads(None)
