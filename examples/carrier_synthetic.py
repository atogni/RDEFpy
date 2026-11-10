"""Extract the carrier frequency of a synthetic tone file, and compare with the truth.

Builds three one-second records of a complex tone with a known baseband frequency,
quantized to the 2k+1 codes of 16-bit samples (§5.3.6), writes them to a temporary
product file and runs extract_carrier on it.

    python examples/carrier_synthetic.py
"""

import tempfile
from pathlib import Path

import numpy as np

from rdefpy import (
    ProductFile, Record, RecordData, RecordHeader, extract_carrier, read_product, write_product,
)

FS = 1000
F_TONE = 123.4567  # baseband frequency [Hz]
FIXED = 8.4e9 + 5.0e6  # rf_to_if + if_to_channel downconversion [Hz]

pf = ProductFile(name="synthetic")
for s in range(3):
    t = s + np.arange(FS) / FS
    phase = 2 * np.pi * F_TONE * t
    # 2k+1 values: odd integers, here 2*floor(v/2) + 1
    i, q = (2 * np.floor(12000 * f(phase) / 2).astype(int) + 1 for f in (np.cos, np.sin))
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
    path = Path(tmp) / "tone.prd"
    write_product(path, pf)
    c = extract_carrier(read_product(path), integration_time=1.0, max_rate_hz_s=0.0)

truth = FIXED + F_TONE
print(f"truth frequency before downconversion {truth:.6f} Hz (baseband {F_TONE} Hz)")
for time, f, base in zip(c["time"], c["frequency_before_downconversion"], c["frequency"]):
    print(f"{time}  before DC {f:.6f} Hz  error {(f - truth) * 1e3:+.3f} mHz  baseband {base:.6f} Hz")
