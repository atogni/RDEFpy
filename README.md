# RDEFpy

[![Documentation Status](https://img.shields.io/readthedocs/rdefpy/latest)](https://rdefpy.readthedocs.io/en/latest/)

Open-source Python reader and writer for the product files of the CCSDS
Delta-DOR Raw Data Exchange Format (RDEF), CCSDS 506.1-B-1 (June 2013), also
published as ISO 20208:2015: binary one-second records of complex I/Q with
full downconversion metadata (§5), as recorded by open-loop receivers for
radio science and Delta-DOR. The ASCII observation files (§4) are read and
validated, not written.

- Reads and writes product files (`.prd`): every header field and the I/Q
  samples of all sample sizes (1, 2, 4, 8, 16 bits), returned as recorded.
- Handles multi-gigabyte files: records above 1 kHz stay on disk and are
  read on access, one at a time.
- Extracts the carrier frequency before downconversion per integration
  time: fractional Fourier transform plus Successive Parabolic
  Interpolation (Togni et al., 2025, Signal Processing 237, 110094).
- Depends on NumPy alone. An optional Cython extension speeds up decoding
  and carrier extraction; without it an equivalent NumPy path runs.

On a laptop it reads and decodes recordings about 68,000 times faster than
real time for 1-kHz, 16-bit data (6 h), about 1,600 times for 100-kHz,
16-bit data (2 h, 2.9 GB), and about 60 times for 100-MHz, 2-bit wideband
data (60 s, 3.0 GB) (RDEFpy 1.1.0, Cython extension; see
[benchmarks](https://rdefpy.readthedocs.io/en/latest/guide/performance.html)).

**Documentation: https://rdefpy.readthedocs.io/en/latest/**
([quick start](https://rdefpy.readthedocs.io/en/latest/quickstart.html),
user guide, [benchmarks](https://rdefpy.readthedocs.io/en/latest/guide/performance.html),
API reference).

## Installation

Python 3.11 or later and NumPy.

```
pip install rdefpy
pip install "rdefpy[browse]"    # with matplotlib, for rdefpy.browse
```

The prebuilt wheels include the Cython extension; a source build needs a C
compiler, or installs the NumPy path only.

## Example

```python
from rdefpy import read_product, write_product, extract_carrier

pf = read_product("scan.prd")  # records above 1 kHz stay on disk until accessed
pf.info()                      # file summary: time span, rate, bits, band

for rec in pf:                 # one one-second record at a time
    d = rec.data               # read and decoded now
    print(rec.header.timetag, d.power_db)

x = pf.samples[:1024]                    # complex I + jQ, across records
t = pf.timetags[:1024]                   # datetime64[ns] of each sample
f = pf.downconversion_frequency[:1024]   # Hz, at each sample (§5.2.6)

c = extract_carrier(pf, integration_time=1.0, max_rate_hz_s=10.0)
c["time"], c["frequency_before_downconversion"]   # per block; time = block midpoint

part = pf.select(start_epoch="2026-02-08T12:00:00", stop_epoch="2026-02-08T12:00:10")
write_product("part.prd", part)
```

The tests (`uv run python -m pytest`) pin the decoding of every sample size
against the standard's bit layout; see
[checking another decoder](https://rdefpy.readthedocs.io/en/latest/reference_files.html).

## Citing

Togni, A., Malphrus, B., Arnold, B., Mayo, D., Sanchez-Net, M., & Asmar, S.
(2026). Extending VLBI Station Capabilities for Delta-DOR and Navigation via
Open-Source RDEF Software Compliant with CCSDS Standards. Poster, 14th
General Meeting of the International VLBI Service for Geodesy and Astrometry
(IVS GM2026), Würzburg.
https://indico.astro.uni-wuerzburg.de/event/2/contributions/143/

## License

MIT. See `LICENSE`. Issues and pull requests are welcome.
