# RDEFpy

Open-source Python reader and writer for the product files of the CCSDS
Delta-DOR Raw Data Exchange Format (RDEF), CCSDS 506.1-B-1 (June 2013), also
published as ISO 20208:2015. It covers the standard's §5: binary one-second
records of complex I/Q with full downconversion metadata, as recorded by
open-loop receivers for radio science and Delta-DOR. The standard's ASCII
observation files (§4) are currently partially supported.

## Introduction

Open-loop recordings of spacecraft signals serve radio-science experiments,
radiometric tracking and Delta-DOR (delta differential one-way ranging), a
navigation technique derived from very long baseline interferometry (VLBI).
The Consultative Committee for Space Data Systems (CCSDS) defines a common
format for these recordings, the Delta-DOR Raw Data Exchange Format (RDEF):
one-second binary records of complex in-phase/quadrature (I/Q) samples with
their downconversion metadata, required when stations of different agencies
share an observation. To our knowledge no licensed open-source library reads
and writes RDEF, so each station or research group builds and validates its
own; VLBI stations, which record in the VLBI Data Interchange Format (VDIF),
face this before joining a navigation campaign. RDEFpy is a Python reader and
writer for RDEF product files that depends on NumPy alone and decodes every
sample size the standard defines, with tests pinning each against the
standard's bit layout. On a laptop workstation it read recordings about
72,000 times faster than real time for 1-kHz, 16-bit data (6 h), about 3,100
times for 100-kHz, 16-bit data (2 h, 2.9 GB), and about 9 times for 100-MHz,
2-bit wideband data (60 s, 3.0 GB). Records above 1 kHz stay on disk until
accessed, so memory stays bounded on multi-gigabyte files. Stations and
research groups can produce and consume RDEF without in-house format
software, and a VLBI station can add it to a VDIF-based workflow without
changing its recording or correlation pipeline.

## Features

- Reads and writes RDEF product files (`.prd`).
- Decodes every record header field and the I/Q samples of all supported sample sizes.
- Handles multi-gigabyte files: records at or below 1 kHz are loaded;
  faster records stay on disk and are read on access, one at a time.
- Depends on NumPy alone. An optional Cython extension speeds up I/Q
  unpacking; without a compiler the package falls back to an equivalent
  NumPy path.

## Installation

TBD.

## Quick start

TBD.

## Testing

```
uv run python -m pytest
```

The tests pin the low-level decoding for every sample size against the
standard's bit layout.

## Contributing

Issues and pull requests are welcome. The low-level decoding (the `2k+1`
transform of §5.3.6, the bit offsets, the header struct) is the format itself
and is meant to stay as it is; changes there need a reference to the standard.
Performance improvements in decoding, reading, or processing data are welcome.

## Publications using RDEFpy

- Togni, A., Malphrus, B., Arnold, B., Mayo, D., Sanchez-Net, M., & Asmar, S.
  (2026). Extending VLBI Station Capabilities for Delta-DOR and Navigation via
  Open-Source RDEF Software Compliant with CCSDS Standards. Poster, 14th
  General Meeting of the International VLBI Service for Geodesy and Astrometry
  (IVS GM2026), Würzburg.
  https://indico.astro.uni-wuerzburg.de/event/2/contributions/143/

## License

MIT. See `LICENSE`.
