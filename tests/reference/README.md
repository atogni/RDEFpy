# Sample product files and their decoded versions

Small RDEF product files with text versions of their contents, for checking
any RDEF decoder (another library, another version of RDEFpy) against the
same answer. One set per sample size: `sample_16bit`, `sample_8bit`,
`sample_4bit`, `sample_2bit` and `sample_1bit`, each with four files:

| File | Contents |
|---|---|
| `.prd` | The product file itself (binary, as recorded). |
| `.hdr` | Every header field of every record (CCSDS 506.1-B-1 Table 5-1). Floats are written so they read back to the same 64-bit value. |
| `.tab` | The decoded samples. One line per 10 samples: record number (from 1), index of the first sample in the record (from 0), then 10 I/Q pairs `I Q I Q ...`. Values are the §5.3.6 sample values 2k+1. |
| `.hex` | The whole file as bytes: offset in hex, then 16 bytes per line. |

The files are synthetic: spacecraft 1, station DS24, 2 kHz, two one-second
records each. They are copies of the test files in `tests/data/`, which keep
the standard's file names (§6).

## Downconversion and agency-block cases

Two more synthetic files, built record by record by `tests/rdef_spec.py`
(not copies of `tests/data/`): 16 bits, 1 kHz, three one-second records, the
same I/Q code pattern in each, shifted by a few codes per record. Each has the four files above plus `.dc`;
`dsn_agency_block` also has `.agency`. `steered_ps0` has `agency_flag` 0.

Both have the same steering: c1, c2 and c3 nonzero, with phase and rate
continuous from one record to the next (the Table 5-1 note on the phase
fields). They differ in TIME TAG PICOSECONDS OF THE SECOND:

| Case | Picoseconds | What it checks |
|---|---|---|
| `steered_ps0` | 0 | First sample on the second boundary. |
| `dsn_agency_block` | 50,000 | First sample 50 ns after the second boundary, inside the 0 to 100,000 ps that DSN 0222 Table 3-1 allows. `agency_flag` 3 and the 40-byte internal-agency block filled with the DSN 820-013 0222-Science Rev. A layout (Table 3-1 / Figure 3-4), every field nonzero and distinct, spare bytes zero. The block is header bytes 132 to 171 counted from 0; 0222 numbers bytes from 1, so it calls them 133 to 172. |

| File | Contents |
|---|---|
| `.dc` | Per record: the total downconversion frequency at the first sample (Hz), its mean over [first sample, first sample + 1 s) (Hz), and the variable-part phase at the first sample (turns), §5.2.6, with t0 = TIME TAG SECOND OF DAY (CCSDS 506.1-B-1 Table 5-1 and §5.2.6.2; DSN 0222 Table 3-1 and §3.4). |
| `.agency` | Every record's internal-agency block decoded field by field with the DSN 0222 layout. |

The 50,000 ps of `dsn_agency_block` show in the `.dc` phase column (c1 x 50 ns,
about 6.2e-5 turns more than `steered_ps0`). Its frequency columns match
`steered_ps0` to the last float64 step: the shift, about 2 c2 x 50 ns = 1.3e-8 Hz,
is below the 9.5e-7 Hz step at 8.4 GHz.

The text files are written by `tests/make_reference.py` from
`tests/rdef_spec.py`, a plain transcription of the standard that does not use
RDEFpy. RDEFpy is tested against them like any other decoder
(`tests/test_reference.py`). To regenerate:

    python tests/make_reference.py
