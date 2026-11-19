Changelog
=========

1.2.1 (2026-11-19)
------------------

Internal changes to reading and carrier extraction; same results, bit for
bit, and same API.

Carrier extraction
   * ``extract_carrier``: blocks of up to 1024 samples are estimated up to
     32 per task, sharing the kernel and FFT calls. ``memory_limit`` bounds
     the tasks in flight.
   * ``extract_carrier``: when a block raises, tasks not yet started are
     cancelled and running ones are waited for.
   * Compiled kernels: SPI evaluates its three stencil points in one pass;
     the FrFT bank is filled column by column, without a scratch array.

Product files (§5)
   * Read-ahead from data sections of 256 KiB (was 512 KiB).
   * Decoded records build their ``RecordData`` without re-checking it;
     records of 501 to 16384 complex samples are built by the compiled
     kernel.
   * Header checks: the file-constant fields in one pass per field; the
     NaN / Inf check ends early when every float field is finite.

1.2.0 (2026-11-17)
------------------

Time-tag and t0_offset warnings on read, and a faster browser with a
frequency-resolution box.

Product files (§5)
   * Reading warns (``RuntimeWarning``, once per kind) for time tags not
     increasing, picoseconds differing between records, a year outside
     1678-2261, day 366 of a non-leap year and second of day 86400, and
     for a ``t0_offset`` of more than 1 s.
   * ``info()``: new rows ``gaps`` and ``invalid_records``.

Browsing
   * ``browse``: the checkbox is now "Compute carrier": unticked,
     ``extract_carrier`` is not run and no carrier is drawn, for faster
     browsing.
   * ``browse``: new "df [Hz]" box: spectrum and spectrogram resolution, by
     averaging the power of consecutive 1/df-s segments (all samples used).
     Blank: 1/T, as before.
   * ``browse``: clearer title, legend ("carrier", "downconversion, variable
     part") and y label in the top middle panel.

Documentation
   * Carrier extraction also cites Bertotti, Iess & Tortora (2003), the
     original presentation of SPI.

1.1.0 (2026-11-10)
------------------

Bug fixes, faster reading, browse improvements and shorter documentation.
Tests and documentation examples use synthetic values.

Fixes
   * ``RecordData``: Q set without I is an error, not an empty record, also
     for ``len``, ``duration_s``, ``time_array`` and writing; I without Q too.
     ``repr`` and ``str`` of such a record show both counts (``I 2 / Q 0``).
   * ``concatenate``: a file joined with itself gives independent records.
   * ``extract_carrier``: ``start_epoch`` takes the forms ``read_product`` does.
   * ``extract_carrier`` / ``frft_frequency``: the compiled final refinement
     no longer stops early (it could end slightly off the NumPy path).
   * Epoch strings keep digits below one microsecond.
   * CCSDS day-of-year epoch strings accept a fraction of a second
     (``"2026-039T12:00:00.25"``).
   * Progress bar full when the file ends early.
   * ``browse``: bad Epoch, T or Blocks values show a message, not an error.
   * ``browse``: zero pad 1 to 10.

Product files (§5)
   * Faster reading of records from 512 KiB (read-ahead, was 8 MiB).

Reference files
   * ``tests/reference/steered_ps0.*`` and ``dsn_agency_block.*`` changed:
     16-bit samples (were 8-bit), new I/Q codes, new steering (c2 and c3
     of the phase cubic, so the phase fields of later records too), and in
     ``dsn_agency_block`` new uplink DSS ID and channel number; still 1 kHz,
     three records. A decoder checked against the 1.0.0 files sees different
     bytes.

Browsing
   * ``browse``: separate Periodogram and Spectrogram limits, a colour bar,
     a checkbox to hide the carrier on the spectrogram, and Tab between
     cells.
   * ``browse``: the "x lim [Hz]", "y lim [dB]", "Carrier t lim [s]", "Top
     y lim [Hz]" and "Spectrogram y [Hz]" boxes are removed; limits are now
     separate f min / f max and P min / P max cells.

Documentation
   * Rewritten and shortened: each page gives a description, its API and one
     example; each topic is explained on one page and linked from the others.
   * README shortened; the benchmark figures moved to :doc:`guide/performance`.
   * API names no longer shown in red; ``agency_fields`` and
     ``DSN_AGENCY_DTYPE`` mentions are links.

1.0.0 (2026-11-10)
------------------

Initial release.
