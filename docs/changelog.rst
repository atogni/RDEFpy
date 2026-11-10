Changelog
=========

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
