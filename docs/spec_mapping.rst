Standard conformance
====================

RDEFpy implements CCSDS 506.1-B-1 (June 2013), the Delta-DOR Raw Data
Exchange Format, also published as ISO 20208:2015: the binary product files
(§5) and the ASCII observation files (§4). Where DSN's version of the
product record, DSN 820-013 module 0222-Science [DSN0222]_, says more, the
guide pages cite it.

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Standard
     - Content
     - RDEFpy
   * - §3.2.2.2-4, §5.1.3
     - A product file holds one station, one scan, one channel.
     - ``read_product`` warns (``RuntimeWarning``) when station, spacecraft,
       agency flag, version, sample size, sample rate or a downconversion
       frequency varies between records. Scan and channel are not in the
       header and are not checked.
   * - §4.1
     - Observation file: line rules and sections ending in a ``Z`` line.
     - :func:`rdefpy.read_observation`. Strict: a breach raises
       ``ValueError`` naming file and line.
   * - §4.2
     - Observation header: version, receive station, optional transmit
       station.
     - :class:`rdefpy.ObservationFile`.
   * - §4.3, Table 4-1
     - Scan line: scan number, source, start and stop time, RA, DEC, TFREQ.
     - :class:`rdefpy.Scan`.
   * - §4.3, Table 4-2
     - Product file line: file name, coherence flag, DOR_MULT, FSUB,
       harmonic.
     - :class:`rdefpy.ProductEntry`.
   * - §4.4
     - Ending section: log lines (``F``), end line ``E *=END=*``.
     - ``ObservationFile.logs``; the end line is required.
   * - §5.1.1, §5.1.2
     - A product file is a run of records, each one second of data, a header
       section and a data section.
     - :class:`rdefpy.ProductFile` (a list of :class:`rdefpy.Record`),
       :func:`rdefpy.read_product`, :func:`rdefpy.read_headers`,
       :func:`rdefpy.scan_product`, :func:`rdefpy.write_product`.
   * - §5.1.4
     - Data section length is fixed by the header (sample rate and size).
     - ``record_length`` is checked against the header; a truncated record
       raises ``EOFError``.
   * - §5.1.5
     - Little-endian multi-byte values.
     - Header struct and sample unpacking; written the same way.
   * - §5.2, Table 5-1
     - The 176-byte record header.
     - :class:`rdefpy.RecordHeader`, :func:`rdefpy.pack_record_header`.
       Label, end label, sample size, word alignment and record length are
       checked on read; the writer checks field ranges and sample parameters.
   * - §5.2.3, §5.2.4
     - IEEE binary64; NaN, infinities and -0 not supported.
     - Warned on read (``UserWarning``, record read as is), refused on
       write; -0 is read, and written, as +0. Exception: :math:`c_1`,
       :math:`c_2` and :math:`c_3` all NaN (:doc:`guide/downconversion`).
   * - §5.2.5
     - Time tags approximate UTC as realized by station time.
     - ``RecordHeader.timetag``: ``datetime64[ns]`` of the record's first
       sample (:doc:`guide/samples_and_time`). Not converted.
   * - §5.2.6
     - Downconversion: a fixed frequency plus a variable one, a cubic phase
       polynomial in turns from :math:`t_0`, TIME TAG SECOND OF DAY.
     - :doc:`guide/downconversion`: ``downconversion_frequency``,
       ``downconversion_phase`` (the variable part only),
       ``record_downconversion_frequency``, ``mean_downconversion_frequency``.
   * - §5.3.1-§5.3.4, Table 5-2
     - I and Q adjacent, packed into 32-bit words from LSB to MSB, 1, 2, 4, 8
       or 16 bits per component.
     - :class:`rdefpy.RecordData`, ``Record.read_samples``,
       ``ProductFile.samples``; :func:`rdefpy.pack_record_data` on write.
       All five sample sizes, with a Cython and a NumPy decoder.
   * - §5.3.5, §5.3.6
     - Truncation leaves an offset of -0.5; every component is mapped to
       2k+1, k the two's complement code.
     - Samples are the 2k+1 values, nothing else done to them. The writer
       accepts only valid 2k+1 values.
   * - §6.2
     - File names ``MMMMnNNNtTsSSSSrRRcCC-YYDDDHHMMSS.XXX``.
     - Not parsed or checked by the library.

Not supported
-------------

* Writing observation files.
* Checks across files: the observation file's version against the product
  files' (§4.2.4), file names against §6.2.2 or the file contents, and that
  the product files listed in an observation file exist.
* Leap seconds: ``hh:mm:60`` in an observation file is rejected, and a
  product record with second of day 86400 gets the ``timetag`` of second 0
  of the next day (:doc:`guide/samples_and_time`).
* Time scale conversion of the time tags.

Conformance tests
-----------------

Run the tests with ``uv run python -m pytest``. ``tests/rdef_spec.py`` is a
plain transcription of the standard (Table 5-1 layout, Table 5-2 packing,
§5.3.6, §5.2.6 downconversion, §6.2.2, and the DSN 0222 internal-agency
layout) that does not import RDEFpy, so a change to the library cannot move
the expectation with it. The spec tests (``tests/test_spec_*.py``) run the
decode tests once per decoder, Cython and NumPy. ``tests/reference/`` holds
small product files with their expected decoding as text, written by the
same transcription: see :doc:`reference_files`.
