Product files
=============

A product file (``.prd``) holds the open-loop recording of one scan, one
channel and one station (§5.1.3). It is a run of **records**, each exactly
one second of data (§5.1.1):

* a 176-byte **header** (Table 5-1): time tag, station and spacecraft IDs,
  sample size and rate, downconversion;
* a **data section** of complex I/Q samples at 1, 2, 4, 8 or 16 bits per
  component.

This page covers reading. See :doc:`samples_and_time` for the samples and
their times, :doc:`downconversion` for the downconversion, and
:doc:`writing` for writing.

.. code-block:: python

   from rdefpy import read_product, read_headers, scan_product

   f = "EXMPn001tSsDS24r01c01-26039120000.prd"
   pf = read_product(f)
   pf.info()                            # same text as str(pf)

   for rec in pf:                       # one one-second record at a time
       d = rec.data                     # read and decoded now: take it once
       print(rec.header.timetag, len(rec), d.power_db)

   sub = pf.select(start_epoch="2026-039T12:00:01")   # a new ProductFile
   hd = read_headers(f)                 # headers only, no samples
   print(len(sub), len(hd), scan_product(f))

.. code-block:: text

   ProductFile EXMPn001tSsDS24r01c01-26039120000
                 path: ...\tests\data\EXMPn001tSsDS24r01c01-26039120000.prd
              records: 2
        first_timetag: 2026-02-08T12:00:00.000000000Z
         last_timetag: 2026-02-08T12:00:01.000000000Z
          consecutive: True
          sample_rate: 2,000 Hz
          sample_size: 16 bit
        spacecraft_id: 1
           station_id: 24
                 band: X
   frequency_steering: False
   2026-02-08T12:00:00.000000000 2000 96.32945284169084
   2026-02-08T12:00:01.000000000 2000 96.32945284169084
   1 2 (2, True)

``consecutive`` is True when the records are 1 s apart (within 1 µs). ``band``
classifies the fixed downconversion into the nominal S, X and Ka downlink
allocations, or ``unknown``.

Lazy loading
------------

:func:`rdefpy.read_product` reads every header but loads samples only for
records **at or below 1 kHz**. Faster records stay on disk:

* ``record.data`` on an unloaded record reads and decodes the samples.
  Nothing keeps them, so a loop over a multi-gigabyte file holds one record
  at a time. Take ``d = record.data`` once and reuse it: each access that
  finds no live result reads the file again.
* ``record.load()`` / ``pf.load()`` keep the samples until ``release()``,
  which also discards in-memory edits.
* Arrays from an unloaded record are read-only. To edit samples, ``load()``
  first or assign a new :class:`rdefpy.RecordData` to ``record.data``.
* The file must stay in place and unchanged while its records are used:
  reading raises :class:`RuntimeError` if its size or modification time
  changed.

See :doc:`performance` for threads and large files.

Options of read_product
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Argument
     - Meaning
   * - ``num_records``
     - Read only the first *N* records. With filters active it caps the
       records examined, not the number returned.
   * - ``start_epoch``, ``stop_epoch``
     - Keep records with ``start_epoch <= timetag < stop_epoch``. Accepts
       ``datetime`` (naive means UTC), ``numpy.datetime64``, or an ISO 8601
       or CCSDS day-of-year string such as ``"2026-039T12:00:00"``.
   * - ``filters``
     - Dict of header field names and expected values, for example
       ``{"station_id": 24}``. A record must match all.
   * - ``min_downconv_hz``, ``max_downconv_hz``
     - Keep records whose total downconversion frequency at the first sample
       lies in the window.
   * - ``band``
     - ``"S"``, ``"X"`` or ``"Ka"``: the nominal downlink allocation widened
       by 1 GHz on each edge. Not with ``min_downconv_hz`` /
       ``max_downconv_hz``.
   * - ``dtype``
     - ``np.complex128`` (default) or ``np.complex64`` for the complex
       samples. Both are exact; ``complex64`` halves the memory.
   * - ``t0_offset``
     - Rarely needed: moves the downconversion polynomial origin. See
       :doc:`downconversion`.
   * - ``agency_layout``
     - Layout of the internal-agency block; see `Internal-agency block`_.
   * - ``progress``
     - Draw a progress bar on stderr.

Filters look at the header only, so a skipped record's data is never read.
Bad arguments raise :class:`ValueError`.

:func:`rdefpy.read_headers` takes the same file and skips every data
section. :func:`rdefpy.scan_product` reads only the 8-byte label and length
of each record and returns ``(count, consistent_length)``; it checks nothing
else.

The objects
-----------

:class:`rdefpy.ProductFile`
   A sequence of records (``len``, indexing, iteration; ``pf.records`` is
   the list) with ``name``, ``path`` and the read-only ``t0_offset`` and
   ``agency_layout``. Header-only arrays, one entry per record:
   ``timetag_array``, ``record_downconversion_frequency``,
   ``mean_downconversion_frequency``. Whole-file views, indexed like one
   array across records: ``samples``, ``timetags``,
   ``downconversion_frequency``, ``downconversion_phase``.

   ``select`` (``filters``, ``min_downconv_hz`` / ``max_downconv_hz`` and
   ``start_epoch`` / ``stop_epoch``, as in ``read_product``), ``slice`` (by index),
   ``copy`` and ``concatenate`` return a new ``ProductFile`` of independent
   records; unloaded copies still read from the original file.
   ``pf_a.concatenate(pf_b, keep="other", sort=True)`` merges two files of
   the same channel; ``keep`` (``"other"``, ``"self"``, ``"both"``) decides
   which record survives a duplicate time tag, with a
   :class:`RuntimeWarning`. Files of different channels raise
   :class:`ValueError`. ``dump_headers`` and ``dump`` write text.

:class:`rdefpy.Record`
   A ``header`` and its ``data``. ``len(record)`` is the number of complex
   samples and does not read the disk. ``read_samples`` decodes straight to
   complex (:doc:`samples_and_time`).

:class:`rdefpy.RecordHeader`
   One attribute per field of Table 5-1, listed in the API reference, plus
   ``timetag``, the ``numpy.datetime64[ns]`` of the record's first sample
   (:doc:`samples_and_time`). Headers are mutable: assign a field, or use
   ``correct_header(field=value)`` on a record or on the whole file, which
   checks the field names. Values are validated on write.

:class:`rdefpy.RecordData`
   ``i_samples``, ``q_samples`` (the ``2k+1`` integers) and ``samples``
   (``I + jQ``, built anew on each access). ``power``, ``power_db``, ``rms``
   and ``full_scale_fraction`` are level checks in ``2k+1`` units, not a
   calibrated power.

Records, headers and product files can be pickled and deep-copied; only
loaded samples travel with the copy.

Internal-agency block
---------------------

Header bytes 132 to 171 (from 0), ``empty_fields_internal_agency_use``, are
for internal agency use: Table 5-1 gives them no layout.
``pf.agency_fields()`` decodes them into a read-only structured array, one
row per record, raw values. The layout is set when reading, with
``agency_layout``:

* ``"auto"`` (default): the DSN layout below when ``agency_flag`` is 3
  (NASA), else one field ``"raw"`` of type ``V40``. If the records disagree
  on ``agency_flag``, raw bytes and a :class:`RuntimeWarning`.
* ``"dsn"``: the DSN layout, whatever the ``agency_flag``.
* any 40-byte dtype without object fields, as ``np.dtype()`` accepts it.

The DSN layout, :data:`rdefpy.DSN_AGENCY_DTYPE`, is that of the DSN Open Loop
Receiver (OLR), little-endian, from DSN 820-013 0222-Science Rev. A,
Table 3-1 and Figure 3-4 [DSN0222]_. ``agency_flag`` 3 means NASA, not
necessarily the OLR: read a file from another recorder with its own layout.

.. list-table::
   :header-rows: 1
   :widths: 8 34 8 50

   * - Offset
     - Field
     - Type
     - Meaning
   * - 0
     - ``predict_pass_number``
     - ``<u2``
     - DSN pass number from the frequency predict file.
   * - 2
     - ``uplink_band``
     - ``u1``
     - 0 unknown or not applicable, 1 S, 2 X, 3 Ka, 4 Ku, 5 L.
   * - 3
     - ``downlink_band``
     - ``u1``
     - 0 unknown, 1 S, 2 X, 3 Ka, 4 Ku, 5 L.
   * - 4
     - ``track_mode``
     - ``u1``
     - 1, 2 or 3; 4 is allowed if the signal is a relay.
   * - 5
     - ``uplink_dss_id``
     - ``u1``
     - Uplink station.
   * - 6
     - ``olr_id``
     - ``u1``
     - 31 is OLR1 ... 38 is OLR8.
   * - 7
     - ``olr_software_version``
     - ``u1``
     - 1 is the initial version.
   * - 8
     - ``channel_power_calibration_factor``
     - ``<f4``
     - Conversion from dB full scale (dBfs) to dBm of the channel at the
       input to the A/D converter (IFD).
   * - 12
     - ``total_frequency_offset``
     - ``<f8``
     - Offset of the channel centre frequency from the predict, Hz.
   * - 20
     - ``channel_number``
     - ``u1``
     - Physical channel number, 0 to 127.
   * - 21
     - ``spare``
     - ``V19``
     - Internal agency use.

Validation on read
------------------

**Errors** (:class:`ValueError` unless noted), when a record cannot be
parsed or the next one cannot be found:

* wrong record label (not ``RDEF``) or end label (not -99999);
* sample size not in 1, 2, 4, 8, 16;
* ``2 * sample_rate * sample_size`` not a multiple of 32;
* ``record_length`` not equal to ``176 + 2 * sample_rate * sample_size / 8``;
* a file that ends inside a header or a data section (:class:`EOFError`).

**Warnings** (:class:`UserWarning`, the record is read as is), for a value
outside its Table 5-1 range; ``write_product`` refuses the same values,
except -0.0 (:doc:`writing`):

* ``record_version_id`` other than 1;
* ``agency_flag`` outside 0 to 3;
* ``time_tag_doy`` outside 1 to 366 or ``time_tag_second_of_day`` outside 0
  to 86400 (the record then has no ``timetag``);
* ``time_tag_picoseconds_of_the_second`` negative or one second or more;
* ``sample_rate`` of 0;
* a float field that is NaN, -Inf or +Inf (§5.2.4), except :math:`c_1`,
  :math:`c_2` and :math:`c_3` all NaN (see :doc:`downconversion`);
* a float field equal to -0.0 (§5.2.4), which is read as +0.0.

A :class:`RuntimeWarning` is raised when station, spacecraft, agency,
version, sample size, sample rate or a fixed downconversion frequency varies
between records.

Not checked: time order and spacing of the records (``info()`` reports
``consecutive``), and the file name against the contents.
