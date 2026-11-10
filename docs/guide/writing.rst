Writing product files
=====================

:func:`rdefpy.write_product` writes a :class:`rdefpy.ProductFile` to a
``.prd`` file, records in list order. It never writes a value the
specification rules out and never alters a sample: anything that has no
on-disk code raises instead of being clipped or rounded.

A record is a :class:`rdefpy.RecordHeader` plus a :class:`rdefpy.RecordData`.
Every header field except ``record_length`` and the two reserved blocks must
be set. Samples are the ``2k+1`` integers of §5.3.6, odd values within
``±(2**sample_size - 1)``: quantize before building the record. A record
holds one second, so the number of samples equals ``sample_rate``.

.. code-block:: python

   import numpy as np
   from rdefpy import ProductFile, Record, RecordData, RecordHeader, write_product

   fs = 1000
   fields = dict(
       record_version_id=1, station_id=24, spacecraft_id=1,
       sample_size=16, sample_rate=fs, validity_flag=0, agency_flag=0,
       rf_to_if_downconv=8.4e9, if_to_channel_downconv=5e6,
       time_tag_year=2026, time_tag_doy=39, time_tag_second_of_day=43200,
       time_tag_picoseconds_of_the_second=0.0,
       channel_accumulated_phase=0.0,
       channel_phase_polynomial_coefficient_0=0.0,
       channel_phase_polynomial_coefficient_1=0.0,
       channel_phase_polynomial_coefficient_2=0.0,
       channel_phase_polynomial_coefficient_3=0.0,
   )
   rng = np.random.default_rng(1)

   pf = ProductFile(name="synthetic")
   for k in range(3):                              # three consecutive seconds
       header = RecordHeader(fields)
       header.time_tag_second_of_day += k
       i = 2 * rng.integers(-1000, 1000, fs).astype(np.int32) + 1
       q = 2 * rng.integers(-1000, 1000, fs).astype(np.int32) + 1
       pf.records.append(Record(header, RecordData(16, fs, i, q)))
   write_product("synthetic.prd", pf)

The phase fields describe the variable part of the downconversion, referred
to TIME TAG SECOND OF DAY (:doc:`downconversion`). Time tags are station
time (§5.2.5).

Editing and round trip
----------------------

* To change samples of a record on disk, ``pf.load()`` first
  (:doc:`product_files`), or assign a new :class:`rdefpy.RecordData` to
  ``record.data``.
* ``correct_header(field=value)``, on a record or on the whole file, edits
  header fields. The 40-byte internal-agency block is written as given; for
  the DSN layout pass the bytes of one row of :data:`rdefpy.DSN_AGENCY_DTYPE`
  as ``empty_fields_internal_agency_use``.
* To cut a window out of a recording, ``write_product("cut.prd",
  pf.select(start_epoch=...))``.
* For a conforming file, ``write_product(read_product(f))`` reproduces
  ``f`` byte for byte, for all five sample sizes. Two exceptions: -0.0 in a
  float field (excluded by §5.2.4) is read with a warning and written as
  +0.0, and :math:`c_1` to :math:`c_3` all NaN are each written as
  ``7fffffffffffffff`` (:doc:`downconversion`).
* The write is atomic: a failure leaves an existing target untouched, and
  writing over the file the records are read from is safe.

What the writer checks
----------------------

A failure raises :class:`ValueError` (as does :class:`~rdefpy.RecordData`
for samples that do not fit the integer type of their sample size):

* the product has records, and every record has data (a header-only product
  from :func:`rdefpy.read_headers` is refused) of ``sample_rate`` samples at
  the header's sample size;
* every required header field is set; integer fields are integers within
  their Table 5-1 width (4 bytes for ``sample_rate`` and
  ``time_tag_second_of_day``, 2 bytes for the others);
* the value ranges the reader only warns about (:doc:`product_files`), so a
  file that was out of range on read must be corrected first;
* ``sample_size`` is 1, 2, 4, 8 or 16 and ``2 * sample_rate * sample_size``
  a multiple of 32; a ``record_length`` set in the header equals the
  computed one;
* samples are valid ``2k+1`` values for the sample size.

:func:`rdefpy.pack_record_header` and :func:`rdefpy.pack_record_data`, which
``write_product`` uses, return the 176 header bytes and the packed data
section (§5.3) of one record.

Not supported: writing observation files, records that are not one second
long, and quantization, clipping or scaling of samples.
